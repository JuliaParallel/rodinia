#!/usr/bin/env julia

using ArgParse
using CUDAdrv, CUDAnative

const OUTPUT = haskey(ENV, "OUTPUT")

# configuration
const BLOCK_SIZE = 16
const MAX_PD = 3.0e6    # maximum power density possible (say 300W for a 10mm x 10mm chip)
const PRECISION = 0.001 # rquired precision in degrees
const SPEC_HEAT_SI = 1.75e6
const K_SI = 100
const FACTOR_CHIP = 0.5 # capacitance fitting factor
const EXPAND_RATE = 2   # add one iteration will extend the pyramid base by 2 per each borderline

# chip parameters
const t_chip = 0.0005f0
const chip_height = 0.016f0
const chip_width = 0.016f0
const amb_temp = 80f0   # ambient temperature, assuming no package at all

function in_range(x, min_x, max_x)
    return x >= min_x && x <= max_x
end

function clamp_range(x, min_x, max_x)
    return (x < min_x) ? min_x : ((x > max_x) ? max_x : x)
end

function writeoutput(vect, grid_rows, grid_cols, file)
    index = 0
    fp = open(file, "w")
    for i = 1:grid_rows, j = 1:grid_cols
        @printf(fp, "%d\t%g\n", index, vect[index + 1])
        index = index + 1
    end
    close(fp)
end

function readinput(vect, grid_rows, grid_cols, file)
    fp = open(file, "r")
    for i = 1:grid_rows, j = 1:grid_cols
        if eof(fp)
            fatal("not enough lines in file")
        end
        val = parse(Float32, chomp(readline(fp)))
        vect[(i - 1) * grid_cols + j] = val
    end
    close(fp)
end

const MATRIX_SIZE = BLOCK_SIZE * BLOCK_SIZE

function calculate_temp(iteration,    # number of iteration
                        power,        # power input
                        temp_src,     # temperature input/output
                        temp_dst,     # temperature input/output
                        grid_cols,    # col of grid
                        grid_rows,    # row of grid
                        border_cols,  # border offset
                        border_rows,  # border offset
                        Cap,          # Capacitance
                        Rx, Ry, Rz, step, time_elapsed)
    temp_on_cuda = @cuStaticSharedMem(Float32, (BLOCK_SIZE, BLOCK_SIZE))
    power_on_cuda = @cuStaticSharedMem(Float32, (BLOCK_SIZE, BLOCK_SIZE))
    # for saving temporary temperature result
    temp_t = @cuStaticSharedMem(Float32, (BLOCK_SIZE, BLOCK_SIZE))

    bx = blockIdx().x - 1
    by = blockIdx().y - 1

    tx = threadIdx().x
    ty = threadIdx().y

    step_div_Cap = step / Cap

    Rx_1 = 1 / Rx
    Ry_1 = 1 / Ry
    Rz_1 = 1 / Rz

    # Each block finally computes the result for a small block after N
    # iterations. It is the non-overlapping small blocks that cover all the
    # input data.
    # calculate the small block size
    small_block_rows = BLOCK_SIZE - iteration * 2
    small_block_cols = BLOCK_SIZE - iteration * 2

    # calculate the boundary for the block according to the boundary of its
    # small block
    blkY = small_block_rows * by - border_rows
    blkX = small_block_cols * bx - border_cols
    blkYmax = blkY + BLOCK_SIZE - 1
    blkXmax = blkX + BLOCK_SIZE - 1

    # calculate the global thread coordination
    yidx = blkY + ty - 1
    xidx = blkX + tx - 1

    # load data if it is within the valid input range
    loadYidx = yidx
    loadXidx = xidx
    index = grid_cols * loadYidx + loadXidx

    if in_range(loadYidx, 0, grid_rows - 1) &&
       in_range(loadXidx, 0, grid_cols - 1)
        # Load the temperature data from global memory to shared memory
        temp_on_cuda[tx, ty] = temp_src[index + 1]
        # Load the power data from global memory to shared memory
        power_on_cuda[tx, ty] = power[index + 1]
    end

    sync_threads()

    # Effective range within this block that falls within the valid range of the
    # input data used to rule out computation outside the boundary.
    validYmin = (blkY < 0) ? -blkY + 1 : 1
    validYmax = (blkYmax > grid_rows - 1) ?
                    BLOCK_SIZE - (blkYmax - grid_rows + 1) : BLOCK_SIZE
    validXmin = (blkX < 0) ? -blkX + 1 : 1
    validXmax = (blkXmax > grid_cols - 1) ?
                    BLOCK_SIZE - (blkXmax - grid_cols + 1) : BLOCK_SIZE

    N = ty - 1
    S = ty + 1
    W = tx - 1
    E = tx + 1

    N = (N < validYmin) ? validYmin : N
    S = (S > validYmax) ? validYmax : S
    W = (W < validXmin) ? validXmin : W
    E = (E > validXmax) ? validXmax : E

    computed = false
    for i = 1:iteration
        computed = false
        if in_range(tx, i + 1, BLOCK_SIZE - i) &&
           in_range(ty, i + 1, BLOCK_SIZE - i) &&
           in_range(tx, validXmin, validXmax) &&
           in_range(ty, validYmin, validYmax)
            computed = true
            t1 = temp_on_cuda[tx, S ] +
                 temp_on_cuda[tx, N ] -
                 temp_on_cuda[tx, ty] * 2.0
            t2 = temp_on_cuda[E , ty] +
                 temp_on_cuda[W , ty] -
                 temp_on_cuda[tx, ty] * 2.0
            temp_t[tx, ty] = Float32(temp_on_cuda[tx, ty] +
                step_div_Cap * (power_on_cuda[tx, ty] + t1 * Ry_1 +
                t2 * Rx_1 + (amb_temp - temp_on_cuda[tx, ty]) * Rz_1))
        end

        sync_threads()
        if i == iteration
            break
        end
        if computed # Assign the computation range
            temp_on_cuda[tx, ty] = temp_t[tx, ty]
        end
        sync_threads()
    end

    # update the global memory
    # after the last iteration, only threads coordinated within the
    # small block perform the calculation and switch on ``computed''
    if computed
        temp_dst[index + 1] = temp_t[tx, ty]
    end

    return nothing
end

# compute N time steps
function compute_tran_temp(MatrixPower, MatrixTemp, col, row, total_iterations,
                           num_iterations, blockCols, blockRows, borderCols,
                           borderRows)
    grid_height = chip_height / row
    grid_width = chip_width / col

    Cap = Float32(FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height)
    Rx = Float32(grid_width / (2.0 * K_SI * t_chip * grid_height))
    Ry = Float32(grid_height / (2.0 * K_SI * t_chip * grid_width))
    Rz = Float32(t_chip / (K_SI * grid_height * grid_width))

    max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI)
    step = Float32(PRECISION / max_slope)
    time_elapsed = 0.001f0
    src = 1
    dst = 0

    for t = 0:num_iterations:total_iterations-1
        temp = src
        src = dst
        dst = temp
        @cuda ((blockCols, blockRows), (BLOCK_SIZE, BLOCK_SIZE)) calculate_temp(
            min(num_iterations, total_iterations - t),
            MatrixPower, MatrixTemp[src + 1], MatrixTemp[dst + 1],
            col, row, borderCols, borderRows, Cap, Rx, Ry, Rz, step, time_elapsed)
    end

    return dst
end

function main(args)
    @printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE)

    s = ArgParseSettings()
    @add_arg_table s begin
        "grid_rows_cols"
            arg_type = Int32
            required = true
            help = "number of rows/cols in the grid (positive integer)"
        "pyramid_height"
            arg_type = Int32
            required = true
            default = Int32(1)
            help = "pyramid height (positive integer)"
        "sim_time"
            arg_type = Int
            required = true
            default = 60
            help = "number of iterations"
        "temp_file"
            required = true
            help = "name of the file containing the initial temperature " *
                "values of each cell"
        "power_file"
            required = true
            help = "name of the file containing the dissipated power values " *
                "of each cell"
        "output_file"
            help = "name of the output file"
    end

    if length(args) != 5
        parse_args(["-h"], s)
    end

    args = parse_args(args, s)

    grid_rows = args["grid_rows_cols"]
    grid_cols = grid_rows
    pyramid_height = args["pyramid_height"]
    total_iterations = args["sim_time"]

    if grid_rows <= 0 || pyramid_height <= 0 || total_iterations <= 0
        parse_args(["-h"], s)
    end

    tfile = args["temp_file"]
    pfile = args["power_file"]
    size = grid_rows * grid_cols

    # --------------- pyramid parameters ---------------
    borderCols = floor(Int32, pyramid_height * EXPAND_RATE / 2)
    borderRows = floor(Int32, pyramid_height * EXPAND_RATE / 2)
    smallBlockCol = BLOCK_SIZE - pyramid_height * EXPAND_RATE
    smallBlockRow = BLOCK_SIZE - pyramid_height * EXPAND_RATE
    blockCols = floor(Int32, grid_cols / smallBlockCol) +
        ((grid_cols % smallBlockCol == 0) ? 0 : 1)
    blockRows = floor(Int32, grid_rows / smallBlockRow) +
        ((grid_rows % smallBlockRow == 0) ? 0 : 1)

    FilesavingTemp = Vector{Float32}(size)
    FilesavingPower = Vector{Float32}(size)
    MatrixOut = Vector{Float32}(size)

    @printf("pyramidHeight: %d\ngridSize: [%d, %d]\nborder:[%d, %d]\n",
        pyramid_height, grid_cols, grid_rows, borderCols, borderRows)
    @printf("blockGrid:[%d, %d]\ntargetBlock:[%d, %d]\n",
        blockCols, blockRows, smallBlockCol, smallBlockRow)

    readinput(FilesavingTemp, grid_rows, grid_cols, tfile)
    readinput(FilesavingPower, grid_rows, grid_cols, pfile)

    MatrixTemp = Array{CuArray{Float32,1}}(2)
    MatrixTemp[1] = CuArray(FilesavingTemp)
    MatrixTemp[2] = CuArray{Float32}(size)
    MatrixPower = CuArray(FilesavingPower)

    println("Start computing the transient temperature")
    ret = compute_tran_temp(MatrixPower, MatrixTemp, grid_cols, grid_rows,
                            total_iterations, pyramid_height, blockCols,
                            blockRows, borderCols, borderRows)
    println("Ending simulation")
    MatrixOut = Array(MatrixTemp[ret + 1])

    if OUTPUT
        writeoutput(MatrixOut, grid_rows, grid_cols, "output.txt")
    end
end


main(ARGS)

if haskey(ENV, "PROFILE")
    CUDAnative.@profile main(ARGS)
end
