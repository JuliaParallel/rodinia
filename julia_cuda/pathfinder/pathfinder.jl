#!/usr/bin/env julia

using CUDAdrv, CUDAnative

include("../../common/julia/crand.jl")
const rng = LibcRNG()

const OUTPUT = haskey(ENV, "OUTPUT")

# configuration
const BLOCK_SIZE = 256
const HALO = 1  # halo width along one direction when advancing to the next iteration


function init(args)
    if length(args) == 3
        cols = parse(Int, args[1])
        rows = parse(Int, args[2])
        pyramid_height = parse(Int, args[3])
    else
        println("Usage: dynproc row_len col_len pyramid_height")
        exit(0)
    end

    srand(rng, 7)

    # Initialize en fill wall
    # Switch semantics of row & col -> easy copy to gpu array in run function
    wall = Array{Int32}(cols, rows)
    for i = 1:length(wall)
        wall[i] = Int32(rand(rng) % 10)
    end

    if OUTPUT
        file = open("output.txt", "w")
        println(file, "wall:")
        for i = 1:rows
            for j = 1:cols
                print(file, "$(wall[j,i]) ")
            end
            println(file, "")
        end
        close(file)
    end

    return wall, rows, cols, pyramid_height
end

inrange(x, min, max) = x >= min && x <= max

function dynproc_kernel(iteration,
                        gpu_wall, gpu_src, gpu_result,
                        cols, rows, start_step, border)
    prev = @cuStaticSharedMem(Int32, BLOCK_SIZE)
    result = @cuStaticSharedMem(Int32, BLOCK_SIZE)

    bx = blockIdx().x
    tx = threadIdx().x

    small_block_cols = BLOCK_SIZE - iteration * HALO * 2

    blk_x = small_block_cols * (bx-1) - border;
    blk_x_max = blk_x + BLOCK_SIZE -1

    xidx = blk_x + tx

    valid_x_min = (blk_x < 0) ? -blk_x : 0
    valid_x_max = (blk_x_max > cols-1) ? BLOCK_SIZE-1-(blk_x_max-cols+1) : BLOCK_SIZE-1
    valid_x_min = valid_x_min+1
    valid_x_max = valid_x_max+1

    W = tx - 1
    E = tx + 1
    W = (W < valid_x_min) ? valid_x_min : W
    E = (E > valid_x_max) ? valid_x_max : E

    is_valid = inrange(tx, valid_x_min, valid_x_max)

    if inrange(xidx, 1, cols)
        prev[tx] = gpu_src[xidx]
    end

    sync_threads()

    computed = false
    for i = 1:iteration
        computed = false
        if inrange(tx, i+1, BLOCK_SIZE -i) && is_valid
            computed = true

            left = prev[W]
            up = prev[tx]
            right = prev[E]

            shortest = min(left, up)
            shortest = min(shortest, right)

            index = cols * (start_step + (i-1)) + xidx
            result[tx] = shortest + gpu_wall[index]
        end
        sync_threads()
        if i == iteration
            break
        end
        if computed
            prev[tx] = result[tx]
        end
        sync_threads()
    end

    if computed
        gpu_result[xidx] = result[tx]
    end

    return nothing
end

"""compute N time steps"""
function calc_path(wall, result, rows, cols, pyramid_height, block_cols, border_cols)
    dim_block = BLOCK_SIZE
    dim_grid = block_cols

    src = 2
    dst = 1

    for t = 0:pyramid_height:rows-1
        src,dst = dst,src
        iter = min(pyramid_height, rows-t-1)

        @cuda (dim_grid, dim_block) dynproc_kernel(iter,
            wall, result[src], result[dst],
            cols, rows, t, border_cols
        )
    end

    return dst
end

function main(args)
    # Initialize data
    wall, rows, cols, pyramid_height = init(args)

    # Calculate parameters
    border_cols = pyramid_height * HALO
    small_block_col = BLOCK_SIZE - pyramid_height*HALO * 2
    block_cols = floor(Int, cols/small_block_col) + ((cols % small_block_col == 0) ? 0 : 1)

    println("""pyramid_height: $pyramid_height
               grid_size: [$cols]
               border: [$border_cols]
               block_size: $BLOCK_SIZE
               block_grid: [$block_cols]
               target_block: [$small_block_col]""")

    # Setup GPU memory
    gpu_result = Array{CuArray{Int32,1}}(2)
    gpu_result[1] = CuArray(wall[:,1])
    gpu_result[2] = CuArray{Int32}(cols)

    gpu_wall = CuArray(wall[cols+1:end])

    final_ret = calc_path(
        gpu_wall, gpu_result,
        rows, cols, pyramid_height,
        block_cols, border_cols)

    result = Array(gpu_result[final_ret])

    # Store the result into a file
    if OUTPUT
        open("output.txt", "a") do fpo
            println(fpo, "data:")

            for i=1:cols
                print(fpo, "$(wall[i]) ")
            end
            println(fpo, "")

            println(fpo, "result:")
            for i=1:cols
                print(fpo, "$(result[i]) ")
            end
            println(fpo, "")
        end
    end
end


main(ARGS)

if haskey(ENV, "PROFILE")
    CUDAnative.@profile main(ARGS)
end
