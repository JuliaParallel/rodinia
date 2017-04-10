#!/usr/bin/env julia

const OUTPUT = haskey(ENV, "OUTPUT")

# maximum power density possible (say 300W for a 10mm x 10mm chip)
MAX_PD = 3.0e6
# required precision in degrees
PRECISION = 0.001
SPEC_HEAT_SI = 1.75e6
# capacitance fitting factor
K_SI = 100
FACTOR_CHIP = 0.5

# chip parameters
t_chip = 0.0005
chip_height = 0.016
chip_width = 0.016
# ambient temperature, assuming no package at all
amb_temp = 80.0


#= Single iteration of the transient solver in the grid model.
 = advances the solution of the discretized difference equations
 = by one time step
 =#

function single_iteration(result::SharedArray{Float64,2}, temp::Matrix{Float64}, power::Matrix{Float64}, cap::Float64, rx::Float64, ry::Float64, rz::Float64, step::Float64)
    row = size(temp,1)
    col = size(temp,2)
    @sync @parallel for r in 1:row
        for c in 1:col
            # Corner 1
            if ((r == 1) & (c == 1))
                delta = (step / cap) * (power[1,1] + (temp[1,2] - temp[1,1]) / rx +
                                        (temp[2,1] - temp[1,1]) / ry +
                                        (amb_temp - temp[1,1]) / rz)
            # Corner 2
            elseif ((r == 1) & (c == col))
                delta =
                    (step / cap) * (power[1,c] + (temp[1,c-1] - temp[1,c]) / rx +
                         (temp[2,c] - temp[1,c]) / ry +
                         (amb_temp - temp[1,c]) / rz)
            # Corner 3
            elseif ((r == row) & (c == col))
                delta = (step / cap) *
                    (power[r,c] +
                     (temp[r,c-1] - temp[r,c]) / rx +
                     (temp[r-1,c] - temp[r,c]) / ry +
                     (amb_temp - temp[r,c]) / rz)
            # Corner 4
            elseif ((r == row) & (c == 1))
                delta =
                    (step / cap) *
                    (power[r,1] + (temp[r,2] - temp[r,1]) / rx +
                     (temp[r-1,1] - temp[r,1]) / ry +
                     (amb_temp - temp[r,1]) / rz)
            # Edge 1
            elseif (r == 1)
                delta = (step / cap) *
                        (power[c] +
                         (temp[1,c+1] + temp[1,c-1] - 2.0*temp[1,c]) / rx +
                         (temp[2,c] - temp[1,c]) / ry +
                         (amb_temp - temp[1,c]) / rz)
            # Edge 2
            elseif (c == col)
                delta = (step / cap) *
                        (power[r,c] +
                         (temp[r+1,c] + temp[r-1,c] -
                          2.0*temp[r,c]) /
                             ry +
                         (temp[r,c-1] - temp[r,c]) / rx +
                         (amb_temp - temp[r,c]) / rz)
            # Edge 3
            elseif (r == row)
                delta = (step / cap) *
                        (power[r,c] +
                         (temp[r,c+1] + temp[r,c-1] -
                          2.0*temp[r,c]) /
                             rx +
                         (temp[r-1,c] - temp[r,c]) / ry +
                         (amb_temp - temp[r,c]) / rz)
            # Edge 4
            elseif (c == 1)
                delta =
                    (step / cap) * (power[r,1] +
                                    (temp[r+1,1] + temp[r-1,1] -
                                     2.0*temp[r,1]) /
                                        ry +
                                    (temp[r,1] - temp[r,1]) / rx +
                                    (amb_temp - temp[r,1]) / rz);
            # Inside the chip
            else
                delta = (step / cap) *
                        (power[r,c] +
                         (temp[r+1,c] + temp[r-1,c] -
                          2.0*temp[r,c]) /
                             ry +
                         (temp[r,c+1] + temp[r,c-1] -
                          2.0*temp[r,c]) /
                             rx +
                         (amb_temp - temp[r,c]) / rz);
            end
            # Update Temperatures
            result[r,c] = temp[r,c]+delta
        end
    end
    temp[:] = result[:]
end

#= Transient solver driver routine: simply converts the heat
 = transfer differential equations to difference equations
 = and solves the difference equations by iterating
 =#
function compute_tran_temp(result::SharedArray{Float64,2}, num_iterations::Int32, temp::Matrix{Float64},
                       power::Matrix{Float64})
    row = size(temp,1)
    col = size(temp,2)
    grid_height = chip_height / row
    grid_width = chip_width / col

    cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height
    rx = grid_width / (2.0 * K_SI * t_chip * grid_height)
    ry = grid_height / (2.0 * K_SI * t_chip * grid_width)
    rz = t_chip / (K_SI * grid_height * grid_width)

    max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI)
    step = PRECISION / max_slope
 
    for i in 1:num_iterations
        single_iteration(result,temp,power,cap,rx,ry,rz,step)
    end
end

function read_input(vect::Matrix{Float64}, file)
    fp = open(file,"r")
    for r in 1:size(vect,1)
        for c in 1:size(vect,2)
            vect[r,c] = parse(Float64,readline(fp))
        end
    end
    close(fp)
end


function usage(args)
    prog = basename(Base.source_path())
# no MT
#    println(STDERR, "Usage: ",prog," <grid_rows> <grid_cols> <sim_time> <no. of threads> <temp_file> <power_file>")
    println(STDERR, "Usage: ",prog," <grid_rows> <grid_cols> <sim_time> <temp_file> <power_file>")
    println(STDERR,
            "\t<grid_rows>  - number of rows in the grid (positive integer)")
    println(
        STDERR,
        "\t<grid_cols>  - number of columns in the grid (positive integer)")
    println(STDERR, "\t<sim_time>   - number of iterations")
#    println(STDERR, "\t<no. of threads>   - number of threads")
    println(STDERR, "\t<temp_file>  - name of the file containing the initial temperature values of each cell")
    println(STDERR, "\t<power_file> - name of the file containing the dissipated power values of each cell")
    exit(1)
end

function main(args)
    # check validity of inputs
    if length(args) != 5
        usage(args)
    end
    grid_rows = get(tryparse(Int64,args[1]),0)
    grid_cols = get(tryparse(Int64,args[2]),0)
    sim_time = get(tryparse(Int32,args[3]),0)
    if (grid_rows <= 0) |
       (grid_cols <= 0) |
       (sim_time <= 0)
        usage(args)
    end


    # allocate memory for the temperature and power arrays
    temp = Matrix{Float64}(grid_rows,grid_cols)
    power = Matrix{Float64}(grid_rows,grid_cols)
    result::SharedArray{Float64,2} = SharedArray(Float64,(grid_rows,grid_cols), init = result -> result[Base.localindexes(result)] = 0)

    # read initial temperatures and input power
    tfile = args[4]
    pfile = args[5]
    read_input(temp,tfile)
    read_input(power,pfile)

    tic()
    println("Start computing the transient temperature")
    compute_tran_temp(result,sim_time,temp,power)
    println("Ending simulation")
    toc()

    # output results
    output = ""
    if OUTPUT
        if ENV["OUTPUT"] != ""
            println("Writing results output.txt")
            f = open("output.txt","w")
            for r in 1:size(temp,1)
                for c in 1:size(temp,2)
                    println(f,@sprintf("%d\t%g",(r-1)*size(temp,2)+c-1,temp[r,c]))
                end
            end
            close(f)
        else
            println("OUTPUT environment variable not set or empty, not writing results")
        end
    end
end

