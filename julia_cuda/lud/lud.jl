#!/usr/bin/env julia

include("common.jl")
include("lud_kernel.jl")

using ArgParse

function main(args)
    @printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE)

    s = ArgParseSettings()
    @add_arg_table s begin
        "-i", "--input"
        "-s", "--size"
            arg_type = Int
            default = 32
        "-v", "--verify"
            action = :store_true
    end

    args = parse_args(args, s)

    verify = args["verify"]
    matrix_dim = args["size"]
    input_file = args["input"]

    if input_file != nothing
        println("Reading matrix from file ", input_file)
        matrix, matrix_dim = create_matrix_from_file(input_file)
    elseif matrix_dim > 0
        println("Creating matrix internally size=", matrix_dim)
        matrix = create_matrix(matrix_dim)
    else
        println("No input file specified!")
        exit(-1)
    end

    if verify
        println("Before LUD")
        matrix_copy = copy(matrix)
    end

    dev = CuDevice(0)
    ctx = CuContext(dev)

    # beginning of timing point
    tic()

    d_matrix = CuArray(matrix)
    lud_cuda(d_matrix, matrix_dim)
    matrix = Array(d_matrix)

    # end of timing point
    println("Time consumed(ms): ", toq() * 1000)

    free(d_matrix)

    if verify
        println("After LUD")
        println(">>>Verify<<<<")
        lud_verify(matrix_copy, matrix, matrix_dim)
    end

    destroy(ctx)
end

main(ARGS)
