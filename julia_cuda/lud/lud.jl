#!/usr/bin/env julia

using ArgParse
using CUDAdrv, CUDAnative

include("common.jl")
include("lud_kernel.jl")

function main(args)
    info("WG size of kernel = $BLOCK_SIZE X $BLOCK_SIZE")

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
        info("Reading matrix from file ", input_file)
        matrix, matrix_dim = create_matrix_from_file(input_file)
    elseif matrix_dim > 0
        info("Creating matrix internally size=", matrix_dim)
        matrix = create_matrix(matrix_dim)
    else
        error("No input file specified!")
    end

    if verify
        info("Before LUD")
        matrix_copy = copy(matrix)
    end

    sec = CUDAdrv.@elapsed begin
        d_matrix = CuArray(matrix)
        lud_cuda(d_matrix, matrix_dim)
        matrix = Array(d_matrix)
    end
    info("Time consumed(ms): ", 1000sec)

    if verify
        info("After LUD")
        info(">>>Verify<<<<")
        lud_verify(matrix_copy, matrix, matrix_dim)
    end
end

function clopts(opts...)
    args = ["", opts...]
    LLVM.API.LLVMParseCommandLineOptions(Int32(length(args)),
        [Base.unsafe_convert(Cstring, arg) for arg in args], C_NULL)
end

# FIXME: for now we increase the unroll threshold to ensure that the nested loops in the
# kernels are unrolled as is the case for the CUDA benchmark. Ideally, we should annotate
# the loops or the kernel(s) with the @unroll macro once it is available.
clopts("--unroll-threshold=1200")

main(ARGS)

if haskey(ENV, "PROFILE")
    CUDAnative.@profile main(ARGS)
end
