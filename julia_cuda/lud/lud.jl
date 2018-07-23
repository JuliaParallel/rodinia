#!/usr/bin/env julia

using CUDAdrv, CUDAnative
using LLVM

include("common.jl")
include("lud_kernel.jl")

function main(args)
    @info "WG size of kernel = $BLOCK_SIZE X $BLOCK_SIZE"

    verify = haskey(ENV, "OUTPUT")

    matrix_dim, input_file = if length(ARGS) == 1
        try
            parse(Int, ARGS[1]), nothing
        catch
            nothing, ARGS[1]
        end
    else
        32, nothing
    end

    if input_file != nothing
        @info "Reading matrix from file $input_file"
        matrix, matrix_dim = create_matrix_from_file(input_file)
    elseif matrix_dim > 0
        @info "Creating matrix internally size=$(matrix_dim)"
        matrix = create_matrix(matrix_dim)
    else
        error("No input file specified!")
    end

    if verify
        @info "Before LUD"
        matrix_copy = copy(matrix)
    end

    sec = CUDAdrv.@elapsed begin
        d_matrix = CuArray(matrix)
        lud_cuda(d_matrix, matrix_dim)
        matrix = Array(d_matrix)
    end
    @info "Time consumed(ms): $(1000sec)"

    if verify
        @info "After LUD"
        @info ">>>Verify<<<<"
        lud_verify(matrix_copy, matrix, matrix_dim)
    end
end

# FIXME: for now we increase the unroll threshold to ensure that the nested loops in the
# kernels are unrolled as is the case for the CUDA benchmark. Ideally, we should annotate
# the loops or the kernel(s) with the @unroll macro once it is available.
LLVM.clopts("--unroll-threshold=1200")

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)

    if haskey(ENV, "PROFILE")
        main(ARGS) # really make sure everything has been compiled
        CUDAdrv.@profile NVTX.@range "application" main(ARGS)
    end
end
