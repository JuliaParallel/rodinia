#!/usr/bin/env julia

using CUDAdrv, CUDAnative, NVTX
using Printf

include("../../common/julia/crand.jl")
const rng = LibcRNG()

include("backprop.jl")
include("backprop_cuda_kernel.jl")

const OUTPUT = haskey(ENV, "OUTPUT")

function backprop_face(layer_size)
    net = bpnn_create(layer_size, 16, 1) # (16, 1 cannot be changed)
    println("Input layer size : ", layer_size)

    units = net.input_units
    for i = 2:layer_size+1
        units[i] = float(rand(rng)) / RAND_MAX
    end

    # Entering the training kernel, only one iteration.
    println("Starting training kernel")
    bpnn_train_cuda(net)

    if OUTPUT
        bpnn_save(net, "output.dat")
    end

    println("Training done")
end

function main(args)
    if length(args) != 1
        println(stderr, "usage: backprop <num of input elements>");
        exit(1)
    end

    layer_size = parse(Int, args[1])

    if layer_size % 16 != 0
        @printf(stderr, "The number of input points must be divisible by 16\n")
        exit(1)
    end

    bpnn_initialize(7)
    backprop_face(layer_size)
end


if abspath(PROGRAM_FILE) == @__FILE__
    NVTX.stop()
    main(ARGS)

    if haskey(ENV, "PROFILE")
        # warm up
        for i in 1:5
            main(ARGS)
            GC.gc()
        end

        empty!(CUDAnative.compilecache)

        NVTX.@activate begin
            for i in 1:5
                GC.gc(true)
            end
            main(ARGS)                                       # measure compile time
            for i in 1:5
                GC.gc(true)
            end
            CUDAdrv.@profile NVTX.@range "host" main(ARGS)   # measure execution time
        end
    end
end
