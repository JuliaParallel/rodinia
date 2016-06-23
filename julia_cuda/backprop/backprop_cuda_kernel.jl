include("backprop_cuda.jl")

@target ptx function idx(x, y)
    return x * WIDTH + y + 1
end

@target ptx function bpnn_layerforward_CUDA(input_cuda,
                                            output_hidden_cuda,
                                            input_hidden_cuda,
                                            hidden_partial_sum,
                                            inp, hid)
    by = blockIdx().y - 1
    tx = threadIdx().x - 1
    ty = threadIdx().y - 1

    index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1)
    index_in = HEIGHT * by + ty + 1

    shared_mem = cuSharedMem(Float32)
    input_node = CuDeviceArray{Float32}(shared_mem)
    weight_matrix = CuDeviceArray{Float32}(
        shared_mem + HEIGHT * sizeof(Float32))

    if tx == 0
        input_node[ty + 1] = input_cuda[index_in + 1]
    end

    sync_threads()

    weight_matrix[idx(ty, tx)] = input_hidden_cuda[index + 1]

    sync_threads()

    weight_matrix[idx(ty, tx)] *= input_node[ty + 1]

    sync_threads()

    # TODO: computing power_two as in the CUDA version of the benchmarks, i.e.,
    # power_two = 2 ^ i results in the following error: LoadError: could not
    # find function 'jlsys_power_by_squaring_33838'.
    # TODO: Since HEIGHT is an integer, there might be a nicer way to express
    # i = 1:log2(HEIGHT) than what is currently used.
    power_two = 1
    for i = 1:Int32(CUDAnative.log2(Float32(HEIGHT)))
        power_two *= 2
        if ty % power_two == 0
            weight_matrix[idx(ty, tx)] +=
                weight_matrix[idx(ty + Int32(power_two / 2), tx)]
        end
        sync_threads()
    end

    input_hidden_cuda[index + 1] = weight_matrix[idx(ty, tx)]

    sync_threads()

    if tx == 0
        hidden_partial_sum[by * hid + ty + 1] = weight_matrix[idx(tx, ty)]
    end

    return nothing
end

@target ptx function bpnn_adjust_weights_cuda(delta, hid, ly, inp, w, oldw)
    by = blockIdx().y - 1
    tx = threadIdx().x - 1
    ty = threadIdx().y - 1

    index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 2 + (hid + 1)
    index_y = HEIGHT * by + ty + 2
    index_x = tx + 2

    w[index] += Float32(ETA * delta[index_x] * ly[index_y] +
        MOMENTUM * oldw[index])
    oldw[index] = Float32(ETA * delta[index_x] * ly[index_y] +
        MOMENTUM * oldw[index])

    sync_threads()

    if ty == 0 && by == 0
        w[index_x] += Float32(ETA * delta[index_x] + MOMENTUM * oldw[index_x])
        oldw[index_x] = Float32(ETA * delta[index_x] + MOMENTUM * oldw[index_x])
    end

    return nothing
end
