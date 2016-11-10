include("backprop_cuda.jl")

function bpnn_layerforward_CUDA(input_cuda,
                                            output_hidden_cuda,
                                            input_hidden_cuda,
                                            hidden_partial_sum,
                                            inp, hid)
    by = blockIdx().y - 1
    tx = threadIdx().x
    ty = threadIdx().y

    index = (hid + 1) * HEIGHT * by + (hid + 1) * (ty - 1) + tx + (hid + 1)
    index_in = HEIGHT * by + ty

    input_node = @cuStaticSharedMem(Float32, HEIGHT)
    weight_matrix = @cuStaticSharedMem(Float32, (HEIGHT, WIDTH))

    if tx == 1
        input_node[ty] = input_cuda[index_in + 1]
    end

    sync_threads()

    weight_matrix[tx, ty] = input_hidden_cuda[index + 1]

    sync_threads()

    weight_matrix[tx, ty] *= input_node[ty]

    sync_threads()

    # TODO: Since HEIGHT is an integer, there might be a nicer way to express
    # i = 1:log2(HEIGHT) than what is currently used.
    for i = 1:trunc(Int32,CUDAnative.log2(Float32(HEIGHT)))
        power_two = CUDAnative.pow(2f0, trunc(Int32,i))
        if ty % power_two == 1
            weight_matrix[tx, ty] +=
                weight_matrix[tx, ty + trunc(Int32,power_two / 2)]
        end
        sync_threads()
    end

    input_hidden_cuda[index + 1] = weight_matrix[tx, ty]

    sync_threads()

    if tx == 1
        hidden_partial_sum[by * hid + ty] = weight_matrix[ty, tx]
    end

    return nothing
end

function bpnn_adjust_weights_cuda(delta, hid, ly, inp, w, oldw)
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
