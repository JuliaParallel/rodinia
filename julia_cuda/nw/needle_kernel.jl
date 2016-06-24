using CUDAdrv
using CUDAnative

const BLOCK_SIZE = 16

@target ptx function cuda_maximum(a, b, c)
    k = (a <= b) ? b : a
    (k <= c) ? c : k
end

@target ptx function tidx(x, y)
    x * (BLOCK_SIZE + 1) + y + 1
end

@target ptx function ridx(x, y)
    x * BLOCK_SIZE + y + 1
end

@target ptx function needle_cuda_shared_1(reference, matrix_cuda, cols, penalty,
                                          i, block_width)
    bx = blockIdx().x - 1
    tx = threadIdx().x - 1

    b_index_x = bx
    b_index_y = i - 1 - bx

    index    = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + cols + 1
    index_n  = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + 1
    index_w  = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + cols
    index_nw = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x

    shared_mem = cuSharedMem(Int32)
    temp = CuDeviceArray(shared_mem)
    ref = CuDeviceArray(shared_mem + (BLOCK_SIZE + 1) * (BLOCK_SIZE + 1) * sizeof(Int32))

    if tx == 0
        temp[tidx(tx, 0)] = matrix_cuda[index_nw + 1]
    end

    for ty = 0:BLOCK_SIZE-1
        ref[ridx(ty, tx)] = reference[index + cols * ty + 1]
    end

    sync_threads()

    temp[tidx(tx + 1, 0)] = matrix_cuda[index_w + cols * tx + 1]

    sync_threads()

    temp[tidx(0, tx + 1)] = matrix_cuda[index_n + 1]

    sync_threads()

    for m = 0:BLOCK_SIZE-1
        if tx <= m
            t_index_x = tx + 1
            t_index_y = m - tx + 1
            temp[tidx(t_index_y, t_index_x)] = cuda_maximum(
                temp[tidx(t_index_y - 1, t_index_x - 1)] +
                 ref[ridx(t_index_y - 1, t_index_x - 1)],
                temp[tidx(t_index_y, t_index_x - 1)] - penalty,
                temp[tidx(t_index_y - 1, t_index_x)] - penalty)
        end
        sync_threads()
    end

    # TODO convert to for loop
    m = BLOCK_SIZE-2
    while m >= 0
        if tx <= m
            t_index_x = tx + BLOCK_SIZE - m
            t_index_y = BLOCK_SIZE - tx
            temp[tidx(t_index_y, t_index_x)] = cuda_maximum(
                temp[tidx(t_index_y - 1, t_index_x - 1)] +
                 ref[ridx(t_index_y - 1, t_index_x - 1)],
                temp[tidx(t_index_y, t_index_x - 1)] - penalty,
                temp[tidx(t_index_y - 1, t_index_x)] - penalty)
        end
        sync_threads()
        m -= 1
    end

    for ty = 0:BLOCK_SIZE-1
        matrix_cuda[index + ty * cols + 1] = temp[tidx(ty + 1, tx + 1)]
    end

    return nothing
end

@target ptx function needle_cuda_shared_2(reference, matrix_cuda, cols, penalty,
                                          i, block_width)
    bx = blockIdx().x - 1
    tx = threadIdx().x - 1

    b_index_x = bx + block_width - i
    b_index_y = block_width - bx - 1

    index    = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + cols + 1
    index_n  = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + 1
    index_w  = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + cols
    index_nw = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x

    shared_mem = cuSharedMem(Int32)
    temp = CuDeviceArray(shared_mem)
    ref = CuDeviceArray(shared_mem + (BLOCK_SIZE + 1) * (BLOCK_SIZE + 1) * sizeof(Int32))

    for ty = 0:BLOCK_SIZE-1
        ref[ridx(ty, tx)] = reference[index + cols * ty + 1]
    end

    sync_threads()

    if tx == 0
        temp[tidx(tx, 0)] = matrix_cuda[index_nw + 1]
    end

    temp[tidx(tx + 1, 0)] = matrix_cuda[index_w + cols * tx + 1]

    sync_threads()

    temp[tidx(0, tx + 1)] = matrix_cuda[index_n + 1]

    sync_threads()

    for m = 0:BLOCK_SIZE-1
        if tx <= m
            t_index_x = tx + 1
            t_index_y = m - tx + 1
            temp[tidx(t_index_y, t_index_x)] = cuda_maximum(
                temp[tidx(t_index_y - 1, t_index_x - 1)] +
                 ref[ridx(t_index_y - 1, t_index_x - 1)],
                temp[tidx(t_index_y, t_index_x - 1)] - penalty,
                temp[tidx(t_index_y - 1, t_index_x)] - penalty)
        end
        sync_threads()
    end

    # TODO convert to for loop
    m = BLOCK_SIZE-2
    while m >= 0
        if tx <= m
            t_index_x = tx + BLOCK_SIZE - m
            t_index_y = BLOCK_SIZE - tx
            temp[tidx(t_index_y, t_index_x)] = cuda_maximum(
                temp[tidx(t_index_y - 1, t_index_x - 1)] +
                 ref[ridx(t_index_y - 1, t_index_x - 1)],
                temp[tidx(t_index_y, t_index_x - 1)] - penalty,
                temp[tidx(t_index_y - 1, t_index_x)] - penalty)
        end
        sync_threads()
        m -= 1
    end

    for ty = 0:BLOCK_SIZE-1
        matrix_cuda[index + ty * cols + 1] = temp[tidx(ty + 1, tx + 1)]
    end

    return nothing
end
