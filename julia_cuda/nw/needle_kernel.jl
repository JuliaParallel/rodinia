using CUDAdrv
using CUDAnative

const BLOCK_SIZE = 16
const BLOCK_SIZE_P1 = BLOCK_SIZE + 1

# FIXME: remove @inbouds (is work-around for shared memory bug)
function needle_cuda_shared_1(reference, matrix_cuda, cols, penalty, i, block_width)
    bx = blockIdx().x - 1
    tx = threadIdx().x - 1

    b_index_x = bx
    b_index_y = i - 1 - bx

    index    = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + cols + 1
    index_n  = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + 1
    index_w  = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + cols
    index_nw = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x

    temp = @cuStaticSharedMem(Int32, (BLOCK_SIZE_P1, BLOCK_SIZE_P1))
    ref = @cuStaticSharedMem(Int32, (BLOCK_SIZE, BLOCK_SIZE))

    if tx == 0
        @inbounds temp[1, tx + 1] = matrix_cuda[index_nw + 1]
    end

    for ty = 0:BLOCK_SIZE-1
        @inbounds ref[tx + 1, ty + 1] = reference[index + cols * ty + 1]
    end

    sync_threads()

    @inbounds temp[1, tx + 2] = matrix_cuda[index_w + cols * tx + 1]

    sync_threads()

    @inbounds temp[tx + 2, 1] = matrix_cuda[index_n + 1]

    sync_threads()

    for m = 0:BLOCK_SIZE-1
        if tx <= m
            t_index_x = tx + 1
            t_index_y = m - tx + 1
            @inbounds temp[t_index_x + 1, t_index_y + 1] = max(
                temp[t_index_x, t_index_y] +
                 ref[t_index_x, t_index_y],
                temp[t_index_x, t_index_y + 1] - penalty,
                temp[t_index_x + 1, t_index_y] - penalty)
        end
        sync_threads()
    end

    # TODO convert to for loop
    m = BLOCK_SIZE-2
    while m >= 0
        if tx <= m
            t_index_x = tx + BLOCK_SIZE - m
            t_index_y = BLOCK_SIZE - tx
            @inbounds temp[t_index_x + 1, t_index_y + 1] = max(
                temp[t_index_x, t_index_y] +
                 ref[t_index_x, t_index_y],
                temp[t_index_x, t_index_y + 1] - penalty,
                temp[t_index_x + 1, t_index_y] - penalty)
        end
        sync_threads()
        m -= 1
    end

    for ty = 0:BLOCK_SIZE-1
        @inbounds matrix_cuda[index + ty * cols + 1] = temp[tx + 2, ty + 2]
    end

    return nothing
end

# FIXME: remove @inbounds (is work-around for shared memory bug)
function needle_cuda_shared_2(reference, matrix_cuda, cols, penalty, i, block_width)
    bx = blockIdx().x - 1
    tx = threadIdx().x - 1

    b_index_x = bx + block_width - i
    b_index_y = block_width - bx - 1

    index    = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + cols + 1
    index_n  = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + 1
    index_w  = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + cols
    index_nw = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x

    temp = @cuStaticSharedMem(Int32, (BLOCK_SIZE_P1, BLOCK_SIZE_P1))
    ref = @cuStaticSharedMem(Int32, (BLOCK_SIZE, BLOCK_SIZE))

    for ty = 0:BLOCK_SIZE-1
        @inbounds ref[tx + 1, ty + 1] = reference[index + cols * ty + 1]
    end

    sync_threads()

    if tx == 0
        @inbounds temp[1, tx + 1] = matrix_cuda[index_nw + 1]
    end

    @inbounds temp[1, tx + 2] = matrix_cuda[index_w + cols * tx + 1]

    sync_threads()

    @inbounds temp[tx + 2, 1] = matrix_cuda[index_n + 1]

    sync_threads()

    for m = 0:BLOCK_SIZE-1
        if tx <= m
            t_index_x = tx + 1
            t_index_y = m - tx + 1
            @inbounds temp[t_index_x + 1, t_index_y + 1] = max(
                temp[t_index_x, t_index_y] +
                 ref[t_index_x, t_index_y],
                temp[t_index_x, t_index_y + 1] - penalty,
                temp[t_index_x + 1, t_index_y] - penalty)
        end
        sync_threads()
    end

    # TODO convert to for loop
    m = BLOCK_SIZE-2
    while m >= 0
        if tx <= m
            t_index_x = tx + BLOCK_SIZE - m
            t_index_y = BLOCK_SIZE - tx
            @inbounds temp[t_index_x + 1, t_index_y + 1] = max(
                temp[t_index_x, t_index_y] +
                 ref[t_index_x, t_index_y],
                temp[t_index_x, t_index_y + 1] - penalty,
                temp[t_index_x + 1, t_index_y] - penalty)
        end
        sync_threads()
        m -= 1
    end

    for ty = 0:BLOCK_SIZE-1
        @inbounds matrix_cuda[index + ty * cols + 1] = temp[tx + 2, ty + 2]
    end

    return nothing
end
