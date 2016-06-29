const BLOCK_SIZE = 16

using CUDAdrv
using CUDAnative

@target ptx function idx(x, y)
    return x * BLOCK_SIZE + y + 1
end

@target ptx function lud_diagonal(matrix, matrix_dim, offset)

    shared_mem = cuSharedMem(Float32)
    shadow = CuDeviceArray(shared_mem)

    array_offset = offset * matrix_dim + offset

    tx = threadIdx().x - 1

    for i = 0:BLOCK_SIZE-1
        shadow[idx(i, tx)] = matrix[array_offset + tx + 1]
        array_offset += matrix_dim
    end

    sync_threads()

    for i = 0:BLOCK_SIZE-2
        if tx > i
            for j = 0:i-1
                shadow[idx(tx, i)] -= shadow[idx(tx, j)] * shadow[idx(j, i)]
            end
            shadow[idx(tx, i)] /= shadow[idx(i, i)]
        end

        sync_threads()

        if tx > i
            for j = 0:i
                shadow[idx(i + 1, tx)] -= shadow[idx(i + 1, j)] * shadow[idx(j, tx)]
            end
        end

        sync_threads()
    end

    # The first row is not modified, it is no need to write it back to the
    # global memory.
    array_offset = (offset + 1) * matrix_dim + offset

    for i = 1:BLOCK_SIZE-1
        matrix[array_offset + tx + 1] = shadow[idx(i, tx)]
        array_offset += matrix_dim
    end

    return nothing
end

@target ptx function lud_perimeter(matrix, matrix_dim, offset)

    shared_mem = cuSharedMem(Float32)
    dia = CuDeviceArray(shared_mem)
    peri_row = CuDeviceArray(shared_mem +
        BLOCK_SIZE * BLOCK_SIZE * sizeof(Float32))
    peri_col = CuDeviceArray(shared_mem +
        BLOCK_SIZE * BLOCK_SIZE * sizeof(Float32) * 2)

    bx = blockIdx().x - 1
    tx = threadIdx().x - 1

    if tx < BLOCK_SIZE
        index = tx
        array_offset = offset * matrix_dim + offset

        for i = 0:Int(BLOCK_SIZE / 2)-1
            dia[idx(i, index)] = matrix[array_offset + index + 1]
            array_offset += matrix_dim
        end

        array_offset = offset * matrix_dim + offset

        for i = 0:BLOCK_SIZE-1
            peri_row[idx(i, index)] =
                matrix[array_offset + (bx + 1) * BLOCK_SIZE + index + 1]
            array_offset += matrix_dim
        end
    else
        index = tx - BLOCK_SIZE
        array_offset = (offset + Int(BLOCK_SIZE / 2)) * matrix_dim + offset

        for i = Int(BLOCK_SIZE / 2):BLOCK_SIZE-1
            dia[idx(i, index)] = matrix[array_offset + index + 1]
            array_offset += matrix_dim
        end

        array_offset = (offset + (bx + 1) * BLOCK_SIZE) * matrix_dim + offset

        for i = 0:BLOCK_SIZE-1
            peri_col[idx(i, index)] = matrix[array_offset + index + 1]
            array_offset += matrix_dim
        end
    end

    sync_threads()

    if tx < BLOCK_SIZE # peri-row
        index = tx
        for i = 1:BLOCK_SIZE-1, j = 0:i-1
            peri_row[idx(i, index)] -= dia[idx(i, j)] * peri_row[idx(j, index)]
        end
    else # peri-col
        index = tx - BLOCK_SIZE
        for i = 0:BLOCK_SIZE-1
            for j = 0:i-1
                peri_col[idx(index, i)] -= peri_col[idx(index, j)] * dia[idx(j, i)]
            end
            peri_col[idx(index, i)] /= dia[idx(i, i)]
        end
    end

    sync_threads()

    if tx < BLOCK_SIZE # peri-row
        index = tx
        array_offset = (offset + 1) * matrix_dim + offset

        for i = 1:BLOCK_SIZE-1
            matrix[array_offset + (bx + 1) * BLOCK_SIZE + index + 1] =
                peri_row[idx(i, index)]
            array_offset += matrix_dim
        end
    else # peri-col
        index = tx - BLOCK_SIZE
        array_offset = (offset + (bx + 1) * BLOCK_SIZE) * matrix_dim + offset

        for i = 0:BLOCK_SIZE-1
            matrix[array_offset + index + 1] = peri_col[idx(i, index)]
            array_offset += matrix_dim
        end
    end

    return nothing
end

@target ptx function lud_internal(matrix, matrix_dim, offset)

    shared_mem = cuSharedMem(Float32)
    peri_row = CuDeviceArray(shared_mem)
    peri_col = CuDeviceArray(shared_mem +
        BLOCK_SIZE * BLOCK_SIZE * sizeof(Float32))

    global_row_id = offset + blockIdx().y * BLOCK_SIZE
    global_col_id = offset + blockIdx().x * BLOCK_SIZE

    tx = threadIdx().x - 1
    ty = threadIdx().y - 1

    peri_row[idx(ty, tx)] = matrix[(offset + ty) * matrix_dim + global_col_id + tx + 1]
    peri_col[idx(ty, tx)] = matrix[(global_row_id + ty) * matrix_dim + offset + tx + 1]

    sync_threads()

    sum::Float32 = 0.0
    for i = 0:BLOCK_SIZE-1
        sum += peri_col[idx(ty, i)] * peri_row[idx(i, tx)]
    end
    matrix[(global_row_id + ty) * matrix_dim + global_col_id + tx + 1] -= sum

    return nothing
end

function lud_cuda(matrix, matrix_dim)

    float_matrix_size = BLOCK_SIZE * BLOCK_SIZE * sizeof(Float32)

    i = 0

    while i < matrix_dim - BLOCK_SIZE

        @cuda (1, BLOCK_SIZE, float_matrix_size) lud_diagonal(
            matrix, matrix_dim, i)

        grid_size = Int((matrix_dim - i) / BLOCK_SIZE) - 1

        @cuda (grid_size, BLOCK_SIZE * 2, float_matrix_size * 3) lud_perimeter(
            matrix, matrix_dim, i)

        @cuda ((grid_size, grid_size), (BLOCK_SIZE, BLOCK_SIZE),
            float_matrix_size * 2) lud_internal(matrix, matrix_dim, i)

        i += BLOCK_SIZE
    end

    @cuda (1, BLOCK_SIZE, float_matrix_size) lud_diagonal(matrix, matrix_dim, i)
end
