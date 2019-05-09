const BLOCK_SIZE = 16
const MATRIX_SIZE = BLOCK_SIZE * BLOCK_SIZE

using CUDAdrv, CUDAnative

function lud_diagonal(matrix, offset)
    shadow = @cuStaticSharedMem(Float32, (BLOCK_SIZE,BLOCK_SIZE))

    tx = threadIdx().x

    for i = 1:BLOCK_SIZE
        @inbounds shadow[tx, i] = matrix[offset + tx, offset+i]
    end

    sync_threads()

    for i = 1:BLOCK_SIZE-1
        if tx > i
            for j = 1:i-1
                @inbounds shadow[i, tx] -= shadow[j, tx] * shadow[i, j]
            end
            @inbounds shadow[i, tx] /= shadow[i, i]
        end

        sync_threads()

        if tx > i
            for j = 1:i
                @inbounds shadow[tx, i + 1] -= shadow[j, i + 1] * shadow[tx, j]
            end
        end

        sync_threads()
    end

    # The first row is not modified, it is no need to write it back to the global memory.
    for i = 2:BLOCK_SIZE
        @inbounds matrix[offset + tx, offset + i] = shadow[tx, i]
    end
    return
end

function lud_perimeter(matrix, offset)
    dia = @cuStaticSharedMem(Float32, (BLOCK_SIZE,BLOCK_SIZE))
    peri_row = @cuStaticSharedMem(Float32, (BLOCK_SIZE,BLOCK_SIZE))
    peri_col = @cuStaticSharedMem(Float32, (BLOCK_SIZE,BLOCK_SIZE))

    if threadIdx().x <= BLOCK_SIZE
        index = threadIdx().x

        for i = 1:BLOCK_SIZE÷2
            @inbounds dia[index, i] = matrix[offset+index, offset+i]
        end

        for i = 1:BLOCK_SIZE
            @inbounds peri_row[index, i] = matrix[offset + index + blockIdx().x * BLOCK_SIZE, offset + i]
        end
    else
        index = threadIdx().x - BLOCK_SIZE

        for i = 1+BLOCK_SIZE÷2:BLOCK_SIZE
            @inbounds dia[index, i] = matrix[offset + index, offset + i]
        end

        for i = 1:BLOCK_SIZE
            @inbounds peri_col[index, i] = matrix[offset + index, offset + i + blockIdx().x * BLOCK_SIZE]
        end
    end

    sync_threads()

    if threadIdx().x <= BLOCK_SIZE # peri-row
        index = threadIdx().x
        for i = 2:BLOCK_SIZE, j = 1:i-1
            @inbounds peri_row[index, i] -= dia[j, i] * peri_row[index, j]
        end
    else # peri-col
        index = threadIdx().x - BLOCK_SIZE
        for i = 1:BLOCK_SIZE
            for j = 1:i-1
                @inbounds peri_col[i, index] -= peri_col[j, index] * dia[i, j]
            end
            @inbounds peri_col[i, index] /= dia[i, i]
        end
    end

    sync_threads()

    if threadIdx().x <= BLOCK_SIZE # peri-row
        index = threadIdx().x
        for i = 2:BLOCK_SIZE
            @inbounds matrix[offset + index + blockIdx().x * BLOCK_SIZE, offset + i] = peri_row[index, i]
        end
    else # peri-col
        index = threadIdx().x - BLOCK_SIZE
        for i = 1:BLOCK_SIZE
            @inbounds matrix[offset + index, offset + blockIdx().x * BLOCK_SIZE + i] = peri_col[index, i]
        end
    end
    return
end

function lud_internal(matrix, offset)
    peri_col = @cuStaticSharedMem(Float32, (BLOCK_SIZE,BLOCK_SIZE))
    peri_row = @cuStaticSharedMem(Float32, (BLOCK_SIZE,BLOCK_SIZE))

    global_row_id = offset + blockIdx().y * BLOCK_SIZE
    global_col_id = offset + blockIdx().x * BLOCK_SIZE

    tx = threadIdx().x
    ty = threadIdx().y

    @inbounds peri_row[tx, ty] = matrix[global_col_id + tx, offset + ty]
    @inbounds peri_col[tx, ty] = matrix[offset + tx, global_row_id + ty]

    sync_threads()

    sum = 0f0
    for i = 1:BLOCK_SIZE
        @inbounds sum += peri_col[i, ty] * peri_row[tx, i]
    end
    @inbounds matrix[global_col_id + tx, global_row_id + ty] -= sum
    return
end

function lud_cuda(matrix, matrix_dim)
    i = 0
    while i < matrix_dim - BLOCK_SIZE
        @cuda threads=BLOCK_SIZE lud_diagonal(matrix, i)

        grid_size = (matrix_dim-i)÷BLOCK_SIZE - 1

        @cuda blocks=grid_size threads=BLOCK_SIZE*2 lud_perimeter(matrix, i)

        @cuda blocks=(grid_size, grid_size) threads=(BLOCK_SIZE, BLOCK_SIZE) lud_internal(matrix, i)

        i += BLOCK_SIZE
    end

    @cuda threads=BLOCK_SIZE lud_diagonal(matrix, i)
end
