function lud_verify(m, lu, matrix_dim)
    tmp = Array{Float32}(matrix_dim, matrix_dim)

    for i = 1:matrix_dim, j = 1:matrix_dim
        sum = 0
        for k = 1:min(i, j)
            if i == k
                l = 1
            else
                l = lu[(i - 1) * matrix_dim + k]
            end
            u = lu[(k - 1) * matrix_dim + j]
            sum += l * u
        end
        tmp[i, j] = sum
    end

    for i = 1:matrix_dim, j = 1:matrix_dim
        if abs(m[(i - 1) * matrix_dim + j] - tmp[i, j]) > 0.0001
            @printf("dismatch at (%d, %d): (o)%f (n)%f\n", i - 1, j - 1,
                m[(i - 1) * matrix_dim + j], tmp[i, j])
        end
    end
end

function create_matrix_from_file(filename)
    println("TODO: not yet implemented.")
    exit(-1)
end

# Generate well-conditioned matrix internally by Ke Wang 2013/08/07 22:20:06
function create_matrix(size)
    lamda = -0.001
    coe = Array{Float32}(2 * size - 1)

    for i = 0:size-1
        coe_i = 10 * exp(lamda * i)
        coe[size + i] = coe_i
        coe[size - i] = coe_i
    end

    matrix = Array{Float32}(size * size)

    for i = 1:size, j = 1:size
        matrix[(i - 1) * size + j] = coe[size - i + j]
    end

    return matrix
end
