include("common.jl")

using ArgParse

function lud_omp(a,size)
    for i = 1:size
        for j = i:size
            sum = a[i, j]
            for k = 1:i-1
                sum -= a[i, k] * a[k, j]
            end
            a[i, j] = sum
        end
        for j = i+1:size
            sum = a[j, i];
            for k = 1:i-1
                sum -= a[j, k] * a[k, i]
            end
            a[j, i] = sum / a[i, i]
        end
    end
end

function main(args)
    s = ArgParseSettings()
    @add_arg_table s begin
        "-i", "--input"
        "-s", "--size"
            arg_type = Int
            default = 32
        "-v", "--verify"
            action = :store_true
    end

    args = parse_args(s)

    verify = args["verify"]
    matrix_dim = args["size"]
    input_file = args["input"]

    if input_file != nothing
        println("Reading matrix from file ", input_file)
        matrix, matrix_dim = create_matrix_from_file(input_file)
    elseif matrix_dim > 0
        println("Creating matrix internally size=", matrix_dim)
        matrix = create_matrix(matrix_dim)
    else
        println("No input file specified!")
        exit(-1)
    end

    if verify
        println("Before LUD")
        matrix_copy = copy(matrix)
    end

    tic()
    lud_omp(matrix, matrix_dim)
    println("Time consumed(ms): ", toq() * 1000)

    if verify
        println("After LUD")
        println(">>>Verify<<<<")
        lud_verify(matrix_copy, matrix, matrix_dim)
    end
end

main(ARGS)
