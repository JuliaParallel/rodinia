include("../../common/julia/wrappers.jl")

const OUTPUT = haskey(ENV, "OUTPUT")

function init(args)
    global rows
    global cols
    global data
    global result

    if length(args) == 2
        cols = parse(Int, args[1])
        rows = parse(Int, args[2])
    else
        println("Usage: pathfinder width num_of_steps")
        exit(0)
    end

    data = Array{Int}(rows, cols)
    result = Array{Int}(cols)

    srand(7)

    for i = 1:rows
        for j = 1:cols
            data[i,j] = rand() % 10
        end
    end

    for j = 1:cols
        result[j] = data[1,j]
    end

    if OUTPUT
        file = open("output.txt", "w")
        @printf(file, "wall:\n")
        for i = 1:rows
            for j = 1:cols
                @printf(file, "%d ", data[i,j])
            end
            @printf(file, "\n")
        end
        close(file)
    end
end

function main(args)
    init(args)

    dst = result
    src = Array{Int}(cols)

    tic()

    for t = 1:rows-1
        temp = src
        src = dst
        dst = temp
# #pragma omp parallel for private(min)
        for n = 1:cols
            min_ = src[n]
            if n > 1
                min_ = min(min_, src[n - 1])
            end
            if n < cols
                min_ = min(min_, src[n + 1])
            end
            dst[n] = data[t + 1, n] + min_
        end
    end

    @printf("timer: %Lu\n", toq())

    if OUTPUT
        file = open("output.txt", "a")
        @printf(file, "data:\n")
        for i = 1:cols
            @printf(file, "%d ", data[1,i])
        end
        @printf(file, "\n")

        @printf(file, "result:\n");
        for i = 1:cols
            @printf(file, "%d ", dst[i])
        end
        @printf(file, "\n");
        close(file)
    end
end

main(ARGS)
