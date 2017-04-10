include("../../common/julia/wrappers.jl")

const OUTPUT = haskey(ENV, "OUTPUT")

function maximum(a, b, c)
    k = (a <= b) ? b : a
    (k <= c) ? c : k
end

blosum62 = [
     4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0 -2 -1  0 -4;
    -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 -1  0 -1 -4;
    -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  3  0 -1 -4;
    -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  4  1 -1 -4;
     0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -3 -2 -4;
    -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0  3 -1 -4;
    -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4;
     0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 -1 -2 -1 -4;
    -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0  0 -1 -4;
    -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 -3 -3 -1 -4;
    -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 -4 -3 -1 -4;
    -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0  1 -1 -4;
    -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 -3 -1 -1 -4;
    -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 -3 -3 -1 -4;
    -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2 -2 -1 -2 -4;
     1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0  0  0 -4;
     0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0 -1 -1  0 -4;
    -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3 -4 -3 -2 -4;
    -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1 -3 -2 -1 -4;
     0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4 -3 -2 -1 -4;
    -2 -1  3  4 -3  0  1 -1  0 -3 -4  0 -3 -3 -2  0 -1 -4 -3 -3  4  1 -1 -4;
    -1  0  0  1 -3  3  4 -2  0 -3 -3  1 -1 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4;
     0 -1 -1 -1 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2  0  0 -2 -1 -1 -1 -1 -1 -4;
    -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4  1;
]

function usage()
    @printf(STDERR, "Usage: %s <max_rows/max_cols> <penalty>\n", "needle.jl")
    @printf(STDERR, "\t<dimension>      - x and y dimensions\n")
    @printf(STDERR, "\t<penalty>        - penalty(positive integer)\n")
    exit(1)
end

#################################################################################
# Program main
#################################################################################
function main(args)
    # The lengths of the two sequences should be divisible by 16.
    # And at current stage max_rows needs to equal max_cols
    if length(args) == 2
        max_rows = parse(Int32, args[1])
        max_cols = parse(Int32, args[1])
        penalty = parse(Int32, args[2])
    else
        usage(argc, argv)
    end

    max_rows = max_rows + 1
    max_cols = max_cols + 1

    reference = Array{Int32}(max_rows, max_cols)
    input_itemsets = zeros(Int32, (max_rows, max_cols))
    output_itemsets = Array{Int32}(max_rows, max_cols)

    srand(7)
    println("Start Needleman-Wunsch")

    for i = 2:max_rows # Please define your own sequence.
        input_itemsets[i,1] = rand() % 10 + 1
    end
    for j = 2:max_cols # Please define your own sequence.
        input_itemsets[1,j] = rand() % 10 + 1
    end

    for i = 2:max_rows
        for j = 2:max_cols
            reference[i,j] = blosum62[input_itemsets[i,1] + 1, input_itemsets[1,j] + 1]
        end
    end

    for i = 2:max_rows
        input_itemsets[i,1] = -(i - 1) * penalty
    end
    for j = 2:max_cols
        input_itemsets[1,j] = -(j - 1) * penalty
    end

    # Compute top-left matrix
    println("Processing top-left matrix")

    for i = 0:max_cols-3
        for idx = 0:i
            input_itemsets[idx + 2, i - idx + 2] = maximum(
                input_itemsets[idx + 1, i + 1 - idx] +
                    reference[idx + 2, i - idx + 2],
                input_itemsets[idx + 2, i + 1 - idx] - penalty,
                input_itemsets[idx + 1, i + 2 - idx] - penalty)
        end
    end

    println("Processing bottom-right matrix")
    # Compute bottom-right matrix
    for i = max_cols-4:-1:0
        for idx = 0:i
            input_itemsets[max_cols - idx - 1, idx - i - 1 + max_cols] = maximum(
                input_itemsets[max_cols - idx - 2, idx - i - 2 + max_cols] +
                    reference[max_cols - idx - 1, idx - i - 1 + max_cols],
                input_itemsets[max_cols - idx - 1, idx - i - 2 + max_cols] - penalty,
                input_itemsets[max_cols - idx - 2, idx - i - 1 + max_cols] - penalty)
        end
    end

    if OUTPUT
        fpo = open("output.txt", "w")
        @printf(fpo, "print traceback value GPU:\n")

        i = max_rows - 2
        j = max_rows - 2

        while i >= 0 && j >= 0

            if i == max_rows - 2 && j == max_rows - 2
                # Print the first element.
                @printf(fpo, "%d ", input_itemsets[i + 1, j + 1])
            end

            if i == 0 && j == 0
                break
            end

            if i > 0 && j > 0
                nw = input_itemsets[i, j]
                w = input_itemsets[i + 1, j]
                n = input_itemsets[i, j + 1]
            elseif i == 0
                nw = n = LIMIT
                w = input_itemsets[i + 1, j]
            elseif j == 0
                nw = w = LIMIT
                n = input_itemsets[i, j + 1]
            end

            new_nw = nw + reference[i + 1, j + 1]
            new_w = w - penalty
            new_n = n - penalty

            traceback = maximum(new_nw, new_w, new_n)
            if traceback == new_nw
                traceback = nw
            end
            if traceback == new_w
                traceback = w
            end
            if traceback == new_n
                traceback = n
            end

            @printf(fpo, "%d ", traceback)

            if traceback == nw
                i = i - 1
                j = j - 1
                continue
            elseif traceback == w
                j = j - 1
                continue
            elseif traceback == n
                i = i - 1
                continue
            end
        end

        close(fpo)
    end
end

main(ARGS)
