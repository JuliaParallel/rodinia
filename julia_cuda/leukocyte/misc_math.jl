const THRESHOLD_double = 0.000001
const C_PI = 3.14159

function double_eq(f1, f2)
    abs(f1 - f2) < THRESHOLD_double
end


# Given a matrix, return the matrix containing an approximation of the gradient
# matrix dM/dx
function gradient_x(input::Matrix)
    result = similar(input)

    for i in 1:size(input,1)
        for j in 1:size(input,2)
            if j == 1
                result[i,j] = input[i,j+1] - input[i,j]
            elseif j == size(input,2)
                result[i,j] = input[i,j] - input[i,j-1]
            else
                result[i,j] = (input[i,j+1] - input[i,j-1]) / 2.0
            end
        end
    end
    result
end


# Given a matrix, return the matrix containing an approximation of the gradient
# matrix dM/dy
function gradient_y(input::Matrix)
    result = similar(input)

    for i in 1:size(input,2)
        for j in 1:size(input,1)
            if j == 1
                result[j,i] = input[j+1,i] - input[j,i]
            elseif j == size(input,1)
                result[j,i] = input[j,i] - input[j-1,i]
            else
                result[j,i] = (input[j+1,i] - input[j-1,i]) / 2.0
            end
        end
    end
    result
end


function mean(in::Array)
# looks nice, but slow because boxed (due to "+" returning ANY, or because lamda?)
#    reducedim(+, in, 1, 0) / size(in,1)
    sum = 0.0
    for i in 1:size(in,1)
        sum += in[i]
    end
    sum / size(in,1)
end


function std_dev(in::Array)
    m = mean(in)
# looks nice, but slow because boxed (due to "+" returning ANY, or because lamda?)
#    sum=reducedim(x -> x - m, in, 1, 0)
    sum = 0.0
    for i in 1:size(in,1)
        sum += (in[i]-m)^2
    end
    sqrt(sum / size(in,1))
end

