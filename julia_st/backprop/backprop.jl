include("../../common/julia/wrappers.jl")

BIGRND   = 0x7fffffff
RAND_MAX = 2147483647

ETA      = 0.3 # eta value
MOMENTUM = 0.3 # momentum value

type BPNN
    input_n  # number of input units
    hidden_n # number of hidden units
    output_n # number of output units

    input_units  # the input units
    hidden_units # the hidden units
    output_units # the output units

    hidden_delta # storage for hidden unit error
    output_delta # storage for output unit error

    target # storage for target vector

    input_weights  # weights from input to hidden layer
    hidden_weights # weights from hidden to output layer

    # The next two are for momentum.
    input_prev_weights  # previous change on input to hidden weights
    hidden_prev_weights # previous change on hidden to output weights

    BPNN(input_n, hidden_n, output_n) =
        new(input_n, hidden_n, output_n,
            Array{Float32}(input_n + 1),
            Array{Float32}(hidden_n + 1),
            Array{Float32}(output_n + 1),
            Array{Float32}(hidden_n + 1),
            Array{Float32}(output_n + 1),
            Array{Float32}(output_n + 1),
            Array{Float32}(input_n + 1, hidden_n + 1),
            Array{Float32}(hidden_n + 1, output_n + 1),
            Array{Float32}(input_n + 1, hidden_n + 1),
            Array{Float32}(hidden_n + 1, output_n + 1))
end

# Returns a random number between 0.0 and 1.0.
function drnd()
    Float32(rand()) / BIGRND
end

# Returns a random number between -1.0 and 1.0.
function dpn1()
    drnd() * 2.0 - 1.0
end

# The squashing function. Currently, it's a sigmoid.
function squash(x)
    1.0 / (1.0 + exp(-x))
end

function bpnn_randomize_weights(w, m, n)
    for i = 1:m+1
        for j = 1:n+1
            w[i,j] = Float32(rand()) / RAND_MAX
        end
    end
end

function bpnn_randomize_row(w, m)
    for i = 1:m+1
        w[i] = 0.1
    end
end


function bpnn_zero_weights(w, m, n)
    for i = 1:m+1
        for j = 1:n+1
            w[i,j] = 0.0
        end
    end
end

function bpnn_initialize(seed)
    println("Random number generator seed: ", seed)
    srand(seed)
end

# Creates a new fully-connected network from scratch, with the given numbers of
# input, hidden, and output units. Threshold units are automatically included.
# All weights are randomly initialized.
# Space is also allocated for temporary storage (momentum weights, error
# computations, etc).
function bpnn_create(n_in, n_hidden, n_out)
    newnet = BPNN(n_in, n_hidden, n_out)
    bpnn_randomize_weights(newnet.input_weights, n_in, n_hidden)
    bpnn_randomize_weights(newnet.hidden_weights, n_hidden, n_out)
    bpnn_zero_weights(newnet.input_prev_weights, n_in, n_hidden)
    bpnn_zero_weights(newnet.hidden_prev_weights, n_hidden, n_out)
    bpnn_randomize_row(newnet.target, n_out)

    return newnet
end

function bpnn_layerforward(l1, l2, conn, n1, n2)
    # Set up thresholding unit.
    l1[1] = 1.0
    # For each unit in second layer, compute the weighted sum of its inputs.
    for j = 2:n2+1
        sum = 0.0
        for k = 1:n1+1
            sum += conn[k, j] * l1[k]
        end
        l2[j] = squash(sum)
    end
end

function bpnn_output_error(delta, target, output, nj)
    errsum = 0.0
    for j = 2:nj+1
        o = output[j]
        t = target[j]
        delta[j] = o * (1.0 - o) * (t - o)
        errsum += abs(delta[j])
    end

    return errsum
end

function bpnn_hidden_error(delta_h, nh, delta_o, no, who, hidden)
    errsum = 0.0
    for j = 2:nh+1
        h = hidden[j]
        sum = 0.0
        for k = 2:no+1
            sum += delta_o[k] * who[j, k]
        end
        delta_h[j] = h * (1.0 - h) * sum
        errsum += abs(delta_h[j])
    end

    return errsum
end

function bpnn_adjust_weights(delta, ndelta, ly, nly, w, oldw)
    ly[1] = 1.0
    for j = 2:ndelta+1
        for k = 1:nly+1
            new_dw = ETA * delta[j] * ly[k] + MOMENTUM * oldw[k,j]
            w[k,j] += new_dw
            oldw[k,j] = new_dw
        end
    end
end

function bpnn_feedforward(net)
    inp = net.input_n
    hid = net.hidden_n
    out = net.output_n

    # Feed forward input activations.
    bpnn_layerforward(net.input_units, net.hidden_units,
                      net.input_weights, inp, hid)
    bpnn_layerforward(net.hidden_units, net.output_units,
                      net.hidden_weights, hid, out)
end

function bpnn_train(net)
    inp = net.input_n
    hid = net.hidden_n
    out = net.output_n

    # Feed forward input activations.
    bpnn_layerforward(net.input_units, net.hidden_units,
                      net.input_weights, inp, hid)
    bpnn_layerforward(net.hidden_units, net.output_units,
                      net.hidden_weights, hid, out)

    # Compute error on output and hidden units.
    out_err = bpnn_output_error(net.output_delta, net.target,
                                net.output_units, out)
    hid_err = bpnn_hidden_error(net.hidden_delta, hid, net.output_delta,
                                out, net.hidden_weights, net.hidden_units)

    # Adjust input and hidden weights.
    bpnn_adjust_weights(net.output_delta, out, net.hidden_units, hid,
                        net.hidden_weights, net.hidden_prev_weights)
    bpnn_adjust_weights(net.hidden_delta, hid, net.input_units, inp,
                        net.input_weights, net.input_prev_weights)

    return (out_err, hid_err)
end

function bpnn_save(net, filename)
    pFile = open(filename, "w+")

    n1::Int32 = net.input_n
    n2::Int32 = net.hidden_n
    n3::Int32 = net.output_n
    @printf("Saving %dx%dx%d network to '%s'\n", n1, n2, n3, filename)

    write(pFile, n1)
    write(pFile, n2)
    write(pFile, n3)

    for i = 1:size(net.input_weights, 1), j = 1:size(net.input_weights, 2)
        write(pFile, net.input_weights[i,j])
    end

    for i = 1:size(net.hidden_weights, 1), j = 1:size(net.hidden_weights, 2)
        write(pFile, net.hidden_weights[i,j])
    end
end

function bpnn_read(filename)
    fd = open(filename, "r")
    println("Reading '", filename, "'")

    n1 = read(fd, Int)
    n2 = read(fd, Int)
    n3 = read(fd, Int)
    net = bpnn_internal_create(n1, n2, n3)

    @printf("'%s' contains a %dx%dx%d network\n", filename, n1, n2, n3)
    print("Reading input weights...")

    for i = 1:n1+1
        read!(fd, net.input_weights[i])
    end

    println("Done")
    print("Reading hidden weights...")

    for i = 1:n2+1
        read!(fd, net.hidden_weights[i])
    end

    println("Done")

    bpnn_zero_weights(net.input_prev_weights, n1, n2)
    bpnn_zero_weights(net.hidden_prev_weights, n2, n3)

    return net
end
