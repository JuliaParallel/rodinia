include("backprop.jl")

const OUTPUT = haskey(ENV, "OUTPUT")

function backprop_face(layer_size)
    net = bpnn_create(layer_size, 16, 1) # (16, 1 cannot be changed)
    println("Input layer size : ", layer_size)

    units = net.input_units
    for i = 2:layer_size+1
        units[i] = float(rand()) / RAND_MAX
    end

    # Entering the training kernel, only one iteration.
    println("Starting training kernel")
    bpnn_train_kernel(net)

    if OUTPUT
        bpnn_save(net, "output.dat")
    end

    println("Training done")
end

function bpnn_train_kernel(net)
    inp = net.input_n
    hid = net.hidden_n
    out = net.output_n

    println("Performing CPU computation")
    bpnn_layerforward(net.input_units, net.hidden_units, net.input_weights, inp, hid)
    bpnn_layerforward(net.hidden_units, net.output_units, net.hidden_weights, hid, out)
    bpnn_output_error(net.output_delta, net.target, net.output_units, out)
    bpnn_hidden_error(net.hidden_delta, hid, net.output_delta, out,
                      net.hidden_weights, net.hidden_units)
    bpnn_adjust_weights(net.output_delta, out, net.hidden_units, hid,
                        net.hidden_weights, net.hidden_prev_weights)
    bpnn_adjust_weights(net.hidden_delta, hid, net.input_units, inp,
                        net.input_weights, net.input_prev_weights)
end

################################################################################
# Program main
################################################################################
function main(args)
    if length(args) != 1
        println(STDERR, "usage: backprop <num of input elements>");
        exit(1)
    end

    layer_size = parse(Int, args[1])

    if layer_size % 16 != 0
        @printf(STDERR, "The number of input points must be divisible by 16\n")
        exit(1)
    end

    bpnn_initialize(7)
    backprop_face(layer_size)
end

main(ARGS)
