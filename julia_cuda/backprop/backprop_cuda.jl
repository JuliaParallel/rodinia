# configuration
const WIDTH = 16    # shared memory width
const HEIGHT = 16   # shared memory height
const THREADS = 256

function bpnn_train_cuda(net)
    inp = net.input_n
    hid = net.hidden_n
    out = net.output_n

    m = 1
    num_blocks = Int(inp / 16)

    input_weights_one_dim = Array{Float32}((inp + 1) * (hid + 1))
    input_weights_prev_one_dim = Array{Float32}((inp + 1) * (hid + 1))
    partial_sum = Array{Float32}(num_blocks * WIDTH)

    # This preprocessing stage is added to correct the bugs of wrong memcopy
    # using two-dimensional net->inputweights.
    for k = 0:inp, j = 0:hid
        input_weights_one_dim[m] = net.input_weights[k + 1, j + 1]
        input_weights_prev_one_dim[m] = net.input_prev_weights[k + 1, j + 1]
        m += 1
    end

    input_cuda = CuArray(net.input_units)
    output_hidden_cuda = CuArray{Float32}(hid + 1)
    input_hidden_cuda = CuArray(input_weights_one_dim)
    hidden_partial_sum = CuArray{Float32}(num_blocks * WIDTH)

    println("Performing GPU computation")

    @cuda ((1, num_blocks), (16, 16)) bpnn_layerforward_CUDA(
        input_cuda, output_hidden_cuda, input_hidden_cuda, hidden_partial_sum, inp, hid)

    partial_sum = Array(hidden_partial_sum)

    for j = 1:hid
        sum = 0.0
        for k = 0:num_blocks-1
            sum += partial_sum[k * hid + j]
        end
        sum += net.input_weights[1, j + 1]
        net.hidden_units[j + 1] = 1.0 / (1.0 + exp(-sum))
    end

    bpnn_layerforward(net.hidden_units, net.output_units, net.hidden_weights,
                      hid, out)
    bpnn_output_error(net.output_delta, net.target, net.output_units, out)
    bpnn_hidden_error(net.hidden_delta, hid, net.output_delta, out,
                      net.hidden_weights, net.hidden_units)
    bpnn_adjust_weights(net.output_delta, out, net.hidden_units, hid,
                        net.hidden_weights, net.hidden_prev_weights)

    hidden_delta_cuda = CuArray(net.hidden_delta)
    input_prev_weights_cuda = CuArray(input_weights_prev_one_dim)

    input_hidden_cuda = CuArray(input_weights_one_dim)

	@cuda ((1, num_blocks), (16, 16)) bpnn_adjust_weights_cuda(
        hidden_delta_cuda, hid, input_cuda, inp, input_hidden_cuda, input_prev_weights_cuda)

	net.input_units = Array(input_cuda)
end
