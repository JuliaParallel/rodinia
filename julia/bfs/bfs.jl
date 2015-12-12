#!/usr/bin/env julia

using CUDA

const MAX_THREADS_PER_BLOCK = 512

immutable Node
    starting::Int32;
    no_of_edges::Int32
end

@target ptx function kernel_1(g_graph_nodes::CuDeviceArray{Node},
                              g_graph_edges::CuDeviceArray{Int32},
                              g_graph_mask::CuDeviceArray{Bool},
                              g_updating_graph_mask::CuDeviceArray{Bool},
                              g_graph_visited::CuDeviceArray{Bool},
                              g_cost::CuDeviceArray{Int32},
                              no_of_nodes::Int32)
    tid = (blockIdx().x - 1) * MAX_THREADS_PER_BLOCK + threadIdx().x;
    if tid <= no_of_nodes && g_graph_mask[tid]
        g_graph_mask[tid] = false
        for i = g_graph_nodes[tid].starting:(g_graph_nodes[tid].starting +
                                             g_graph_nodes[tid].no_of_edges - 1)
            id = g_graph_edges[i]
            if !g_graph_visited[id]
                g_cost[id] = g_cost[tid] + Int32(1)
                g_updating_graph_mask[id] = true
            end
        end
    end

    return nothing
end

@target ptx function kernel_2(g_graph_mask::CuDeviceArray{Bool},
                              g_updating_graph_mask::CuDeviceArray{Bool},
                              g_graph_visited::CuDeviceArray{Bool},
                              g_over::CuDeviceArray{Bool},
                              no_of_nodes::Int32)
    tid = (blockIdx().x - 1) * MAX_THREADS_PER_BLOCK + threadIdx().x;
    if tid <= no_of_nodes && g_updating_graph_mask[tid]
        g_graph_mask[tid] = true
        g_graph_visited[tid] = true
        g_over[1] = true
        g_updating_graph_mask[tid] = false
    end

    return nothing
end

function parseline{T<:Number}(::Type{T}, f)
    while true
        line = chomp(readline(f))
        if length(line) > 0
            return map(s -> parse(T, s), split(line))
        end
    end
    return T[]
end

function main(args)
    dev = CuDevice(0)
    ctx = CuContext(dev)
    cgctx = CuCodegenContext(ctx, dev)

    if length(args) != 1
        error("Usage: bfs.jl <input_file)")
    end
    input_f = args[1]

    info("Reading File")
    fp = open(input_f)

    no_of_nodes, = parseline(Int, fp)

    num_of_blocks = 1
    num_of_threads_per_block = no_of_nodes

    # Make execution Parameters according to the number of nodes
    # Distribute threads across multiple Blocks if necessary
    if no_of_nodes > MAX_THREADS_PER_BLOCK
        num_of_blocks = ceil(Integer, no_of_nodes / MAX_THREADS_PER_BLOCK)
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK
    end

    # allocate host memory
    h_graph_nodes = Array{Node, 1}(no_of_nodes)
    h_graph_mask = Array{Bool, 1}(no_of_nodes)
    h_updating_graph_mask = Array{Bool, 1}(no_of_nodes)
    h_graph_visited = Array{Bool, 1}(no_of_nodes)
    h_cost = Array{Int32, 1}(no_of_nodes)

    # initalize the memory
    for i = 1:no_of_nodes
        start, edgeno = parseline(Int, fp)
        h_graph_nodes[i] = Node(start+1, edgeno)
        h_graph_mask[i] = false
        h_updating_graph_mask[i] = false
        h_graph_visited[i] = false
        h_cost[i] = -1
    end

    # read the source node from the file
    source, = parseline(Int, fp)
    source += 1

    # set the source node as true in the mask
    h_graph_mask[source] = true
    h_graph_visited[source] = true
    h_cost[source] = 0

    edge_list_size, = parseline(Int, fp)

    h_graph_edges = Array{Int32, 1}(edge_list_size)
    for i = 1:edge_list_size
        id = parse(Int, readuntil(fp, " "))+1
        cost = parse(Int, readuntil(fp, "\n"))
        h_graph_edges[i] = id
    end

    close(fp)
    info("Read File")

    # setup execution parameters
    grid = (num_of_blocks, 1, 1)
    threads = (num_of_threads_per_block, 1, 1)

    k = 0
    info("Start traversing the tree")
    stop = Bool[1]
    while true
        # if no thread changes this value then the loop stops
        stop[1] = false

        @cuda (grid, threads, 0) kernel_1(
            h_graph_nodes, h_graph_edges, h_graph_mask,
            h_updating_graph_mask, h_graph_visited,
            h_cost, Int32(no_of_nodes)
        )

        @cuda (grid, threads, 0) kernel_2(
            h_graph_mask, h_updating_graph_mask, h_graph_visited,
            stop, Int32(no_of_nodes)
        )

        k += 1
        if !stop[1]
            break
        end
    end

    info("Kernel Executed $k times")

    # Store the result into a file
    if haskey(ENV, "OUTPUT")
        open("output.txt", "w") do fpo
            for i = 1:no_of_nodes
                write(fpo, "$(i-1)) cost:$(h_cost[i])\n")
            end
        end
    end
end

main(ARGS)
