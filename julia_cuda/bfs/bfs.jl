#!/usr/bin/env julia

using CUDAdrv, CUDAnative

const OUTPUT = haskey(ENV, "OUTPUT")

# configuration
const MAX_THREADS_PER_BLOCK = UInt32(512)

struct Node
    starting::Int32
    no_of_edges::Int32
end

function Kernel(g_graph_nodes, g_graph_edges, g_graph_mask,
                  g_updating_graph_mask, g_graph_visited, g_cost,
                  no_of_nodes)
    tid = (blockIdx().x - 1) * MAX_THREADS_PER_BLOCK + threadIdx().x;
    if tid <= no_of_nodes && g_graph_mask[tid]
        g_graph_mask[tid] = false
        for i = g_graph_nodes[tid].starting:(g_graph_nodes[tid].starting +
                                             g_graph_nodes[tid].no_of_edges - Int32(1))
            id = g_graph_edges[i]
            if !g_graph_visited[id]
                g_cost[id] = g_cost[tid] + Int32(1)
                g_updating_graph_mask[id] = true
            end
        end
    end

    return nothing
end

function Kernel2(g_graph_mask, g_updating_graph_mask, g_graph_visited,
                  g_over, no_of_nodes)
    tid = (blockIdx().x - UInt32(1)) * MAX_THREADS_PER_BLOCK + threadIdx().x;
    if tid <= no_of_nodes && g_updating_graph_mask[tid]
        g_graph_mask[tid] = true
        g_graph_visited[tid] = true
        g_over[1] = true
        g_updating_graph_mask[tid] = false
    end

    return nothing
end

# generate an expression (strs=split(str,delim); tuple(parse(Ts[1],strs[1], ...)))
@generated function parse_tuple(::Type{Ts}, str, delim) where Ts <: Tuple
    @gensym strs
    ex = Expr(:tuple)
    for i in 1:length(Ts.parameters)
        push!(ex.args, :(parse($(Ts.parameters[i]), $strs[$i])))
    end
    quote
        $(Expr(:meta, :inline))
        $strs = split(str, delim)
        $ex
    end
end

# a more sane alternative, 20% slower because of the Vector allocation:
Base.parse(::Type{Vector{T}}, str, delim) where T = map(x->parse(T, x), split(str, delim))

function main(args)
    if length(args) != 1
        error("Usage: bfs.jl <input_file>")
    end
    input_f = args[1]

    info("Reading File")
    fp = open(input_f)

    no_of_nodes = parse(Int, readline(fp))

    num_of_blocks = 1
    num_of_threads_per_block = no_of_nodes

    # Make execution Parameters according to the number of nodes
    # Distribute threads across multiple Blocks if necessary
    if no_of_nodes > MAX_THREADS_PER_BLOCK
        num_of_blocks = ceil(Integer, no_of_nodes / MAX_THREADS_PER_BLOCK)
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK
    end

    # allocate host memory
    h_graph_nodes = Vector{Node}(no_of_nodes)
    h_graph_mask = Vector{Bool}(no_of_nodes)
    h_updating_graph_mask = Vector{Bool}(no_of_nodes)
    h_graph_visited = Vector{Bool}(no_of_nodes)
    h_cost = Vector{Int32}(no_of_nodes)

    # initalize the memory
    for i = 1:no_of_nodes
        start, edgeno = parse_tuple(Tuple{Int, Int}, readline(fp), " ")
        h_graph_nodes[i] = Node(start+1, edgeno)
        h_graph_mask[i] = false
        h_updating_graph_mask[i] = false
        h_graph_visited[i] = false
        h_cost[i] = -1
    end

    skipchars(fp, isspace)

    # read the source node from the file
    source = parse(Int, readline(fp))
    source += 1

    skipchars(fp, isspace)

    # set the source node as true in the mask
    h_graph_mask[source] = true
    h_graph_visited[source] = true
    h_cost[source] = 0

    edge_list_size = parse(Int, readline(fp))

    skipchars(fp, isspace)

    h_graph_edges = Vector{Int32}(edge_list_size)
    for i = 1:edge_list_size
        id, cost = parse_tuple(Tuple{Int, Int}, readline(fp), " ")
        h_graph_edges[i] = id+1
    end

    close(fp)
    info("Read File")

    # setup execution parameters
    grid = num_of_blocks
    threads = num_of_threads_per_block


    # Manual copies to device
    g_graph_nodes = CuArray(h_graph_nodes)
    g_graph_edges = CuArray(h_graph_edges)
    g_graph_mask  = CuArray(h_graph_mask)
    g_updating_graph_mask = CuArray(h_updating_graph_mask)
    g_graph_visited = CuArray(h_graph_visited)
    g_cost = CuArray(h_cost)
    g_stop = CuArray{Bool}(1)

    k = 0
    info("Start traversing the tree")
    stop = Bool[1]

    while true
        # if no thread changes this value then the loop stops
        stop[1] = false
        copy!(g_stop, stop)

        @cuda (grid, threads) Kernel(
            g_graph_nodes, g_graph_edges, g_graph_mask,
            g_updating_graph_mask, g_graph_visited,
            g_cost, no_of_nodes
        )

        @cuda (grid, threads) Kernel2(
            g_graph_mask, g_updating_graph_mask, g_graph_visited,
            g_stop, no_of_nodes
        )

        k += 1
        copy!(stop, g_stop)
        if !stop[1]
            break
        end
    end

    # Copy result back + free
    h_cost = Array(g_cost)

    info("Kernel Executed $k times")

    # Store the result into a file
    if OUTPUT
        open("output.txt", "w") do fpo
            for i = 1:no_of_nodes
                write(fpo, "$(i-1)) cost:$(h_cost[i])\n")
            end
        end
    end
end


main(ARGS)

if haskey(ENV, "PROFILE")
    CUDAnative.@profile main(ARGS)
end
