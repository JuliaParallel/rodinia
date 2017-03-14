using CUDAdrv

struct Invocation
    start::CuEvent
    stop::CuEvent

    Invocation() = new(CuEvent(), CuEvent())
end

launch(inv::Invocation) = record(inv.start)
finish(inv::Invocation) = record(inv.stop)

const kernels = Dict{String,Vector{Invocation}}()
function launch(id::String)
    inv = Invocation()
    push!(get!(kernels, id, Invocation[]), inv)
    launch(inv)
    return inv
end

macro measure(id::String, expr)
    quote
        inv = launch($id)
        $(esc(expr))
        finish(inv)
    end
end

function report()
    for (id, invs) in kernels
        times = Float64[]
        for inv in invs
            synchronize(inv.stop)
            push!(times, elapsed(inv.start, inv.stop))
        end
        shift!(times)
        info("Kernel launch $id: " *
             "min $(trunc(Int, 1_000_000*minimum(times))) µs, " *
             "mean $(trunc(Int, 1_000_000*mean(times))) " *
             "± $(trunc(Int, 1_000_000*std(times))) µs")
    end
    empty!(kernels)
end
