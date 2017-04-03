using CUDAdrv

struct Invocation
    start::CuEvent
    stop::CuEvent

    Invocation() = new(CuEvent(), CuEvent())
end

measure_launch(inv::Invocation) = record(inv.start)
measure_finish(inv::Invocation) = record(inv.stop)

const kernels = Dict{String,Vector{Invocation}}()
function measure_launch(id::String)
    inv = Invocation()
    push!(get!(kernels, id, Invocation[]), inv)
    measure_launch(inv)
    return inv
end

macro measure(id::String, expr)
    quote
        inv = measure_launch($id)
        $(esc(expr))
        measure_finish(inv)
    end
end

function report()
    println()
    info("Kernel profile report for $(length(kernels)) kernels:")
    for (id, invs) in kernels
        times = Float64[]
        for inv in invs
            synchronize(inv.stop)
            push!(times, elapsed(inv.start, inv.stop))
        end
        if length(times) > 1
            shift!(times)
        end
        info(" - $id (x $(length(invs))): " *
             "min $(round(Int, 1_000_000*minimum(times))) µs, " *
             "mean $(round(Int, 1_000_000*mean(times))) " *
             (length(times)>1 ? "± $(round(Int, 1_000_000*std(times))) µs" : " µs"))
    end
    empty!(kernels)
end
