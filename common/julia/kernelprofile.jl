module KernelProfile

export @measure

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

macro measure(id, expr)
    quote
        inv = measure_launch($(esc(id)))
        $(esc(expr))
        measure_finish(inv)
    end
end

clear() = empty!(kernels)

function report()
    benchmark = basename(dirname(Base.source_path()))
    for (id, invs) in kernels
        # gather data
        times = Float64[]
        for inv in invs
            synchronize(inv.stop)
            push!(times, elapsed(inv.start, inv.stop))
        end
        times .*= 1_000_000     # s to μs

        # calculate metric and print csv
        fields = [benchmark
                  id
                  length(times)
                  minimum(times)
                  median(times)
                  mean(times)
                  maximum(times)
                  length(times)>1 ? round(Int, std(times)) : 0
                 ]
        writecsv(STDOUT, reshape(fields, (1, length(fields))))
    end
end

end

using KernelProfile
