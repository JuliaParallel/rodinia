module KernelProfile

export @measure

using CUDAdrv

struct Invocation
    start::CuEvent
    stop::CuEvent

    Invocation() = new(CuEvent(), CuEvent())
end

const enabled = Ref(false)
const profiling = Ref(false)

function measure_launch(inv::Invocation)
    if enabled[] && !profiling[]
        # lazy-start the profiler to get a narrow scope
        CUDAdrv.start_profiler()
        profiling[] = true
    end
    record(inv.start)
end
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

enable() = enabled[] = true

function report()
    if enabled[] && profiling[]
        CUDAdrv.stop_profiler()
        profiling[] = false
    end

    benchmark = basename(dirname(Base.source_path()))
    for (id, invs) in kernels
        # gather data
        for inv in invs
            synchronize(inv.stop)

            # print csv
            println("$benchmark,$id,$(1_000_000*elapsed(inv.start, inv.stop))")
        end
    end

    empty!(kernels)
end

end

using KernelProfile
