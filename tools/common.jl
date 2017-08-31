using DataFrames
using Measurements
using Distributions

const suites = ["cuda", "julia_cuda"]   # which benchmark suites to process
const baseline = "cuda"
const non_baseline = filter(suite->suite!=baseline, suites)
const root = dirname(@__DIR__)

# some benchmarks spend a variable amount of time per kernel,
# making it impossible to aggregate across kernel iterations.
const irregular_kernels = Dict(
    "bfs"             => ["Kernel", "Kernel2"],
    "leukocyte"       => ["IMGVF_kernel"],
    "lud"             => ["lud_perimeter", "lud_internal"],
    "particlefilter"  => ["find_index_kernel"],
    "nw"              => ["needle_cuda_shared_1", "needle_cuda_shared_2"],
)

# measurement parameters
const MIN_KERNEL_ITERATIONS = 5
const MAX_KERNEL_ERROR      = 0.01
const MAX_BENCHMARK_RUNS    = 100
const MAX_BENCHMARK_SECONDS = 300

# tools for accessing analysis results
function suite_stats(analysis, suite)
    df = analysis[analysis[:kernel] .== "total", [:benchmark, Symbol(suite)]]
    names!(df, [:benchmark, :ratio])
    return df
end
benchmark_stats(analysis, benchmark) = analysis[analysis[:benchmark] .== benchmark, :]

# helper function for averaging measurements using a lognormal distribution
function summarize(data, across::Vector{Symbol}, key::Symbol; fields...)
    function f(dt)
        if length(unique(dt[key])) == 1
            # all values identical, cannot estimate a distribution.
            # assume a value without error (hence a minimal iteration parameter)
            x = measurement(first(dt[key]))
        else
            d = fit(LogNormal, dt[key])
            x = measurement(d.μ, d.σ)
        end
        kwargs = Any[(key, x)]

        for field in fields
            push!(kwargs, (field[1], field[2](dt)))
        end

        return DataFrame(;kwargs...)
    end
    return by(data, across, f)
end 
