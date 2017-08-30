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
const MIN_KERNEL_ITERATIONS = 25
const MAX_KERNEL_ERROR      = 0.02
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
        d = fit(LogNormal, dt[key])
        kwargs = Any[(key, measurement(d.μ, d.σ))]
        for field in fields
            push!(kwargs, (field[1], field[2](dt)))
        end
        return DataFrame(;kwargs...)
    end
    return by(data, across, f)
end 
