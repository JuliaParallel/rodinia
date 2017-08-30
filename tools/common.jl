using DataFrames
using Measurements

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
# NOTE: because of how we calculate totals (per-benchmark totals based on time x iterations,
#       per-suite benchmarks based on flat performance difference) it is possible to gather
#       more data for individual benchmarks, but not for individual kernels (as that would
#       skew the per-benchmark totals)
const MIN_KERNEL_ITERATIONS  = 10
const MAX_KERNEL_UNCERTAINTY = 0.02
const MAX_BENCHMARK_RUNS     = 100
const MAX_BENCHMARK_SECONDS  = 300

# tools for accessing analysis results
function suite_stats(analysis, suite)
    df = analysis[analysis[:kernel] .== "total", [:benchmark, Symbol(suite)]]
    names!(df, [:benchmark, :ratio])
    return df
end
benchmark_stats(analysis, benchmark) = analysis[analysis[:benchmark] .== benchmark, :]
