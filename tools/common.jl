using DataFrames, CSV
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
const MIN_KERNEL_ITERATIONS = 5
const MAX_KERNEL_ERROR      = 0.01
const MAX_BENCHMARK_RUNS    = 100
const MAX_BENCHMARK_SECONDS = 300

function allequal(it)
    items = unique(it)
    @assert length(items) == 1
    return first(items)
end

# summarize across executions
function summarize(measurements)
    # first, sum timings of identical kernels within an execution
    # (or the sum of all device timings within an execution wouldn't be correct,
    # after taking the average below)
    grouped = by(measurements, [:suite, :benchmark, :target, :execution],
                 df->DataFrame(time = sum(df[:time]),
                               target_iterations = length(df[:time])))

    # next, take the average across all executions
    grouped = by(grouped, [:suite, :benchmark, :target],
                 df->DataFrame(time = measurement(mean(df[:time]), std(df[:time])),
                               target_iterations = allequal(df[:target_iterations]),
                               benchmark_iterations = length(df[:time])))

    grouped
end

# tools for accessing analysis results
function suite_stats(analysis, suite)
    df = filter(row->startswith(row[:target], '#'),
                analysis)[[:benchmark, :target, Symbol(suite)]]
    names!(df, [:benchmark, :timing, :ratio])
    return df
end
benchmark_stats(analysis, benchmark) = analysis[analysis[:benchmark] .== benchmark, :]
