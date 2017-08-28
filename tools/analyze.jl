#!/usr/bin/env julia

include("common.jl")


measurements = readtable("measurements.dat")

# create a summary with a column per suite (except the baseline)
analysis = DataFrame(benchmark=String[], kernel=String[])
for suite in non_baseline
    analysis[Symbol(suite)] = Float64[]
end

# add time totals for each benchmark
# NOTE: we do this before summarizing across iterations, to make totals more fair
#       (ie. the totals are affected by the amount of iterations for each kernel)
# NOTE: we must normalize these timings by the number of iterations,
#       as not every suite might have executed the same number of times
append!(measurements, by(measurements, [:suite, :benchmark],
                         dt->DataFrame(kernel="total",
                                       time=sum(dt[:time])/size(dt, 1))))

# summarize across iterations
grouped_measurements = by(measurements, [:suite, :benchmark, :kernel],
                          dt->DataFrame(time=minimum(dt[:time])))

# calculate the slowdown/improvement compared against the baseline
for benchmark in unique(grouped_measurements[:benchmark])
    # get the measurements for this benchmark
    benchmark_data = grouped_measurements[grouped_measurements[:benchmark] .== benchmark, :]
    for kernel in unique(benchmark_data[:kernel])
        # get the measurements for this kernel
        kernel_data = benchmark_data[benchmark_data[:kernel] .== kernel, :]
        if sort(kernel_data[:suite]) != sort(suites)
            warn("$benchmark - $kernel: don't have measurements for all suites")
            continue
        end

        # compare other suites against the chosen baseline
        baseline_data = kernel_data[kernel_data[:suite] .== baseline, :]
        others_data = kernel_data[kernel_data[:suite] .!= baseline, :]
        for suite in others_data[:suite]
            suite_data = kernel_data[kernel_data[:suite] .== suite, :]
            difference = suite_data[:time][1] / baseline_data[:time][1]
            push!(analysis, [benchmark kernel difference])
        end
    end
end

# add difference totals for each suite (based on previous totals)
# NOTE: this total only looks at each benchmark's performance increase/loss,
#       not only ignores the iteration count, but the execution time altogether
# FIXME: can't we do this with a `by`, summarizing over all remaining columns?
totals = []
for suite in names(analysis)[3:end]
    push!(totals, mean(analysis[analysis[:kernel] .== "total", suite]))
end
push!(analysis, ["total", "total", totals...])

writetable("analysis.dat", analysis)
for suite in non_baseline
    info("$suite vs $baseline:")
    println(suite_stats(analysis, suite))
end
