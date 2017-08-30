#!/usr/bin/env julia

include("common.jl")


measurements = readtable("measurements.dat")

# summarize across executions
# NOTE: this isn't entirely correct, because it also aggregates
#       across kernel iterations within a single execution
grouped = summarize(measurements, [:suite, :benchmark, :kernel], :time)

# add time totals for each benchmark
append!(grouped, by(grouped, [:suite, :benchmark],
                    dt->DataFrame(kernel = "total",
                                    time = sum(dt[:time]))))

info("Aggregated timings:")
println(grouped[grouped[:kernel] .== "total", :])

# create a summary with one column per suite (except for the baseline)
analysis = DataFrame(benchmark=String[], kernel=String[])
for suite in non_baseline
    analysis[Symbol(suite)] = Measurement{Float64}[]
end

# calculate the slowdown/improvement compared against the baseline
for benchmark in unique(grouped[:benchmark])
    # get the measurements for this benchmark
    benchmark_data = grouped[grouped[:benchmark] .== benchmark, :]
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
        ratios = Measurement{Float64}[]
        for suite in non_baseline
            suite_data = kernel_data[kernel_data[:suite] .== suite, :]
            ratio = suite_data[:time][1] / baseline_data[:time][1]
            push!(ratios, ratio)
        end
        push!(analysis, [benchmark kernel ratios...])
    end
end

# calculate per-suite totals
geomean(x) = prod(x)^(1/length(x))  # analysis contains normalized numbers, so use geomean
totals = Measurement{Float64}[]
for suite in non_baseline
    total = geomean(analysis[analysis[:kernel] .== "total", Symbol(suite)])
    push!(totals, total)
end
push!(analysis, ["total", "total", totals...])

writetable("analysis.dat", analysis)
for suite in non_baseline
    info("Performance ratio of $suite vs $baseline:")
    println(suite_stats(analysis, suite))
end
