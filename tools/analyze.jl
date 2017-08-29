#!/usr/bin/env julia

using Distributions

include("common.jl")


measurements = readtable("measurements.dat")

# ∆ = absolute uncertainty
# ε = relative uncertainty

# summarize across executions
grouped = by(measurements, [:suite, :benchmark, :kernel],
             dt->DataFrame( time = fit(LogNormal, dt[:time]).μ,
                           ∆time = fit(LogNormal, dt[:time]).σ))

# add time totals for each benchmark
append!(grouped, by(grouped, [:suite, :benchmark],
                    dt->DataFrame(kernel = "total",
                                    time = sum(dt[:time]),
                                   ∆time = sum(dt[:∆time]))))
grouped[:εtime] = grouped[:∆time] ./ abs(grouped[:time])

info("Aggregated timings:")
println(grouped[grouped[:kernel] .== "total", :])

# create a summary with a column per suite (except the baseline)
analysis = DataFrame(benchmark=String[], kernel=String[])
for suite in non_baseline
    analysis[Symbol(suite)] = Float64[]
    analysis[Symbol(:∆, suite)] = Float64[]
    analysis[Symbol(:ε, suite)] = Float64[]
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
        for suite in others_data[:suite]
            suite_data = kernel_data[kernel_data[:suite] .== suite, :]
            difference = suite_data[:time][1] / baseline_data[:time][1]
            εdifference = suite_data[:εtime][1] + baseline_data[:εtime][1]
            ∆difference = difference * εdifference
            push!(analysis, [benchmark kernel difference ∆difference εdifference])
        end
    end
end

# calculate per-suite totals
geomean(x) = prod(x)^(1/length(x))  # analysis contains normalized numbers, so use geomean
totals = []
for suite in non_baseline
    total = geomean(analysis[analysis[:kernel] .== "total", Symbol(suite)])
    εtotal = sum(analysis[analysis[:kernel] .== "total", Symbol(:ε, suite)])
    ∆total = εtotal * total
    append!(totals, [total, ∆total, εtotal])
end
push!(analysis, ["total", "total", totals...])

writetable("analysis.dat", analysis)
for suite in non_baseline
    info("Speedup of $suite vs $baseline:")
    println(suite_stats(analysis, suite))
end
