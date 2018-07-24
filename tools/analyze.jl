#!/usr/bin/env julia

# include("common.jl")


function analyze(host=gethostname())
    measurements = CSV.read("measurements_$host.dat")
    grouped = summarize(measurements)
    delete!(grouped, [:kernel_invocations, :benchmark_executions])

    # add device totals for each benchmark
    grouped_kernels = filter(entry->entry[:target] != "host", grouped)
    grouped = vcat(grouped,
                   by(grouped_kernels, [:suite, :benchmark],
                      dt->DataFrame(target = "device",
                                    time = sum(dt[:time]))))

    info("Aggregated timings:")
    println(filter(entry->entry[:target] == "device" ||
                          entry[:target] == "host",
                   grouped))

    # create a summary with one column per suite (except for the baseline)
    analysis = DataFrame(benchmark=String[], target=String[])
    for suite in non_baseline
        analysis[Symbol(suite)] = Measurement{Float64}[]
    end

    # calculate the slowdown/improvement compared against the baseline
    for benchmark in unique(grouped[:benchmark])
        # get the measurements for this benchmark
        benchmark_data = grouped[grouped[:benchmark] .== benchmark, :]
        for target in unique(benchmark_data[:target])
            # get the measurements for this target
            target_data = benchmark_data[benchmark_data[:target] .== target, :]
            if sort(target_data[:suite]) != sort(suites)
                warn("$benchmark - $target: don't have measurements for all suites")
                continue
            end

            # compare other suites against the chosen baseline
            baseline_data = target_data[target_data[:suite] .== baseline, :]
            others_data = target_data[target_data[:suite] .!= baseline, :]
            ratios = Measurement{Float64}[]
            for suite in non_baseline
                suite_data = target_data[target_data[:suite] .== suite, :]
                ratio = suite_data[:time][1] / baseline_data[:time][1]
                push!(ratios, ratio)
            end
            push!(analysis, [benchmark target ratios...])
        end
    end

    # calculate grand totals
    geomean(x) = prod(x)^(1/length(x))  # analysis contains normalized numbers, so use geomean
    for total in ["host", "device"]
        total_times = Measurement{Float64}[]
        for suite in non_baseline
            total_time = geomean(analysis[analysis[:target] .== total, Symbol(suite)])
            push!(total_times, total_time)
        end
        push!(analysis, ["total", total, total_times...])
    end

    CSV.write("analysis_$host.dat", analysis)
    for suite in non_baseline
        info("Performance ratio of $suite vs $baseline:")
        println(suite_stats(analysis, suite))
    end
end
