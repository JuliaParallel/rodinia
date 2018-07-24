#!/usr/bin/env julia

# include("common.jl")


function analyze(host=gethostname())
    measurements = CSV.read("measurements_$host.dat")
    grouped = summarize(measurements)
    delete!(grouped, [:kernel_invocations, :benchmark_executions])

    # add device totals for each benchmark
    grouped_kernels = filter(entry->!startswith(entry[:target], '#'), grouped)
    grouped = vcat(grouped,
                   by(grouped_kernels, [:suite, :benchmark],
                      dt->DataFrame(target = "#device",
                                    time = sum(dt[:time]))))

    # create summary dataframes with one column per suite
    ## with timings
    time_analysis = DataFrame(benchmark=String[], target=String[])
    for suite in suites
        time_analysis[Symbol(suite)] = Union{Missing,Measurement{Float64}}[]
    end
    ## with ratios
    ratio_analysis = DataFrame(benchmark=String[], target=String[])
    for suite in non_baseline
        ratio_analysis[Symbol(suite)] = Union{Missing,Measurement{Float64}}[]
    end

    for benchmark in unique(grouped[:benchmark])
        # get the measurements for this benchmark
        benchmark_data = grouped[grouped[:benchmark] .== benchmark, :]
        for target in unique(benchmark_data[:target])
            # get the measurements for this target
            target_data = benchmark_data[benchmark_data[:target] .== target, :]

            timings = Union{Missing,Measurement{Float64}}[]
            for suite in suites
                measurement = target_data[target_data[:suite] .== suite, :time]
                if size(measurement, 1) == 1
                    push!(timings, first(measurement))
                else
                    @assert size(measurement, 1) == 0
                    push!(timings, missing)
                end
            end
            push!(time_analysis, [benchmark target timings...])

            # TODO: make sure multiple #compilation's are dealt with properly

            # calculate performance ratios against the baseline
            ratios = Union{Missing,Measurement{Float64}}[]
            if baseline in target_data[:suite]
                baseline_data = target_data[target_data[:suite] .== baseline, :]
                others_data = target_data[target_data[:suite] .!= baseline, :]
                for suite in non_baseline
                    ratio = if suite in others_data[:suite]
                        suite_data = target_data[target_data[:suite] .== suite, :]
                        suite_data[:time][1] / baseline_data[:time][1]
                    else
                        missing
                    end
                    push!(ratios, ratio)
                end
            else
                for suite in non_baseline
                    push!(ratios, missing)
                end
            end
            push!(ratio_analysis, [benchmark target ratios...])
        end
    end

    # calculate ratio grand totals
    geomean(x) = prod(x)^(1/length(x))  # analysis contains normalized numbers, so use geomean
    ranges = filter(target->startswith(target, '#'), unique(ratio_analysis[:target]))
    for range in ranges
        times = Union{Missing,Measurement{Float64}}[]
        for suite in non_baseline
            total_time = geomean(ratio_analysis[ratio_analysis[:target] .== range, Symbol(suite)])
            push!(times, total_time)
        end
        push!(ratio_analysis, ["#all", range, times...])
    end

    @info("Analysis complete",
          timings           = filter(row->startswith(row[:target], '#'), time_analysis),
          performance_ratio = filter(row->startswith(row[:target], '#'), ratio_analysis))
    CSV.write("analysis_time_$host.dat", time_analysis)
    CSV.write("analysis_ratio_$host.dat", ratio_analysis)

    return
end
