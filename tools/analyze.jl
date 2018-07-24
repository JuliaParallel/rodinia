#!/usr/bin/env julia

# include("common.jl")


function analyze(host=gethostname(), suite="julia_cuda")
    measurements = CSV.read("measurements_$host.dat")
    grouped = summarize(measurements)
    delete!(grouped, [:kernel_invocations, :benchmark_executions])

    # add device totals for each benchmark
    grouped_kernels = filter(entry->!startswith(entry[:target], '#'), grouped)
    grouped = vcat(grouped,
                   by(grouped_kernels, [:suite, :benchmark],
                      dt->DataFrame(target = "#device",
                                    time = sum(dt[:time]))))

    # XXX: add compilation time to host measurements
    # grouped[(grouped[:suite].==suite) .& (grouped[:target].=="#host"), :time] +=
    #     grouped[(grouped[:suite].==suite) .& (grouped[:target].=="#compilation"), :time]

    # select values for the requested suite, adding ratios against the baseline
    analysis = by(grouped, [:benchmark, :target]) do df
        baseline_data = df[df[:suite] .== baseline, :]
        suite_data = df[df[:suite] .== suite, :]
        time = if size(suite_data, 1) == 1
            suite_data[1, :time]
        else
            missing
        end
        ratio = if size(baseline_data,1) == 1 && size(suite_data, 1) == 1
            suite_data[1, :time] / baseline_data[1, :time]
        else
            missing
        end
        DataFrame(time = time, ratio = ratio)
    end

    # calculate ratio grand totals
    geomean(x) = prod(x)^(1/length(x))  # ratios are normalized, so use geomean
    totals = by(filter(row->startswith(row[:target], '#'), analysis), [:target]) do df
        DataFrame(time = sum(df[:time]), ratio = geomean(df[:ratio]))
    end
    totals = hcat(DataFrame(benchmark=fill("#all", size(totals,1))), totals)
    analysis = vcat(analysis, totals)

    @info "Analysis complete"
    println(filter(row->startswith(row[:target], '#'), analysis))
    CSV.write("analysis_$(host)_$suite.csv", analysis; header=true)

    return
end
