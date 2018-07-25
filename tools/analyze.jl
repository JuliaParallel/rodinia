#!/usr/bin/env julia

# include("common.jl")


function analyze(host=gethostname(), dst=nothing, suite="julia_cuda")
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

        reference = size(baseline_data, 1) == 1 ? baseline_data[1, :time] : missing
        time      = size(suite_data, 1) == 1    ?  suite_data[1, :time]   : missing
        ratio = ismissing(reference) || ismissing(time) ? missing : time / reference
        DataFrame(reference = reference, time = time, ratio = ratio)
    end

    # calculate ratio grand totals
    geomean(x) = prod(x)^(1/length(x))  # ratios are normalized, so use geomean
    totals = by(filter(row->startswith(row[:target], '#'), analysis), [:target]) do df
        DataFrame(reference = sum(df[:reference]), time = sum(df[:time]),
                  ratio = geomean(df[:ratio]))
    end
    totals = hcat(DataFrame(benchmark=fill("#all", size(totals,1))), totals)
    analysis = vcat(analysis, totals)

    @info "Analysis complete"
    println(filter(row->startswith(row[:target], '#'), analysis))

    # prepare measurements for PGF
    for column in [:reference, :time, :ratio]
        analysis[Symbol("$(column)_val")] = map(datum->ismissing(datum)?missing:datum.val,
                                                analysis[column])
        analysis[Symbol("$(column)_err")] = map(datum->ismissing(datum)?missing:datum.err,
                                                analysis[column])
        delete!(analysis, column)
    end

    let analysis = filter(row->!startswith(row[:benchmark], '#') && startswith(row[:target], '#'), analysis)
        analysis = analysis[[:benchmark, :target, :time_val, :time_err, :reference_val, :reference_err]]
        sort!(analysis, cols=[:benchmark, :target]; rev=true)
        if dst != nothing
            CSV.write(joinpath(dst, "perf.csv"), analysis; header=true)
        end
    end

    let analysis = filter(row->!startswith(row[:benchmark], '#') && row[:target] == "#device", analysis)
        analysis = analysis[[:benchmark, :ratio_val, :ratio_err]]
        analysis[:ratio_val] = 1 - analysis[:ratio_val]
        rename!(analysis, :ratio_val=>:speedup_val, :ratio_err=>:speedup_err)
        sort!(analysis, cols=:speedup_val; rev=true)
        if dst != nothing
            CSV.write(joinpath(dst, "perf_device.csv"), analysis; header=true)
        end
    end

    let analysis = analysis[(analysis[:benchmark] .== "#all") .& (analysis[:target] .== "#device"), :ratio_val]
        if dst != nothing
            open(joinpath(dst, "perf_device_total.csv"), "w") do io
                println(io, analysis[1])
            end
        end
    end

    return
end
