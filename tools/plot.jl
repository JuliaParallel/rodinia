#!/usr/bin/env julia

const hasplots = try
    using Plots
    pyplot()
    true
catch
    false
end

function plot(host=gethostname())
    analysis = readtable("analysis_$host.dat")
    for suite in non_baseline
        analysis[Symbol(suite)] = map(str->measurement(str), analysis[Symbol(suite)])
    end

    for suite in non_baseline
        df = suite_stats(analysis, suite)
        df[:speedup] = -1.*df[:ratio].+1
        delete!(df, :ratio)

        df[:error]   = map(x->x.err, df[:speedup])
        df[:speedup] = map(x->x.val, df[:speedup])

        total = df[df[:benchmark] .== "total", :speedup][1]
        df = df[df[:benchmark] .!= "total", :]
        writedlm("$(host)-$(suite)_total.csv", total)

        sort!(df, cols=:speedup; rev=true)
        writetable("$(host)-$suite.csv", df; header=true)

        if hasplots
            labels = df[:benchmark]

            # speed-ups
            speedup = map(i->min(0,i), df[:speedup])
            bar(labels,
                100*speedup;
                legend=false,
                rotation=45,
                color=:red,
                xlabel = "benchmark",
                ylabel = "speedup difference (%)")

            # slow-downs
            speedup = map(i->max(0,i), df[:speedup])
            bar!(labels,
                100*speedup;
                color=:green)

            png(suite)
        end
    end
end
