#!/usr/bin/env julia

using Plots
pyplot()

include("common.jl")


function generate_plot(data, suite)
    df = suite_stats(analysis, suite)
    delete!(df, Symbol(:ε, suite))
    names!(df, [:benchmark, :performance, :error])
    df[:performance] = -1.*df[:performance].+1

    total = df[df[:benchmark] .== "total", :performance][1]
    df = df[df[:benchmark] .!= "total", :]
    writedlm("$(suite)_total.csv", total)

    sort!(df, cols=:performance; rev=true)
    writetable("$suite.csv", df; header=true)

    labels = df[:benchmark]

    # speed-ups
    performance = map(i->min(0,i), df[:performance])
    bar(labels,
        100*performance;
        legend=false,
        rotation=45,
        color=:red,
        xlabel = "benchmark",
        ylabel = "performance difference (%)")

    # slow-downs
    performance = map(i->max(0,i), df[:performance])
    bar!(labels,
        100*performance;
        color=:green)

    png(suite)
end


analysis = readtable("analysis.dat")

for suite in non_baseline
    generate_plot(analysis, suite)
end
