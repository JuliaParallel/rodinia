#!/usr/bin/env julia

using Plots
pyplot()

include("common.jl")


function generate_plot(data, suite)
    col = Symbol(suite)
    df = suite_stats(analysis, suite)
    df = df[df[:benchmark] .!= "total", :]
    sort!(df, cols=col; rev=true)
    labels = df[:benchmark]

    # speed-ups
    performance = map(i->max(1,i), df[col])
    bar(labels,
        -100.*performance.+100;
        legend=false,
        rotation=45,
        color=:red,
        xlabel = "benchmark",
        ylabel = "performance difference (%)")

    # slow-downs
    performance = map(i->min(1,i), df[col])
    bar!(labels,
        -100.*performance.+100;
        color=:green)

    png(suite)
end


analysis = readtable("analysis.dat")

for suite in non_baseline
    generate_plot(analysis, suite)
end
