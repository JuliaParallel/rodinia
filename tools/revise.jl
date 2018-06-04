using Revise

for file in ["common.jl", "measure.jl", "analyze.jl", "plot.jl"]
    Revise.track(file)
    include(file)
end
