using Revise

for file in ["common.jl", "measure.jl", "analyze.jl"]
    Revise.track(file)
    include(file)
end
