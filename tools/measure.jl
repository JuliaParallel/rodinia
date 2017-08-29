#!/usr/bin/env julia

using Glob

include("common.jl")


# run a benchmark once, returning the measurement data
function run_benchmark(dir, suite, benchmark)
    output_file = "nvprof.csv"
    output_pattern = "nvprof.csv.*"

    # delete old profile output
    rm.(glob(output_pattern, dir))

    # run and measure
    out = Pipe()
    cmd = ```
        nvprof
        --profile-from-start off
        --profile-child-processes
        --unified-memory-profiling off
        --print-gpu-trace
        --normalized-time-unit us
        --csv
        --log-file $output_file.%p
        ./profile --depwarn=no
    ```
    cmd_success = cd(dir) do
        success(pipeline(ignorestatus(cmd), stdout=out, stderr=out))
    end
    close(out.in)
    output_files = glob(output_pattern, dir)

    # read all data
    if !cmd_success
        println(readstring(out))
        error("benchmark did not succeed")
    else
        # when precompiling, an additional process will be spawned,
        # but the resulting CSV should not contain any data.
        output_data = []
        for output_file in output_files
            data = read_data(output_file, suite, benchmark)
            if data != nothing
                push!(output_data, data)
            end
        end

        if length(output_data) == 0
            error("no output files")
        elseif length(output_data) > 1
            error("too many output files")
        else
            return output_data[1]
        end
    end
end

function read_data(output_path, suite, benchmark)
    output = readlines(output_path; chomp=false)
    contains(join(output), "No kernels were profiled.") && return nothing

    # nuke the headers and parse the data
    raw_data = mktemp() do path,io
        write(io, output[4])
        write(io, output[6:end])
        flush(io)
        readtable(path)
    end

    # remove API calls
    raw_data = raw_data[!startswith.(raw_data[:Name], "[CUDA"), :]

    # demangle kernel names
    kernels = raw_data[:Name]
    for i = 1:length(kernels)
        jl_match = match(r"ptxcall_(.*)_[0-9]+ .*", kernels[i])
        if jl_match != nothing
            kernels[i] = jl_match.captures[1]
            continue
        end

        cu_match = match(r"(.*)\(.*", kernels[i])
        if cu_match != nothing
            kernels[i] = cu_match.captures[1]
            continue
        end
    end

    return raw_data
end

function process_data(raw_data)
    # generate a nicer table
    rows = size(raw_data, 1)
    data = DataFrame(suite = repeat([suite]; inner=rows),   # DataFramesMeta.jl/#46
                     benchmark = benchmark,
                     kernel = kernels,
                     time = raw_data[:Duration])

    # pull apart iterations of irregular kernels
    if haskey(irregular_kernels, benchmark)
        kernels = irregular_kernels[benchmark]
        for kernel in kernels
            for i in 1:size(data, 1)
                if data[:kernel][i] in kernels
                    data[:kernel][i] *= "#$i"
                end
            end
        end
    end

    return data
end

# check if measurements are accurate enough
function is_accurate(data)
    # group across iterations
    grouped = by(data, [:suite, :benchmark, :kernel],
                 dt->DataFrame(iterations=length(dt[:time]),
                               abs_uncert=std(dt[:time]),     # TODO: lognormal
                               best=minimum(dt[:time]))
                )

    # calculate relative uncertainty
    grouped[:rel_uncert] = grouped[:abs_uncert] ./ abs(grouped[:best])

    return all(i->i>=MIN_KERNEL_ITERATIONS, grouped[:iterations]) &&
           all(val->val<MAX_KERNEL_UNCERTAINTY, grouped[:rel_uncert])
end


# find benchmarks common to all suites
benchmarks = Dict()
for suite in suites
    entries = readdir(joinpath(root, suite))
    benchmarks[suite] = filter(entry->(isdir(joinpath(root,suite,entry)) &&
                                       isfile(joinpath(root,suite,entry,"profile"))), entries)
end
common_benchmarks = intersect(values(benchmarks)...)

# collect measurements
measurements = DataFrame(suite=String[], benchmark=String[], kernel=String[], time=Float64[])
for suite in suites, benchmark in common_benchmarks
    info("Collecting measurements for $suite/$benchmark")
    dir = joinpath(root, suite, benchmark)
    cache_path = joinpath(dir, "profile.csv")

    if isfile(cache_path)
        local_data = readtable(cache_path)
    else
        t0 = time()
        collect_data() = process_data(run_benchmark(dir, suite, benchmark))

        # iteration 0
        iter = 1
        local_data = collect_data()

        # additional iterations
        while (time()-t0) < MAX_BENCHMARK_SECONDS &&
               iter < MAX_BENCHMARK_RUNS &&
               !is_accurate(local_data)
            iter += 1
            append!(local_data, collect_data())
        end

        writetable(cache_path, local_data)
    end

    append!(measurements, local_data)
end

writetable("measurements.dat", measurements)
