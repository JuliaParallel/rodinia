#!/usr/bin/env julia

using Glob

include("common.jl")


# run a benchmark once, returning the measurement data
function run_benchmark(dir)
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
            data = read_data(output_file)
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

function read_data(output_path)
    output = readlines(output_path; chomp=false)
    contains(join(output), "No kernels were profiled.") && return nothing

    # nuke the headers and parse the data
    raw_data = mktemp() do path,io
        write(io, output[4])
        write(io, output[6:end])
        flush(io)
        readtable(path)
    end

    return raw_data
end

function process_data(raw_data, suite, benchmark)
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

    # generate a nicer table
    rows = size(raw_data, 1)
    data = DataFrame(suite = repeat([suite]; inner=rows),   # DataFramesMeta.jl/#46
                     benchmark = benchmark,
                     kernel = kernels,
                     time = raw_data[:Duration])

    # pull apart iterations of irregular kernels
    if haskey(irregular_kernels, benchmark)
        counters = Dict{String,Int64}()
        bad_kernels = irregular_kernels[benchmark]
        for i in 1:size(data, 1)
            kernel = data[:kernel][i]
            if kernel in bad_kernels
                j = get(counters, kernel, 0) + 1
                data[:kernel][i] *= "#$j"
                counters[kernel] = j
            end
        end
    end

    return data
end

# check if measurements are accurate enough
function is_accurate(data)
    # group across iterations
    grouped = summarize(data, [:suite, :benchmark, :kernel], :time;
                        iterations=dt->length(dt[:time]))

    # calculate relative error
    grouped[:ε] = map(t->t.err / abs(t.val), grouped[:time])

    return all(i->i>=MIN_KERNEL_ITERATIONS, grouped[:iterations]) &&
           all(val->val<MAX_KERNEL_ERROR, grouped[:ε])
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
measurements = DataFrame(suite=String[], benchmark=String[], kernel=String[],
                         time=Float64[], execution=Int64[])
for suite in suites, benchmark in common_benchmarks
    info("Processing $suite/$benchmark")
    dir = joinpath(root, suite, benchmark)
    cache_path = joinpath(dir, "profile.csv")

    if isfile(cache_path)
        data = readtable(cache_path)
    else
        iter = 1
        t0 = time()
        data = nothing
        while true
            new_data = process_data(run_benchmark(dir), suite, benchmark)
            new_data[:execution] = repeat([iter]; inner=size(new_data,1))
            iter += 1

            if data == nothing
                data = new_data
            else
                append!(data, new_data)
            end

            is_accurate(data)                    && break
            iter >= MAX_BENCHMARK_RUNS           && break
            (time()-t0) >= MAX_BENCHMARK_SECONDS && break
        end

        writetable(cache_path, data)
    end

    append!(measurements, data)
end

writetable("measurements.dat", measurements)
