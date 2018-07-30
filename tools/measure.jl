#!/usr/bin/env julia

using Glob

# include("common.jl")


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
        --concurrent-kernels off
        --profile-child-processes
        --unified-memory-profiling off
        --print-api-trace
        --print-gpu-trace
        --normalized-time-unit us
        --csv
        --log-file $output_file.%p
        ./profile --depwarn=no --check-bounds=no --math-mode=fast
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
    any(line->contains(line, "No kernels were profiled."), output) && return nothing

    # skip nvprof comments
    comments = findlast(line->occursin(r"^==\d+==", line), output)

    # NOTE: the first row often contains missing values (because it is the NVTX range),
    #       which confuses CSV.jl (https://github.com/JuliaData/CSV.jl/issues/156)
    CSV.read(output_path; header=comments+1, datarow=comments+3,
             types=Dict("Duration"              => Union{Missing,Float64},
                        "Grid X"                => Union{Missing,Int},
                        "Grid Y"                => Union{Missing,Int},
                        "Grid Z"                => Union{Missing,Int},
                        "Block X"               => Union{Missing,Int},
                        "Block Y"               => Union{Missing,Int},
                        "Block Z"               => Union{Missing,Int},
                        "Registers Per Thread"  => Union{Missing,Int},
                        "Static SMem"           => Union{Missing,Float64},
                        "Dynamic SMem"          => Union{Missing,Float64},
                        "Size"                  => Union{Missing,Float64},
                        "Throughput"            => Union{Missing,Float64},
                        "SrcMemType"            => Union{Missing,String},
                        "DstMemType"            => Union{Missing,String},
                        "Device"                => Union{Missing,String},
                        "Context"               => Union{Missing,String},
                        "Stream"                => Union{Missing,String},
                        "Correlation_ID"        => Union{Missing,Int}))
end

function process_data(raw_data, suite, benchmark)
    # extract kernel timings
    kernel_data = filter(entry -> begin
            !(startswith(entry[:Name], "cu") && ismissing(entry[:Device])) &&
            !occursin(r"^\[CUDA .+\]$", entry[:Name]) &&
            !occursin(r"^\[Range .+\] .+", entry[:Name])
        end, raw_data)

    # demangle kernel names
    for kernel in eachrow(kernel_data)
        jl_match = match(r"ptxcall_(.*)_[0-9]+", kernel[:Name])
        if jl_match != nothing
            kernel[:Name] = jl_match.captures[1]
            continue
        end

        cu_match = match(r"(.*)\(.*", kernel[:Name])
        if cu_match != nothing
            kernel[:Name] = cu_match.captures[1]
            continue
        end

        error("could not match kernel name $(kernel[:Name])")
    end

    # extract NVTX range timings
    if any(name->startswith(name, "[Range"), raw_data[:Name])
        range_start = filter(row -> startswith(row[:Name], "[Range start]"), raw_data)
        range_end   = filter(row -> startswith(row[:Name], "[Range end]"),   raw_data)
        @assert size(range_start) == size(range_end)

        ranges = range_start
        ranges[:Name] = map(name->'#' * match(r"\[Range start\] (.+) \(Domain: .+\)",
                                              name).captures[1],
                            ranges[:Name])
        ranges[:Duration] = range_end[:Start] - range_start[:Start]

        kernel_data = vcat(kernel_data, ranges)
    end

    # generate a nicer table
    rows = size(kernel_data, 1)
    data = DataFrame(suite = suite,
                     benchmark = benchmark,
                     target = kernel_data[:Name],
                     time = kernel_data[:Duration])

    # extract additional host-side compilation time measurements from compiles.dat:
    # -    main(ARGS)
    # +    open("compiles.dat", "w") do io
    # +        ccall(:jl_dump_compiles, Nothing, (Ptr{Cvoid},), io.handle)
    # +        main(ARGS)
    # +        ccall(:jl_dump_compiles, Nothing, (Ptr{Cvoid},), C_NULL)
    # +    end
    if suite == "julia_cuda"
        compiles = joinpath(root, suite, benchmark, "compiles.dat")
        # measure everything that's defined in Main (modules can be precompiled)
        # except for kernels (already contained in the CUDAnative timings)
        kernels = filter(target->!startswith(target, '#'), unique(data[:target]))
        measurements = 0
        open(compiles) do io
            for line in eachline(io)
                if occursin(r"\bMain\.", line) &&
                   !any(occursin(Regex("\\b$kernel\\b"), line) for kernel in kernels)
                    duration = parse(Int, first(split(line)))
                    push!(data, [suite, benchmark, "#julia", duration/1000])
                    measurements += 1
                end
            end
        end

        # if we don't have any measurement (eg. all functions interpreted),
        # add a dummy timing
        if measurements == 0
            push!(data, [suite, benchmark, "#julia", 0])
        end
    end

    # pull apart iterations of irregular kernels
    if haskey(irregular_kernels, benchmark)
        counters = Dict{String,Int64}()
        bad_kernels = irregular_kernels[benchmark]
        for i in 1:size(data, 1)
            target = data[:target][i]
            if target in bad_kernels
                j = get(counters, target, 0) + 1
                data[:target][i] *= "#$j"
                counters[target] = j
            end
        end
    end


    return data
end

# check if measurements are accurate enough
function is_accurate(data)
    grouped = summarize(data)

    # calculate relative error
    grouped[:ε] = map(t->t.err / abs(t.val), grouped[:time])

    return all(i->i>=MIN_KERNEL_ITERATIONS, grouped[:target_iterations]) &&
           all(val->val<MAX_KERNEL_ERROR, grouped[:ε])
end

function measure(host=gethostname())
    # find benchmarks common to all suites
    benchmarks = Dict()
    for suite in suites
        entries = readdir(joinpath(root, suite))
        benchmarks[suite] = filter(entry->(isdir(joinpath(root,suite,entry)) &&
                                           isfile(joinpath(root,suite,entry,"profile"))), entries)
    end
    common_benchmarks = intersect(values(benchmarks)...)

    # collect measurements
    measurements = DataFrame(suite=String[], benchmark=String[], target=String[],
                             time=Float64[], execution=Int64[])
    for suite in suites, benchmark in common_benchmarks
        @info "Processing $suite/$benchmark"
        dir = joinpath(root, suite, benchmark)
        cache_path = joinpath(dir, "profile_$host.csv")

        data = if isfile(cache_path)
            try
                CSV.read(cache_path)
            catch
                @warn "Corrupt cache file at $cache_path"
                nothing
            end
        else
            nothing
        end

        if data == nothing
            iter = 1
            t0 = time()
            data = nothing
            while true
                new_data = process_data(run_benchmark(dir), suite, benchmark)
                new_data[:execution] = iter
                iter += 1

                if data == nothing
                    data = new_data
                else
                    data = vcat(data, new_data)
                end

                is_accurate(data)                    && break
                iter >= MAX_BENCHMARK_RUNS           && break
                (time()-t0) >= MAX_BENCHMARK_SECONDS && break
            end

            CSV.write(cache_path, data)
        end

        measurements = vcat(measurements, data)
    end

    CSV.write("measurements_$host.dat", measurements)

    return
end
