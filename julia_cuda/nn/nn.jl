#!/usr/bin/env julia

using ArgParse
using CUDAdrv, CUDAnative

const OUTPUT = haskey(ENV, "OUTPUT")

# configuration
const DEFAULT_THREADS_PER_BLOCK = 256
const LATITUDE_POS = 28 # character position of the latitude value in each record
const OPEN = 10000 # initial value of nearest neighbors

ceilDiv(a, b) = ceil(Int, a / b)

immutable LatLong
    lat::Float32
    lng::Float32
end

type Record
    recString::String
    distance::Float32
end

# Calculates the Euclidean distance from each record in the database to the
# target position.
function euclid(d_locations, d_distances, numRecords, lat, lng)
    globalId = threadIdx().x + blockDim().x *
                (gridDim().x * (blockIdx().y - UInt32(1)) + (blockIdx().x - UInt32(1)))
    if globalId <= numRecords
        latLong = d_locations[globalId]
        d_distances[globalId] =
            sqrt((lat - latLong.lat) * (lat - latLong.lat) +
                 (lng - latLong.lng) * (lng - latLong.lng))
    end

    return nothing
end

# This program finds the k-nearest neighbors.
function main(args)
    # Parse command line
    filename, resultsCount, lat, lng, quiet, timing, platform, dev =
        parseCommandline(args)
    numRecords, records, locations = loadData(filename)

    if resultsCount > numRecords
        resultsCount = numRecords
    end

    ctx = CuCurrentContext()
    dev = device(ctx)

    # Scaling calculations - added by Sam Kauffman
    synchronize()

    maxGridX = attribute(dev, CUDAdrv.MAX_GRID_DIM_X)
    maxThreadsPerBlock = attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
    threadsPerBlock = min(maxThreadsPerBlock, DEFAULT_THREADS_PER_BLOCK)

    freeDeviceMemory = Mem.free()

    synchronize()

    usableDeviceMemory = floor(UInt, freeDeviceMemory * 85 / 100) # 85% arbitrary throttle
        # to compensate for known CUDA bug
    maxThreads = floor(UInt, usableDeviceMemory / 12) # 4 bytes in 3 vectors per thread

    if numRecords > maxThreads
        println(STDERR, "Error: Input too large.")
        exit(1)
    end

    blocks = ceilDiv(numRecords, threadsPerBlock) # extra threads do nothing
    gridY = ceilDiv(blocks, maxGridX)
    gridX = ceilDiv(blocks, gridY)

    # Allocate memory on device and copy data from host to device.
    d_locations = CuArray(locations)
    d_distances = CuArray{Float32}(numRecords)

    # Execute kernel. There will be no more than (gridY - 1) extra blocks.
    @cuda ((gridX, gridY), threadsPerBlock) euclid(
        d_locations, d_distances, UInt32(numRecords), lat, lng)

    # Copy data from device memory to host memory.
    distances = Array(d_distances)

    # Find the resultsCount least distances.
    findLowest(records, distances, numRecords, resultsCount);

    # Print out results.
    if OUTPUT
        out = open("output.txt", "w")
        @printf(out, "The %d nearest neighbors are:\n", resultsCount)
        for i = 1:resultsCount
            @printf(out, "%s --> %f\n", records[i].recString, records[i].distance)
        end
        close(out)
    end
end

function loadData(filename)
    recNum = 0
    records = Record[]
    locations = LatLong[]

    # Main processing
    flist = open(filename, "r")

    while !eof(flist)
        # Read in all records. If this is the last file in the filelist, then
        # we are done. Otherwise, open next file to be read next iteration.
        fp = open(chomp(readline(flist)), "r");

        # read each record
        while !eof(fp)
            record = chomp(readline(fp))
            # Parse for lat and lng
            lat, lng = map(x -> parse(Float32, x),
                split(record[LATITUDE_POS:end]))

            push!(records, Record(record, OPEN))
            push!(locations, LatLong(lat, lng))
            recNum = recNum + 1
        end

        close(fp)
    end

    close(flist)

    return (recNum, records, locations)
end

function findLowest(records, distances, numRecords, topN)
    for i = 1:topN
        minLoc = i-1
        for j = i-1:numRecords-1
            val = distances[j + 1]
            if val < distances[minLoc + 1]
                minLoc = j
            end
        end
        # swap locations and distances
        tmp = records[i]
        records[i] = records[minLoc + 1]
        records[minLoc + 1] = tmp

        tmp = distances[i]
        distances[i] = distances[minLoc + 1]
        distances[minLoc + 1] = tmp

        # Add distance to the min we just found.
        records[i].distance = distances[i]
    end
end

function parseCommandline(args)
    s = ArgParseSettings("Nearest Neighbor")
    @add_arg_table s begin
        "-r" # number of results
            arg_type = Int
            default = 10
            help = "the number of records to return (default: 10)"
        "--lat"
            arg_type = Float32
            help = "the latitude for nearest neighbors (default: 0)"
        "--lng"
            arg_type = Float32
            help = "the longitude for nearest neighbors (default: 0)"
        "-q" # quiet
            action = :store_true
            help = "Quiet mode. Suppress all text output."
        "-t" # timing
            action = :store_true
            help = "Print timing information."
        "-p" # platform
            arg_type = Int
            default = 0
            help = "Choose the platform (must choose both platform and device)"
        "-d" # device
            arg_type = Int
            default = 0
            help = "Choose the device (must choose both platform and device)"
        "filename"
            help = "the filename that lists the data input files"
            required = true
    end

    options = parse_args(args, s)

    d = options["d"]
    p = options["p"]

    # Both p and d must be specified if either are specified.
    if (d >= 0 && p < 0) || (p >= 0 && d < 0)
        println("Yep")
        parse_args(["-h"], args)
    end

    return (options["filename"], options["r"], options["lat"], options["lng"],
            options["q"], options["t"], p, d)
end


main(ARGS)

if haskey(ENV, "PROFILE")
    CUDAnative.@profile main(ARGS)
end
