#!/usr/bin/env julia

using CUDAdrv, CUDAnative

include("../../common/julia/crand.jl")
const rng = LibcRNG48()

include("streamcluster_cuda.jl")

const OUTPUT = haskey(ENV, "OUTPUT")

# configuration
const SP = 1 # number of repetitions of speedy must be >=1
const SEED = 1
const ITER  = 3 # iterate ITER* k log k times ITER >= 1
const PRINT_INFO = false

# global state
g_isCoordChanged = false


# shuffle points into random order
function shuffle(points)

    global g_time_shuffle

    for i = 0:points.num-2
        j = (rand(rng, Culong) % (points.num - i)) + i
        temp = points.p[i + 1]
        points.p[i + 1] = points.p[j + 1]
        points.p[j + 1] = temp
    end

    return nothing
end

# shuffle an array of integers
function intshuffle(intarray, length)

    global g_time_shuffle

    for i = 0:length-1
        j = (rand(rng, Culong) % (length - i)) + i
        temp = intarray[i + 1]
        intarray[i + 1] = intarray[j + 1]
        intarray[j + 1] = temp
    end

    return nothing
end

# compute Euclidean distance squared between two points
function dist(p1, p2, dim)

    result = 0f0

    for i = 1:dim
        result += (p1.coord[i] - p2.coord[i]) * (p1.coord[i] - p2.coord[i])
    end

    return result
end

const __pspeedy_i = Ref{Int32}(0)
const __pspeedy_open = Ref{Bool}(false)
const __pspeedy_costs = Ref{Array{Float32,1}}() # cost for each thread.
const __pspeedy_totalcost = Ref{Float32}(0f0)

# run speedy on the points, return total cost of solution
function pspeedy(points, z, kcenter, pid)

    global g_time_speedy
    global __pspeedy_i
    global __pspeedy_open
    global __pspeedy_costs
    global __pspeedy_totalcost

    # my block
    bsize = Int(points.num / nproc[])
    k1 = bsize * pid
    k2 = k1 + bsize

    if pid == nproc[] - 1
        k2 = points.num
    end

    if PRINT_INFO && pid == 0
        @printf(STDERR, "Speedy: facility cost %lf\n", z)
    end

    # create center at first point, send it to itself
    for k = k1+1:k2
        distance = dist(points.p[k], points.p[1], points.dim)
        points.p[k].cost = distance * points.p[k].weight
        points.p[k].assign = 0
    end

    if pid == 0
        kcenter[] = 1
        __pspeedy_costs[] = Array{Float32}(nproc[])

        # I am the master thread. I decide whether to open a center and notify others if so.
        for __pspeedy_i[] = 1:points.num-1
            if (Float32(rand(rng, Culong)) / INT_MAX) < (points.p[__pspeedy_i[]+1].cost / z)
                kcenter[] += 1
                __pspeedy_open[] = true

                for k = k1+1:k2
                    distance = dist(points.p[__pspeedy_i[]+1], points.p[k], points.dim)
                    if distance * points.p[k].weight < points.p[k].cost
                        points.p[k].cost = distance * points.p[k].weight
                        points.p[k].assign = __pspeedy_i[]
                    end
                end
                __pspeedy_open[] = false
            end
        end
        __pspeedy_open[] = true
    else
        # We are not the master threads. We wait until a center is opened.
        while true
            if __pspeedy_i[] >= points.num
                break
            end

            for k = k1+1:k2
                distance = dist(points.p[__pspeedy_i[]+1], points.p[k], points.dim)
                if distance * points.p[k].weight < points.p[k].cost
                    points.p[k].cost = distance * points.p[k].weight
                    points.p[k].assign = __pspeedy_i[]
                end
            end
        end
    end

    __pspeedy_open[] = false
    mytotal = 0f0

    for k = k1+1:k2
        mytotal += points.p[k].cost
    end

    __pspeedy_costs[][pid + 1] = mytotal

    # aggregate costs from each thread
    if pid == 0
        __pspeedy_totalcost[] = z * kcenter[]
        for i = 1:nproc[]
            __pspeedy_totalcost[] += __pspeedy_costs[][i]
        end
    end

    if PRINT_INFO && pid == 0
        @printf(STDERR, "Speedy opened %d facilities for total cost %lf\n", kcenter[],
            totalcost)
        @printf(STDERR, "Distance Cost %lf\n", __pspeedy_totalcost[] - z * kcenter[])
    end

    return __pspeedy_totalcost[]
end

# facility location on the points using local search
# z is the facility cost, returns the total cost and # of centers
# assumes we are seeded with a reasonable solution
# cost should represent this solution's cost
# halt if there is < e improvement after iter calls to gain
# feasible is an array of numfeasible points which may be centers
function pFL(points, feasible, z, k, kmax, cost, iter, e, pid)

    change = cost
    numfeasible = length(feasible)

    # continue until we run iter iterations without improvement
    # stop instead if improvement is less than e
    while change / cost > 1.0 * e

        change = 0f0
        numberOfPoints = points.num

        # randomize order in which centers are considered
        if pid == 0
            intshuffle(feasible, numfeasible)
        end

        for i = 0:iter-1
            x = i % numfeasible
            change += pgain(feasible[x + 1], points, z, k, kmax, g_is_center,
                            g_center_table, g_switch_membership, g_isCoordChanged)
        end

        cost -= change

        if PRINT_INFO && pid == 0
            @printf(STDERR, "%d centers, cost %lf, total distance %lf\n", k[],
                cost, cost - z * k[])
        end
    end

    return cost
end

function selectfeasible_fast(points, kmin, pid)

    global g_time_select_feasible

    numfeasible = points.num

    if numfeasible > ITER * kmin * log(kmin)
        numfeasible = floor(Int32, ITER * kmin * log(kmin))
    end

    feasible = Array{Int32}(numfeasible)

    # Calcuate my block.
    # For now this routine does not seem to be the bottleneck, so it is not parallelized.
    # When necessary, this can be parallelized by setting k1 and k2 to proper values and
    # calling this routine from all threads (it is called only by thread 0 for now).
    # Note that when parallelized, the randomization might not be the same and it might not
    # be difficult to measure the parallel speed-up for the whole program.

    k1 = 0
    k2 = numfeasible

    # not many points, all will be feasible
    if numfeasible == points.num
        for i = k1:k2-1
            feasible[i+1] = i
        end
        return feasible
    end

    accumweight = Array{Float32}(points.num)
    accumweight[1] = points.p[1].weight
    totalweight = 0

    for i = 2:points.num
        accumweight[i] = accumweight[i - 1] + points.p[i].weight
    end

    totalweight = accumweight[points.num]

    for i = k1+1:k2

        w = (rand(rng, Culong) / Float32(INT_MAX)) * totalweight

        # binary search
        l = 0
        r = points.num - 1

        if accumweight[1] > w
            feasible[i] = 0
            continue
        end

        while l + 1 < r
            k = floor(Int32, (l + r) / 2)
            if accumweight[k + 1] > w
                r = k
            else
                l = k
            end
        end

        feasible[i] = r
    end

    return feasible
end

const __pkmedian_k = Ref{Int64}(0)
const __pkmedian_hizs = Ref{Array{Float32,1}}()
const __pkmedian_feasible = Ref{Array{Int32,1}}()

# compute approximate kmedian on the points
function pkmedian(points, kmin, kmax, kfinal, pid)
    global g_is_center
    global __pkmedian_k
    global __pkmedian_hizs
    global __pkmedian_feasible

    if pid == 0
        __pkmedian_hizs[] = zeros(Float32, nproc[])
    end

    hiz = 0f0
    loz = 0f0

    numberOfPoints = points.num
    ptDimension = points.dim

    # my block
    bsize = Int(points.num / nproc[])
    k1 = bsize * pid
    k2 = k1 + bsize

    if pid == nproc[] - 1
        k2 = points.num
    end

    if PRINT_INFO && pid == 0
        println("Starting Kmedian procedure")
        @printf("%i points in %i dimensions\n", numberOfPoints, ptDimension)
    end

    myhiz = 0f0

    for kk = k1+1:k2
        myhiz += dist(points.p[kk], points.p[1], ptDimension) * points.p[kk].weight
    end

    __pkmedian_hizs[][pid + 1] = myhiz

    for i = 1:nproc[]
        hiz += __pkmedian_hizs[][i]
    end

    loz = 0f0
    z = (hiz + loz) / 2

    # NEW: Check whether more centers than points!
    if points.num <= kmax

        # just return all points as facilities
        for kk = k1+1:k2
            points.p[kk].assign = kk - 1
            points.p[kk].cost = 0
        end

        if pid == 0
            kfinal[] = __pkmedian_k[]
        end

        return 0
    end

    if pid == 0
        shuffle(points)
    end

    cost = pspeedy(points, z, __pkmedian_k, pid)

    if PRINT_INFO && pid == 0
        @printf("thread %d: Finished first call to speedy, cost=%lf, k=%i\n", pid, cost,
            __pkmedian_k[])
    end

    i = 0

    # give speedy SP chances to get at least kmin/2 facilities
    while __pkmedian_k[] < kmin && i < SP
        cost = pspeedy(points, z, __pkmedian_k, pid)
        i += 1
    end

    if PRINT_INFO && pid == 0
        @printf("thread %d: second call to speedy, cost=%lf, k=%d\n", pid, cost,
            __pkmedian_k[])
    end

    # if still not enough facilities, assume z is too high
    while __pkmedian_k[] < kmin

        if PRINT_INFO && pid == 0
            @printf("%lf %lf\n", loz, hiz)
            println("Speedy indicates we should try lower z")
        end

        if i >= SP
            hiz = z
            z = (hiz + loz) / 2
            i = 0
        end

        if pid == 0
            shuffle(points)
        end

        cost = pspeedy(points, z, __pkmedian_k, pid)
        i += 1
    end

    # now we begin the binary search for real
    # must designate some points as feasible centers
    # this creates more consistancy between FL runs
    # helps to guarantee correct # of centers at the end

    if pid == 0
        __pkmedian_feasible[] = selectfeasible_fast(points, kmin, pid)

        for i = 1:points.num
            g_is_center[points.p[i].assign + 1] = true
        end
    end

    while true
        if PRINT_INFO && pid == 0
            @printf("loz = %lf, hiz = %lf\n", loz, hiz)
            println("Running Local Search...")
        end

        # first get a rough estimate on the FL solution
        lastcost = cost
        cost = pFL(points, __pkmedian_feasible[], z, __pkmedian_k, kmax, cost,
            floor(Int64, ITER * kmax * log(kmax)), 0.1, pid)

        # if number of centers seems good, try a more accurate FL
        if (__pkmedian_k[] <= 1.1 * kmax && __pkmedian_k[] >= 0.9 * kmin) ||
           (__pkmedian_k[] <= kmax + 2   && __pkmedian_k[] >= kmin - 2)

            if PRINT_INFO && pid == 0
                println("Trying a more accurate local search...")
            end

            # may need to run a little longer here before halting without improvement
            cost = pFL(points, __pkmedian_feasible[], z, __pkmedian_k, kmax, cost,
                floor(Int64, ITER * kmax * log(kmax)), 0.001, pid)
        end

        if __pkmedian_k[] > kmax
            # facilities too cheap
            # increase facility cost and up the cost accordingly
            loz = z
            z = (hiz + loz) / 2
            cost += (z - loz) * __pkmedian_k[]
        end

        if __pkmedian_k[] < kmin
            # facilities too expensive
            # decrease facility cost and reduce the cost accordingly
            hiz = z
            z = (hiz + loz) / 2
            cost += (z - hiz) * __pkmedian_k[]
        end

        # if k is good, return the result
        # if we're stuck, just give up and return what we have
        if (__pkmedian_k[] <= kmax && __pkmedian_k[] >= kmin) || (loz >= 0.999 * hiz)
            break
        end
    end

    if pid == 0
        kfinal[] = __pkmedian_k[]
    end

    return cost
end

# compute the means for the k clusters
function contcenters(points)
    for i = 1:points.num
        # compute relative weight of this point to the cluster
        if points.p[i].assign != (i - 1)
            relweight = points.p[points.p[i].assign + 1].weight +
                points.p[i].weight
            relweight = points.p[i].weight / relweight

            for ii = 1:points.dim
                points.p[points.p[i].assign + 1].coord[ii] *= 1f0 - relweight
                points.p[points.p[i].assign + 1].coord[ii] +=
                    points.p[i].coord[ii] * relweight
            end

            points.p[points.p[i].assign + 1].weight += points.p[i].weight
        end
    end
end

# copy centers from points to centers
function copycenters(points, centers, centerIDs, offset)

    is_a_median = [false for i = 1:points.num]

    # mark the centers
    for i = 1:points.num
        is_a_median[points.p[i].assign + 1] = 1
    end

    k = centers.num

    # count how many
    for i = 1:points.num
        if is_a_median[i]
            centers.p[k+1].coord[1:points.dim] = points.p[i].coord[1:points.dim]
            centers.p[k+1].weight = points.p[i].weight
            centerIDs[k+1] = (i - 1) + offset
            k += 1
        end
    end

    centers.num = k
end

function localSearch(points, kmin, kmax, kfinal)
    global g_time_local_search

    for i = 0:nproc[]-1
        pkmedian(points, kmin, kmax, kfinal, i)
    end
end

function outcenterIDs(centers, centerIDs, outfile)
    try
        fp = open(outfile, "w")
        is_a_median = [false for i = 1:centers.num]

        for i = 1:centers.num
            is_a_median[centers.p[i].assign + 1] = true
        end

        for i = 1:centers.num
            if is_a_median[i]
                @printf(fp, "%u\n", centerIDs[i])
                @printf(fp, "%lf\n", centers.p[i].weight)

                for k = 1:centers.dim
                    @printf(fp, "%lf ", centers.p[i].coord[k])
                end

                @printf(fp, "\n\n")
            end
        end

        close(fp)
    catch
        @printf(STDERR, "error opening %s\n", outfile)
        exit(1)
    end
end

function streamCluster(stream, kmin, kmax, dim, chunksize, centersize, outfile)
    global g_is_center
    global g_center_table
    global g_isCoordChanged
    global g_switch_membership

    centerIDs = Array{Int64}(centersize * dim)
    points = Points(dim, chunksize, chunksize)
    centers = Points(dim, 0, centersize)

    for i = 1:chunksize
        points.p[i] = Point(0f0, Array{Float32}(dim), 0, 0f0)
    end

    for i = 1:centersize
        centers.p[i] = Point(1f0, Array{Float32}(dim), 0, 0f0)
    end

    IDoffset = 0
    kfinal = Ref{Int64}(0)

    while true
        numRead = 0

        for i = 1:chunksize
            numRead += read(stream, points.p[i].coord, dim, 1)
        end

        @printf(STDERR, "read %d points\n", numRead)

        if numRead < chunksize && !feof(stream)
            @printf(STDERR, "error reading data!\n")
            exit(1)
        end

        points.num = numRead

        for i = 1:points.num
            points.p[i].weight = 1f0
        end

        g_switch_membership = Array{Bool}(points.num)
        g_is_center = [false for i = 1:points.num]
        g_center_table = Array{Int32}(points.num)

        localSearch(points, kmin, kmax, kfinal)

        @printf(STDERR, "finish local search\n")

        contcenters(points)
        g_isCoordChanged = true

        if kfinal[] + centers.num > centersize
            # here we don't handle the situation where # of centers gets too large.
            @printf(STDERR, "oops! no more space for centers\n")
            exit(1)
        end

        if PRINT_INFO
            println("finish cont center")
        end

        copycenters(points, centers, centerIDs, IDoffset)
        IDoffset += numRead

        if PRINT_INFO
            println("finish copy centers")
        end

        if feof(stream)
            break
        end
    end

    # finally cluster all temp centers
    g_switch_membership = Array{Bool}(points.num)
    g_is_center = [false for i = 1:points.num]
    g_center_table = Array{Int32}(points.num)

    localSearch(centers, kmin, kmax, kfinal)
    contcenters(centers)
    if OUTPUT
        outcenterIDs(centers, centerIDs, outfile)
    end
end

const nproc = Ref{Int32}(0)

function main(args)
    global nproc

    if length(args) < 9
        @printf(STDERR, "usage: %s k1 k2 d n chunksize clustersize infile outfile nproc\n",
            "streamcluster")
        @printf(STDERR, "  k1:          Min. number of centers allowed\n")
        @printf(STDERR, "  k2:          Max. number of centers allowed\n")
        @printf(STDERR, "  d:           Dimension of each data point\n")
        @printf(STDERR, "  n:           Number of data points\n")
        @printf(STDERR, "  chunksize:   Number of data points to handle per step\n")
        @printf(STDERR, "  clustersize: Maximum number of intermediate centers\n")
        @printf(STDERR, "  infile:      Input file (if n<=0)\n")
        @printf(STDERR, "  outfile:     Output file\n")
        @printf(STDERR, "  nproc:       Number of threads to use\n")
        @printf(STDERR, "\n")
        @printf(STDERR, "if n > 0, points will be randomly generated instead of ")
        @printf(STDERR, "reading from infile.\n")
        exit(1)
    end

    kmin = parse(Int32, args[1])
    kmax = parse(Int32, args[2])
    dim = parse(Int32, args[3])
    n = parse(Int32, args[4])
    chunksize = parse(Int32, args[5])
    clustersize = parse(Int32, args[6])
    infilename = args[7]
    outfilename = args[8]
    nproc[] = parse(Int32, args[9])

    # reset global state
    global g_isCoordChanged
    g_isCoordChanged = false
    srand(rng, SEED)

    stream = n > 0 ? SimStream(n) : FileStream(infilename)

    streamCluster(stream, kmin, kmax, dim, chunksize, clustersize, outfilename)

    delete(stream)
end


main(ARGS)

if haskey(ENV, "PROFILE")
    CUDAnative.@profile main(ARGS)
end
