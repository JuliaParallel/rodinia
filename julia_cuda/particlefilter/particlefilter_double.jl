#!/usr/bin/env julia

using CUDAdrv, CUDAnative
include("../../common/julia/kernelprofile.jl")

# Variables

const M = typemax(Int32)
const A = Int32(1103515245)
const C = Int32(12345)

const PI = 3.1415926535897932

const threads_per_block = 512

const OUTPUT = haskey(ENV, "OUTPUT")

# Utility functions

function gettime()
    current = now()
    elapsed = Dates.hour(current) * 60 * 60 * 1000000
            + Dates.minute(current)    * 60 * 1000000
            + Dates.second(current)       * 1000000
            + Dates.millisecond(current)
    return elapsed
end

function elapsedtime(start_time, end_time)
    return (end_time - start_time)/(1000 * 1000)
end

function rounddouble(value)
    new_value = convert(Int, floor(value))
    if (value - new_value < 0.5)
        return new_value
    else
        return new_value    # Copy bug from original
    end
end

function randu(seed, index)
    num::Int32 = A * seed[index] + C
    seed[index] = num % M
    q = seed[index]/M
    return abs(q)
end

function randn(seed, index)
    u = randu(seed, index)
    v = randu(seed, index)
    cosine = cos(2 * PI * v)
    rt = -2 * log(u)
    return sqrt(rt) * cosine
end

# Video sequence

function setif(test_value, new_value, array3D::Array{UInt8}, dimX, dimY, dimZ)
    for x=0:dimX-1
        for y=0:dimY-1
            for z=0:dimZ-1
                if array3D[x*dimY*dimZ + y*dimZ + z + 1] == test_value
                    array3D[x*dimY*dimZ + y*dimZ + z + 1] = new_value
                end
            end
        end
    end
end

function addnoise(array3D::Array{UInt8}, dimX, dimY, dimZ, seed)
    for x=0:dimX-1
        for y=0:dimY-1
            for z=0:dimZ-1
                noise = convert(Int, trunc(5 * randn(seed, 1)))
                array3D[x*dimY*dimZ + y*dimZ + z + 1] =
                    array3D[x*dimY*dimZ + y*dimZ + z + 1] + noise
            end
        end
    end
end

function dilate_matrix(matrix, posX, posY, posZ, dimX, dimY, dimZ, error)
    startX = posX - error
    while startX < 0
        startX += 1
    end
    startY = posY - error
    while startY < 0
        startY += 1
    end
    endX = posX + error
    while endX > dimX
        endX -= 1
    end
    endY = posY + error
    while endY > dimY
        endY -= 1
    end
    for x=startX:endX-1
        for y=startY:endY-1
            distance = sqrt((x-posX)^2 + (y-posY)^2)
            if distance < error
                matrix[x * dimY * dimZ + y * dimZ + posZ + 1] = 1
            end
        end
    end
end

function imdilate_disk(matrix::Array{UInt8}, dimX, dimY, dimZ, error, new_matrix)
    for z=0:dimZ-1
        for x=0:dimX-1
            for y=0:dimY-1
                if matrix[x * dimY * dimZ + y * dimZ + z + 1] == 1 
                    dilate_matrix(new_matrix, x, y, z, dimX, dimY, dimZ, error)
                end
            end
        end
    end
end

function videosequence(I::Array{UInt8}, IszX, IszY, Nfr, seed::Array{Int32})

    max_size = IszX * IszY * Nfr
    # get object centers
    x0 = convert(Int, rounddouble(IszX/2.0))
    y0 = convert(Int, rounddouble(IszY/2.0))
    I[x0 * IszY * Nfr + y0 * Nfr + 1] = 1       # TODO: +1 instead of 0????

    # Move point
    xk = yk = 0
    for k = 2:Nfr
        xk = abs(x0 + (k - 2));
        yk = abs(y0 - 2 * (k - 2));
        pos = yk * IszY * Nfr + xk * Nfr + k;
        if pos > max_size
            pos = 1;
        end
        I[pos] = 1;
    end

    # Dialate matrix
    new_matrix = zeros(UInt8, IszX * IszY * Nfr)
    imdilate_disk(I, IszX, IszY, Nfr, 5, new_matrix)

    for x=1:IszX
        for y=1:IszY
            for k=1:Nfr
                I[(x-1) * IszY * Nfr + (y-1) * Nfr + k] = 
                    new_matrix[(x-1) * IszY * Nfr + (y-1) * Nfr + k]
            end
        end
    end

    # Define background, add noise
    setif(0, UInt8(100), I, IszX, IszY, Nfr)
    setif(1, UInt8(228), I, IszX, IszY, Nfr)
    # Add noise
    addnoise(I, IszX, IszY, Nfr, seed)
end

# Particle filter

function streldisk(disk, radius)
    diameter = radius * 2 -1
    for x=1:diameter
        for y=1:diameter
            distance = sqrt((x-radius)^2 + (y-radius)^2)
            if distance < radius
                disk[(x-1)*diameter + y] = 1
            else
                disk[(x-1)*diameter + y] = 0
            end
        end
    end
end

# Particle filter
# Helper functions

@inline function cdf_calc(
    CDF,        # Out
    weights,    # Int
    Nparticles)
    CDF[1] = weights[1]
    for x=2:Nparticles
        CDF[x] = weights[x] + CDF[x-1]
    end
end

@inline function d_randu(seed, index)
    num = A * seed[index] + C
    seed[index] = num % M
    return CUDAnative.abs(seed[index]/M)
end

@inline function d_randn(seed, index)
    pi = 3.14159265358979323846
    u = d_randu(seed, index)
    v = d_randu(seed, index)
    cosine = CUDAnative.cos(2*pi*v)
    rt = -2 * CUDAnative.log(u)
    return CUDAnative.sqrt(rt) * cosine
end

@inline function calc_likelihood_sum(I, ind, num_ones, index)
    likelihood_sum = Float64(0)
    for x=1:num_ones
        i = ind[(index-1) * num_ones + x]
        v = ((I[i] - 100)*(I[i] - 100)
            - (I[i] -228)*(I[i] -228))/50
        likelihood_sum += v
    end
    return likelihood_sum
end

@inline function dev_round_double(value)
    if value < 0
        new_value = trunc(Int,CUDAnative.ceil(value))
    else
        new_value = trunc(Int,CUDAnative.floor(value))
    end
    if value - new_value < 0.5
        return new_value
    else
        return new_value # keep buggy semantics of original, should be new_value+1
    end
end


# Kernels

function find_index_kernel(arrayX_ptr, arrayX_len,
                           arrayY_ptr, arrayY_len,
                           CDF_ptr, CDF_len,
                           u_ptr, u_len,
                           xj_ptr, xj_len,
                           yj_ptr, yj_len,
                           weights_ptr, weights_len,
                           Nparticles)
    # reconstruct array arguments
    arrayX = CuDeviceArray(arrayX_len, arrayX_ptr)
    arrayY = CuDeviceArray(arrayY_len, arrayY_ptr)
    CDF = CuDeviceArray(CDF_len, CDF_ptr)
    u = CuDeviceArray(u_len, u_ptr)
    xj = CuDeviceArray(xj_len, xj_ptr)
    yj = CuDeviceArray(yj_len, yj_ptr)
    weights = CuDeviceArray(weights_len, weights_ptr)
    
    block_id = blockIdx().x
    i = blockDim().x * (block_id-1) + threadIdx().x

    if i <= Nparticles
        index = 0   # an invalid index
        for x=1:Nparticles
            if CDF[x] >= u[i]
                index = x
                break
            end
        end
        if index == 0
            index = Nparticles
        end

        xj[i] = arrayX[index]
        yj[i] = arrayY[index]
    end
    sync_threads()
end

function normalize_weights_kernel(weights_ptr, weights_len,
                                  Nparticles,
                                  partial_sums_ptr, partial_sums_len,
                                  CDF_ptr, CDF_len,
                                  u_ptr, u_len,
                                  seed_ptr, seed_len)
    # reconstruct array arguments
    weights = CuDeviceArray(weights_len, weights_ptr)
    partial_sums = CuDeviceArray(partial_sums_len, partial_sums_ptr)
    CDF = CuDeviceArray(CDF_len, CDF_ptr)
    u = CuDeviceArray(u_len, u_ptr)
    seed = CuDeviceArray(seed_len, seed_ptr)

    block_id = blockIdx().x
    i = blockDim().x * (block_id-1) + threadIdx().x

    shared = @cuStaticSharedMem(Float64, 2)
    u1_i = 1
    sum_weights_i = 2
    # shared[1] == u1, shared[2] = sum_weights

    if threadIdx().x == 1
        shared[sum_weights_i] = partial_sums[1]
    end
    sync_threads()

    if i <= Nparticles
        weights[i] = weights[i] / shared[sum_weights_i]
    end
    sync_threads()

    if i==1
        cdf_calc(CDF, weights, Nparticles)
        u[1] = (1/Nparticles) * d_randu(seed, i)
    end
    sync_threads()

    if threadIdx().x == 1
        shared[u1_i] = u[1]
    end
    sync_threads()

    if i <= Nparticles
        u1 = shared[u1_i]
        u[i] = u1 + i / Nparticles
    end

    return nothing
end

function sum_kernel(partial_sums_ptr, partial_sums_len, Nparticles)
    # reconstruct array arguments
    partial_sums = CuDeviceArray(partial_sums_len, partial_sums_ptr)

    block_id = blockIdx().x
    i = blockDim().x * (block_id-1) + threadIdx().x

    if i==1
        sum = 0.0
        num_blocks = trunc(Int,CUDAnative.ceil(Nparticles/threads_per_block))
        for x=1:num_blocks
            sum += partial_sums[x]
        end
        partial_sums[1] = sum
    end

    return nothing
end

function likelihood_kernel(arrayX_ptr, arrayX_len,
                           arrayY_ptr, arrayY_len,
                           xj_ptr, xj_len,
                           yj_ptr, yj_len,
                           ind_ptr, ind_len,
                           objxy_ptr, objxy_len,
                           likelihood_ptr, likelihood_len,
                           I_ptr, I_len,
                           weights_ptr, weights_len,
                           Nparticles, count_ones, max_size, k, IszY,
                           Nfr,
                           seed_ptr, seed_len,
                           partial_sums_ptr, partial_sums_len)
    # reconstruct array arguments
    arrayX = CuDeviceArray(arrayX_len, arrayX_ptr)
    arrayY = CuDeviceArray(arrayY_len, arrayY_ptr)
    xj = CuDeviceArray(xj_len, xj_ptr)
    yj = CuDeviceArray(yj_len, yj_ptr)
    ind = CuDeviceArray(ind_len, ind_ptr)
    objxy = CuDeviceArray(objxy_len, objxy_ptr)
    likelihood = CuDeviceArray(likelihood_len, likelihood_ptr)
    I = CuDeviceArray(I_len, I_ptr)
    weights = CuDeviceArray(weights_len, weights_ptr)
    seed = CuDeviceArray(seed_len, seed_ptr)
    partial_sums = CuDeviceArray(partial_sums_len, partial_sums_ptr)

    block_id = blockIdx().x
    i::Int = blockDim().x * (block_id-1) + threadIdx().x

    buffer = @cuStaticSharedMem(Float64, 512)
    if i <= Nparticles
        arrayX[i] = xj[i]
        arrayY[i] = yj[i]
        weights[i] = 1/Nparticles

        arrayX[i] = arrayX[i] + 1.0 + 5.0 * d_randn(seed, i)
        arrayY[i] = arrayY[i] - 2.0 + 2.0 * d_randn(seed, i)
    end

    sync_threads()

    if i <= Nparticles
        for y=0:count_ones-1
            indX = dev_round_double(arrayX[i]) + objxy[y*2 + 2]
            indY = dev_round_double(arrayY[i]) + objxy[y*2 + 1]

            val = indX*IszY*Nfr + indY*Nfr + k -1 #CUDAnative.abs(val)
            if val < 0
                val = -val
            end
            val += 1
            index = (i-1)*count_ones + y + 1
            ind[index] = val
            if ind[(i-1)*count_ones + y + 1] > max_size
                ind[(i-1)*count_ones + y + 1] = 1
            end
        end
        likelihood[i] = calc_likelihood_sum(I, ind, count_ones, i)
        likelihood[i] = likelihood[i]/count_ones
        weights[i] = weights[i] * CUDAnative.exp(likelihood[i])
    end
    buffer[threadIdx().x] = 0.0

    sync_threads()

    if i<=Nparticles
        buffer[threadIdx().x] = weights[i]
    end
    sync_threads()

    s = div(blockDim().x,2)
    while s > 0
        if threadIdx().x <= s
            v = buffer[threadIdx().x]
            v += buffer[threadIdx().x + s]
            buffer[threadIdx().x] = v
        end
        sync_threads()
        s>>=1
    end
    if threadIdx().x == 1
        partial_sums[blockIdx().x] = buffer[1]
    end
    sync_threads()
    return nothing
end

function getneighbors(se::Array{Int}, num_ones, neighbors::Array{Int}, radius)
    neighY = 1
    center = radius -1
    diameter = radius * 2 -1
    for x=0:diameter-1
        for y=0:diameter-1
            if se[x*diameter + y + 1] != 0
                neighbors[neighY * 2 - 1] = y - center
                neighbors[neighY * 2] = x - center
                neighY += 1
            end
        end
    end
end

function particlefilter(I::Array{UInt8}, IszX, IszY, Nfr, seed::Array{Int32}, Nparticles)

    max_size = IszX * IszY * Nfr
    start = gettime()
    # Original particle centroid
    xe = rounddouble(IszY/2.0)
    ye = rounddouble(IszX/2.0)

    # Expected object locations, compared to cneter
    radius = 5
    diameter = radius * 2 -1
    disk = Array{Int, 1}(diameter * diameter)
    streldisk(disk, radius)
    count_ones = 0
    for x=1:diameter
        for y=1:diameter
            if disk[(x-1) * diameter + y] == 1
                count_ones += 1
            end
        end
    end

    objxy = Array{Int, 1}(count_ones * 2)
    getneighbors(disk, count_ones, objxy, radius)
    get_neighbors = gettime()
    println("TIME TO GET NEIGHBORS TOOK: $(elapsedtime(start, get_neighbors))")

    # Initial weights are all equal (1/Nparticles)
    weights = Array{Float64, 1}(Nparticles)
    for x=1:Nparticles
        weights[x] = 1 / Nparticles
    end

    get_weights = gettime()
    println("TIME TO GET WEIGHTS TOOK: $(elapsedtime(get_neighbors, get_weights))")

    # Initial likelihood to 0.0
    g_likelihood = CuArray(zeros(Float64, Nparticles))
    g_arrayX = CuArray{Float64}(Nparticles)
    g_arrayY = CuArray{Float64}(Nparticles)
    xj = Array{Float64, 1}(Nparticles)
    yj = Array{Float64, 1}(Nparticles)
    g_CDF = CuArray{Float64}(Nparticles)

    g_ind = CuArray{Int}(count_ones * Nparticles)
    g_u = CuArray{Float64}(Nparticles)
    g_partial_sums = CuArray{Float64}(Nparticles)

    for x=1:Nparticles
        xj[x] = xe
        yj[x] = ye
    end

    num_blocks = Int(ceil(Nparticles/threads_per_block))

    g_xj = CuArray(xj)
    g_yj = CuArray(yj)
    g_objxy = CuArray(objxy)
    g_I = CuArray(I)
    g_weights = CuArray(weights)
    g_seed = CuArray(seed)

    for k=2:Nfr
        @measure "likelihood" @cuda (num_blocks, threads_per_block) likelihood_kernel(
            pointer(g_arrayX), length(g_arrayX),
            pointer(g_arrayY), length(g_arrayY),
            pointer(g_xj), length(g_xj),
            pointer(g_yj), length(g_yj),
            pointer(g_ind), length(g_ind),
            pointer(g_objxy), length(g_objxy),
            pointer(g_likelihood), length(g_likelihood),
            pointer(g_I), length(g_I),
            pointer(g_weights), length(g_weights),
            Nparticles, count_ones, max_size, k, IszY, Nfr,
            pointer(g_seed), length(g_seed),
            pointer(g_partial_sums), length(g_partial_sums))

        @measure "sum" @cuda (num_blocks, threads_per_block) sum_kernel(
            pointer(g_partial_sums), length(g_partial_sums), Nparticles)

        @measure "normalize_weights" @cuda (num_blocks, threads_per_block) normalize_weights_kernel(
            pointer(g_weights), length(g_weights),
            Nparticles,
            pointer(g_partial_sums), length(g_partial_sums),
            pointer(g_CDF), length(g_CDF),
            pointer(g_u), length(g_u),
            pointer(g_seed), length(g_seed))

        @measure "find_index" @cuda (num_blocks, threads_per_block) find_index_kernel(
            pointer(g_arrayX), length(g_arrayX),
            pointer(g_arrayY), length(g_arrayY),
            pointer(g_CDF), length(g_CDF),
            pointer(g_u), length(g_u),
            pointer(g_xj), length(g_xj),
            pointer(g_yj), length(g_yj),
            pointer(g_weights), length(g_weights),
            Nparticles)
    end

    arrayX = Array(g_arrayX)
    arrayY = Array(g_arrayY)
    weights = Array(g_weights)

    xe = ye = 0
    for x=1:Nparticles
        xe += arrayX[x] * weights[x]
        ye += arrayY[x] * weights[x]
    end

    if OUTPUT
        outf = open("output.txt", "w")
    else
        outf = STDOUT
    end
    println(outf,"XE: $xe")
    println(outf,"YE: $ye")
    distance = sqrt((xe - Int(rounddouble(IszX/2.0)))^2
                   +(ye - Int(rounddouble(IszY/2.0)))^2)
    println(outf,"distance: $distance")

    if OUTPUT
      close(outf)
    end

end

# Main

function main(args)

    # Check usage

    usage = "naive.out -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>"
    if length(args) != 8
        println(usage)
        exit(0)
    end

    if args[1] != "-x" || args[3] != "-y" || args[5] != "-z" || args[7] != "-np"
        println(usage)
        exit(0)
    end

    # Parse arguments

    IszX = parse(Int, args[2])
    if IszX <= 0
        println("dimX must be > 0")
        exit(0)
    end

    IszY = parse(Int, args[4])
    if IszY <= 0
        println("dimY must be > 0")
        exit(0)
    end

    Nfr = parse(Int, args[6])
    if Nfr <= 0
        println("number of frames must be > 0")
        exit(0)
    end

    Nparticles = parse(Int, args[8])
    if Nparticles <= 0
        println("number of particles must be > 0")
        exit(0)
    end

    # Initialize stuff
    seed = Array{Int32, 1}(Nparticles)
    for i = 1:Nparticles
        seed[i] = i-1
    end
    I = zeros(UInt8, IszX * IszY * Nfr)

    # Call videao sequence
    start = gettime()

    videosequence(I, IszX, IszY, Nfr, seed)
    end_video_sequence = gettime()
    println("VIDEO SEQUENCE TOOK $(elapsedtime(start, end_video_sequence))")

    # Call particle filter
    particlefilter(I, IszX, IszY, Nfr, seed, Nparticles)
    end_particle_filter = gettime()
    println("PARTICLE FILTER TOOK $(elapsedtime(end_video_sequence, end_particle_filter))")

    println("ENTIRE PROGRAM TOOK $(elapsedtime(start, end_video_sequence))")
end


dev = CuDevice(0)
ctx = CuContext(dev)

main(ARGS)
report()

destroy(ctx)
