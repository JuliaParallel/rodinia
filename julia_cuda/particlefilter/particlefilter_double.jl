#!/usr/bin/env julia

using CUDAdrv, CUDAnative

const OUTPUT = haskey(ENV, "OUTPUT")

# configuration
const M = typemax(Int32)
const A = Int32(1103515245)
const C = Int32(12345)
const threads_per_block = 512


# Utility functions

function rounddouble(value)
    new_value = convert(Int, floor(value))
    if (value - new_value < 0.5)
        return new_value
    else
        # NOTE: this is wrong, but we mimic the behavior
        return new_value
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
    cosine = cos(2 * pi * v)
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
    I[x0 * IszY * Nfr + y0 * Nfr + 1] = 1       # TODO: +1 instead of 0?

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

@inline function dev_round_double(value)::Int32
    new_value = unsafe_trunc(Int32, value)
    if value - new_value < .5f0
        return new_value
    else
        # NOTE: keep buggy semantics of original, should be new_value+1
        return new_value
    end
end


# Kernels

function find_index_kernel(arrayX, arrayY, CDF, u, xj, yj, weights, Nparticles)
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

function normalize_weights_kernel(weights, Nparticles, partial_sums, CDF, u, seed)
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

function sum_kernel(partial_sums, Nparticles)
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

function likelihood_kernel(arrayX, arrayY, xj, yj, ind, objxy, likelihood, I, weights,
                           Nparticles, count_ones, max_size, k, IszY, Nfr, seed, partial_sums)
    block_id = blockIdx().x
    i = blockDim().x * (block_id-1) + threadIdx().x

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

            ind[(i-1)*count_ones + y + 1] = CUDAnative.abs(indX*IszY*Nfr + indY*Nfr + k - 1) + 1
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

    # Original particle centroid
    xe = rounddouble(IszY/2.0)
    ye = rounddouble(IszX/2.0)

    # Expected object locations, compared to cneter
    radius = 5
    diameter = radius * 2 -1
    disk = Vector{Int}(diameter * diameter)
    streldisk(disk, radius)
    count_ones = 0
    for x=1:diameter
        for y=1:diameter
            if disk[(x-1) * diameter + y] == 1
                count_ones += 1
            end
        end
    end

    objxy = Vector{Int}(count_ones * 2)
    getneighbors(disk, count_ones, objxy, radius)

    # Initial weights are all equal (1/Nparticles)
    weights = Vector{Float64}(Nparticles)
    for x=1:Nparticles
        weights[x] = 1 / Nparticles
    end

    # Initial likelihood to 0.0
    g_likelihood = CuArray(zeros(Float64, Nparticles))
    g_arrayX = CuArray{Float64}(Nparticles)
    g_arrayY = CuArray{Float64}(Nparticles)
    xj = Vector{Float64}(Nparticles)
    yj = Vector{Float64}(Nparticles)
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
        @cuda (num_blocks, threads_per_block) likelihood_kernel(
            g_arrayX, g_arrayY, g_xj, g_yj, g_ind,
            g_objxy, g_likelihood, g_I, g_weights,
            Nparticles, count_ones, max_size, k, IszY, Nfr,
            g_seed, g_partial_sums)

        @cuda (num_blocks, threads_per_block) sum_kernel(
            g_partial_sums, Nparticles)

        @cuda (num_blocks, threads_per_block) normalize_weights_kernel(
            g_weights, Nparticles, g_partial_sums, g_CDF, g_u, g_seed)

        @cuda (num_blocks, threads_per_block) find_index_kernel(
            g_arrayX, g_arrayY, g_CDF, g_u, g_xj, g_yj, g_weights, Nparticles)
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

    # initialize
    seed = Vector{Int32}(Nparticles)
    for i = 1:Nparticles
        seed[i] = i-1
    end
    I = zeros(UInt8, IszX * IszY * Nfr)

    videosequence(I, IszX, IszY, Nfr, seed)

    particlefilter(I, IszX, IszY, Nfr, seed, Nparticles)
end


main(ARGS)

if haskey(ENV, "PROFILE")
    CUDAnative.@profile main(ARGS)
end
