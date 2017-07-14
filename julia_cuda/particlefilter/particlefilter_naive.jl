#!/usr/bin/env julia

using CUDAdrv, CUDAnative

# Variables

const M = typemax(Int32)
const A = Int32(1103515245)
const C = Int32(12345)

const PI = 3.1415926535897932

const threads_per_block = 128

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

function calc_likelihood_sum(I, ind, num_ones)
    likelihood_sum = Float64(0)
    for y=1:num_ones
        v = ((I[ind[y]] -100)^2 - (I[ind[y]] -228)^2)/50
        likelihood_sum += v
    end
    return likelihood_sum
end


function particlefilter(I::Array{UInt8}, IszX, IszY, Nfr, seed, Nparticles)

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
    likelihood = Array{Float64, 1}(Nparticles)
    arrayX = Array{Float64, 1}(Nparticles)
    arrayY = Array{Float64, 1}(Nparticles)
    xj = Array{Float64, 1}(Nparticles)
    yj = Array{Float64, 1}(Nparticles)
    CDF = Array{Float64, 1}(Nparticles)

    ind = Array{Int, 1}(count_ones)
    u = Array{Float64, 1}(Nparticles)

    for x=1:Nparticles
        arrayX[x] = xe
        arrayY[x] = ye
    end

    g_arrayX = CuArray(Float64, Nparticles)
    g_arrayY = CuArray(Float64, Nparticles)
    g_xj = CuArray(Float64, Nparticles)
    g_yj = CuArray(Float64, Nparticles)
    g_CDF = CuArray(Float64, Nparticles)
    g_u = CuArray(Float64, Nparticles)

    for k=2:Nfr
        set_arrays = gettime()

        for x=1:Nparticles
            arrayX[x] = arrayX[x] + 1 + 5 * randn(seed, x)
            arrayY[x] = arrayY[x] - 2 + 2 * randn(seed, x)
        end

        error = gettime()
        println("TIME TO SET ERROR TOOK: $(elapsedtime(set_arrays, error))")

        # Particle filter likelihood
        for x=1:Nparticles
            for y=1:count_ones
                #d("objxy[(y-1)*2 + 1]", objxy[(y-1)*2 + 1])
                indX::Int = rounddouble(arrayX[x]) + objxy[(y-1)*2 + 2]
                indY::Int = rounddouble(arrayY[x]) + objxy[(y-1)*2 + 1]
                v::Float64 = abs(indX * IszY * Nfr + indY * Nfr + k -1)
                ind[y] = v+1
                if ind[y] >= max_size
                    ind[y] = 1
                end
            end
            likelihood[x] = calc_likelihood_sum(I, ind, count_ones)
            likelihood[x] = likelihood[x] / count_ones
        end
        
        likelihood_time = gettime()
        println("TIME TO GET LIKELIHOODS TOOK: $(elapsedtime(error, likelihood_time))")

        # Update & normalize weights
        for x=1:Nparticles
            weights[x] = weights[x] * exp(likelihood[x])
        end

        exponential = gettime()
        println("TIME TO GET EXP TOOK: $(elapsedtime(likelihood_time, exponential))")
        sum_weights = 0;
        for x = 1:Nparticles
            sum_weights += weights[x];
        end

        sum_time = gettime();
        println("TIME TO SUM WEIGHTS TOOK: $(elapsedtime(exponential, sum_time))");
        for x = 1:Nparticles
            weights[x] = weights[x] / sum_weights;
        end

        normalize = gettime();
        println("TIME TO NORMALIZE WEIGHTS TOOK: $(elapsedtime(sum_time, normalize))");

        xe = ye = 0
        for x=1:Nparticles
            xe += arrayX[x] * weights[x]
            ye += arrayY[x] * weights[x]
        end
        move_time = gettime()
        println("TIME TO MOVE OBJECT TOOK: $(elapsedtime(normalize, move_time))")
        println("XE: $xe")
        println("YE: $ye")
        distance = sqrt(
            (xe - trunc(rounddouble(IszY/2.0)))^2  
            + (ye - trunc(rounddouble(IszX/2.0)))^2)

        # Resampling

        CDF[1] = weights[1]
        for x=2:Nparticles
            CDF[x] = weights[x] + CDF[x-1]
        end

        cumsum = gettime()
        println("TIME TO CALC CUM SUM: $(elapsedtime(move_time, cumsum))")

        u1 = (1/Nparticles) * randu(seed, 1)
        for x=1:Nparticles
            u[x] = u1 + (x-1)/Nparticles
        end

        utime = gettime()
        println("TIME TO CALC U TOOK: $(elapsedtime(cumsum, utime))")

        # Set number of threads
        num_blocks = convert(Int,ceil(Nparticles/threads_per_block))
        # Kernel call

        copy!(g_arrayX, arrayX)
        copy!(g_arrayY, arrayY)
        copy!(g_CDF, CDF)
        copy!(g_u, u)

        @cuda (num_blocks, threads_per_block) kernel_kernel(
            g_arrayX, g_arrayY, g_CDF, g_u, g_xj, g_yj, Nparticles)
        copy!(xj, g_xj)
        copy!(yj, g_yj)
        exec_time = gettime()

        println("CUDA EXEC TOOK: $(elapsedtime(utime, exec_time))")

        for x=1:Nparticles
            arrayX[x] = xj[x]
            arrayY[x] = yj[x]
            weights[x] = 1 / Nparticles
        end

        reset = gettime()
        println("TIME TO RESET: $(elapsedtime(exec_time, reset))")
    end
end

# Kernel

function kernel_kernel(arrayX, arrayY, CDF, u, xj, yj, Nparticles)
    
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

    return nothing
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


main(ARGS)

if haskey(ENV, "PROFILE")
    CUDAnative.@profile main(ARGS)
end
