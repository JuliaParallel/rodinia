using CUDA

# Variables

const M = typemax(Int32)
const A = Int32(1103515245)
const C = Int32(12345)

const PI = 3.1415926535897932

const threads_per_block = 512

# Utility functions

function gettime()
	current = now()
	elapsed = Dates.hour(current) * 60 * 60 * 1000000
	        + Dates.minute(current)    * 60 * 1000000
	        + Dates.second(current)		  * 1000000
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
		return new_value	# Copy bug from original
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
	I[x0 * IszY * Nfr + y0 * Nfr + 1] = 1 		# TODO: +1 instead of 0????

	# Move point
	xk = yk = 0
	for k = 2:Nfr-1
		xk = abs(x0 + (k - 2));
        yk = abs(y0 - 2 * (k - 2));
        pos = yk * IszY * Nfr + xk * Nfr + k;
        if pos >= max_size
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

function particlefilter(I::Array{UInt8}, IszX, IszY, Nfr, seed::Array{Int32}, Nparticles)

	max_size = IszX * IszY * Nfr
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
	# Initial weights are all equal (1/Nparticles)
	weights = Array{Float64, 1}(Nparticles)
	for x=1:Nparticles
		weights[x] = 1 / Nparticles
	end

	# Initial likelihood to 0.0
	likelihood = Array{Float64, 1}(Nparticles)
	arrayX = Array{Float64, 1}(Nparticles)
	arrayY = Array{Float64, 1}(Nparticles)
	xj = Array{Float64, 1}(Nparticles)
	yj = Array{Float64, 1}(Nparticles)
	CDF = Array{Float64, 1}(Nparticles)

	ind = Array{Int, 1}(count_ones * Nparticles)
	u = Array{Float64, 1}(Nparticles)
	partial_sums = Array{Float64, 1}(Nparticles)

	for x=1:Nparticles
		xj[x] = xe
		yj[x] = ye
	end

	num_blocks = Int(ceil(Nparticles/threads_per_block))

	for k=2:Nfr

		@cuda (num_blocks, threads_per_block, 8*512) kernel_likelihood(
			arrayX, arrayY, xj, yj, CDF, ind, objxy, likelihood, I, u, weights, 
			Nparticles, count_ones, max_size, k, IszY, Nfr, 
			seed, partial_sums)
		#=@cuda (num_blocks, threads_per_block) kernel_sum(partial_sums, Nparticles)
		@cuda (num_blocks, threads_per_block, 8*2) kernel_normalize_weights(
			weights, Nparticles,
			partial_sums, CDF, u, seed)
		@cuda (num_blocks, threads_per_block) kernel_find_index(
			arrayX, arrayY, CDF, u, xj, yj, weights, Nparticles)=#
    end
    synchronize(ctx)

    xe = ye = 0
    for x=1:Nparticles
    	xe += arrayX[x] * weights[x]
    	ye += arrayY[x] * weights[x]
    end

    println("XE: $xe")
    println("YE: $ye")
    distance = sqrt((xe - Int(rounddouble(IszX/2.0)))^2
    			   +(ye - Int(rounddouble(IszY/2.0)))^2)
    println(distance)

end

# Device code

@target ptx function kernel_find_index(
	arrayX::CuDeviceArray{Float64}, arrayY::CuDeviceArray{Float64}, CDF::CuDeviceArray{Float64},
	u::CuDeviceArray{Float64}, xj::CuDeviceArray{Float64}, yj::CuDeviceArray{Float64},
	weights::CuDeviceArray{Float64}, Nparticles)
	
	block_id = blockIdx().x
	i = blockDim().x * (block_id-1) + threadIdx().x

	if i <= Nparticles
		index = 0 	# an invalid index
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

@target ptx function cdf_calc(CDF::CuDeviceArray{Float64}, weights::CuDeviceArray{Float64}, Nparticles)
	CDF[1] = weights[1]
	for x=2:Nparticles
		CDF[x] = weights[x] + CDF[x-1]
	end
end

@target ptx function d_randu(seed::CuDeviceArray{Int32}, index)
	num = A * seed[index] + C
	seed[index] = num % M
	return abs(seed[index]/M)
end

@target ptx function kernel_normalize_weights(
	weights::CuDeviceArray{Float64}, Nparticles,
	partial_sums::CuDeviceArray{Float64}, CDF::CuDeviceArray{Float64},
	u::CuDeviceArray{Float64}, seed::CuDeviceArray{Int})

	block_id = blockIdx().x
	i = blockDim().x * (block_id-1) + threadIdx().x

	shared = cuSharedMem_double()	# size of 2 doubles
	u1_i = 1
	sum_weights_i = 2
	# shared[1] == u1, shared[2] = sum_weights

	if threadIdx().x == 1
		setCuSharedMem_double(shared, sum_weights_i, partial_sums[0])
	end
	sync_threads()

	if i <= Nparticles
		weights[i] = weights[i] / getCuSharedMem_double(shared, sum_weights_i)
	end
	sync_threads()

	if i==1
		cdf_calc(CDF, weights, Nparticles)
		u[1] = (1/Nparticles) * d_randu(seed, i)
	end
	sync_threads()

	if threadIdx().x == 1
		setCuSharedMem_double(shared, u1_i, u[1])
	end
	sync_threads()

	if i <= Nparticles
		u1 = getCuSharedMem_double(shared, u1_i)
		u[i] = u1 + i / Nparticles
	end

	return nothing
end

@target ptx function kernel_sum(partial_sums::CuDeviceArray{Float64}, Nparticles)
	block_id = blockIdx().x
	i = blockDim().x * (block_id-1) + threadIdx().x

	if i==1
		sum = 0.0
		num_blocks = Int(ceil(Nparticles/threads_per_block))
		for x=1:num_blocks
			sum += partial_sums[x]
		end
		partial_sums[1] = sum
	end

	return nothing
end

@target ptx function calc_likelihood_sum(I, ind, num_ones, index)
	likelihood_sum = Float64(0)
	for x=1:num_ones
		v = ((I[ind[index * num_ones + x]] -100)^2 
			- (I[ind[index * num_ones + x]] -228)^2)/50
		likelihood_sum += v
	end
	return likelihood_sum
end

@target ptx function kernel_likelihood(
	arrayX::CuDeviceArray{Float64}, arrayY::CuDeviceArray{Float64}, xj::CuDeviceArray{Float64}, 
	yj::CuDeviceArray{Float64}, CDF::CuDeviceArray{Float64}, ind::CuDeviceArray{Int}, objxy::CuDeviceArray{Int}, 
	likelihood::CuDeviceArray{Float64}, I::CuDeviceArray{UInt8}, 
	u::CuDeviceArray{Float64}, weights::CuDeviceArray{Float64}, 
	Nparticles, count_ones, max_size, k, IszY, 
	Nfr, seed::CuDeviceArray{Int32}, partial_sums::CuDeviceArray{Float64})

	block_id = blockIdx().x
	i::Int = blockDim().x * (block_id-1) + threadIdx().x

	buffer = cuSharedMem_double()	# 512 doubles
	if i <= Nparticles
		arrayX[i] = xj[i]
		arrayY[i] = yj[i]
		weights[i] = 1/Nparticles

		d_randn(seed, i)
		#arrayX[i] = arrayX[i] + 1.0 + 5.0 * d_randn(seed, i)
		#arrayY[i] = arrayY[i] - 2.0 + 2.0 * d_randn(seed, i)
	end

	#=sync_threads()

	if i <= Nparticles
		for y=0:count_ones-1
			indX = dev_round_double(arrayX[i]) + objxy[y*2 + 2]
			indY = dev_round_double(arrayY[i]) + objxy[y*2 + 1]

			ind[(i-1)*count_ones + y + 1] = abs(indX*IszY*Nfr + indY*Nfr + k) + 1
			if ind[(i-1)*count_ones + y + 1] > max_size
				ind[(i-1)*count_ones + y + 1] = 1
			end
		end
		likelihood[i] = calc_likelihood_sum(I, ind, count_ones, i)
		likelihood[i] = likelihood[i]/count_ones
		weights[i] = weights[i] * CUDA.exp(likelihood[i])
	end
	setCuSharedMem_double(buffer, threadIdx().x, 0.0)

	sync_threads()

	if i<Nparticles
		buffer[threadIdx().x] = weights[i]
	end
	sync_threads()

	s = UInt(blockDim().x/2)
	while s > 0
		if threadIdx().x < s+1
			v = getCuSharedMem_double(buffer, threadIdx().x)
			v += getCuSharedMem_double(buffer, threadIdx().x + s)
			setCuSharedMem_double(buffer, threadIdx().x, v)
		end
		s>>=1
	end
	if threadIdx().x == 1
		partial_sums[blockIdx().x] = getCuSharedMem_double(buffer, 1)
	end
	sync_threads()=#
	return nothing
end

# Utility device functions

@target ptx function dev_round_double(value)
	new_value = trunc(value)
	if value - new_value < 0.5
		return new_value
	else
		return new_value # keep buggy semantics of original, should be new_value+1
	end
end

# Main

function main()

	# Check usage

	usage = "naive.out -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>"
	if length(ARGS) != 8
		println(usage)
		exit(0)
	end

	if ARGS[1] != "-x" || ARGS[3] != "-y" || ARGS[5] != "-z" || ARGS[7] != "-np"
		println(usage)
		exit(0)
	end

	# Parse arguments

	IszX = parse(Int, ARGS[2])
	if IszX <= 0
		println("dimX must be > 0")
		exit(0)
	end

	IszY = parse(Int, ARGS[4])
	if IszY <= 0
		println("dimY must be > 0")
		exit(0)
	end

	Nfr = parse(Int, ARGS[6])
	if Nfr <= 0
		println("number of frames must be > 0")
		exit(0)
	end

	Nparticles = parse(Int, ARGS[8])
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

# Setup context etc
num_dev = devcount()
if num_dev > 0
	const dev = CuDevice(0)
	const ctx = CuContext(dev)
	const cgctx = CuCodegenContext(ctx, dev)

	# Run the code with given params
	main()

	destroy(ctx)
	destroy(cgctx)
end
