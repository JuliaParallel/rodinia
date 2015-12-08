# Variables

M = typemax(Int32)
A = 1103515245
C = 12345

const PI = 3.1415926535897932

const threads_per_block = 128

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
	new_value = floor(value)
	if (value - new_value < 0.5)
		return new_value
	else
		return new_value+1
	end
end

function randu(seed, index)
	num = A * seed[index] + C
	seed[index] = num % M
	return abs(seed[index]/M)
end

function randn(seed, index)
	u = randu(seed, index)
	v = randu(seed, index)
	cosine = cos(2 * PI * v)
	rt = -2 * log(u)
	return sqrt(rt) * cosine
end

# Video sequence

function setif(test_value, new_value, array3D::Array{Int}, dimX, dimY, dimZ)
	for x=1:dimX
		for y=1:dimY
			for z=1:dimZ
				if array3D[(x-1)*dimY+(y-1)*dimZ+z] == test_value
					array3D[(x-1)*dimY+(y-1)*dimZ+z] = new_value
				end
			end
		end
	end
end

function addnoise(array3D::Array{Int}, dimX, dimY, dimZ, seed)
	for x=1:dimX
		for y=1:dimY
			for z=1:dimZ
				noise = randn(seed, 1)
				array3D[(x-1)*dimY+(y-1)*dimZ+z] =
					array3D[(x-1)*dimY+(y-1)*dimZ+z] + round(noise)
			end
		end
	end
end

function dilate_matrix(matrix, posX, posY, posZ, dimX, dimY, dimZ, error)
	startX = posX - error
	while startX < 1
		startX += 1
	end
	startY = posY - error
	while startY < 1
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
	for x=startX:endX
		for y=startY:endY
			distance = sqrt((x-posX)^2 + (y-posY)^2)
			if distance < error
				matrix[(x-1) * dimY * dimZ + (y-1) * dimZ + posZ] = 1
			end
		end
	end
end

function imdilate_disk(matrix, dimX, dimY, dimZ, error, new_matrix)
	for z=1:dimZ
		for x=1:dimX
			for y=1:dimY
				if matrix[(x-1) * dimY * dimZ + (y-1) * dimZ + z] == 1 
					dilate_matrix(new_matrix, x, y, z, dimX, dimY, dimZ, error)
				end
			end
		end
	end
end

function videosequence(I::Array{Int}, IszX, IszY, Nfr, seed::Array{Int})

	max_size = IszX * IszY * Nfr
	# get object centers
	x0 = convert(Int, rounddouble(IszX/2.0))
	y0 = convert(Int, rounddouble(IszY/2.0))
	I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1 		# TODO: +1 instead of 0????

	# Move point
	xk = yk = 0
	for k = 1:Nfr-1
		xk = abs(x0 + (k - 1));
        yk = abs(y0 - 2 * (k - 1));
        pos = yk * IszY * Nfr + xk * Nfr + k;
        if pos >= max_size
            pos = 1;
        end
        I[pos] = 1;
    end

    # Dialate matrix
    new_matrix = Array{Int, 1}(IszX * IszY * Nfr)
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
    setif(0, 200, I, IszX, IszY, Nfr)
    setif(0, 200, I, IszX, IszY, Nfr)
    # Add noise
    addnoise(I, IszX, IszY, Nfr, seed)
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
	seed = Array{Int, 1}(Nparticles)
	for i = 1:Nparticles
		t = round(Int32, time())
		seed[i] = t * (i-1)
	end
	I = Array{Int, 1}(IszX * IszY * Nfr)

	# Call videao sequence
	start = gettime()
	videosequence(I, IszX, IszY, Nfr, seed)
	end_video_sequence = gettime()
	println("VIDEO SEQUENCE TOOK $(elapsedtime(start, end_video_sequence))")

	# Call particle filter
	#particlefilter(I, IszX, IszY, Nfr, seed, Nparticles)
	end_particle_filter = gettime()
	#println("PARTICLE FILTER TOOK $(elapsedtime(end_video_sequence, end_particle_filter))")

	println("ENTIRE PROGRAM TOOK $(elapsedtime(start, end_video_sequence))")
end

main()