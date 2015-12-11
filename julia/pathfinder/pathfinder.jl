using CUDA

# Configuration
const BLOCK_SIZE = 256
const HALO = 1

const M_SEED = 7
const OUTPUT = true

# Helper function

@target ptx function inrange(x, min, max)
	return x >= min && x <= max
end

function min(a, b)
	return a <= b ? a : b
end

@target ptx function dev_min(a, b)
	return a <= b ? a : b
end

# Override rng functions with libc implementations
function srand(seed)
	ccall( (:srand, "libc"), Void, (Int,), seed)
end

function rand()
	r = ccall( (:rand, "libc"), Int, ())
	return r
end

# Define global variables
wall = Array{Int64, 2}
rows = cols = pyramid_height = 0

# Device code

@target ptx function kernel_dynproc(
	iteration,
	gpu_wall::CuDeviceArray{Int64}, gpu_src::CuDeviceArray{Int64}, gpu_results::CuDeviceArray{Int64}, 
	cols, rows, start_step, border)
	
	# Define shared memory
	#=
	#cuShareMem_i64() returns the same pointer twice, we want separate values, 
	prev = cuSharedMem_i64()
	result = cuSharedMem_i64()
	=#

	#For now use 1 shared mem block together with offsets
	shared_mem = cuSharedMem_i64()	# size: 2*256*8 bytes (Int64 -> 8 bytes), indicated when calling using @cuda macro
	# prev = shared_mem[0:256]
	# result = shared_mem[265:512]
	prev_offset = 0
	result_offset = 256

	bx = blockIdx().x
	tx = threadIdx().x

	# Will this be a problem: references to global vars
	# but will likely be replaced by a constant when jitting, or not?
	small_block_cols = BLOCK_SIZE - iteration * HALO * 2

	blk_x = small_block_cols * (bx-1) - border;
	blk_x_max = blk_x + BLOCK_SIZE -1

	xidx = blk_x + tx

	valid_x_min = (blk_x < 0) ? -blk_x : 0
	valid_x_max = (blk_x_max > cols -1)  ? BLOCK_SIZE -1 -(blk_x_max - cols +1) : BLOCK_SIZE -1
	valid_x_min = valid_x_min +1
	valid_x_max = valid_x_max +1

	W = tx - 1
	E = tx + 1
	W = (W < valid_x_min) ? valid_x_min : W
	E = (E > valid_x_max) ? valid_x_max : E

	is_valid = inrange(tx, valid_x_min, valid_x_max)

	if inrange(xidx, 1, cols)
		#prev[tx] = gpu_src[xidx]
		# setCuSharedMem_i64(prev, tx, gpu_src[xidx])
		setCuSharedMem_i64(shared_mem, prev_offset +tx, gpu_src[xidx])
	end

	sync_threads()

	computed = false
	for i = 1:iteration
		computed = false
		if inrange(tx, i+1, BLOCK_SIZE -i) && is_valid
			computed = true

			#left = getCuSharedMem_i64(prev, W)	#left = prev[W]
			left = getCuSharedMem_i64(shared_mem, prev_offset +W)
			#up = getCuSharedMem_i64(prev, tx)	#up = prev[tx]
			up = getCuSharedMem_i64(shared_mem, prev_offset +tx)
			#right = getCuSharedMem_i64(prev, E)	# right = prev[E]
			right = getCuSharedMem_i64(shared_mem, prev_offset +E)	

			shortest = dev_min(left, up)
			shortest = dev_min(shortest, right)

			index = cols * (start_step + (i-1)) + xidx
			#result[tx] = shortest + gpu_wall[index]
			#setCuSharedMem_i64(result, tx, shortest + gpu_wall[index]) 
			setCuSharedMem_i64(shared_mem, result_offset +tx, shortest + gpu_wall[index])
		end
		sync_threads()
		if i == iteration
			break
        end
        if computed
			#prev[tx] = result[tx]
			#value = getCuSharedMem_i64(result, tx)
			value = getCuSharedMem_i64(shared_mem, result_offset +tx)
			#setCuSharedMem_i64(prev, tx, value)	
			setCuSharedMem_i64(shared_mem, prev_offset +tx, value)
		end
		sync_threads()
	end

	if computed
		# gpu_result[xidx] = result[tx]
		#gpu_results[xidx] =  getCuSharedMem_i64(result, tx)
		gpu_results[xidx] =  getCuSharedMem_i64(shared_mem, result_offset +tx)
	end
	return nothing	# pretty important this
end

# Host code

function init(args)

	if length(args) == 3 
		global cols = parse(Int, args[1])
		global rows = parse(Int, args[2])
		global pyramid_height = parse(Int, args[3])
	else
		println("Usage: dynproc row_len col_len pyramid_height")
		exit(0) 
	end

	srand(M_SEED)

	# Initialize en fill wall
	# Switch semantics of row & col -> easy copy to gpu array in run function
	global wall = Array{Int64}(cols, rows)
	for i = 1:length(wall)
		wall[i] = Int64(rand() % 10)
	end

	# Print wall
	if OUTPUT
		file = open("output.txt", "w")
		println(file, "wall:")
		for i = 1:rows
			for j = 1:cols
				print(file, "$(wall[j,i]) ")
			end
			println(file, "")
		end
		close(file)
	end

end

function calcpath(gpu_wall, gpu_result, rows, cols, 
	pyramid_height, block_cols, border_cols)

	dim_block = BLOCK_SIZE
	dim_grid = block_cols

	src = 2
	dst = 1
	for t = 0:pyramid_height:rows-1
		tmp = src
		src = dst
		dst = tmp

		# TODO: Copy slice, because slices can't be passed as kernerl args
		gpu_src = gpu_result[src,:]
		gpu_dst = gpu_result[dst,:]
		iter = min(pyramid_height, rows -t -1)

		@cuda (dim_grid, dim_block, 256*2*8) kernel_dynproc(
			iter,
			CuIn(gpu_wall), 
			CuIn(gpu_src),		# Does not work with slice: CuIn(gpu_result[src,:])
			CuOut(gpu_dst),
			cols, rows, t, border_cols
		)

		gpu_result[src,:] = gpu_src 
		gpu_result[dst,:] = gpu_dst 
	end

	return dst
end

function run(args)

	# Initialize data
	init(args)

	# Calculate parameters
	border_cols = pyramid_height * HALO
	small_block_col = BLOCK_SIZE - pyramid_height*HALO * 2
	block_cols = floor(Int, cols/small_block_col) + ((cols % small_block_col == 0) ? 0 : 1)

	println(
"pyramid_height: $pyramid_height
grid_size: [$cols]
border: [$border_cols]
block_size: $BLOCK_SIZE
block_grid: [$block_cols]
target_block: [$small_block_col]")

	# Setup GPU memory
	gpu_result = Array{Int64}(2, cols)
	gpu_result[1,:] = wall[:,1]	# 1st row (column major)

	gpu_wall = wall[cols+1:end]

	# First do a dummy call to force compilation of the 
	# calcpath function and the kernel
	final_ret = calcpath(
		gpu_wall, gpu_result,
		rows, cols, pyramid_height,
		block_cols, border_cols)

	result = gpu_result[final_ret, :]

	if OUTPUT
		file = open("output.txt", "a")
		println(file, "data:")

		for i=1:cols
			print(file, "$(wall[i]) ")
		end
		println(file, "")

		println(file, "result:")
		for i=1:cols
			print(file, "$(result[i]) ")
		end
		println(file, "")

		close(file)
	end

end



# Set the CUDA device
num_dev = devcount()
if num_dev > 0
	dev = CuDevice(0)
	ctx = CuContext(dev)
	cgctx = CuCodegenContext(ctx, dev)

	# Run the code with given params
	run(ARGS)

	destroy(ctx)
	destroy(cgctx)

end
