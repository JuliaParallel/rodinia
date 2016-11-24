include("streamcluster_header.jl")

using CUDAdrv
using CUDAnative

const CUDA_TIME = true
const THREADS_PER_BLOCK = 512
const MAXBLOCKS = 65536

g_iter = 0 # counter for total # of g_iterations

#=======================================#
# Euclidean Distance
#=======================================#
function d_dist(p1, p2, num, dim, coord_d)
    retval = 0f0

    for i = 0:dim-1
        tmp = coord_d[(i * num) + p1 + 1] - coord_d[(i * num) + p2 + 1]
        retval += tmp * tmp
    end

    return retval
end

#=======================================#
# Kernel - Compute Cost
#=======================================#
function kernel_compute_cost(num, dim, x, p_w, p_a, p_c, K, stride, coord_d,
	                                     work_mem_d, center_table_d, switch_membership_d)
    # block ID and global thread ID
    const bid = blockIdx().x - 1 + gridDim().x * (blockIdx().y - 1)
    const tid = blockDim().x * bid + threadIdx().x - 1

    if tid < num
        lower_idx = tid * stride

        # cost between this point and point[x]: euclidean distance multiplied by weight
        x_cost = d_dist(tid, x, num, dim, coord_d) * p_w[tid + 1]

        # if computed cost is less then original (it saves), mark it as to reassign
        if x_cost < p_c[tid + 1]
            switch_membership_d[tid + 1] = true
            work_mem_d[lower_idx + K + 1] += x_cost - p_c[tid + 1]
        # if computed cost is larger, save the difference
        else
            work_mem_d[lower_idx + center_table_d[p_a[tid + 1] + 1] + 1] +=
                p_c[tid + 1] - x_cost
        end
    end

    return nothing
end

const g_coord_h = Ref{Array{Float32}}()

#=======================================#
# pgain Entry - CUDA SETUP + CUDA CALL
#=======================================#
function pgain(ctx, x, points, z, numcenters, kmax, is_center, center_table,
               switch_membership, isCoordChanged, serial_t, cpu_to_gpu_t,
               gpu_to_cpu_t, alloc_t, kernel_t)

    global g_iter
    global g_coord_h

    if CUDA_TIME
        start = CuEvent()
        stop = CuEvent()
        record(start)
    end

    stride = numcenters[] + 1 # size of each work_mem segment
    K = numcenters[]          # number of centers
    num = points.num              # number of points
    dim = points.dim              # number of dimension
    nThread = num                 # number of threads == number of data points

    #=========================================#
    # ALLOCATE HOST MEMORY + DATA PREPARATION
    #=========================================#
    work_mem_h = Array{Float32}(stride * (nThread + 1))

    # Only on the first iteration
    if g_iter == 0
        g_coord_h[] = Array{Float32}(num * dim)
    end

    # build center-index table
    count = 0

    for i = 1:num
        if is_center[i]
            center_table[i] = count
            count += 1
        end
    end

    # Extract 'coord'
    # Only if first iteration OR coord has changed
    if isCoordChanged || g_iter == 0
        for i = 1:dim
            for j = 1:num
                g_coord_h[][num * (i - 1) + j] = points.p[j].coord[i]
            end
        end
    end

    if CUDA_TIME
        record(stop)
        synchronize(stop)
        serial_t[] += elapsed(start, stop)
        record(start)
    end

    #==============================================#
    # ALLOCATE GPU MEMORY + CPU-TO-GPU MEMORY COPY
    #==============================================#
    # Only if first iteration OR coord has changed
    if isCoordChanged || g_iter == 0
        global g_coord_d = CuArray(g_coord_h[])
    end

    center_table_d = CuArray(center_table)

    p_w::Array{Float32} = [p.weight for p in points.p]
    p_a::Array{Int64}   = [p.assign for p in points.p]
    p_c::Array{Float32} = [p.cost   for p in points.p]

    p_wd = CuArray(p_w)
    p_ad = CuArray(p_a)
    p_cd = CuArray(p_c)

    work_mem_d = CuArray(zeros(Float32, stride * (nThread + 1)))
    switch_membership_d = CuArray([false for i = 1:num])

    if CUDA_TIME
        record(stop)
        synchronize(stop)
        tmp = elapsed(start, stop)
        alloc_t[] += tmp
        cpu_to_gpu_t[] += tmp
        record(start)
    end

    #=======================================#
    # KERNEL: CALCULATE COST
    #=======================================#
    # Determine the number of thread blocks in the x- and y-dimension
    num_blocks = floor(Int64, (num + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)
    num_blocks_y = floor(Int64, (num_blocks + MAXBLOCKS - 1) / MAXBLOCKS)
    num_blocks_x = floor(Int64, (num_blocks + num_blocks_y - 1) / num_blocks_y)

    @cuda ((num_blocks_x, num_blocks_y, 1), THREADS_PER_BLOCK) kernel_compute_cost(
        num,                # in:  # of data
        dim,                # in:  dimension of point coordinates
        x,                  # in:  point to open a center at
        p_wd, p_ad, p_cd,   # in:  data point array
        K,                  # in:  number of centers
        stride,             # in:  size of each work_mem segment
        g_coord_d,          # in:  array of point coordinates
        work_mem_d,         # out: cost and lower field array
        center_table_d,     # in:  center index table
        switch_membership_d # out: changes in membership
    )

    # TODO
    # // error check
    # error = cudaGetLastError();
    # if (error != cudaSuccess) {
    #     printf("kernel error: %s\n", cudaGetErrorString(error));
    #     exit(EXIT_FAILURE);
    # }

    if CUDA_TIME
        record(stop)
        synchronize(stop)
        kernel_t[] += elapsed(start, stop)
        record(start)
    end

    #=======================================#
    # GPU-TO-CPU MEMORY COPY
    #=======================================#
    work_mem_h = Array(work_mem_d)
    switch_membership = Array(switch_membership_d)

    if CUDA_TIME
        record(stop)
        synchronize(stop)
        gpu_to_cpu_t[] += elapsed(start, stop)
        record(start)
    end

    #=======================================#
    # CPU (SERIAL) WORK
    #=======================================#
    number_of_centers_to_close = 0
    gl_cost_of_opening_x = z
    gl_lower_idx = stride * nThread
    # compute the number of centers to close if we are to open i
    for i = 0:num-1
        if is_center[i + 1]
            low = z
            for j = 0:num-1
                low += work_mem_h[j * stride + center_table[i + 1] + 1]
            end

            work_mem_h[gl_lower_idx + center_table[i + 1] + 1] = low

            if low > 0
                number_of_centers_to_close += 1
                work_mem_h[i * stride + K + 1] -= low
            end
        end
        gl_cost_of_opening_x += work_mem_h[i * stride + K + 1]
    end

    # if opening a center at x saves cost (i.e. cost is negative) do so;
    # otherwise, do nothing
    if gl_cost_of_opening_x < 0
        for i = 1:num
            close_center = work_mem_h[gl_lower_idx + center_table[
                points.p[i].assign + 1] + 1] > 0

            if switch_membership[i] || close_center
                points.p[i].cost =
                    dist(points.p[i], points.p[x + 1], dim) * points.p[i].weight
                points.p[i].assign = x
            end
        end

        for i = 1:num
            if is_center[i] && work_mem_h[gl_lower_idx + center_table[i] + 1] > 0
                is_center[i] = false
            end
        end

        if x >= 0 && x < num
            is_center[x + 1] = true
        end

        numcenters[] += 1 - number_of_centers_to_close
    else
        gl_cost_of_opening_x = 0f0
    end

    if CUDA_TIME
        record(stop)
        synchronize(stop)
        serial_t[] += elapsed(start, stop)
        record(start)
    end

    g_iter += 1

    return -gl_cost_of_opening_x
end
