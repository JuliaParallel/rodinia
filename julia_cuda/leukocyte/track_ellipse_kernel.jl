using CUDAdrv, CUDAnative

# The number of threads per thread block
const threads_per_block = 320
# next_lowest_power_of_two = 2^(floor(log2(threads_per_block)))
const next_lowest_power_of_two = 256

# Regularized version of the Heaviside step function:
# He(x) = (atan(x) / pi) + 0.5
function heaviside(z)
    result = CUDAnative.atan(z) * (1.0 / C_PI) + 0.5
    return result
end


function IMGVF_kernel(I, IMGVF_global, m::Integer, n::Integer, vx, vy, e, max_iterations, cutoff)
    # Constants
    const mu = 0.5
    const lambda = 8.0 * mu + 1.0
    # 41 * 81, @cuStaticSharedMem can't deal with expressions -- even in constants
    const IMGVF_SIZE = 3321

    # Shared copy of the matrix being computed
    IMGVF = @cuStaticSharedMem(Float32, (IMGVF_SIZE,))

    # Shared buffer used for two purposes:
    # 1) To temporarily store newly computed matrix values so that only
    #    values from the previous iteration are used in the computation.
    # 2) To store partial sums during the tree reduction which is performed
    #    at the end of each iteration to determine if the computation has
    #    converged.
    buffer = @cuStaticSharedMem(Float32, (threads_per_block,))
    # keep track of whether the current cell has converged
    cell_converged = @cuStaticSharedMem(Int32, 1)

    # Compute the number of virtual thread blocks
    max = div(m * n + threads_per_block - 1,threads_per_block)

    # avoid error checks for undefined i later on (the loop below
    # always entered and hence i always initialised)
    i::Int32 = 0

    # Load the initial IMGVF matrix into shared memory
    thread_id::Int32 = threadIdx().x
    for thread_block in 0:max-1
        offset = div(thread_block,threads_per_block)
        i = div(thread_id + offset, n)
        j = mod1(thread_id + offset, n)
        if i < m
            @inbounds IMGVF[i * n + j] = IMGVF_global[i * n + j]
        end
    end
    sync_threads()

    # Set the converged flag to false
    if thread_id == 0
        @inbounds cell_converged[1] = 0
    end
    sync_threads()


    # Constants used to iterate through virtual thread blocks
    const one_nth = 1.0/convert(Float32,n)
    const tid_mod = thread_id % n
    const tbsize_mod = threads_per_block % n

    # Constant used in the computation of Heaviside values
    const one_over_e = 1.0/e

     
    # Iteratively compute the IMGVF matrix until the computation has
    #  converged or we have reached the maximum number of iterations
    iterations = 0

    @inbounds while (cell_converged[1] == 0) && (iterations < max_iterations)
        # The total change to this thread's matrix elements in the current
        # iteration
        total_diff::Float32 = 0.0

        old_i::Int32 = 0
        old_j::Int32 = 0
        j = tid_mod - tbsize_mod

        # Iterate over virtual thread blocks
        for thread_block in 1:max
            # Store the index of this thread's previous matrix element
            #  (used in the buffering scheme below)
            old_i = i
            old_j = j

            # Determine the index of this thread's current matrix element
            offset = thread_block * threads_per_block
            i = trunc(Int32,(thread_id + offset) * one_nth)
            j += tbsize_mod
            if j >= n
                j -= n
            end

            new_val::Float32 = 0.0
            old_val::Float32 = 0.0
            
            # Make sure the thread has not gone off the end of the matrix
            if i < m
                # Compute neighboring matrix element indices
                rowU = (i == 0) ? 0  : i - 1
                rowD = (i == m - 1) ? m - 1: i + 1
                colL = (j == 0) ? 0 : j - 1
                colR = (j == n - 1) ? n -1 : j + 1

                # Compute the difference between the matrix element and its
                # eight neighbors
                @inbounds old_val = IMGVF[(i * n) + j + 1]
                @inbounds U = IMGVF[(rowU * n) + j + 1] - old_val
                @inbounds D = IMGVF[(rowD * n) + j + 1] - old_val
                @inbounds L = IMGVF[(i * n) + colL + 1] - old_val
                @inbounds R = IMGVF[(i * n) + colR + 1] - old_val
                @inbounds UR = IMGVF[(rowU * n) + colR + 1] - old_val
                @inbounds DR = IMGVF[(rowD * n) + colR + 1] - old_val
                @inbounds UL = IMGVF[(rowU * n) + colL + 1] - old_val
                @inbounds DL = IMGVF[(rowD * n) + colL + 1] - old_val

                # Compute the regularized heaviside value for these differences
                UHe = heaviside((U * -vy) * one_over_e)
                DHe = heaviside((D * vy) * one_over_e)
                LHe = heaviside((L * -vx) * one_over_e)
                RHe = heaviside((R * vx) * one_over_e)
                URHe = heaviside((UR * (vx - vy)) * one_over_e)
                DRHe = heaviside((DR * (vx + vy)) * one_over_e)
                ULHe = heaviside((UL * (-vx - vy)) * one_over_e)
                DLHe = heaviside((DL * (-vx + vy)) * one_over_e)

                # Update the IMGVF value in two steps:
                # 1) Compute IMGVF += (mu / lambda)(UHe .*U  + DHe .*D  + LHe
                # .*L  + RHe .*R +
                #                                   URHe.*UR + DRHe.*DR +
                #                                   ULHe.*UL + DLHe.*DL)
                new_val = old_val +
                          (mu / lambda) *
                              (UHe * U + (DHe * D + LHe * L) + RHe * R +
                               URHe * UR + (DRHe * DR + ULHe * UL) + DLHe * DL)
                # 2) Compute IMGVF -= (1 / lambda)(I .* (IMGVF - I))
                @inbounds vI = I[i * n + j + 1]
                new_val -= ((1.0 / lambda) * vI * (new_val - vI));
            end

            # Save the previous virtual thread block's value (if it exists)
            if thread_block > 0
                offset = (thread_block - 1) * threads_per_block
                if old_i < m
                    @inbounds IMGVF[(old_i * n) + old_j + 1] = buffer[thread_id + 1]
                end
            end
            if thread_block < max - 1
                # Write the new value to the buffer
                @inbounds buffer[thread_id + 1] = new_val
            else
                # We've reached the final virtual thread block,
                #  so write directly to the matrix
                if i < m
                    @inbounds IMGVF[(i * n) + j + 1] = new_val
                end
            end

            # Keep track of the total change of this thread's matrix elements
            total_diff += abs(new_val - old_val)

            # We need to synchronize between virtual thread blocks to prevent
            #  threads from writing the values from the buffer to the actual
            #  IMGVF matrix too early
            sync_threads()
        end

        # We need to compute the overall sum of the change at each matrix
        # element
        #  by performing a tree reduction across the whole threadblock
        @inbounds buffer[thread_id+1] = total_diff
        sync_threads()

        # Account for thread block sizes that are not a power of 2
        if thread_id >= next_lowest_power_of_two
            @inbounds buffer[thread_id - next_lowest_power_of_two + 1] += buffer[thread_id + 1]
        end
        sync_threads()
        
        # Perform the tree reduction
        th = div(next_lowest_power_of_two,2)
        while th > 0
            if thread_id < th
                @inbounds buffer[thread_id + 1] += buffer[thread_id + th + 1]
            end
            th = div(th,2)
            sync_threads()
        end

        # Figure out if we have converged
        if thread_id == 0
            @inbounds mean = buffer[1] / (m * n)
            if mean < cutoff
                # We have converged, so set the appropriate flag
                @inbounds cell_converged[1] = 1
            end
        end

        # We need to synchronize to ensure that all threads
        #  read the correct value of the convergence flag
        sync_threads()

        # Keep track of the number of iterations we have performed
        iterations += 1
    end

    # Save the final IMGVF matrix to global memory
    for thread_block in 0:max-1
        offset = thread_block * threads_per_block
        i = div(thread_id + offset, n)
        j = mod1(thread_id + offset, n)
        if (i < m)
            @inbounds IMGVF_global[i * n + j] = IMGVF[(i * n) + j]
        end
    end
    return nothing
end


function IMGVF_cuda(dev, I, vx, vy, e, max_iterations, cutoff)

    # Copy input matrices to device
    I_dev::Array{CuArray{Float32,1},1} = Array{CuArray{Float32,1}}(size(I,1))
    IMGVF_dev::Array{CuArray{Float32,1},1} = Array{CuArray{Float32,1}}(size(I,1))
    for i in eachindex(I)
        # Transpose to go from column major to row major (or rewrite the kernel indexing)
        I_Float32 = convert(Array{Float32,1},reshape(I[i]',size(I[i],1)*size(I[i],2)))
        I_dev[i] = CuArray(I_Float32)
        IMGVF_dev[i] = copy(I_dev[i])
        # I_dev[i] should be CuIn(), but I get "identifier not found"?
        @cuda dev (1,threads_per_block) IMGVF_kernel(I_dev[i], IMGVF_dev[i], size(I[i],1), size(I[i],2), vx, vy, e, max_iterations, cutoff)
    end

    synchronize(default_stream())

    # Copy results back to host
    IMGVF = similar(I)
    for i in eachindex(IMGVF)
        # Reverse transposition
        IMGVF[i] = reshape(Array(IMGVF_dev[i]),size(I[i],2),size(I[i],1))'
    end
    IMGVF
end
