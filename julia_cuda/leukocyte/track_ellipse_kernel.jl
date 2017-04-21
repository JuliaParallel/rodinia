using CUDAdrv, CUDAnative

# The number of threads per thread block
const threads_per_block = 320
# next_lowest_power_of_two = 2^(floor(log2(threads_per_block)))
const next_lowest_power_of_two = 256

# Regularized version of the Heaviside step function:
# He(x) = (atan(x) / pi) + 0.5
function heaviside(z)
    CUDAnative.atan(z) * (Float32(1) / Float32(C_PI)) + Float32(0.5)
end

function IMGVF_kernel(I_flat, IMGVF_flat, m_array, n_array, offsets, vx, vy, e,
                      max_iterations, cutoff)

    # Constants
    const mu = 0.5
    const lambda = 8.0 * mu + 1.0
    # 41 * 81, @cuStaticSharedMem can't deal with expressions -- even in constants
    const IMGVF_SIZE = 3321

    cell_num = blockIdx().x
    m = m_array[cell_num]
    n = n_array[cell_num]
    cell_offset = offsets[cell_num]

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
    thread_id::Int32 = threadIdx().x - 1

    for thread_block in 0:max-1
        offset = thread_block * threads_per_block
        i = div(thread_id + offset, n)
        j = mod(thread_id + offset, n)
        if i < m
            IMGVF[i * n + j + 1] = IMGVF_flat[cell_offset + i * n + j + 1]
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
    const one_over_e = Float32(1) / Float32(e)

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
        for thread_block in 0:max-1
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
                @inbounds vI = I_flat[cell_offset + i * n + j + 1]
                new_val -= ((1.0 / lambda) * vI * (new_val - vI));
            end

            sync_threads()

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
        j = mod(thread_id + offset, n)
        if (i < m)
            IMGVF_flat[cell_offset + i * n + j + 1] = IMGVF[(i * n) + j + 1]
        end
    end
    return nothing
end


function IMGVF_cuda(I, vx, vy, e, max_iterations, cutoff)

    # Copy input matrices to device
    num_cells = size(I,1)
    m_array = Array{Int32}(num_cells)
    n_array = Array{Int32}(num_cells)
    offsets = Array{Int32}(num_cells)

    total_size = 0

    for c = 1:num_cells

        m = size(I[c],1)
        n = size(I[c],2)

        m_array[c] = m
        n_array[c] = n
        offsets[c] = total_size
        total_size += m * n
    end

    I_flat = Array{Float32}(total_size)

    for c = 1:num_cells

        m = m_array[c]
        n = n_array[c]
        offset = offsets[c]
        I_c = I[c]

        for i = 1:m, j = 1:n
            I_flat[offset + (i - 1) * n + j] = I_c[i,j]
        end
    end

    dev_I_flat = CuArray(I_flat)
    dev_IMGVF_flat = CuArray(I_flat)
    dev_m_array = CuArray(m_array)
    dev_n_array = CuArray(n_array)
    dev_offsets = CuArray(offsets)

    @cuda (num_cells, threads_per_block) IMGVF_kernel(dev_I_flat,
        dev_IMGVF_flat, dev_m_array, dev_n_array, dev_offsets, Float32(vx),
        Float32(vy), Float32(e), max_iterations, Float32(cutoff))

    # Copy results back to host
    IMGVF = Array{Array{Float32,2}}(num_cells)
    IMGVF_flat = Array(dev_IMGVF_flat)

    for c = 1:num_cells

        m = m_array[c]
        n = n_array[c]
        offset = offsets[c]
        IMGVF_c = Array{Float32}(m, n)
        IMGVF[c] = IMGVF_c

        for i = 1:m, j = 1:n
            IMGVF_c[i,j] = IMGVF_flat[offset + (i - 1) * n + j]
        end
    end
    IMGVF
end
