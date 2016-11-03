using CUDAdrv, CUDAnative

type CUDAConstValues
        c_sin_angle::CuArray{Float32,1}
        c_cos_angle::CuArray{Float32,1}
        c_tX::CuArray{Int32,2}
        c_tY::CuArray{Int32,2}
        c_strel::CuArray{Float32,2}
end

# Transfers pre-computed constants used by the two kernels to the GPU
function transfer_constants(host_sin_angle, host_cos_angle, host_tX, host_tY,
        host_strel)
  # TODO: should all be put in constant memory
  c_sin_angle = CuArray(convert(Array{Float32,1},host_sin_angle))
  c_cos_angle = CuArray(convert(Array{Float32,1},host_cos_angle))
  c_tX = CuArray(convert(Array{Int32,2},host_tX))
  c_tY = CuArray(convert(Array{Int32,2},host_tY))
  c_strel = CuArray(convert(Array{Float32,2},host_strel))
  CUDAConstValues(c_sin_angle,c_cos_angle,c_tX,c_tY,c_strel)
end


# Kernel to compute the dilation of the GICOV matrix produced by the GICOV
# kernel
# Each element (i, j) of the output matrix is set equal to the maximal value in
#  the neighborhood surrounding element (i, j) in the input matrix
# Here the neighborhood is defined by the structuring element (c_strel)
function dilate_kernel(img_dev, c_strel, dilated_out)
    # Find the center of the structuring element
    el_center_i = div(size(c_strel,1),2)
    el_center_j = div(size(c_strel,2),2)

    img_m = size(img_dev,1)
    img_n = size(img_dev,2)

    # Determine this thread's location in the matrix
    thread_id = ((blockIdx().x -1) * blockDim().x) + threadIdx().x - 1
    i = mod(thread_id, img_m)
    j = div(thread_id, img_m)

    # Initialize the maximum GICOV score seen so far to zero
    max::Float32 = 0.0

    # Iterate across the structuring element
    for el_i in 0:size(c_strel,1)-1, el_j in 0:size(c_strel,2)-1
        y = i - el_center_i + el_i
        x = j - el_center_j + el_j
        # Make sure we have not gone off the edge of the matrix
        @inbounds if (0 <= y < size(img_dev,1)) & (0 <= x < size(img_dev,2)) & (c_strel[el_i+1,el_j+1] != 0.0)
            @inbounds temp = img_dev[y+1,x+1]
            if temp > max
                max = temp
            end
        end
    end
    # Store the maximum value found
    @inbounds dilated_out[i+1,j+1] = max

    return nothing    
end


# Sets up and invokes the dilation kernel and returns its output
function dilate_CUDA(dev, img_in, GICOV_constants)
    # TODO: should be put in texture memory
    img_dev = CuArray(img_in)
    dilated_out = CuArray(Float32,(size(img_in,1),size(img_in,2)))

    num_threads = size(img_in,1) * size(img_in,2)
    threads_per_block = 176
    num_blocks = trunc(Int64,num_threads / threads_per_block + 0.5)

    @cuda dev (num_blocks,threads_per_block) dilate_kernel(img_dev,GICOV_constants.c_strel,dilated_out)
    synchronize(default_stream())

    Array(dilated_out)
end

