include("../../common/julia/libavi.jl")

# Defines the region in the video frame containing the blood vessel
const TOP = 110
const BOTTOM = 328

# The number of sample points per ellipse
const NPOINTS = 150
# The expected radius (in pixels) of a cell
const RADIUS = 10
# The range of acceptable radiuses
const MIN_RAD = RADIUS - 2
const MAX_RAD = RADIUS * 2
# The number of different sample ellipses to try
const NCIRCLES = 7

# Returns the specified frame from the specified video file
# If cropped == true, the frame is cropped to pre-determined dimensions
#  (hardcoded to the boundaries of the blood vessel in the test video)
# If scaled == true, all values are scaled to the range [0.0, 1.0]
function get_frame(cell_file, frame_num, cropped::Bool, scaled::Bool)
    width = AVI_video_width(cell_file)
    height = AVI_video_height(cell_file)

    AVI_set_video_position(cell_file, frame_num)

    image_buf = Array{UInt8,2}(width,height)
    dummy = zeros(Int32,1)
    frame = AVI_read_frame(cell_file, image_buf, dummy)
    if frame == -1
        AVI_print_error("Error with AVI_read_frame")
        exit(1)
    end

    # convert from row-major to column-major
    image_buf = image_buf'

    if cropped
        image_chopped = chop_flip_image(image_buf, TOP, BOTTOM, 0, width - 1, scaled)
    else
        image_chopped = chop_flip_image(image_buf, 0, height - 1, 0, width - 1, scaled)
    end
    image_chopped
end


# Flips the specified image and crops it to the specified dimensions
function chop_flip_image(image, top, bottom, left, right, scaled::Bool)
    scale = scaled ? 1.0/255.0 : 1.0
    image[size(image,1)-top:-1:size(image,1)-bottom,left+1:right+1] * scale
end


# Given x- and y-gradients of a video frame, computes the GICOV
#  score for each sample ellipse at every pixel in the frame
function ellipsematching(grad_x, grad_y)
    # Compute the sine and cosine of the angle to each point in each sample
    # circle
    #  (which are the same across all sample circles)
    theta = [n * 2.0 * C_PI/NPOINTS for n in 0:NPOINTS-1]
    cos_angle = cos(theta)
    sin_angle = sin(theta)

    # Compute the (x,y) pixel offsets of each sample point in each sample
    # circle
    tX = [trunc(Int32,(MIN_RAD+2*k)*cos_angle[n+1]) for k in 0:NCIRCLES-1,n in 0:NPOINTS-1]
    tY = [trunc(Int32,(MIN_RAD+2*k)*sin_angle[n+1]) for k in 0:NCIRCLES-1,n in 0:NPOINTS-1]

    const MaxR = MAX_RAD + 2

    # Allocate the result matrix
    height = size(grad_x,1)
    width = size(grad_x,2)
    gicov = zeros(Float64,height,width)

    # Scan from left to right, top to bottom, computing GICOV values
    for i in MaxR:width-MaxR-1
        Grad = Array{Float64}(NPOINTS)
        for j in MaxR:height-MaxR-1
            # Initialize the maximal GICOV score to 0
            max_GICOV = 0.0

            # Iterate across each stencil
            for k in 0:NCIRCLES-1
                # Iterate across each sample point in the current stencil
                for n in 0:NPOINTS-1
                    # Determine the x- and y-coordinates of the current sample
                    # point
                    y = j + tY[k+1,n+1]
                    x = i + tX[k+1,n+1]

                    # Compute the combined gradient value at the current sample
                    # point
                    Grad[n+1] = grad_x[y+1,x+1] * cos_angle[n+1] +
                                grad_y[y+1,x+1] * sin_angle[n+1]
                end

                # Compute the mean gradient value across all sample points
                mean = sum(Grad)/NPOINTS

                # Compute the variance of the gradient values
                var = 0.0
                for n in 1:NPOINTS
                    sum = Grad[n] - mean
                    var += sum^2
                end
                var = var / (NPOINTS - 1)

                # Keep track of the maximal GICOV value seen so far
                if mean^2 / var > max_GICOV
                    gicov[j+1,i+1] = mean / sqrt(var)
                    max_GICOV = mean^2 / var
                end
            end
        end
    end

    gicov
end


# Returns a circular structuring element of the specified radius
function structuring_element(radius)
    [sqrt((i-radius)^2 + (j-radius)^2)<=radius?1.0:0.0 for i in 0:radius*2, j in 0:radius*2]
end


# Performs an image dilation on the specified matrix
#  using the specified structuring element
function dilate_f(img_in, strel)

    dilated = similar(img_in)

    # Find the center of the structuring element
    el_center_i = div(size(strel,1),2)
    el_center_j = div(size(strel,2),2)

    # Iterate across the input matrix
    for i in 0:size(img_in,1)-1, j in 0:size(img_in,2)-1
        max = 0.0

        # Iterate across the structuring element
        for el_i in 0:size(strel,1)-1, el_j in 0:size(strel,2)-1
            y = i - el_center_i + el_i
            x = j - el_center_j + el_j
            # Make sure we have not gone off the edge of the matrix
            if (0 <= y < size(img_in,1)) & (0 <= x < size(img_in,2)) &
               (strel[el_i+1,el_j+1] != 0.0)
                temp = img_in[y+1,x+1]
                if temp > max
                    max = temp
                end
            end
        end
        # Store the maximum value found
        dilated[i+1,j+1] = max
    end
    dilated
end


function get_abcd_indices(N)
    aindex = [i-1 for i in 1:N]
    aindex[1] = N

    bindex = [i for i in 1:N]

    cindex = [i+1 for i in 1:N]
    cindex[end] = 1

    dindex = [i+2 for i in 1:N]
    dindex[end-1] = 1
    dindex[end] = 2

    (aindex, bindex, cindex, dindex)
end


# M = # of sampling points in each segment
# N = number of segment of curve
# Get special TMatrix
function TMatrix(N, M)
    aindex, bindex, cindex, dindex = get_abcd_indices(N)

    B = Array{Float64}(N*M,N)

    for i in 1:N
        LB = zeros(Float64,M,N)

        for j in 1:M
            s = (j-1)/M

            a = (-1.0 * s * s * s + 3.0 * s * s - 3.0 * s + 1.0) / 6.0
            b = (3.0 * s * s * s - 6.0 * s * s + 4.0) / 6.0
            c = (-3.0 * s * s * s + 3.0 * s * s + 3.0 * s + 1.0) / 6.0
            d = s * s * s / 6.0

            LB[j,aindex[i]] = a
            LB[j,bindex[i]] = b
            LB[j,cindex[i]] = c
            LB[j,dindex[i]] = d
        end

        for m in (i-1)*M+1:i*M, n in 1:N
            B[m,n] =
              LB[mod1(m,M),n]
        end
    end

    B_TEMP_INV = inv(B' * B)
    B_TEMP_INV * B'
end


function uniformseg(cellx_row, celly_row, x, y)
    dx = [cellx_row[i] - cellx_row[mod1(i-1,36)] for i in 1:36]
    dy = [celly_row[i] - celly_row[mod1(i-1,36)] for i in 1:36]
    dist = sqrt(dx.^2 + dy.^2)


    perm = dist[1]
    dsum = Array{Float64}(36)
    dsum[1] = dist[1]
    for i in 2:size(dist,1)
      perm += dist[i]
      dsum[i] = dsum[i-1] + dist[i]
    end
    uperm = perm/36.0
    # the inner array does not necessarily need constructing,
    # since we only use it for indmin(). A for loop may be
    # faster
    index = [indmin([abs(dsum[j]-i*uperm) for j in 1:36]) for i in 0:35]

    for i in 1:36
        x[1,i] = cellx_row[index[i]]
        y[1,i] = celly_row[index[i]]
    end
end


function getsampling(m, ns)
    N = size(m,1)
    M = ns

    aindex, bindex, cindex, dindex = get_abcd_indices(N)

    retval = Array{Float64,1}(N*M)
    for i in 1:N, j in 1:M
        s = (j-1)/M
        # This indexing is inverted compared to the C version, but that is
        # because it uses a hack to index a (9,1) matrix as if it's a (1,9) one
        a = m[aindex[i],1] * (-1.0 * s * s * s + 3.0 * s * s - 3.0 * s + 1.0)
        b = m[bindex[i],1] * (3.0 * s * s * s - 6.0 * s * s + 4.0)
        c = m[cindex[i],1] * (-3.0 * s * s * s + 3.0 * s * s + 3.0 * s + 1.0)
        d = m[dindex[i],1] * s * s * s
        retval[(i-1)*M+j] = (a + b + c + d) / 6.0
    end
    retval
end


function getfdriv(m, ns)
    N = size(m,1)
    M = ns

    aindex, bindex, cindex, dindex = get_abcd_indices(N)

    retval = Array{Float64,1}(M*N)
    for i in 1:N, j in 1:M
        s = (j-1)/M
        # This indexing is inverted compared to the C version, but that is
        # because it uses a hack to index a (9,1) matrix as if it's a (1,9) one
        a = m[aindex[i],1] * (-3.0 * s * s + 6.0 * s - 3.0)
        b = m[bindex[i],1] * (9.0 * s * s - 12.0 * s)
        c = m[cindex[i],1] * (-9.0 * s * s + 6.0 * s + 3.0)
        d = m[dindex[i],1] * (3.0 * s * s)
        retval[(i-1)*M+j] = (a + b + c + d) / 6.0
    end
    retval
end


# Performs bilinear interpolation, getting the values of m specified in the
# vectors X and Y
function linear_interp2(m, X, Y)
    # Kind of assumes X and Y have same len!
    retval = Array{Float64}(1,length(X))

    for i in 1:length(X)
        x_coord = X[i]
        y_coord = Y[i]
        l = trunc(Int32,x_coord)
        k = trunc(Int32,y_coord)

        a = x_coord - l
        b = y_coord - k

#        print(@sprintf("xc: %f \t yc: %f \t i: %d \t l: %d \t k: %d \t a: %f \t b: %f,", x_coord, y_coord, i, l, k, a, b))
#        print(@sprintf(" \t m[k,l]: %f \t m[k,l+1]: %f \t m[l+1,l]: %f \t m[k+1,l+1]: %f\n", m[k+1,l+1], m[k+1,l+2], m[k+2,l+1], m[k+2,l+2]))
        new_val = (1.0 - a) * (1.0 - b) * m[k+1,l+1] +
            a * (1.0 - b) * m[k+1,l+2] +
            (1.0 - a) * b * m[k+2,l+1] +
            a * b * m[k+2,l+2]

        retval[1,i] = new_val
    end
    retval
end


function splineenergyform01(Cx, Cy, Ix, Iy, ns, delta, dt, typeofcell)
    X = getsampling(Cx, ns)
    Y = getsampling(Cy, ns)
    Xs = getfdriv(Cx, ns)
    Ys = getfdriv(Cy, ns)

    Nx = Ys ./ sqrt(Xs.^2 + Ys.^2)
    Ny = -1.0 * Xs ./ sqrt(Xs.^2 + Ys.^2)

    X1 = X + delta * Nx
    Y1 = Y + delta * Ny
    X2 = X - delta * Nx
    Y2 = Y + delta * Ny

    Ix1_mat = linear_interp2(Ix, X1, Y1)
    Iy1_mat = linear_interp2(Iy, X1, Y1)
    Ix2_mat = linear_interp2(Ix, X2, Y2)
    Iy2_mat = linear_interp2(Iy, X2, Y2)

    Ix1 = slice(Ix1_mat,1,:)
    Iy1 = slice(Iy1_mat,1,:)
    Ix2 = slice(Ix2_mat,1,:)
    Iy2 = slice(Iy2_mat,1,:)

    aindex, bindex, cindex, dindex = get_abcd_indices(size(Cx,1))
    XY = Xs .* Ys
    XX = Xs .* Xs
    YY = Ys .* Ys

    dCx = zeros(Float64,size(Cx,1))
    dCy = zeros(Float64,size(Cy,1))

    # get control points for splines
    for i in 1:size(Cx,1), j in 1:ns
        s = (j-1)/ns

        A1 = (-1.0 * (s - 1.0) * (s - 1.0) * (s - 1.0)) / 6.0
        A2 = (3.0 * s * s * s - 6.0 * s * s + 4.0) / 6.0
        A3 = (-3.0 * s * s * s + 3.0 * s * s + 3.0 * s + 1.0) / 6.0
        A4 = s * s * s / 6.0

        B1 = (-3.0 * s * s + 6.0 * s - 3.0) / 6.0
        B2 = (9.0 * s * s - 12.0 * s) / 6.0
        B3 = (-9.0 * s * s + 6.0 * s + 3.0) / 6.0
        B4 = 3.0 * s * s / 6.0

        k = (i-1) * ns + j

        D = sqrt(Xs[k]^2 + Ys[k]^2)
        D_3 = D^3

        # 1st control point

        Tx1 = A1 - delta * XY[k] * B1 / D_3
        Tx2 = -1.0 * delta * (B1 / D - XX[k] * B1 / D_3)
        Tx3 = A1 + delta * XY[k] * B1 / D_3
        Tx4 = delta * (B1 / D - XX[k] * B1 / D_3)

        Ty1 = delta * (B1 / D - YY[k] * B1 / D_3)
        Ty2 = A1 + delta * (XY[k] * B1 / D_3)
        Ty3 = -1.0 * delta * (B1 / D - YY[k] * B1 / D_3)
        Ty4 = A1 - delta * (XY[k] * B1 / D_3)

        dCx[aindex[i]] +=
            Tx1 * Ix1[k] +
            Tx2 * Iy1[k] - Tx3 * Ix2[k] -
            Tx4 * Iy2[k]
        dCy[aindex[i]] += Ty1 * Ix1[k] +
            Ty2 * Iy1[k] - Ty3 * Ix2[k] -
            Ty4 * Iy2[k]

        # 2nd control point

        Tx1 = A2 - delta * XY[k] * B2 / D_3
        Tx2 = -1.0 * delta * (B2 / D - XX[k] * B2 / D_3)
        Tx3 = A2 + delta * XY[k] * B2 / D_3
        Tx4 = delta * (B2 / D - XX[k] * B2 / D_3)

        Ty1 = delta * (B2 / D - YY[k] * B2 / D_3)
        Ty2 = A2 + delta * XY[k] * B2 / D_3
        Ty3 = -1.0 * delta * (B2 / D - YY[k] * B2 / D_3)
        Ty4 = A2 - delta * XY[k] * B2 / D_3

        dCx[bindex[i]] += Tx1 * Ix1[k] +
                      Tx2 * Iy1[k] - Tx3 * Ix2[k] -
                      Tx4 * Iy2[k]
        dCy[bindex[i]] += Ty1 * Ix1[k] +
                      Ty2 * Iy1[k] - Ty3 * Ix2[k] -
                      Ty4 * Iy2[k]

        # 3nd control point

        Tx1 = A3 - delta * XY[k] * B3 / D_3
        Tx2 = -1.0 * delta * (B3 / D - XX[k] * B3 / D_3)
        Tx3 = A3 + delta * XY[k] * B3 / D_3
        Tx4 = delta * (B3 / D - XX[k] * B3 / D_3)

        Ty1 = delta * (B3 / D - YY[k] * B3 / D_3)
        Ty2 = A3 + delta * XY[k] * B3 / D_3
        Ty3 = -1.0 * delta * (B3 / D - YY[k] * B3 / D_3)
        Ty4 = A3 - delta * XY[k] * B3 / D_3

        dCx[cindex[i]] += Tx1 * Ix1[k] +
                      Tx2 * Iy1[k] - Tx3 * Ix2[k] -
                      Tx4 * Iy2[k]
        dCy[cindex[i]] += Ty1 * Ix1[k] +
                      Ty2 * Iy1[k] - Ty3 * Ix2[k] -
                      Ty4 * Iy2[k]

        # 4nd control point

        Tx1 = A4 - delta * XY[k] * B4 / D_3
        Tx2 = -1.0 * delta * (B4 / D - XX[k] * B4 / D_3)
        Tx3 = A4 + delta * XY[k] * B4 / D_3
        Tx4 = delta * (B4 / D - XX[k] * B4 / D_3)

        Ty1 = delta * (B4 / D - YY[k] * B4 / D_3)
        Ty2 = A4 + delta * XY[k] * B4 / D_3
        Ty3 = -1.0 * delta * (B4 / D - YY[k] * B4 / D_3)
        Ty4 = A4 - delta * XY[k] * B4 / D_3

        dCx[dindex[i]] += Tx1 * Ix1[k] +
                      Tx2 * Iy1[k] - Tx3 * Ix2[k] -
                      Tx4 * Iy2[k]
        dCy[dindex[i]] += Ty1 * Ix1[k] +
                      Ty2 * Iy1[k] - Ty3 * Ix2[k] -
                      Ty4 * Iy2[k]
    end

    if typeofcell == 1
        Cx[1,:] = Cx[2,:] - dt * dCx[1:size(Cx,2)]
        Cy[1,:] = Cy[2,:] - dt * dCy[1:size(Cy,2)]
    else
        Cx[1,:] = Cx[2,:] + dt * dCx[1:size(Cx,2)]
        Cy[1,:] = Cy[2,:] + dt * dCy[1:size(Cy,2)]
    end
end
