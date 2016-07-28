include("../../common/julia/libavi.jl")
include("misc_math.jl")
include("find_ellipse.jl")
include("track_ellipse.jl")

function main(args)
    # Keep track of the start time of the program
    program_start_time = time()

    # Let the user specify the number of frames to process
    num_frames = 1

    if length(args) != 2
        println(STDERR,"usage: ",basename(Base.source_path())," <num of frames> <input file>")
        exit(1)
    end

    num_frames = parse(Int32,args[1])

    video_file_name = args[2]

    cell_file = AVI_open_input_file(video_file_name, 1)

    if cell_file == C_NULL
        AVI_print_error("Error with AVI_open_input_file")
        exit(1)
    end

    Iter = 20
    ns = 4
    k_count = 0
    
    const threshold = 1.8
    const radius = 10.0
    const delta = 3.0
    const dt = 0.01
    const b = 5.0

    # Extract a cropped version of the first frame from the video file
    image_chopped = get_frame(cell_file, 0, true, false)
    println("Detecting cells in frame 0")

    # Get gradient matrices in x and y directions
    grad_x = gradient_x(image_chopped)
    grad_y = gradient_y(image_chopped)

    # Get GICOV matrix corresponding to image gradients
    tic()
    gicov = ellipsematching(grad_x, grad_y)

    # Square GICOV values
    max_gicov = gicov.^2
    GICOV_end_time = toq()

    # Dilate the GICOV matrix
    tic()
    strel = structuring_element(12)
    img_dilated = dilate_f(max_gicov, strel)
    dilate_end_time = toq()

    # Find possible matches for cell centers based on GICOV and record the
    # rows/columns in which they are found
    crow,ccol = findn((max_gicov .!= 0.0) & (img_dilated .== max_gicov))
    # convert to zero-based indices for parity with the C code (don't
    # use map in the above expression, because it's very slow)
    crow,ccol = crow.-1,ccol.-1
    # sort based on increasing y-coordinate to have the same order as the C benchmark
    sortedy = sortperm(crow)
    crow = crow[sortedy]
    ccol = ccol[sortedy]

    GICOV_spots = [gicov[crow[i]+1,ccol[i]+1] for i in 1:size(crow,1)]

    result_indices = find(crow .> 26 & crow .< BOTTOM - TOP + 39)
    x_result = ccol[result_indices]
    y_result = crow[result_indices] .- 40
    G = GICOV_spots[result_indices]

    #  Make an array t which holds each "time step" for the possible cells
    t = [i * 2.0 * C_PI / 36.0 for i in 0:35]

    # Store cell boundaries (as simple circles) for all cells
    cellx = [x_result[i] + radius * cos(t[j]) for i in 1:size(x_result,1), j in 1:36]
    celly = [y_result[i] + radius * sin(t[j]) for i in 1:size(x_result,1), j in 1:36]

    A = TMatrix(9,4)
    cell_width = AVI_video_width(cell_file)
    cell_height = AVI_video_height(cell_file)

    V = zeros(Float64,size(x_result,1))
    QAX_CENTERS = zeros(Float64,size(x_result,1))
    QAY_CENTERS = zeros(Float64,size(x_result,1))

    # For all possible results, find the ones that are feasibly leukocytes and
    # store their centers
    k_count = 0

    for n in 0:size(x_result,1)-1
        if (G[n+1] < -1 * threshold) | (G[n+1] > threshold)

            x = Array{Float64,2}(1,36)
            y = Array{Float64,2}(1,36)

            # Get current values of possible cells from cellx/celly matrices
            uniformseg(slice(cellx,n+1,:), slice(celly,n+1,:), x, y)

            # Make sure that the possible leukocytes are not too close to the
            # edge of the frame
            if (minimum(x) > b) & (minimum(y) > b) &
               (maximum(x) < cell_width - b) & (maximum(y) < cell_height - b)
                Cx = A * x'
                Cy = A * y'
                Cy = Cy .+ 40.0

                # Iteratively refine the snake/splin
                for i in 0:Iter-1
                    if G[n+1] > 0.0
                        typeofcell = 0
                    else
                        typeofcell = 1
                    end

                    splineenergyform01(Cx, Cy, grad_x, grad_y, ns, delta, 2.0 * dt, typeofcell)
                end

                X = getsampling(Cx, ns)
                Cy_temp = similar(Cy)
                Cy_temp[:,1] = Cy[:,1] .- 40.0
                Y = getsampling(Cy_temp, ns)

                Ix1 = linear_interp2(grad_x, X, Y)
                Iy1 = linear_interp2(grad_x, X, Y)
                Xs = getfdriv(Cx, ns)
                Ys = getfdriv(Cy, ns)

                Nx = Ys ./ sqrt(Xs.^2 + Ys.^2)
                Ny = -1.0 .* Ys ./ sqrt(Xs.^2 + Ys.^2)
                W = slice(Ix1,1) .* Nx + slice(Iy1,1) .* Ny

                V[n+1] = mean(W) / std_dev(W)

                # get means of X and Y values for all "snaxels" of the spline
                # contour, thus finding the cell centers
                QAX_CENTERS[k_count+1] = mean(X)
                QAY_CENTERS[k_count+1] = mean(Y) + TOP
                k_count += 1
            end
        end
    end
    # Report the total number of cells detected
    println("Cells detected: ",k_count)
    println()

    # Report the breakdown of the detection runtime
    println("Detection runtime");
    println("-----------------");
    print(@sprintf("GICOV computation: %.5f seconds\n",GICOV_end_time))
    print(@sprintf("   GICOV dilation: %.5f seconds\n",dilate_end_time))
    print(@sprintf("            Total: %.5f seconds\n",time() - program_start_time))

    # Now that the cells have been detected in the first frame,
    #  track the ellipses through subsequent frames
    println()
    if (num_frames > 1)
        println("Tracking cells across ",num_frames," frames")
    else
        println("Tracking cells across 1 frame")
    end
    tic()
    num_snaxels = 20
    ellipsetrack(cell_file, QAX_CENTERS, QAY_CENTERS, k_count, radius, num_snaxels, num_frames)
    toc()

    # Report total program execution time
    print(@sprintf("\nTotal application run time: %.5f seconds\n",time() - program_start_time))
end

main(ARGS)
