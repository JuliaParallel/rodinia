include("track_ellipse_kernel.jl")

const OUTPUT = haskey(ENV, "OUTPUT")

function ellipsetrack(video, xc0, yc0, Nc, R, Np, Nf)
    #=
    % ELLIPSETRACK tracks cells in the movie specified by 'video', at
    %  locations 'xc0'/'yc0' with radii R using an ellipse with Np discrete
    %  points, starting at frame number one and stopping at frame number 'Nf'.
    %
    % INPUTS:
    %   video.......pointer to avi video object
    %   xc0,yc0.....initial center location (Nc entries)
    %   Nc..........number of cells
    %   R...........initial radius
    %   Np..........nbr of snaxels points per snake
    %   Nf..........nbr of frames in which to track
    %
    % Matlab code written by: DREW GILLIAM (based on code by GANG DONG /
    %                                                        NILANJAN RAY)
    % Ported to C by: MICHAEL BOYER
    =#

    # Compute angle paramter
    const increment = (2.0 * C_PI) / Np

    t = [increment * i for i in 0:Np-1]

    # Allocate space for a snake for each cell in each frame
    xc = zeros(Float64,Nc,Nf+1)
    yc = zeros(Float64,Nc,Nf+1)
    r = zeros(Float64,Nc,Np,Nf+1)
    x = zeros(Float64,Nc,Np,Nf+1)
    y = zeros(Float64,Nc,Np,Nf+1)

    # Save the first snake for each cell
    xc[:,1] = xc0[1:Nc]
    yc[:,1] = yc0[1:Nc]
    r[:,:,1] = R

    # Generate ellipse points for each cell
    for i in 1:Nc, j in 1:Np
        x[i,j,1] = xc[i,1] + r[i,j,1] .* cos(t[j])
        y[i,j,1] = yc[i,1] + r[i,j,1] .* sin(t[j])
    end

    # Keep track of the total time spent on computing
    #  the MGVF matrix and evolving the snakes
    MGVF_time = 0
    snake_time = 0

    xci = Array{Float64}(Nc)
    yci = Array{Float64}(Nc)
    ri = Array{Array{Float64,1}}(Nc)
    ycavg = Array{Float64}(Nc)
    u1 = Array{Int32}(Nc)
    u2 = Array{Int32}(Nc)
    v1 = Array{Int32}(Nc)
    v2 = Array{Int32}(Nc)
    Isub = Array{Array{Float64,2}}(Nc)
    IE = Array{Array{Float64,2}}(Nc)

    # Process each frame sequentially
    for frame_num in 1:Nf
        print("\rProcessing frame ",frame_num," / ", Nf)

        # Get the current video frame and its dimensions
        I = get_frame(video, frame_num, false, true)
        Ih = size(I,1)
        Iw = size(I,2)

        # Initialize the current positions equal to be equal to the previous
        # positions
        xc[:,frame_num+1] = xc[:,frame_num]
        yc[:,frame_num+1] = yc[:,frame_num]
        r[:,:,frame_num+1] = r[:,:,frame_num]

        # Sequentially extract the subimage near each cell
        # Make copies of the current cell's location
        xci[:] = xc[:,frame_num+1]
        yci[:] = yc[:,frame_num+1]
        for cell_num in eachindex(ri)
            ri[cell_num] = reshape(r[cell_num,:,frame_num+1],Np)
        end

        # Add up the last ten y-values for this cell
        #  (or fewer if there are not yet ten previous frames)
        for cell_num in eachindex(ycavg)
            ycavg[cell_num] = sum(yc[cell_num,(frame_num>10?frame_num-10:0)+1:frame_num])
        end

        # Compute the average of the last ten y-values
        #  (this represents the expected y-location of the cell)
        ycavg = ycavg / (frame_num>10?10:frame_num)

        # Determine the range of the subimage surrounding the current
        # position
        u1[:] = trunc.(Int32,max.(xci[:] - 4.0 * R + 0.5, 0))
        u2[:] = trunc.(Int32,min.(xci[:] + 4.0 * R + 0.5, Iw - 1))
        v1[:] = trunc.(Int32,max.(yci[:] - 2.0 * R + 1.5, 0))
        v2[:] = trunc.(Int32,min.(yci[:] + 2.0 * R + 1.5, Ih - 1))

        # Extract the subimage
        for cell_num in eachindex(Isub)
            Isub[cell_num] = I[v1[cell_num]+1:v2[cell_num]+1,u1[cell_num]+1:u2[cell_num]+1]
            # Compute the subimage gradient magnitude
            IE[cell_num] = sqrt.(gradient_x(Isub[cell_num]).^2 + gradient_y(Isub[cell_num]).^2)
        end

        # Compute the motion gradient vector flow (MGVF) edgemaps for all cells
        # concurrently
        MGVF_start_time = time()
        IMGVF = MGVF(IE, 1, 1, Nc)
        MGVF_time += time() - MGVF_start_time

        # Sequentially determine the position of the cell in the subimage
        for cell_num in 1:Nc
            xci[cell_num] = xci[cell_num] - u1[cell_num]
            yci[cell_num] = yci[cell_num] - (v1[cell_num] - 1)
            ycavg[cell_num] = ycavg[cell_num] - (v1[cell_num] - 1)

            # Evolve the snake
            snake_start_time = time()
            xci[cell_num], yci[cell_num] = ellipseevolve(IMGVF[cell_num], xci[cell_num], yci[cell_num], ri[cell_num], t, Np, convert(Float64,R), ycavg[cell_num])
            snake_time += time() - snake_start_time

            # Compute the cell's new position in the full image
            xci[cell_num] = xci[cell_num] + u1[cell_num]
            yci[cell_num] = yci[cell_num] + (v1[cell_num] - 1)

            # Store the new location of the cell and the snake
            xc[cell_num,frame_num+1] = xci[cell_num]
            yc[cell_num,frame_num+1] = yci[cell_num]
            r[cell_num,:,frame_num+1] = ri[cell_num][:]
            x[cell_num,:,frame_num+1] = xc[cell_num,frame_num+1] + ri[cell_num] .* cos.(t)
            y[cell_num,:,frame_num+1] = yc[cell_num,frame_num+1] + ri[cell_num] .* sin.(t)
        end
        if OUTPUT & (frame_num == Nf)
            pFile = open("result.txt","w+")
            for cell_num in 1:Nc
                println(pFile)
                print(pFile,@sprintf("%d,%f,%f",cell_num-1,xc[cell_num,Nf+1],
                                yc[cell_num,Nf+1]))
            end
            close(pFile)
        end
    end

    println()
    println()
    println("\n\nTracking runtime (average per frame):")
    println("-------------------------------------")
    print(@sprintf("MGVF computation: %.5f seconds\n",MGVF_time / Nf))
    print(@sprintf(" Snake evolution: %.5f seconds\n",snake_time / Nf))
end

function MGVF(IE, vx, vy, Nc)
    #=
    % MGVF calculate the motion gradient vector flow (MGVF)
    %  for the image 'I'
    %
    % Based on the algorithm in:
    %  Motion gradient vector flow: an external force for tracking rolling
    %   leukocytes with shape and size constrained active contours
    %  Ray, N. and Acton, S.T.
    %  IEEE Transactions on Medical Imaging
    %  Volume: 23, Issue: 12, December 2004
    %  Pages: 1466 - 1478
    %
    % INPUTS
    %   I...........image
    %   vx,vy.......velocity vector
    %
    % OUTPUT
    %   IMGVF.......MGVF vector field as image
    %
    % Matlab code written by: DREW GILLIAM (based on work by GANG DONG /
    %                                                        NILANJAN RAY)
    % Ported to C by: MICHAEL BOYER
    =#

    # Constants
    const converge = 0.00001
    const epsilon = 0.0000000001
    const iterations = Int32(500)   # Maximum number of iterations to compute the MGVF matrix

    for cell_num in eachindex(IE)
        m = size(IE[cell_num],1)
        n = size(IE[cell_num],2)

        # Find the maximum and minimum values in IE[cell_num]
        Imax = maximum(IE[cell_num])
        Imin = minimum(IE[cell_num])

        # Normalize the image IE[cell_num]
        scale = 1.0 / (Imax - Imin + eps(Float64))
        IE[cell_num] = (IE[cell_num] .- Imin) * scale
    end

    # Offload the MGVF computation to the GPU
    IMGVF_cuda(IE, vx, vy, epsilon, iterations, converge)
end


function ellipseevolve(f, xc0, yc0, r0, t, Np, Er, Ey)
    #=
    % ELLIPSEEVOLVE evolves a parametric snake according
    %  to some energy constraints.
    %
    % INPUTS:
    %   f............potential surface
    %   xc0,yc0......initial center position
    %   r0,t.........initial radii & angle vectors (with Np elements each)
    %   Np...........number of snaxel points per snake
    %   Er...........expected radius
    %   Ey...........expected y position
    %
    % OUTPUTS
    %   xc0,yc0.......final center position
    %   r0...........final radii
    %
    % Matlab code written by: DREW GILLIAM (based on work by GANG DONG /
    %                                                        NILANJAN RAY)
    % Ported to C by: MICHAEL BOYER
    =#

    # Constants
    const deltax = 0.2
    const deltay = 0.2
    const deltar = 0.2
    const converge = 0.1
    const lambdaedge = 1
    const lambdasize = 0.2
    const lambdapath = 0.05
    const iterations = 1000 # maximum number of iterationsa

    # Initialize variables
    xc = xc0
    yc = yc0
    r = copy(r0)

    # Compute the x- and y-gradients of the MGVF matrix
    fx = gradient_x(f)
    fy = gradient_y(f)

    const fh = size(f,1)
    const fw = size(f,2)

    # Normalize the gradients
    fmag = sqrt.(fx.^2 + fy.^2)
    fx = fx ./ fmag
    fy = fy ./ fmag

    # Evolve the snake
    iter::Int32 = 0
    snakediff = 1.0

    while (iter < iterations) & (snakediff > converge)
        # Save the values from the previous iteration
        xc_old = xc
        yc_old = yc
        r_old = copy(r)

        # Compute the locations of the snaxels
        x = xc .+ r .* cos.(t)
        y = yc .+ r .* sin.(t)

        # See if any of the points in the snake are off the edge of the image
        if (minimum(x) < 0.0) | (maximum(x) > (fw - 1.0)) |
           (minimum(y) < 0.0) | (maximum(y) > (fh - 1.0))
            break
        end

        # Compute the length of the snake
        L = sum(sqrt.((x[2:end] - x[1:end-1]).^2 +
                     (y[2:end] - y[1:end-1]).^2))
        L += sqrt.((x[1] - x[end])^2 + (y[1] - y[end])^2)

        # Compute the potential surface at each snaxel
        vf = linear_interp2(f,x,y)
        vfx = linear_interp2(fx,x,y)
        vfy = linear_interp2(fy,x,y)

        # Compute the average potential surface around the snake
        vfmean = sum(vf) / L
        vfxmean = sum(vfx) / L
        vfymean = sum(vfy) / L

        # Compute the radial potential surface
        m = size(vf,1)
        n = size(vf,2)
        vfr = (vf[1,:] + vfx[1,:] .* (x - xc) +
               vfy[1,:] .* (y - yc) - vfmean) / L

        # Update the snake center and snaxels
        xc = xc + deltax * lambdaedge * vfxmean
        yc = (yc + deltay * lambdaedge * vfymean + deltay * lambdapath * Ey) / (1.0 + deltay * lambdapath)
        r = (r + (deltar * lambdaedge * vfr + deltar * lambdasize * Er)) / (1.0 + deltar * lambdasize)
        r_diff = sum(abs.(r-r_old))

        # Test for convergence
        snakediff = abs(xc - xc_old) + abs(yc - yc_old) + r_diff

        iter+=1
    end

    r0[:] = r[:]
    xc, yc
end

