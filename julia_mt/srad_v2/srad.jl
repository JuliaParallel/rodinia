#!/usr/bin/env julia

const OUTPUT = haskey(ENV, "OUTPUT")

function usage()
    prog = basename(Base.source_path())
#    println(STDERR,"Usage ",prog," <rows> <cols> <y1> <y2> <x1> <x2> <no. of threads><lamda> <no. of iter>")
    println(STDERR,"Usage ",prog," <rows> <cols> <y1> <y2> <x1> <x2> <lamda> <no. of iter>")
    println(STDERR,"\t<rows>   - number of rows")
    println(STDERR, "\t<cols>    - number of cols\n")
    println(STDERR, "\t<y1> 	 - y1 value of the speckle\n")
    println(STDERR, "\t<y2>      - y2 value of the speckle\n")
    println(STDERR, "\t<x1>       - x1 value of the speckle\n")
    println(STDERR, "\t<x2>       - x2 value of the speckle\n")
#    println(STDERR, "\t<no. of threads>  - no. of threads\n")
    println(STDERR, "\t<lamda>   - lambda (0,1)\n")
    println(STDERR, "\t<no. of iter>   - number of iterations\n")
    exit(1)
end


function main(args)
    if length(args) != 8
      usage()
    end
    rows = parse(Int64,args[1])
    cols = parse(Int64,args[2])
    if ((rows % 16) != 0) | ((cols % 16) != 0)
        println(STDERR,"rows and cols must be mutiples of 16")
        exit(1)
    end
    r1 = parse(Int32,args[3]) + 1
    r2 = parse(Int32,args[4]) + 1
    c1 = parse(Int32,args[5]) + 1
    c2 = parse(Int32,args[6]) + 1
    lambda = parse(Float32,args[7])
    niter = parse(Int32,args[8])

    size_R = (r2 - r1 + 1) * (c2 - c1 + 1);
    
    iN::Array{Int32,1} = [i-1 for i in 1:rows] 
    iS::Array{Int32,1} = [i+1 for i in 1:rows]
    jW::Array{Int32,1} = [j-1 for j in 1:cols] 
    jE::Array{Int32,1} = [j+1 for j in 1:cols]
    iN[1] = 1
    iS[rows] = rows
    jW[1] = 1
    jE[cols] = cols

    dN::SharedArray{Float32,2} = SharedArray(Float32,(rows,cols))
    dS::SharedArray{Float32,2} = SharedArray(Float32,(rows,cols))
    dW::SharedArray{Float32,2} = SharedArray(Float32,(rows,cols))
    dE::SharedArray{Float32,2} = SharedArray(Float32,(rows,cols))

    println("Randomizing the input matrix")

    srand(7)
    I = rand(Float32,rows,cols)
    J::SharedArray{Float32,2} = SharedArray(Float32,(rows,cols),init = J -> J[Base.localindexes(J)] = exp.(I[Base.localindexes(J)]))
    c::SharedArray{Float32,2} = SharedArray(Float32,(rows,cols))

    println("Start the SRAD main loop")

    tic()
 
    sum::Float32 = 0
    sum2::Float32 = 0
    for iter in 1:niter
        sum = 0
        sum2 = 0
        for i in r1:r2
            for j in c1:c2
                tmp = J[i,j]
                sum += tmp
                sum2 += tmp * tmp
            end
        end
    end

    meanROI::Float32 = sum / size_R
    varROI::Float32 = (sum / size_R) - meanROI * meanROI
    q0sqr::Float32 = varROI / (meanROI * meanROI)

    @sync @parallel for i in 1:size(J,1)
        for j in size(J,2)
            Jc = J[i,j]

            # directional derivates
            dN[i,j] = J[iN[i],j] - Jc
            dS[i,j] = J[iS[i],j] - Jc
            dW[i,j] = J[i,jW[j]] - Jc
            dE[i,j] = J[i,jE[j]] - Jc

            G2 = (dN[i,j] * dN[i,j] + dS[i,j] * dS[i,j] + dW[i,j] * dW[i,j] +
                  dE[i,j] * dE[i,j]) /
                 (Jc * Jc)

            L = (dN[i,j] + dS[i,j] + dW[i,j] + dE[i,j]) / Jc

            num = (0.5 * G2) - ((1.0 / 16.0) * (L * L))
            den = 1 + (0.25 * L)
            qsqr = num / (den * den)

            # diffusion coefficent (equ 33)
            den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr))
            c[i,j] = 1.0 / (1.0 + den)

            # saturate diffusion coefficent
            if (c[i,j] < 0)
                c[i,j] = 0
            elseif (c[i,j] > 1)
                c[i,j] = 1
            end
        end
    end

    @sync @parallel for i in 1:size(J,1)
        for j in 1:size(J,2)
            # diffusion coefficient
            cN = c[i,j]
            cS = c[iS[i],j]
            cW = c[i,j]
            cE = c[i,jE[j]]

            # divergence (equ 58)
            D = cN * dN[i,j] + cS * dS[i,j] + cW * dW[i,j] + cE * dE[i,j]

            # image update (equ 61)
            J[i,j] += 0.25 * lambda * D
        end
    end

    toc()

    println("Computation done")

    if OUTPUT
        f = open("output.txt","w")
        for i in 1:size(J,1)
            for j in 1:size(J,2)
               print(f,@sprintf("%.5f ",J[i,j]))
            end
            println(f)
        end
        close(f)
    end
end

