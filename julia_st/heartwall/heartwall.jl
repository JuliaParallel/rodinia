#!/usr/bin/env julia

const OUTPUT = haskey(ENV, "OUTPUT")

include("../../common/julia/libavi.jl")
include("../../common/julia/wrappers.jl")

include("define.jl")
include("kernel.jl")

function write_data(filename, frameNo, frames_processed, endoPoints,
                    input_a, input_b, epiPoints, input_2a, input_2b)

    try
        # Open file for reading
        fid = open(filename, "w+")

        # Write values to the file
        @printf(fid, "Total AVI Frames: %d\n", frameNo)
        @printf(fid, "Frames Processed: %d\n", frames_processed)
        @printf(fid, "endoPoints: %d\n", endoPoints)
        @printf(fid, "epiPoints: %d", epiPoints)

        for j = 0:frames_processed-1

            @printf(fid, "\n---Frame %d---", j)
            @printf(fid, "\n--endo--\n")

            for i = 0:endoPoints-1
                @printf(fid, "%d\t", input_a[j + i * frameNo + 1])
            end

            @printf(fid, "\n")

            for i = 0:endoPoints-1
                @printf(fid, "%d\t", input_b[j + i * frameNo + 1])
            end

            @printf(fid, "\n--epi--\n")

            for i = 0:epiPoints-1
                @printf(fid, "%d\t", input_2a[j + i * frameNo + 1])
            end

            @printf(fid, "\n")

            for i = 0:epiPoints-1
                @printf(fid, "%d\t", input_2b[j + i * frameNo + 1])
            end
        end

        close(fid)
    catch
        println("The file was not opened for writing")
    end
end

function main(args)

    public = public_struct()
    private = Array{private_struct}(ALL_POINTS)

    # Frames
    if length(args) != 2

        println("ERROR: usage: heartwall <inputfile> <num of frames>")
        exit(1)
    end

    d_frames = AVI_open_input_file(args[1], 1)

    # TODO
    # if (d_frames == NULL) {
    #     AVI_print_error((char *)"Error with AVI_open_input_file");
    #     return -1;
    # }

    public.d_frames = d_frames
    public.frames = AVI_video_frames(public.d_frames)
    public.frame_rows = AVI_video_height(public.d_frames)
    public.frame_cols = AVI_video_width(public.d_frames)
    public.frame_elem = public.frame_rows * public.frame_cols
    public.frame_mem = sizeof(Float32) * public.frame_elem

    # Check input arguments
    frames_processed = parse(Int32, args[2])

    if frames_processed < 0 || frames_processed > public.frames

        @printf("ERROR: %d is an incorrect number of frames specified, ", frames_processed)
        @printf("select in the range of 0-%d\n", public.frames)
        exit(0)
    end

    # Endo points
    public.endoPoints = ENDO_POINTS
    public.d_endo_mem = sizeof(Int32) * public.endoPoints
    public.d_endoRow = Array{Int32}([369, 400, 429, 452, 476, 486, 479, 458, 433, 404,
                                     374, 346, 318, 294, 277, 269, 275, 287, 311, 339])

    public.d_endoCol = Array{Int32}([408, 406, 397, 383, 354, 322, 294, 270, 250, 237,
                                     235, 241, 254, 273, 300, 328, 356, 383, 401, 411])

    public.d_tEndoRowLoc = Array{Int32}(public.endoPoints * public.frames)
    public.d_tEndoColLoc = Array{Int32}(public.endoPoints * public.frames)

    # Epi points
    public.epiPoints = EPI_POINTS
    public.d_epi_mem = sizeof(Int32) * public.epiPoints
    public.d_epiRow = Array{Int32}([390, 419, 448, 474, 501, 519, 535, 542, 543, 538,
                                    528, 511, 491, 466, 438, 406, 376, 347, 318, 291,
                                    275, 259, 256, 252, 252, 257, 266, 283, 305, 331,
                                    360])
    public.d_epiCol = Array{Int32}([457, 454, 446, 431, 411, 388, 361, 331, 301, 273,
                                    243, 218, 196, 178, 166, 157, 155, 165, 177, 197,
                                    218, 248, 276, 304, 333, 361, 391, 415, 434, 448,
                                    455])

    public.d_tEpiRowLoc = Array{Int32}(public.epiPoints * public.frames)
    public.d_tEpiColLoc = Array{Int32}(public.epiPoints * public.frames)

    # All points
    public.allPoints = ALL_POINTS

    # Constants
    public.tSize = 25
    public.sSize = 40
    public.maxMove = 10
    public.alpha = 0.87f0

    # Sums
    for i = 1:public.allPoints

        private[i] = private_struct()
        private[i].in_partial_sum = Array{Float32}(2 * public.tSize + 1)
        private[i].in_sqr_partial_sum = Array{Float32}(2 * public.tSize + 1)
        private[i].par_max_val =
            Array{Float32}(2 * public.tSize + 2 * public.sSize + 1)
        private[i].par_max_coo =
            Array{Int32}(2 * public.tSize + 2 * public.sSize + 1)
    end

    # Input 2 (sample around point)
    public.in2_rows = 2 * public.sSize + 1
    public.in2_cols = 2 * public.sSize + 1
    public.in2_elem = public.in2_rows * public.in2_cols
    public.in2_mem = sizeof(Float32) * public.in2_elem

    for i = 1:public.allPoints

        private[i].d_in2 = Array{Float32}(public.in2_elem)
        private[i].d_in2_sqr = Array{Float32}(public.in2_elem)
    end

    # Input (point template)
    public.in_mod_rows = public.tSize + 1 + public.tSize
    public.in_mod_cols = public.in_mod_rows
    public.in_mod_elem = public.in_mod_rows * public.in_mod_cols
    public.in_mod_mem = sizeof(Float32) * public.in_mod_elem

    for i = 1:public.allPoints

        private[i].d_in_mod = Array{Float32}(public.in_mod_elem)
        private[i].d_in_sqr = Array{Float32}(public.in_mod_elem)
    end

    # Array of templates for all points
    public.d_endoT = Array{Float32}(public.in_mod_elem * public.endoPoints)
    public.d_epiT = Array{Float32}(public.in_mod_elem * public.epiPoints)

    # Setup private pointers to rows, cols and template
    for i = 1:public.endoPoints

        private[i].point_no = i - 1
        private[i].in_pointer = private[i].point_no * public.in_mod_elem
        private[i].d_Row = public.d_endoRow # original row coordinates
        private[i].d_Col = public.d_endoCol # original col coordinates
        private[i].d_tRowLoc = public.d_tEndoRowLoc # updated row coordinates
        private[i].d_tColLoc = public.d_tEndoColLoc # updated row coordinates
        private[i].d_T = public.d_endoT # templates
    end

    for i = public.endoPoints+1:public.allPoints

        private[i].point_no = i - 1 - public.endoPoints
        private[i].in_pointer = private[i].point_no * public.in_mod_elem
        private[i].d_Row = public.d_epiRow
        private[i].d_Col = public.d_epiCol
        private[i].d_tRowLoc = public.d_tEpiRowLoc
        private[i].d_tColLoc = public.d_tEpiColLoc
        private[i].d_T = public.d_epiT
    end

    # Convolution
    public.ioffset = 0
    public.joffset = 0
    public.conv_rows = public.in_mod_rows + public.in2_rows - 1 # number of rows in I
    public.conv_cols = public.in_mod_cols + public.in2_cols - 1 # number of columns in I
    public.conv_elem = public.conv_rows * public.conv_cols # number of elements
    public.conv_mem = sizeof(Float32) * public.conv_elem

    for i = 1:public.allPoints
        private[i].d_conv = Array{Float32}(public.conv_elem)
    end

    # Pad array
    public.in2_pad_add_rows = public.in_mod_rows
    public.in2_pad_add_cols = public.in_mod_cols
    public.in2_pad_rows = public.in2_rows + 2 * public.in2_pad_add_rows
    public.in2_pad_cols = public.in2_cols + 2 * public.in2_pad_add_cols
    public.in2_pad_elem = public.in2_pad_rows * public.in2_pad_cols
    public.in2_pad_mem = sizeof(Float32) * public.in2_pad_elem

    for i = 1:public.allPoints
        private[i].d_in2_pad = Array{Float32}(public.in2_pad_elem)
    end

    # Selection, selection 2, subtraction
    # Horizontal cumulative sum
    public.in2_pad_cumv_sel_rowlow = 1 + public.in_mod_rows # (1 to n+1)
    public.in2_pad_cumv_sel_rowhig = public.in2_pad_rows - 1
    public.in2_pad_cumv_sel_collow = 1
    public.in2_pad_cumv_sel_colhig = public.in2_pad_cols
    public.in2_pad_cumv_sel2_rowlow = 1
    public.in2_pad_cumv_sel2_rowhig = public.in2_pad_rows - public.in_mod_rows - 1
    public.in2_pad_cumv_sel2_collow = 1
    public.in2_pad_cumv_sel2_colhig = public.in2_pad_cols
    public.in2_sub_rows =
        public.in2_pad_cumv_sel_rowhig - public.in2_pad_cumv_sel_rowlow + 1
    public.in2_sub_cols =
        public.in2_pad_cumv_sel_colhig - public.in2_pad_cumv_sel_collow + 1
    public.in2_sub_elem = public.in2_sub_rows * public.in2_sub_cols
    public.in2_sub_mem = sizeof(Float32) * public.in2_sub_elem

    for i = 1:public.allPoints
        private[i].d_in2_sub = Array{Float32}(public.in2_sub_elem)
    end

    # Selection, selection 2, subtraction, square, numerator
    public.in2_sub_cumh_sel_rowlow = 1
    public.in2_sub_cumh_sel_rowhig = public.in2_sub_rows
    public.in2_sub_cumh_sel_collow = 1 + public.in_mod_cols
    public.in2_sub_cumh_sel_colhig = public.in2_sub_cols - 1
    public.in2_sub_cumh_sel2_rowlow = 1
    public.in2_sub_cumh_sel2_rowhig = public.in2_sub_rows
    public.in2_sub_cumh_sel2_collow = 1
    public.in2_sub_cumh_sel2_colhig = public.in2_sub_cols - public.in_mod_cols - 1
    public.in2_sub2_sqr_rows =
        public.in2_sub_cumh_sel_rowhig - public.in2_sub_cumh_sel_rowlow + 1
    public.in2_sub2_sqr_cols =
        public.in2_sub_cumh_sel_colhig - public.in2_sub_cumh_sel_collow + 1
    public.in2_sub2_sqr_elem = public.in2_sub2_sqr_rows * public.in2_sub2_sqr_cols
    public.in2_sub2_sqr_mem = sizeof(Float32) * public.in2_sub2_sqr_elem

    for i = 1:public.allPoints
        private[i].d_in2_sub2_sqr = Array{Float32}(public.in2_sub2_sqr_elem)
    end

    # Template mask create
    public.tMask_rows = public.in_mod_rows + (public.sSize + 1 + public.sSize) - 1
    public.tMask_cols = public.tMask_rows
    public.tMask_elem = public.tMask_rows * public.tMask_cols
    public.tMask_mem = sizeof(Float32) * public.tMask_elem

    for i = 1:public.allPoints
        private[i].d_tMask = Array{Float32}(public.tMask_elem)
    end

    # Point mask initialize
    public.mask_rows = public.maxMove
    public.mask_cols = public.mask_rows
    public.mask_elem = public.mask_rows * public.mask_cols
    public.mask_mem = sizeof(Float32) * public.mask_elem

    # Mask convolution
    public.mask_conv_rows = public.tMask_rows # number of rows in I
    public.mask_conv_cols = public.tMask_cols # number of columns in I
    public.mask_conv_elem =
        public.mask_conv_rows * public.mask_conv_cols # number of elements
    public.mask_conv_mem = sizeof(Float32) * public.mask_conv_elem
    public.mask_conv_ioffset = div(public.mask_rows - 1, 2)

    if (public.mask_rows - 1) % 2 > 0
        public.mask_conv_ioffset = public.mask_conv_ioffset + 1
    end

    public.mask_conv_joffset = div(public.mask_cols - 1, 2)

    if (public.mask_cols - 1) % 2 > 0
        public.mask_conv_joffset = public.mask_conv_joffset + 1
    end

    for i = 1:public.allPoints
        private[i].d_mask_conv = Array{Float32}(public.mask_conv_elem)
    end

    # Print frame progress start
    print("frame progress: ")
    flush(STDOUT)

    # Kernel
    for public.frame_no = 0:frames_processed-1

        # Getting frame
        # Extract a cropped version of the first frame from the video file
        public.d_frame = AVI_get_frame(
            public.d_frames, # pointer to video file
            public.frame_no, # number of frame that needs to be returned
            false,           # cropped?
            false,           # scaled?
            true)            # converted

        # Processing
        for i = 1:public.allPoints
            kernel(public, private[i])
        end

        # Free memory for frame after each loop iteration, since AVI library allocates
        # memory for every frame fetched
        cfree(public.d_frame)

        # Print frame progress
        @printf("%d ", public.frame_no)
        flush(STDOUT)
    end

    # Print frame progress end
    println()
    flush(STDOUT)

    # Dump data to file
    if OUTPUT
        write_data("output.txt", public.frames, frames_processed,
                   public.endoPoints, public.d_tEndoRowLoc,
                   public.d_tEndoColLoc, public.epiPoints, public.d_tEpiRowLoc,
                   public.d_tEpiColLoc)
    end
end

main(ARGS)
