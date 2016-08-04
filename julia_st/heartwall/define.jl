const ENDO_POINTS = 20
const EPI_POINTS  = 31
const ALL_POINTS  = 51

type public_struct

    # What used to be inputs from Matlab
    tSize::Int32
    sSize::Int32
    maxMove::Int32
    alpha::Float32

    endoPoints::Int32
    d_endo_mem::Int32
    d_endoRow::Array{Int32,1}
    d_endoCol::Array{Int32,1}
    d_tEndoRowLoc::Array{Int32,1}
    d_tEndoColLoc::Array{Int32,1}
    d_endoT::Array{Float32,1}

    epiPoints::Int32
    d_epi_mem::Int32
    d_epiRow::Array{Int32,1}
    d_epiCol::Array{Int32,1}
    d_tEpiRowLoc::Array{Int32,1}
    d_tEpiColLoc::Array{Int32,1}
    d_epiT::Array{Float32,1}

    allPoints

    # Frame
    d_frames::Ptr{avi_t}
    frames::Int32
    frame_no::Int32
    d_frame::Ptr{Float32}
    frame_rows::Int32
    frame_cols::Int32
    frame_elem::Int32
    frame_mem::Int32

    # Input 2
    in2_rows::Int32
    in2_cols::Int32
    in2_elem::Int32
    in2_mem::Int32

    # Input
    in_mod_rows::Int32
    in_mod_cols::Int32
    in_mod_elem::Int32
    in_mod_mem::Int32

    # Convolution
    ioffset::Int32
    joffset::Int32

    conv_rows::Int32
    conv_cols::Int32
    conv_elem::Int32
    conv_mem::Int32

       # Pad array
    in2_pad_add_rows::Int32
    in2_pad_add_cols::Int32

    in2_pad_rows::Int32
    in2_pad_cols::Int32
    in2_pad_elem::Int32
    in2_pad_mem::Int32

    # Selection, selection 2, subtration, horizontal cumulative sum
    in2_pad_cumv_sel_rowlow::Int32
    in2_pad_cumv_sel_rowhig::Int32
    in2_pad_cumv_sel_collow::Int32
    in2_pad_cumv_sel_colhig::Int32

    in2_pad_cumv_sel2_rowlow::Int32
    in2_pad_cumv_sel2_rowhig::Int32
    in2_pad_cumv_sel2_collow::Int32
    in2_pad_cumv_sel2_colhig::Int32

    in2_sub_rows::Int32
    in2_sub_cols::Int32
    in2_sub_elem::Int32
    in2_sub_mem::Int32

    # Selection, selection 2, subtraction, square, numerator
    in2_sub_cumh_sel_rowlow::Int32
    in2_sub_cumh_sel_rowhig::Int32
    in2_sub_cumh_sel_collow::Int32
    in2_sub_cumh_sel_colhig::Int32

    in2_sub_cumh_sel2_rowlow::Int32
    in2_sub_cumh_sel2_rowhig::Int32
    in2_sub_cumh_sel2_collow::Int32
    in2_sub_cumh_sel2_colhig::Int32

    in2_sub2_sqr_rows::Int32
    in2_sub2_sqr_cols::Int32
    in2_sub2_sqr_elem::Int32
    in2_sub2_sqr_mem::Int32

    # Template mask create
    tMask_rows::Int32
    tMask_cols::Int32
    tMask_elem::Int32
    tMask_mem::Int32

    # Point mask initialize
    mask_rows::Int32
    mask_cols::Int32
    mask_elem::Int32
    mask_mem::Int32

    # Mask convolution
    mask_conv_rows::Int32
    mask_conv_cols::Int32
    mask_conv_elem::Int32
    mask_conv_mem::Int32
    mask_conv_ioffset::Int32
    mask_conv_joffset::Int32

    public_struct() = new()
end

type private_struct

    # Point-specific
    point_no::Int32
    in_pointer::Int32

    d_Row::Array{Int32,1}
    d_Col::Array{Int32,1}
    d_tRowLoc::Array{Int32,1}
    d_tColLoc::Array{Int32,1}
    d_T::Array{Float32,1}

    # Input 2
    d_in2::Array{Float32,1}
    d_in2_sqr::Array{Float32,1}

    # Input
    d_in_mod::Array{Float32,1}
    d_in_sqr::Array{Float32,1}

    # Convolution
    d_conv::Array{Float32,1}

    # Pad array
    d_in2_pad::Array{Float32,1}

    # Horizontal cumulative sum
    d_in2_sub::Array{Float32,1}

    # Selection, selection 2, subtraction, square, numerator
    d_in2_sub2_sqr::Array{Float32,1}

    # Template mask create
    d_tMask::Array{Float32,1}

    # Mask convolution
    d_mask_conv::Array{Float32,1}

    # Sum
    in_partial_sum::Array{Float32,1}
    in_sqr_partial_sum::Array{Float32,1}
    par_max_val::Array{Float32,1}
    par_max_coo::Array{Int32,1}

    private_struct() = new()
end
