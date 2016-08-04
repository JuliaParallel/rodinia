function kernel(public, private)

    largest_value_current = 0f0
    largest_value = 0f0
    largest_coordinate_current = 0
    largest_coordinate = 0
    fin_max_val = 0f0
    fin_max_coo = 0

    # Generate template
    # generate templates based on the first frame only
    if public.frame_no == 0

        # update temporary row/col coordinates
        pointer = private.point_no * public.frames + public.frame_no
        private.d_tRowLoc[pointer + 1] = private.d_Row[private.point_no + 1]
        private.d_tColLoc[pointer + 1] = private.d_Col[private.point_no + 1]

        # update template, limit the number of working threads to the size of template
        for col = 0:public.in_mod_cols-1, row = 0:public.in_mod_rows-1

            # figure out row/col location in corresponding new template area in image and
            # give to every thread (get top left corner and progress down and right)
            ori_row = private.d_Row[private.point_no + 1] - 25 + row - 1
            ori_col = private.d_Col[private.point_no + 1] - 25 + col - 1
            ori_pointer = ori_col * public.frame_rows + ori_row

            # update template
            private.d_T[private.in_pointer + col * public.in_mod_rows + row + 1] =
                unsafe_load(public.d_frame, ori_pointer + 1)
        end
    end

    # Process points in all frames except for the first one
    if public.frame_no != 0

        # 1) setup pointer to point to current frame from batch
        # 2) select input 2 (sample around point) from frame; save in d_in2 (not linear
        #    in memory, so need to save output for later use)
        # 3) square input 2; save in d_in2_sqr

        # pointers and variables
        in2_rowlow = private.d_Row[private.point_no + 1] - public.sSize # (1 to n+1)
        in2_collow = private.d_Col[private.point_no + 1] - public.sSize

        for col = 0:public.in2_cols-1, row = 0:public.in2_rows-1

            # figure out corresponding location in old matrix and
            # copy values to new matrix
            ori_row = row + in2_rowlow - 1
            ori_col = col + in2_collow - 1
            temp = unsafe_load(public.d_frame, ori_col * public.frame_rows + ori_row + 1)
            private.d_in2[col * public.in2_rows + row + 1] = temp
            private.d_in2_sqr[col * public.in2_rows + row + 1] = temp * temp
        end

        # 1) get pointer to input 1 (template for this point) in template array
        #    (linear in memory, so don't need to save, just get pointer)
        # 2) rotate input 1; save in d_in_mod
        # 3) square input 1; save in d_in_sqr

        for col = 0:public.in_mod_cols-1, row = 0:public.in_mod_rows-1

            # rotated coordinates
            rot_row = (public.in_mod_rows - 1) - row
            rot_col = (public.in_mod_rows - 1) - col
            pointer = rot_col * public.in_mod_rows + rot_row

            # execution
            temp = private.d_T[private.in_pointer + pointer + 1]
            private.d_in_mod[col * public.in_mod_rows + row + 1] = temp
            private.d_in_sqr[pointer + 1] = temp * temp
        end

        # 1) get sum of input 1
        # 2) get sum of input 1 squared

        pip = private.in_pointer
        in_final_sum = sum(private.d_T[pip+1:pip+public.in_mod_elem])
        in_sqr_final_sum = sum(private.d_in_sqr[1:public.in_mod_elem])

        # 3) do statistical calculations
        # 4) get denominator T

        mean = in_final_sum / public.in_mod_elem # mean value of element in ROI
        mean_sqr = mean * mean
        variance = (in_sqr_final_sum / public.in_mod_elem) - mean_sqr # variance of ROI
        deviation = sqrt(variance) # standard deviation of ROI
        denomT = sqrt(public.in_mod_elem - 1) * deviation

        # 1) convolve input 2 with rotated input 1; save in d_conv
        for col::Int32 = 1:public.conv_cols

            # column setup
            j = col + public.joffset
            jp1 = j + 1

            if public.in2_cols < jp1
                ja1 = jp1 - public.in2_cols
            else
                ja1 = 1
            end

            if public.in_mod_cols < j
                ja2 = public.in_mod_cols
            else
                ja2 = j
            end

            for row::Int32 = 1:public.conv_rows

                # row range setup
                i = row + public.ioffset
                ip1 = i + 1

                if public.in2_rows < ip1
                    ia1 = ip1 - public.in2_rows
                else
                    ia1 = 1
                end

                if public.in_mod_rows < i
                    ia2 = public.in_mod_rows
                else
                    ia2 = i
                end

                s = 0f0

                # getting data
                for ja = ja1:ja2

                    jb = jp1 - ja

                    for ia = ia1:ia2

                        ib = ip1 - ia

                        s = s + private.d_in_mod[public.in_mod_rows * (ja - 1) + ia] *
                                private.d_in2[public.in2_rows * (jb - 1) + ib]
                    end
                end

                private.d_conv[(col - 1) * public.conv_rows + row] = s
            end
        end

        # Local sum 1
        # 1) padd array; save in d_in2_pad
        for col = 0:public.in2_pad_cols-1, row = 0:public.in2_pad_rows-1

            # execution
            if row > (public.in2_pad_add_rows - 1) && # if has numbers in original array
                row < (public.in2_pad_add_rows + public.in2_rows) &&
                col > (public.in2_pad_add_cols - 1) &&
                col < (public.in2_pad_add_cols + public.in2_cols)

                ori_row = row - public.in2_pad_add_rows
                ori_col = col - public.in2_pad_add_cols
                private.d_in2_pad[col * public.in2_pad_rows + row + 1] =
                    private.d_in2[ori_col * public.in2_rows + ori_row + 1]
            else
                private.d_in2_pad[col * public.in2_pad_rows + row + 1] = 0f0
            end
        end

        # 1) get vertical cumulative sum, save in d_in2_pad
        for ei_new = 0:public.in2_pad_cols-1

            # figure out column position
            pos_ori = ei_new * public.in2_pad_rows

            # loop through all rows
            sum = 0f0

            for position = pos_ori+1:pos_ori+public.in2_pad_rows

                private.d_in2_pad[position] = private.d_in2_pad[position] + sum
                sum = private.d_in2_pad[position]
            end
        end

        # 1) make 1st selection from vertical cumulative sum
        # 2) make 2nd selection from vertical cumulative sum
        # 3) subtract the two selections; save in d_in2_sub
        for col = 0:public.in2_sub_cols-1, row = 0:public.in2_sub_rows-1

            # figure out corresponding location in old matrix
            # and copy values to new matrix
            ori_row = row + public.in2_pad_cumv_sel_rowlow - 1
            ori_col = col + public.in2_pad_cumv_sel_collow - 1
            temp = private.d_in2_pad[ori_col * public.in2_pad_rows + ori_row + 1]

            # figure out corresponding location in old matrix
            # and copy values to new matrix
            ori_row = row + public.in2_pad_cumv_sel2_rowlow - 1
            ori_col = col + public.in2_pad_cumv_sel2_collow - 1
            temp2 = private.d_in2_pad[ori_col * public.in2_pad_rows + ori_row + 1]

            # subtraction
            private.d_in2_sub[col * public.in2_sub_rows + row + 1] = temp - temp2
        end

        # 1) get horizontal cumulative sum; save in d_in2_sub
        for ei_new = 0:public.in2_sub_rows-1

            # figure out row position
            pos_ori = ei_new

            # loop through all rows
            sum = 0f0

            for position = pos_ori+1:public.in2_sub_rows:pos_ori+public.in2_sub_elem

                private.d_in2_sub[position] = private.d_in2_sub[position] + sum
                sum = private.d_in2_sub[position]
            end
        end

        # 1) make 1st selection from horizontal cumulative sum
        # 2) make 2nd selection from horizontal cumulative sum
        # 3) subtract the two selections to get local sum 1
        # 4) get cumulative sum 1 squared; save in d_in2_sub2_sqr
        # 5) get numerator; save in d_conv
        for col = 0:public.in2_sub2_sqr_cols-1, row = 0:public.in2_sub2_sqr_rows-1

            # figure out corresponding location in old matrix
            # and copy values to new matrix
            ori_row = row + public.in2_sub_cumh_sel_rowlow - 1
            ori_col = col + public.in2_sub_cumh_sel_collow - 1
            temp = private.d_in2_sub[ori_col * public.in2_sub_rows + ori_row + 1]

            # figure out corresponding location in old matrix
            # and copy values to new matrix
            ori_row = row + public.in2_sub_cumh_sel2_rowlow - 1
            ori_col = col + public.in2_sub_cumh_sel2_collow - 1
            temp2 = private.d_in2_sub[ori_col * public.in2_sub_rows + ori_row + 1]

            # subtraction
            temp2 = temp - temp2

            # squaring
            private.d_in2_sub2_sqr[col * public.in2_sub2_sqr_rows + row + 1] =
                temp2 * temp2

            # numerator
            private.d_conv[col * public.in2_sub2_sqr_rows + row + 1] =
                private.d_conv[col * public.in2_sub2_sqr_rows + row + 1] -
                temp2 * in_final_sum / public.in_mod_elem
        end

        # Local sum 2
        # 1) pad array; save in d_in2_pad
        for col = 0:public.in2_pad_cols-1, row = 0:public.in2_pad_rows-1

            # execution
            if row > (public.in2_pad_add_rows - 1) && # if has numbers in original array
                row < (public.in2_pad_add_rows + public.in2_rows) &&
                col > (public.in2_pad_add_cols - 1) &&
                col < (public.in2_pad_add_cols + public.in2_cols)

                ori_row = row - public.in2_pad_add_rows
                ori_col = col - public.in2_pad_add_cols
                private.d_in2_pad[col * public.in2_pad_rows + row + 1] =
                    private.d_in2_sqr[ori_col * public.in2_rows + ori_row + 1]
            else
                private.d_in2_pad[col * public.in2_pad_rows + row + 1] = 0f0
            end
        end

        # 2) get vertical cumulative sum; save in d_in2_pad
        for ei_new = 0:public.in2_pad_cols-1

            # figure out column position
            pos_ori = ei_new * public.in2_pad_rows

            # loop through all rows
            sum = 0f0

            for position = pos_ori+1:pos_ori+public.in2_pad_rows

                private.d_in2_pad[position] = private.d_in2_pad[position] + sum
                sum = private.d_in2_pad[position]
            end
        end

        # 1) make 1st selection from vertical cumulative sum
        # 2) make 2nd selection from vertical cumulative sum
        # 3) subtract the two selections; save in d_in2_sub
        for col = 0:public.in2_sub_cols-1, row = 0:public.in2_sub_rows-1

            # figure out corresponding location in old matrix
            # and copy values to new matrix
            ori_row = row + public.in2_pad_cumv_sel_rowlow - 1
            ori_col = col + public.in2_pad_cumv_sel_collow - 1
            temp = private.d_in2_pad[ori_col * public.in2_pad_rows + ori_row + 1]

            # figure out corresponding location in old matrix
            # and copy values to new matrix
            ori_row = row + public.in2_pad_cumv_sel2_rowlow - 1
            ori_col = col + public.in2_pad_cumv_sel2_collow - 1
            temp2 = private.d_in2_pad[ori_col * public.in2_pad_rows + ori_row + 1]

            # subtract
            private.d_in2_sub[col * public.in2_sub_rows + row + 1] = temp - temp2
        end

        # 1) get horizontal cumulative sum; save in d_in2_sub
        for ei_new = 0:public.in2_sub_rows-1

            # figure out row position
            pos_ori = ei_new

            # loop through all rows
            sum = 0f0

            for position = pos_ori+1:public.in2_sub_rows:pos_ori+public.in2_sub_elem

                private.d_in2_sub[position] = private.d_in2_sub[position] + sum
                sum = private.d_in2_sub[position]
            end
        end

        # 1) make 1st selection from horizontal cumulative sum
        # 2) make 2nd selection from horizontal cumulative sum
        # 3) subtract the two selections to get local sum 2
        # 4) get differential local sum
        # 5) get denominator A
        # 6) get denominator
        # 7) divide numerator by denominator to get correlation; save in d_conv
        for col = 0:public.conv_cols-1, row = 0:public.conv_rows-1

            # figure out corresponding location in old matrix
            # and copy values to new matrix
            ori_row = row + public.in2_sub_cumh_sel_rowlow - 1
            ori_col = col + public.in2_sub_cumh_sel_collow - 1
            temp = private.d_in2_sub[ori_col * public.in2_sub_rows + ori_row + 1]

            # figure out corresponding location in old matrix
            # and copy values to new matrix
            ori_row = row + public.in2_sub_cumh_sel2_rowlow - 1
            ori_col = col + public.in2_sub_cumh_sel2_collow - 1
            temp2 = private.d_in2_sub[ori_col * public.in2_sub_rows + ori_row + 1]

            # subtract
            temp2 = temp - temp2

            # diff_local_sums
            temp2 = temp2 - private.d_in2_sub2_sqr[col * public.conv_rows + row + 1] /
                public.in_mod_elem

            # denominator A
            if temp2 < 0
                temp2 = 0f0
            end

            temp2 = sqrt(temp2)

            # denominator
            temp2 = denomT * temp2

            # correlation
            private.d_conv[col * public.conv_rows + row + 1] =
                private.d_conv[col * public.conv_rows + row + 1] / temp2
        end

        # template mask create
        # parameters
        cent = public.sSize + public.tSize + 1
        pointer = public.frame_no - 1 + private.point_no * public.frames
        tMask_row = cent + private.d_tRowLoc[pointer + 1] -
                    private.d_Row[private.point_no + 1] - 1
        tMask_col = cent + private.d_tColLoc[pointer + 1] -
                    private.d_Col[private.point_no + 1] - 1

        for ei_new = 1:public.tMask_elem
          private.d_tMask[ei_new] = 0
        end

        private.d_tMask[tMask_col * public.tMask_rows + tMask_row + 1] = 1

        # 1) mask convolution
        # 2) multiplication
        for col::Int32 = 1:public.mask_conv_cols

            # col setup
            j = col + public.mask_conv_joffset
            jp1 = j + 1

            if public.mask_cols < jp1
                ja1 = jp1 - public.mask_cols
            else
                ja1 = 1
            end

            if public.tMask_cols < j
                ja2 = public.tMask_cols
            else
                ja2 = j
            end

            for row::Int32 = 1:public.mask_conv_rows

                # row setup
                i = row + public.mask_conv_ioffset
                ip1 = i + 1

                if public.mask_rows < ip1
                    ia1 = ip1 - public.mask_rows
                else
                    ia1 = 1
                end

                if public.tMask_rows < i
                    ia2 = public.tMask_rows
                else
                    ia2 = i
                end

                s = 0f0

                # get data
                for ja = ja1:ja2

                    jb = jp1 - ja

                    for ia = ia1:ia2

                        ib = ip1 - ia
                        s = s + private.d_tMask[public.tMask_rows * (ja - 1) + ia]
                    end
                end

                private.d_mask_conv[(col - 1) * public.conv_rows + row] =
                    private.d_conv[(col - 1) * public.conv_rows + row] * s
            end
        end

        # Maximum value
        # search
        fin_max_val = 0f0
        fin_max_coo = 0

        for i = 1:public.mask_conv_elem

            if private.d_mask_conv[i] > fin_max_val

                fin_max_val = private.d_mask_conv[i]
                fin_max_coo = i - 1
            end
        end

        # Offset
        # convert coordinate to row/col form
        largest_row = (fin_max_coo + 1) % public.mask_conv_rows - 1 # (0-n) row
        largest_col = div(fin_max_coo + 1, public.mask_conv_rows) # (0-n) column

        if (fin_max_coo + 1) % public.mask_conv_rows == 0

            largest_row = public.mask_conv_rows - 1
            largest_col = largest_col - 1
        end

        # calculate offset
        largest_row = largest_row + 1 # compensate to match MATLAB format (1-n)
        largest_col = largest_col + 1 # compensate to match MATLAB format (1-n)
        offset_row = largest_row - public.in_mod_rows - (public.sSize - public.tSize)
        offset_col = largest_col - public.in_mod_cols - (public.sSize - public.tSize)
        pointer = private.point_no * public.frames + public.frame_no
        private.d_tRowLoc[pointer + 1] = private.d_Row[private.point_no + 1] + offset_row
        private.d_tColLoc[pointer + 1] = private.d_Col[private.point_no + 1] + offset_col
    end

    # if the last frame in the batch, update template
    if public.frame_no != 0 && public.frame_no % 10 == 0

        # update coordinate
        loc_pointer = private.point_no * public.frames + public.frame_no
        private.d_Row[private.point_no + 1] = private.d_tRowLoc[loc_pointer + 1]
        private.d_Col[private.point_no + 1] = private.d_tColLoc[loc_pointer + 1]

        # update template, limit the number of working threads to the size of template
        for col = 0:public.in_mod_cols-1, row = 0:public.in_mod_rows-1

            # figure out row/col location in corresponding new template area in image and
            # give to every thread (get top left corner and progress down and right)
            ori_row = private.d_Row[private.point_no + 1] - 25 + row - 1
            ori_col = private.d_Col[private.point_no + 1] - 25 + col - 1
            ori_pointer = ori_col * public.frame_rows + ori_row

            # update template
            idx = private.in_pointer + col * public.in_mod_rows + row
            private.d_T[idx + 1] = public.alpha * private.d_T[idx + 1] +
                (1f0 - public.alpha) * unsafe_load(public.d_frame, ori_pointer + 1)
        end
    end
end
