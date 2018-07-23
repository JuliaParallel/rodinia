const AVI_MAX_TRACKS = 8
const AVI_MODE_WRITE = 0
const AVI_MODE_READ = 1
const AVI_ERR_SIZELIM = 1
const AVI_ERR_OPEN = 2
const AVI_ERR_READ = 3
const AVI_ERR_WRITE = 4
const AVI_ERR_WRITE_INDEX = 5
const AVI_ERR_CLOSE = 6
const AVI_ERR_NOT_PERM = 7
const AVI_ERR_NO_MEM = 8
const AVI_ERR_NO_AVI = 9
const AVI_ERR_NO_HDRL = 10
const AVI_ERR_NO_MOVI = 11
const AVI_ERR_NO_VIDS = 12
const AVI_ERR_NO_IDX = 13
const WAVE_FORMAT_UNKNOWN = 0x0000
const WAVE_FORMAT_PCM = 0x0001
const WAVE_FORMAT_ADPCM = 0x0002
const WAVE_FORMAT_IBM_CVSD = 0x0005
const WAVE_FORMAT_ALAW = 0x0006
const WAVE_FORMAT_MULAW = 0x0007
const WAVE_FORMAT_OKI_ADPCM = 0x0010
const WAVE_FORMAT_DVI_ADPCM = 0x0011
const WAVE_FORMAT_DIGISTD = 0x0015
const WAVE_FORMAT_DIGIFIX = 0x0016
const WAVE_FORMAT_YAMAHA_ADPCM = 0x0020
const WAVE_FORMAT_DSP_TRUESPEECH = 0x0022
const WAVE_FORMAT_GSM610 = 0x0031
const IBM_FORMAT_MULAW = 0x0101
const IBM_FORMAT_ALAW = 0x0102
const IBM_FORMAT_ADPCM = 0x0103

struct avi_t
# opaque to Julia
end

const libavi = joinpath(dirname(@__DIR__), "avi", "libavi.so")

function AVI_open_output_file(filename::AbstractString)
    ccall((:AVI_open_output_file,libavi),Ptr{avi_t},(Cstring,),filename)
end

function AVI_set_video(AVI::Ptr{avi_t},width::Cint,height::Cint,fps::Cdouble,compressor::AbstractString)
    ccall((:AVI_set_video,libavi),Cvoid,(Ptr{avi_t},Cint,Cint,Cdouble,Cstring),AVI,width,height,fps,compressor)
end

function AVI_set_audio(AVI::Ptr{avi_t},channels::Cint,rate::Clong,bits::Cint,format::Cint,mp3rate::Clong)
    ccall((:AVI_set_audio,libavi),Cvoid,(Ptr{avi_t},Cint,Clong,Cint,Cint,Clong),AVI,channels,rate,bits,format,mp3rate)
end

function AVI_write_frame(AVI::Ptr{avi_t},data::Array{UInt8,1},bytes::Clong,keyframe::Cint)
    ccall((:AVI_write_frame,libavi),Cint,(Ptr{avi_t},Ptr{UInt8},Clong,Cint),AVI,data,bytes,keyframe)
end

function AVI_dup_frame(AVI::Ptr{avi_t})
    ccall((:AVI_dup_frame,libavi),Cint,(Ptr{avi_t},),AVI)
end

function AVI_write_audio(AVI::Ptr{avi_t},data::Ptr{UInt8},bytes::Clong)
    ccall((:AVI_write_audio,libavi),Cint,(Ptr{avi_t},Ptr{UInt8},Clong),AVI,data,bytes)
end

function AVI_append_audio(AVI::Ptr{avi_t},data::Ptr{UInt8},bytes::Clong)
    ccall((:AVI_append_audio,libavi),Cint,(Ptr{avi_t},Ptr{UInt8},Clong),AVI,data,bytes)
end

function AVI_bytes_remain(AVI::Ptr{avi_t})
    ccall((:AVI_bytes_remain,libavi),Clong,(Ptr{avi_t},),AVI)
end

function AVI_close(AVI::Ptr{avi_t})
    ccall((:AVI_close,libavi),Cint,(Ptr{avi_t},),AVI)
end

function AVI_bytes_written(AVI::Ptr{avi_t})
    ccall((:AVI_bytes_written,libavi),Clong,(Ptr{avi_t},),AVI)
end

function AVI_open_input_file(filename::AbstractString,getIndex)
    ccall((:AVI_open_input_file,libavi),Ptr{avi_t},(Cstring,Cint),filename,getIndex)
end

function AVI_open_fd(fd::Cint,getIndex::Cint)
    ccall((:AVI_open_fd,libavi),Ptr{avi_t},(Cint,Cint),fd,getIndex)
end

function avi_parse_input_file(AVI::Ptr{avi_t},getIndex::Cint)
    ccall((:avi_parse_input_file,libavi),Cint,(Ptr{avi_t},Cint),AVI,getIndex)
end

function AVI_audio_mp3rate(AVI::Ptr{avi_t})
    ccall((:AVI_audio_mp3rate,libavi),Clong,(Ptr{avi_t},),AVI)
end

function AVI_video_frames(AVI::Ptr{avi_t})
    ccall((:AVI_video_frames,libavi),Clong,(Ptr{avi_t},),AVI)
end

function AVI_video_width(AVI::Ptr{avi_t})
    ccall((:AVI_video_width,libavi),Cint,(Ptr{avi_t},),AVI)
end

function AVI_video_height(AVI::Ptr{avi_t})
    ccall((:AVI_video_height,libavi),Cint,(Ptr{avi_t},),AVI)
end

function AVI_frame_rate(AVI::Ptr{avi_t})
    ccall((:AVI_frame_rate,libavi),Cdouble,(Ptr{avi_t},),AVI)
end

function AVI_video_compressor(AVI::Ptr{avi_t})
    ccall((:AVI_video_compressor,libavi),Ptr{UInt8},(Ptr{avi_t},),AVI)
end

function AVI_audio_channels(AVI::Ptr{avi_t})
    ccall((:AVI_audio_channels,libavi),Cint,(Ptr{avi_t},),AVI)
end

function AVI_audio_bits(AVI::Ptr{avi_t})
    ccall((:AVI_audio_bits,libavi),Cint,(Ptr{avi_t},),AVI)
end

function AVI_audio_format(AVI::Ptr{avi_t})
    ccall((:AVI_audio_format,libavi),Cint,(Ptr{avi_t},),AVI)
end

function AVI_audio_rate(AVI::Ptr{avi_t})
    ccall((:AVI_audio_rate,libavi),Clong,(Ptr{avi_t},),AVI)
end

function AVI_audio_bytes(AVI::Ptr{avi_t})
    ccall((:AVI_audio_bytes,libavi),Clong,(Ptr{avi_t},),AVI)
end

function AVI_audio_chunks(AVI::Ptr{avi_t})
    ccall((:AVI_audio_chunks,libavi),Clong,(Ptr{avi_t},),AVI)
end

function AVI_max_video_chunk(AVI::Ptr{avi_t})
    ccall((:AVI_max_video_chunk,libavi),Clong,(Ptr{avi_t},),AVI)
end

function AVI_frame_size(AVI::Ptr{avi_t},frame::Clong)
    ccall((:AVI_frame_size,libavi),Clong,(Ptr{avi_t},Clong),AVI,frame)
end

function AVI_audio_size(AVI::Ptr{avi_t},frame::Clong)
    ccall((:AVI_audio_size,libavi),Clong,(Ptr{avi_t},Clong),AVI,frame)
end

function AVI_seek_start(AVI::Ptr{avi_t})
    ccall((:AVI_seek_start,libavi),Cint,(Ptr{avi_t},),AVI)
end

function AVI_set_video_position(AVI::Ptr{avi_t},frame::Clong)
    ccall((:AVI_set_video_position,libavi),Cint,(Ptr{avi_t},Clong),AVI,frame)
end

function AVI_get_video_position(AVI::Ptr{avi_t},frame::Clong)
    ccall((:AVI_get_video_position,libavi),Clong,(Ptr{avi_t},Clong),AVI,frame)
end

function AVI_read_frame(AVI::Ptr{avi_t},vidbuf,keyframe)
    ccall((:AVI_read_frame,libavi),Clong,(Ptr{avi_t},Ptr{UInt8},Ptr{Cint}),AVI,vidbuf,keyframe)
end

function AVI_set_audio_position(AVI::Ptr{avi_t},byte::Clong)
    ccall((:AVI_set_audio_position,libavi),Cint,(Ptr{avi_t},Clong),AVI,byte)
end

function AVI_set_audio_bitrate(AVI::Ptr{avi_t},bitrate::Clong)
    ccall((:AVI_set_audio_bitrate,libavi),Cint,(Ptr{avi_t},Clong),AVI,bitrate)
end

function AVI_read_audio(AVI::Ptr{avi_t},audbuf::Ptr{UInt8},bytes::Clong)
    ccall((:AVI_read_audio,libavi),Clong,(Ptr{avi_t},Ptr{UInt8},Clong),AVI,audbuf,bytes)
end

function AVI_audio_codech_offset(AVI::Ptr{avi_t})
    ccall((:AVI_audio_codech_offset,libavi),Clong,(Ptr{avi_t},),AVI)
end

function AVI_audio_codecf_offset(AVI::Ptr{avi_t})
    ccall((:AVI_audio_codecf_offset,libavi),Clong,(Ptr{avi_t},),AVI)
end

function AVI_video_codech_offset(AVI::Ptr{avi_t})
    ccall((:AVI_video_codech_offset,libavi),Clong,(Ptr{avi_t},),AVI)
end

function AVI_video_codecf_offset(AVI::Ptr{avi_t})
    ccall((:AVI_video_codecf_offset,libavi),Clong,(Ptr{avi_t},),AVI)
end

function AVI_read_data(AVI::Ptr{avi_t},vidbuf::Ptr{UInt8},max_vidbuf::Clong,audbuf::Ptr{UInt8},max_audbuf::Clong,len::Ptr{Clong})
    ccall((:AVI_read_data,libavi),Cint,(Ptr{avi_t},Ptr{UInt8},Clong,Ptr{UInt8},Clong,Ptr{Clong}),AVI,vidbuf,max_vidbuf,audbuf,max_audbuf,len)
end

function AVI_print_error(str::AbstractString)
    ccall((:AVI_print_error,libavi),Cvoid,(Cstring,),str)
end

function AVI_strerror()
    ccall((:AVI_strerror,libavi),Ptr{UInt8},())
end

function AVI_syserror()
    ccall((:AVI_syserror,libavi),Ptr{UInt8},())
end

function AVI_scan(name::Ptr{UInt8})
    ccall((:AVI_scan,libavi),Cint,(Ptr{UInt8},),name)
end

function AVI_dump(name::Ptr{UInt8},mode::Cint)
    ccall((:AVI_dump,libavi),Cint,(Ptr{UInt8},Cint),name,mode)
end

function AVI_codec2str(cc::Int16)
    ccall((:AVI_codec2str,libavi),Ptr{UInt8},(Int16,),cc)
end

function AVI_file_check(import_file::Ptr{UInt8})
    ccall((:AVI_file_check,libavi),Cint,(Ptr{UInt8},),import_file)
end

function AVI_info(avifile::Ptr{avi_t})
    ccall((:AVI_info,libavi),Cvoid,(Ptr{avi_t},),avifile)
end

function AVI_max_size()
    ccall((:AVI_max_size,libavi),UInt64,())
end

function avi_update_header(AVI::Ptr{avi_t})
    ccall((:avi_update_header,libavi),Cint,(Ptr{avi_t},),AVI)
end

function AVI_set_audio_track(AVI::Ptr{avi_t},track::Cint)
    ccall((:AVI_set_audio_track,libavi),Cint,(Ptr{avi_t},Cint),AVI,track)
end

function AVI_get_audio_track(AVI::Ptr{avi_t})
    ccall((:AVI_get_audio_track,libavi),Cint,(Ptr{avi_t},),AVI)
end

function AVI_audio_tracks(AVI::Ptr{avi_t})
    ccall((:AVI_audio_tracks,libavi),Cint,(Ptr{avi_t},),AVI)
end

function AVI_chop_flip_image(image::Ptr{UInt8},height::Cint,width::Cint,cropped::Bool,scaled::Bool,converted::Bool)
    ccall((:chop_flip_image,libavi),Ptr{Cfloat},(Ptr{UInt8},Cint,Cint,Cint,Cint,Cint),image,height,width,cropped,scaled,converted)
end

function AVI_get_frame(cell_file::Ptr{avi_t},frame_num::Cint,cropped::Bool,scaled::Bool,converted::Bool)
    ccall((:get_frame,libavi),Ptr{Cfloat},(Ptr{avi_t},Cint,Cint,Cint,Cint),cell_file,frame_num,cropped,scaled,converted)
end
