#!/usr/bin/env/env bash

if [ -z "${RUN_1D}" ]; then
    echo "Runing 1D version"
    exe=$EXE
else
    exe=$EXE_1D
fi

# FIXME seperate enable var and arg var
# FIXME I donot understand previous line
args=$EXE_ARG

if [ ! -z "${OFFLOAD}" ]; then
    if [ ! -z "${EXE_OFFLOAD_ARG}" ]; then args=$EXE_OFFLOAD_ARG; fi
fi

if [ ! -z "${RUN_LARGE}" ]; then
    if [ ! -z "${EXE_LARGE_ARG}" ]; then args=$EXE_LARGE_ARG; fi

    if [ ! -z "${OFFLOAD}" ]; then
        if [ ! -z "${EXE_LARGE_OFFLOAD_ARG}" ]; then args=$EXE_LARGE_OFFLOAD_ARG; fi
    fi
fi

echo ./$exe $args
./$exe $args
