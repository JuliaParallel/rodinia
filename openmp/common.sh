#!/usr/bin/env/env bash

if [ -z "${RUN_1D}" ]; then
    exe=$EXE
else
    exe=$EXE_1D
fi

if [ -z "${OMP_OFFLOAD}" ]; then
    if [ -z "${RUN_LARGE}" ]; then
        args=$EXE_ARG
    else
        args=$EXE_LARGE_ARG
    fi
else
    # Run omp
    if [ -z "${RUN_LARGE}" ]; then
        args=$EXE_OFFLOAD_ARG
    else
        args=$EXE_LARGE_OFFLOAD_ARG
    fi
fi

echo ./$exe $args
./$exe $args
