#pragma once

#include "helper_cuda.h"
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "nvToolsExt.h"

extern bool enabled;
extern bool started;

#define PROFILE(launch)                                                        \
    {                                                                          \
        if (enabled && !started) {                                             \
            started = true;                                                    \
            checkCudaErrors(cudaProfilerStart());                              \
        }                                                                      \
        launch;                                                                \
    }
