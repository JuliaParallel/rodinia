#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "helper_cuda.h"

struct Invocation {
    cudaEvent_t start;
    cudaEvent_t stop;

    Invocation() {
        checkCudaErrors(cudaEventCreate(&start, 0));
        checkCudaErrors(cudaEventCreate(&stop, 0));
    }
};

extern bool enabled;
extern bool profiling;
static inline void measure_launch(const Invocation &inv) {
    if (enabled && !profiling) {
        // lazy-start the profiler to get a narrow scope
        cudaProfilerStart();
        profiling = true;
    }
    checkCudaErrors(cudaEventRecord(inv.start, 0));
}

static inline void measure_finish(const Invocation &inv) {
    checkCudaErrors(cudaEventRecord(inv.stop, 0));
}

extern std::map<std::string, std::vector<Invocation> > kernels;
static inline Invocation measure_launch(const std::string &id) {
    if (kernels.find(id) == kernels.end())
        kernels[id] = std::vector<Invocation>();
    Invocation inv;
    kernels[id].push_back(inv);
    measure_launch(inv);
    return inv;
}

#define MEASURE(id, launch)                                                    \
    {                                                                          \
        Invocation inv = measure_launch(id);                                   \
        launch;                                                                \
        measure_finish(inv);                                                   \
    }

