#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "helper_cuda.h"
#include <cuda.h>

struct Invocation {
    CUevent start;
    CUevent stop;

    Invocation() {
        checkCudaErrors(cuEventCreate(&start, 0));
        checkCudaErrors(cuEventCreate(&stop, 0));
    }
};

void measure_launch(const Invocation &inv) {
    checkCudaErrors(cuEventRecord(inv.start, 0));
}

void measure_finish(const Invocation &inv) {
    checkCudaErrors(cuEventRecord(inv.stop, 0));
}

std::map<std::string, std::vector<Invocation> > kernels;
Invocation measure_launch(const std::string &id) {
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

void measure_report() {
    for (std::map<std::string, std::vector<Invocation> >::const_iterator it =
             kernels.begin();
         it != kernels.end(); it++) {
        const std::string &id = it->first;
        const std::vector<Invocation> &invs = it->second;
        std::vector<float> times(invs.size());
        for (size_t i = 0; i < invs.size(); i++) {
            checkCudaErrors(cuEventSynchronize(invs[i].stop));
            checkCudaErrors(
                cuEventElapsedTime(&times[i], invs[i].start, invs[i].stop));
        }
        float min = *std::min_element(times.begin(), times.end());
        float sum = std::accumulate(times.begin(), times.end(), 0.0);
        float mean = sum / times.size();
        float sq_sum =
            std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
        float stdev = std::sqrt(sq_sum / times.size() - mean * mean);
        std::cout << "Kernel launch " << id << ": min " << int(1000 * min)
                  << " µs, mean " << int(1000 * mean) << " ± "
                  << int(1000 * stdev) << " µs" << std::endl;
    }
    kernels.clear();
}
