#pragma once

#include "kernelprofile.h"

std::map<std::string, std::vector<Invocation> > kernels;

void measure_report() {
    std::cout << "\nKernel profile report for " << kernels.size() << " kernels:" << std::endl;
    for (std::map<std::string, std::vector<Invocation> >::const_iterator it =
             kernels.begin();
         it != kernels.end(); it++) {
        const std::string &id = it->first;
        const std::vector<Invocation> &invs = it->second;
        std::vector<float> times(invs.size());
        for (size_t i = 0; i < invs.size(); i++) {
            checkCudaErrors(cudaEventSynchronize(invs[i].stop));
            checkCudaErrors(
                cudaEventElapsedTime(&times[i], invs[i].start, invs[i].stop));
        }
        float min = *std::min_element(times.begin(), times.end());
        float sum = std::accumulate(times.begin(), times.end(), 0.0);
        float mean = sum / times.size();
        float sq_sum =
            std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
        float stdev = std::sqrt(sq_sum / times.size() - mean * mean);
        std::cout << " - " << id << " (x " << invs.size() << "): min " << int(1000 * min)
                  << " µs, mean " << int(1000 * mean) << " ± "
                  << int(1000 * stdev) << " µs" << std::endl;
    }
    kernels.clear();
}