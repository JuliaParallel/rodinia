#pragma once

#include <cmath>

#include "kernelprofile.h"

std::map<std::string, std::vector<Invocation> > kernels;

void measure_report(std::string benchmark) {
    for (std::map<std::string, std::vector<Invocation> >::const_iterator it =
             kernels.begin();
         it != kernels.end(); it++) {
        const std::string &id = it->first;
        const std::vector<Invocation> &invs = it->second;

        // gather data
        std::vector<float> times(invs.size());
        for (size_t i = 0; i < invs.size(); i++) {
            checkCudaErrors(cudaEventSynchronize(invs[i].stop));
            checkCudaErrors(
                cudaEventElapsedTime(&times[i], invs[i].start, invs[i].stop));
        }
        std::transform(
            times.begin(), times.end(), times.begin(),
            std::bind1st(std::multiplies<float>(), 1000)); // ms to μs

        // calculate metrics
        float min = *std::min_element(times.begin(), times.end());
        float max = *std::max_element(times.begin(), times.end());
        float sum = std::accumulate(times.begin(), times.end(), 0.0);
        float mean = sum / times.size();
        float sq_sum =
            std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
        float stdev = std::sqrt(sq_sum / times.size() - mean * mean);
        std::nth_element(times.begin(), times.begin() + times.size() / 2,
                         times.end());
        float median = times[times.size() / 2];

        // print csv
        std::cout << benchmark << "," << id << "," << times.size() << "," << min
                  << "," << median << "," << mean << "," << max << "," << stdev
                  << std::endl;
    }
    kernels.clear();
}