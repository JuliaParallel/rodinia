#pragma once

#include "kernelprofile.h"

bool enabled = false;
bool profiling = false;

std::map<std::string, std::vector<Invocation> > kernels;

void measure_enable() {
    enabled = true;
}

void measure_report(std::string benchmark) {
    if (enabled && profiling) {
        cudaProfilerStop();
        profiling = false;
    }

    for (std::map<std::string, std::vector<Invocation> >::const_iterator it =
             kernels.begin();
         it != kernels.end(); it++) {
        const std::string &id = it->first;
        const std::vector<Invocation> &invs = it->second;

        // gather data
        for (size_t i = 0; i < invs.size(); i++) {
            float elapsed;
            checkCudaErrors(cudaEventSynchronize(invs[i].stop));
            checkCudaErrors(cudaEventElapsedTime(&elapsed, invs[i].start, invs[i].stop));

            // print csv
            std::cout << benchmark << "," << id << "," << 1000*elapsed << "\n";
        }
    }
    std::cout << std::flush;

    kernels.clear();
}