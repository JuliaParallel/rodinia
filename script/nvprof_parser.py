#!/usr/bin/env python3

import re

def parse(output, result):
    state=0
    kernelIdx=1
    for line in result.splitlines():
        first = False
        offset = 0
        # result.data -> dict
        if state == 0:
            # Get "GPU activities:"
            if line.find("GPU activities:") != -1:
                state = 1
                first = True
        if state == 1:
            tokens=re.split(" +", line)
            if first:
                offset = 2
            # Get API calls
            if line.find("API calls") != -1:
                state = 2
                first = True
            elif line.find("CUDA memcpy HtoD") != -1:
                output.data["H2D.count"] = tokens[3 + offset]
                output.data["H2D.total"] = tokens[2 + offset]
            elif line.find("CUDA memcpy DtoH") != -1:
                output.data["D2H.count"] = tokens[3 + offset]
                output.data["D2H.total"] = tokens[2 + offset]
            elif line.find("CUDA memcpy DtoH") != -1:
                output.data["D2H.count"] = tokens[3 + offset]
                output.data["D2H.total"] = tokens[2 + offset]
            elif line.find("(") != -1:
                output.data["kernel" + str(kernelIdx) + ".count"] = tokens[3 + offset]
                output.data["kernel" + str(kernelIdx) + ".total"] = tokens[2 + offset]
                kernelIdx += kernelIdx
        if state == 2:
            tokens=re.split(" +", line)
            if first:
                offset = 2
            # list of cared API
            APIs = ["cudaMalloc", "cudaMemcpy", "cudaFree", "cudaThreadSynchronize"]
            for API in APIs:
                if line.find(API) != -1:
                    output.data[API + ".count"] = tokens[3 + offset]
                    output.data[API + ".total"] = tokens[2 + offset]
                    break
    #for item in output.data:
    #    print(item, end=' ')
    #    print(output.data[item])
