#!/usr/bin/env python3
# TODO Add unify memory
import math
import re
from dataTy import dataTy

def parse(outputs, result):
    output = do_parse(result)
    if len(output) > 0:
        outputs.logs.append(result)
        #print("Valid nvprof")
        outputs.nvprof_datas.append(output)
        # FIXME add to list
        return 0
    #print("Invalid nvprof")
    return -1

def do_parse(result):
    state = ""
    kernelIdx=1
    out = {}
    for line in result.splitlines():
        first = False
        state, line = getState(line, state)
        tokens=re.split(" +", line)
        if state == "GPU":
            if line.find("[CUDA memcpy HtoD]") != -1:
                name = "GPU-H2D"
            elif line.find("[CUDA memcpy DtoH]") != -1:
                name = "GPU-D2H"
            else: # kernel
                kernel_name = tokens[7]
                name = "kernel-" + kernel_name
            d = dataTy(name, int(tokens[3]), toSec(tokens[2]), toSec(tokens[4]), toSec(tokens[5]), toSec(tokens[6]))
            out[name] = d
        elif state == "API":
            API_name = tokens[7]
            if API_name[0:2] != "cu":
                print("Error unknown cuda API: " + API_name)
            name = "API-" + API_name
            d = dataTy(name, int(tokens[3]), toSec(tokens[2]), toSec(tokens[4]), toSec(tokens[5]), toSec(tokens[6]))
            out[name] = d
    return out

def getState(line, state):
    section_list = {}
    section_list["GPU activities:"] = "GPU"
    section_list["API calls:"] = "API"
    # TODO unify

    for section in section_list:
        pos = line.find(section)
        if pos != -1:
            #print("Get section \"" + section + "\"")
            line = line[pos + len(section):]
            state = section_list[section]
            return state, line
    # if not found
    if line[1] != ' ' or line[2] != ' ':
        #print("Ignore section \"" + line + "\"")
        state = ""
    return state, line

def toSec(string):
    factor = 1
    number_end = len(string)
    if string.find("ms") != -1:
        number_end = string.find("ms")
        factor = math.pow(0.1,3)
    elif string.find("us") != -1:
        number_end = string.find("us")
        factor = math.pow(0.1,6)
    elif string.find("ns") != -1:
        number_end = string.find("ns")
        factor = math.pow(0.1,9)
    elif string.find("s") != -1:
        number_end = string.find("s")
        factor = 1
    t = float(string[0:number_end])
    return t
