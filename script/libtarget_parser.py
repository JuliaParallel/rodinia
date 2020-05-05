#!/usr/bin/env python3

import re
from dataTy import dataTy

def parse(outputs, result):
    outputs.logs.append(result)
    state = 0
    out = {}
    for line in result.splitlines():
        if state == 0:
            if line.find("PerfRecord") >= 0:
                state = 1
        elif state == 1:
            newline = line.replace(',', " ")
            words = newline.split()
            if len(words) == 3:
                d = dataTy(words[0], int(words[1]), float(words[2]))
                out[words[0]] = d
            elif len(words) == 0:
                return
            else:
                print("Unknown data format")
                print(line)
    if len(out) < 1:
        print("No result parsed")
    outputs.prof_datas.append(out)
