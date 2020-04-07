#!/usr/bin/env python3

import re
from dataTy import dataTy

def parse(output, result):
    output.log = result
    state = 0
    for line in result.splitlines():
        if state == 0:
            if line.find("PerfRecord") >= 0:
                state = 1
        elif state == 1:
            newline = line.replace(',', " ")
            words = newline.split()
            if len(words) == 3:
                d = dataTy(words[0], words[1], words[2])
                output.data[words[0]] = d
            elif len(words) == 0:
                return
            else:
                print("Unknown data format")
                print(line)
