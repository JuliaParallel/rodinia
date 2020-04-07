#!/usr/bin/env python3

import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt

from dataTy import dataTy
from dataTy import Output


class Printer:
    def __init__(self, result):
        for config in result:
            output_of_proj = result[config]
            for proj in output_of_proj:
                output = output_of_proj[proj]
                # Config-proj
                print("{0}-{1}, {2}".format(config, proj, output.times), end=' ')
                if output.CE:
                    print("CE")
                    continue
                if output.RE:
                    print("RE")
                    continue
                if output.TL:
                    print("TL")
                    continue
                if output.Failed:
                    print("WA")
                else:
                    print("Pass")
                print("Profiling Times: " + str(output.prof_time))
                for name in output.data:
                    d = output.data[name]
                    print("  {0:20}| {1:10}| {2}".format(name, d.Count, d.Value))
                    continue

class config:
    width = 0.35

class StackChartPrinter:
    def plot(self, name):
        N = 5
        menMeans = [1, 2, 3, 4, 5]
        womenMeans = menMeans
        #womenMeans = (25, 32, 34, 20, 25)
        ind = np.arange(N)    # the x locations for the groups
        width = 0.35       # the width of the bars: can also be len(x) sequence

        p1 = plt.bar(ind, menMeans, config.width+0.1)
        p2 = plt.bar(ind, womenMeans, config.width, bottom=menMeans)

        plt.ylabel('Execution Time(sec)')
        plt.title('Breakdown')
        plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
        plt.yticks(np.arange(0, 81, 10))
        plt.legend((p1[0], p2[0]), ('Men', 'Women'))

        plt.show()

if __name__ == "__main__":
    # Read from pickle
    # FIXME
    pfile = "./results/result.p"
    if len(sys.argv) > 1:
        # first arg is filename
        pfile = sys.argv[1]
    result = pickle.load(open(pfile, "rb"))
    print("Open " + pfile)
    Printer(result)
    StackChartPrinter().plot("NN")
