#!/usr/bin/env python3

import os
import subprocess
import threading
import time
import datetime

class Opt:
    timeout = 1000
    clean_cmd = "make clean".split()
    make_cmd = "make".split()
    run_cmd = "./run".split()
    verify_cmd = "./verify".split()
    timer_prefix = "/usr/bin/time -o .time -f %e".split()
    # Not working need to wrap to command
    #timer_suffix = "&> /dev/null".split()
    test_count = 5
    def __init__(self):
        self.env = os.environ
        self.offload = False
        self.bulk = False
        self.at = False

class Output:
    CE = False
    RE = False
    TL = False
    Failed=False

    # Perf
    times = 1


class Test:
    def __init__(self, name, path, opt):
        self.name = name
        self.root = path
        self.opt = opt

    def run (self,projs):
        print(self.name)
        for p in projs:
            os.chdir(self.root)
            print(p, end=',')
            output = self.runOnProj(p)
            # Print output
            if output.CE:
                print("CE", end=',')
            elif output.RE:
                print("RE", end=',')
            elif output.TL:
                print("TL", end=',')
            else:
                print("Pass", end=',')
            print("")

    def runOnProj(self, proj):
        output = Output()
        os.chdir(proj)

        #  TODO test verify

        # make clean
        subprocess.run(self.opt.clean_cmd, capture_output=True)
        # make
        CP = subprocess.run(self.opt.make_cmd, capture_output=True, env=self.opt.env)
        if CP.returncode != 0 :
            output.CE = True
            print(CP.stdout)
            print(CP.stderr)
            return output

        # Run/verify w/ timeout
        # Profiling and Get data
        for i in range(self.opt.test_count):
            ret = self.runWithTimerOrProfiler(output)
            if ret != 0:
                return output


        # make clean
        subprocess.run(self.opt.clean_cmd, capture_output=True)
        return output

    def runWithTimerOrProfiler(self, output):
        time.sleep(1)
        time_cmd = self.opt.timer_prefix + self.opt.run_cmd
        #print(time_cmd)
        try:
            CP = subprocess.run(time_cmd, capture_output=True, timeout=self.opt.timeout, env=self.opt.env)
            if CP.returncode != 0 :
                output.RE = True
                return -1
            else:
                f = open(".time", "r")
                out = f.read()
                print(float(out), end=',', flush=True)
                return 0
                #secs = datetime.datetime.strptime(CP.stderr.decode("utf-8"), '%M:%S.%f')
                #print(secs.time())
        except subprocess.TimeoutExpired:
            output.TL = True
            return -1

script_dir = os.path.dirname(os.path.realpath(__file__))
rodinia_root = os.path.dirname(os.path.dirname(script_dir))
print(rodinia_root)

os.chdir(rodinia_root)

projects = ["backprop", "kmeans", "pathfinder"]

Opt.test_count = 1
opt1 = Opt()
T1 = Test("omp", os.path.join(rodinia_root, "openmp"), opt1)
T1.run(projects)

opt2 = Opt()
opt2.env["OFFLOAD"] = "1"
T2 = Test("omp-offload", os.path.join(rodinia_root, "openmp"), opt2)
T2.run(projects)

opt3 = Opt()
opt3.env["OFFLOAD"] = "1"
opt3.env["OMP_BULK"] = "1"
T3 = Test("omp-offload-bulk", os.path.join(rodinia_root, "openmp"), opt3)
T3.run(projects)

opt4 = Opt()
opt4.env["OFFLOAD"] = "1"
opt4.env["OMP_BULK"] = "1"
opt4.env["OMP_AT"] = "1"
T4 = Test("omp-offload-at", os.path.join(rodinia_root, "openmp"), opt4)
T4.run(projects)

opt5 = Opt()
T5 = Test("cuda", os.path.join(rodinia_root, "cuda"), opt5)
T5.run(projects)

