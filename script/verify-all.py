#!/usr/bin/env python3

import os
import subprocess
import threading
import time
import datetime
import sys

import nvprof_parser

class output_file(object):
    name = "test.csv"
    name = ""
    fd = ""
    bd_name = "bd.csv"
    bd_fd =""

if len(output_file.name) > 0:
    output_file.fd=open(output_file.name,"w+")
if len(output_file.bd_name) > 0:
    output_file.bd_fd=open(output_file.bd_name,"w+")

class Opt:
    # Not working need to wrap to command
    #timer_suffix = "&> /dev/null".split()
    test_count = 3
    prof = True
    def __init__(self):
        self.env = os.environ
        self.offload = False
        self.bulk = False
        self.at = False
        self.timeout = 1000
        self.clean_cmd = "make clean".split()
        self.make_cmd = "make".split()
        self.run_cmd = "./run".split()
        self.verify_cmd = "./verify".split()
        self.timer_prefix = "/usr/bin/time -o .time -f %e".split()
        self.nvprof_prefix = "nvprof --profile-child-processes".split()
        self.cuda = False

class Output:
    def __init__(self):
        self.CE = False
        self.RE = False
        self.TL = False
        self.Failed=False
        # Perf
        self.times = []
        self.data = {}


def fprint(str, end='\n'):
    if len(output_file.name) > 0:
        output_file.fd.write(str)
        output_file.fd.write(end)
        # write to file
    else:
        print(str, end=end)
        sys.stdout.flush()
def bdprint(str, end='\n'):
    if len(output_file.bd_name) > 0:
        output_file.bd_fd.write(str)
        output_file.bd_fd.write(end)
        # write to file
    else:
        return

class Test:
    def __init__(self, name, path, opt):
        self.name = name
        self.root = path
        self.opt = opt
    def run (self, projs):
        fprint(self.name)
        for p in projs:
            os.chdir(self.root)
            fprint(p, end=',')
            output = self.runOnProj(p)
            # Print output
            if output.CE:
                fprint("CE", end=',')
            elif output.RE:
                fprint("RE", end=',')
            elif output.TL:
                fprint("TL", end=',')
            else:
                fprint("Pass", end=',')
            for t in output.times:
                fprint(str(t), end=',')
            fprint("")

            # print break down
            bdprint(self.name+"-"+p)
            for d in output.data:
                bdprint(d, end = ",")
                bdprint(output.data[d])
            bdprint("")

        fprint("")

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
            ret = self.runWithTimer(output)
            if ret != 0:
                return output

        if self.opt.prof:
            ret = self.runWithProfiler(output)
            if ret != 0:
                return output

        # make clean
        subprocess.run(self.opt.clean_cmd, capture_output=True)
        return output

    def runWithTimer(self, output):
        time.sleep(1)
        time_cmd = self.opt.timer_prefix + self.opt.run_cmd
        #print(time_cmd)
        try:
            CP = subprocess.run(time_cmd, capture_output=True, timeout=self.opt.timeout, env=self.opt.env)
            if CP.returncode != 0 :
                output.RE = True
                return -1
            elif os.path.exists(".time"):
                    f = open(".time", "r")
                    f = float(f.read())
                    output.times.append(f)
                    os.remove(".time")
                    return 0
            else:
                print("No .time gen")
                return -1
        except subprocess.TimeoutExpired:
            output.TL = True
            return -1
    def runWithProfiler(self, output):
        time.sleep(1)
        prof_cmd = self.opt.run_cmd
        if self.opt.cuda:
            prof_cmd = self.opt.nvprof_prefix + self.opt.run_cmd
        else:
            self.opt.env["Perf"] = "1"

        try:
            CP = subprocess.run(prof_cmd, capture_output=True, timeout=self.opt.timeout, env=self.opt.env)
            if CP.returncode != 0 :
                output.RE = True
                return -1
            else:
                result = CP.stderr.decode("utf-8")
                if self.opt.cuda:
                    nvprof_parser.parse(output, result)
                else:
                    output.data["1"] = result
                return 0
        except subprocess.TimeoutExpired:
            output.TL = True
            return -1


script_dir = os.path.dirname(os.path.realpath(__file__))
rodinia_root = os.path.dirname(script_dir)
os.chdir(rodinia_root)

projects = ["backprop", "kmeans", "myocyte", "pathfinder"]
#projects = ["backprop"]

def run1():
    opt1 = Opt()
    T1 = Test("omp", os.path.join(rodinia_root, "openmp"), opt1)
    T1.run(projects)

def run2():
    opt2 = Opt()
    opt2.env["OFFLOAD"] = "1"
    T2 = Test("omp-offload", os.path.join(rodinia_root, "openmp"), opt2)
    T2.run(projects)

def run3():
    opt3 = Opt()
    opt3.env["OFFLOAD"] = "1"
    opt3.env["OMP_BULK"] = "1"
    T3 = Test("omp-offload-bulk", os.path.join(rodinia_root, "openmp"), opt3)
    T3.run(projects)

def run4():
    opt4 = Opt()
    opt4.env["OFFLOAD"] = "1"
    opt4.env["OMP_BULK"] = "1"
    opt4.env["OMP_AT"] = "1"
    T4 = Test("omp-offload-at", os.path.join(rodinia_root, "openmp"), opt4)
    T4.run(projects)

def run5():
    opt5 = Opt()
    opt5.cuda = True
    T5 = Test("cuda", os.path.join(rodinia_root, "cuda"), opt5)
    T5.run(projects)

def run6():
    opt6 = Opt()
    opt6.env["OFFLOAD"] = "1"
    opt6.run_cmd = "./run1d".split()
    T6 = Test("omp-offload-1d", os.path.join(rodinia_root, "openmp"), opt6)
    T6.run(projects)

Opt.test_count = 2
#Opt.prof = False
#run1()
#run5()
run2()
run6()
#run3()
#run4()

if len(output_file.name) > 0:
    output_file.fd.close()
if len(output_file.bd_name) > 0:
    output_file.bd_fd.close()
