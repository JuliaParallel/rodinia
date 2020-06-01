#!/usr/bin/env python3

import os
import subprocess
import threading
import time
import datetime
import sys
import pickle
import copy

import nvprof_parser
import libtarget_parser
from dataTy import dataTy
from dataTy import Output

# global config
class Config:
    dry_run = False
    verifying = False
    enalbeProfile = True
    test_count = 3
    test_delay = 0.5
    _run_cmd = "./run".split()
    _verify_cmd = "./verify".split()
    _clean_cmd = "make clean".split()
    _make_cmd = "make".split()

# Indivial option
class Opt:
    def __init__(self):
        self.env = copy.deepcopy(os.environ)
        self.timeout = 1000 # (s) ??
        self.reg_run_cmd = Config._run_cmd
        self.verify_cmd = Config._verify_cmd
        self.make_cmd = Config._make_cmd
        self.clean_cmd = Config._clean_cmd

        self.timer_out = ".time_out"
        # %p is the pid
        self.nvprof_out_prefix = ".nvprof_out"
        self.nvprof_out = self.nvprof_out_prefix + "%p"

        self.timer_prefix = ("/usr/bin/time -o " + self.timer_out + " -f %e").split()
        self.nvprof_prefix = ("nvprof -u s --log-file " + self.nvprof_out + " --profile-child-processes").split()

        self.cuda = False

class Test:
    def __init__(self, name, path, opt):
        self.name = name  # config name
        self.root = path
        self.opt = opt
    def run (self, projs, Result):
        print(self.name)

        # init result
        self.result = {}
        for proj in projs:
            os.chdir(self.root)
            # Real run
            output = self.runOnProj(proj)
            print("* {0} {1}".format(proj, "Failed" if output.HasError() else ""))
            self.result[proj] = output

        Result[self.name] = self.result

    def runOnProj(self, proj):
        output = Output()
        os.chdir(proj)
        self.opt.run_cmd = self.opt.reg_run_cmd

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
        for i in range(Config.test_count):
            ret = self.runWithTimer(output)
            if output.HasError(ret):
                return output
            pass
        if Config.enalbeProfile == True:
            for i in range(Config.test_count):
                ret = self.runWithProfiler(output)
                if output.HasError(ret):
                    return output
                ret = self.runWithNvprof(output)
                if output.HasError(ret):
                    return output
        # make clean
        subprocess.run(self.opt.clean_cmd, capture_output=True)
        return output

    def runWithTimer(self, output):
        time.sleep(Config.test_delay)
        time_cmd = self.opt.timer_prefix + self.opt.run_cmd
        if Config.dry_run:
            print(time_cmd)
            return 0
        try:
            CP = subprocess.run(time_cmd, capture_output=True, timeout=self.opt.timeout, env=self.opt.env)
        except subprocess.TimeoutExpired:
            output.TL = True
            return -1
        ret, timing = self.checkRet(CP, output, True)
        if ret != 0:
            return -1
        output.times.append(timing)
        return 0
    def runWithNvprof(self, output):
        time.sleep(Config.test_delay)
        cmd = self.opt.nvprof_prefix + self.opt.run_cmd
        if Config.dry_run:
            print(cmd)
            return 0
        try:
            CP = subprocess.run(cmd, capture_output=True, timeout=self.opt.timeout, env=self.opt.env)
        except subprocess.TimeoutExpired:
            output.TL = True
            return -1
        ret = self.checkRet(CP, output, False)
        if ret != 0:
            return -1
        # Process output in multiple self.opt.nvprof_out
        out_files = [ f for f in os.listdir(os.getcwd()) if self.opt.nvprof_out_prefix in f]
        for f in out_files:
            with open(f, 'r') as out:
                nvprof_result = out.read()
                ret = nvprof_parser.parse(output, nvprof_result)
                if ret == 0:
                    for f in out_files:
                        os.remove(f)
                    return 0
        for f in out_files:
            os.remove(f)
        return -1

    def runWithProfiler(self, output):
        time.sleep(Config.test_delay)
        prof_cmd = self.opt.timer_prefix + self.opt.run_cmd
        env = copy.deepcopy(self.opt.env)
        env["PERF"] = "1"
        if Config.dry_run:
            print(prof_cmd)
            return 0
        try:
            CP = subprocess.run(prof_cmd, capture_output=True, timeout=self.opt.timeout, env=env)
        except subprocess.TimeoutExpired:
            output.TL = True
            return -1
        ret, timing = self.checkRet(CP, output, True)
        if ret != 0:
            return -1
        output.prof_times.append(timing)
        # Process output
        libtarget_parser.parse(output, CP.stderr.decode("utf-8"))
        return 0
    def checkRet(self, CP, output, getTime=False):
        # Store output
        output.stdouts += CP.stdout.decode("utf-8")
        output.stderrs += CP.stderr.decode("utf-8")
        if CP.returncode != 0 :
            output.RE = True
            print(CP.returncode)
            print(CP.stdout.decode("utf-8"))
            print(CP.stderr.decode("utf-8"))
            return -1
        if getTime:
            with open(self.opt.timer_out, 'r') as out:
                timing = float(out.read())
                os.remove(self.opt.timer_out)
                return 0, timing
            return -1, 0
        return 0

def run_cpu():
    opt1 = Opt()
    return Test("omp", os.path.join(rodinia_root, "openmp"), opt1)

def run_omp():
    opt2 = Opt()
    opt2.env["OFFLOAD"] = "1"
    return Test("omp-offload", os.path.join(rodinia_root, "openmp"), opt2)

def run_dce():
    opt_dce = Opt()
    opt_dce.env["OFFLOAD"] = "1"
    # Compile DCE with DC
    opt_dce.env["DC"] = "1"
    return Test("omp-dce", os.path.join(rodinia_root, "openmp"), opt_dce)

def run_bulk():
    opt3 = Opt()
    opt3.env["OFFLOAD"] = "1"
    opt3.env["OMP_BULK"] = "1"
    return Test("omp-offload-bulk", os.path.join(rodinia_root, "openmp"), opt3)

def run_dce_bulk():
    opt3 = Opt()
    opt3.env["OFFLOAD"] = "1"
    opt3.env["OMP_BULK"] = "1"
    opt3.env["DC"] = "1"
    return Test("dce-bulk", os.path.join(rodinia_root, "openmp"), opt3)

def run_at():
    opt4 = Opt()
    opt4.env["OFFLOAD"] = "1"
    opt4.env["OMP_BULK"] = "1"
    opt4.env["OMP_AT"] = "1"
    return Test("omp-offload-at", os.path.join(rodinia_root, "openmp"), opt4)

def run_dce_at():
    opt4 = Opt()
    opt4.env["OFFLOAD"] = "1"
    opt4.env["OMP_BULK"] = "1"
    opt4.env["OMP_AT"] = "1"
    opt4.env["DC"] = "1"
    return Test("dce-at", os.path.join(rodinia_root, "openmp"), opt4)

def run_cuda():
    opt5 = Opt()
    opt5.cuda = True
    return Test("cuda", os.path.join(rodinia_root, "cuda"), opt5)

def run_1d():
    opt6 = Opt()
    opt6.env["OFFLOAD"] = "1"
    opt6.env["RUN_1D"] = "1"
    return Test("omp-offload-1d", os.path.join(rodinia_root, "openmp"), opt6)

def Setup():
    Tests = []
    # Options
    #Config.dry_run = True
    #Config.verifying = 1

    projects = ["backprop", "kmeans", "myocyte", "pathfinder"]
    #projects = ["backprop", "myocyte", "pathfinder"]
    #projects = ["backprop", "pathfinder"]
    #projects = ["kmeans", "myocyte"]
    #projects = ["backprop", "kmeans",  "pathfinder"]
    #projects = ["myocyte"]
    #projects = ["pathfinder"]
    #projects = ["backprop"]

    # Final result
    Config.test_count = 1
    #Tests.append(run_cpu)
    #Tests.append(run_cuda)
    Tests.append(run_omp)
    Tests.append(run_1d)
    Tests.append(run_bulk)
    Tests.append(run_at)
    #Tests.append(run_dce)
    #Tests.append(run_dce_bulk)
    #Tests.append(run_dce_at)
    return Tests , projects

def Pickle(Result):
    # save result to pickle
    os.chdir(script_dir)
    now = datetime.datetime.now()
    timestamp = now.strftime("%d%m_%H%M")
    pickle_file = "./results/result_" + timestamp + ".p"
    with open(pickle_file, "wb") as f:
        pickle.dump(Result, f)
    # save as last result
    pickle_file = "./results/result.p"
    with open(pickle_file, "wb") as f:
        pickle.dump(Result, f)

if __name__ == "__main__":
    # Moving
    script_dir = os.path.dirname(os.path.realpath(__file__))
    rodinia_root = os.path.dirname(script_dir)
    os.chdir(rodinia_root)

    TestGens, projects = Setup()
    if Config.verifying == True:
        Config.enalbeProfile = False
        Config.test_count = 1
        Config._run_cmd = Config._verify_cmd
    else:
        os.environ["RUN_LARGE"] = "1"

    # print info
    if Config.dry_run == True:
        print("Dry-run, no data produced")
    if Config.enalbeProfile == False:
        print("Start in verify mode")
    print("Start running test with test count: {0}".format(Config.test_count))
    print("Projects: {0}".format(', '.join(projects)))
    Result = {}
    for TG in TestGens:
        test = TG()
        test.run(projects, Result)
    Pickle(Result)
