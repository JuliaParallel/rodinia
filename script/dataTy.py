#!/usr/bin/env python3
import sys
class dataTy:
    Name = ""
    Count = 0
    val = 0
    min = 0
    max = 0
    avg = 0
    def __init__(self, name, count : int, val : float, _avg = 0, _min = 0, _max = 0):
        self.Name = name
        if not isinstance(count, int):
            print("dataTy count is not int")
        if not isinstance(val, float):
            print("dataTy val is not float")
        self.Count = count
        self.Value = val
        self.max = _max
        self.min = _min
        self.avg = _avg

# Per config, per proj
class Output:
    def __init__(self):
        self.CE = False
        self.RE = False
        self.TL = False
        self.Failed = False
        self.Error = False
        # output
        self.stdouts=""
        self.stderrs=""
        # list of execution time
        self.times = []
        self.time = 0 # avg
        # list of profiling result(dataTy for omp) dict
        self.nvprof_datas = []
        self.nvprof_times = []
        self.prof_datas = []
        self.prof_times = []
        # Finalized profiling result
        self.prof_data = {}
        self.nvprof_data = {}
        self.prof_time = 0
        # stderr of profiling mode
        # TODO
        self.logs = []
    def HasError(self, ret = 0):
        if ret != 0 or self.CE or self.RE or self.TL or self.TL or self.Failed:
            self.Error = True
            return True
        return False

class ResultHelper:
    # get Projs of first config
    def getProjs(result):
        if len(result) < 1:
            print("No result")
            return []
        for config in result:
            ret = []
            output_of_proj = result[config]
            for proj in output_of_proj:
                ret.append(proj)
            return ret
    def getConfigs(result):
        if len(result) < 1:
            print("No result")
            return []
        ret = []
        for config in result:
            ret.append(config)
        return ret
    def getAvg(flist):
        if len(flist) < 1:
            return 0
        n = len(flist)
        sum = 0
        for s in flist:
            sum += s
        return sum / n
    def hasError(output):
        if output.CE:
            return True
        if output.RE:
            return True
        if output.TL:
            return True
        if output.Failed:
            return True
        return False
    def getErrorOrAvgTime(output):
        if output.CE:
            return "CE"
        if output.RE:
            return "RE"
        if output.TL:
            return "TL"
        if output.Failed:
            return "WA"
        return sum(output.times)/ len(output.times)

    def preprocessing(result):
        # Remove data if the result is error
        for config in result:
            for proj in list(result[config]):
                proj_out = result[config]
                if ResultHelper.hasError(proj_out[proj]) == True:
                    #del proj_out[proj]
                    print(config + "-" + proj + " has error, removed??")
        # vg times
        for config in result:
            for proj in result[config]:
                output = result[config][proj]
                output.time = sum(output.times)/ len(output.times)
        # Avg all profiling result
        for config in result:
            proj_out = result[config]
            for proj in proj_out:
                output = proj_out[proj]
                if len (output.prof_times) == 0:
                    print("Ignore no profiling entry: " + config + ":" + proj)
                    continue
                the_sum = 0
                # avg time
                if len(output.prof_times) == 0:
                    print(config)
                    print(proj)
                    output.prof_time = 0

                for t in output.prof_times:
                    the_sum += t
                output.prof_time += the_sum / len(output.prof_times)

                # avg data
                names = list(output.prof_datas[0].keys())
                for name in names:
                    the_sum = 0
                    for pdata in output.prof_datas:
                        the_sum += pdata[name].Value
                    val = the_sum / len(output.prof_datas)
                    count = output.prof_datas[0][name].Count
                    output.prof_data[name] = dataTy(name, count, val)
                # Do the nvprof data
                nvprof_names = list(output.nvprof_datas[0].keys())
                # Do the kernel first
                kernel_sum = 0
                kernel_count = 0
                for name in nvprof_names:
                    if name[:7] == "kernel-":
                        for pdata in output.nvprof_datas:
                            kernel_sum += pdata[name].Value
                        kernel_count += output.nvprof_datas[0][name].Count
                # FIXME put into nvprof_data??
                output.prof_data["GPU-kernel"] = dataTy("GPU-kernel", kernel_count, kernel_sum/len(output.nvprof_datas))
                for name in nvprof_names:
                    if name[:7] == "kernel-":
                        continue
                    the_sum = 0
                    for pdata in output.nvprof_datas:
                        the_sum += pdata[name].Value # possible has no same entry??
                    val = the_sum / len(output.nvprof_datas)
                    count = output.nvprof_datas[0][name].Count
                    output.prof_data[name] = dataTy(name, count, val)

        # Substract runtime with others
        for config in result:
            proj_out = result[config]
            for proj in proj_out:
                pdata = proj_out[proj].prof_data
                member = ["Kernel", "H2DTransfer", "D2HTransfer", "UpdatePtr"]
                get = pdata.get("Runtime")
                if get == None:
                    #del proj_out[proj]
                    print(config + "-" + proj + " does not has key - Runtime, Abort")
                    sys.exit()


                sumup = get.Value
                for m in member:
                    sumup -= pdata[m].Value
                if sumup < 0:
                    sumup = 0
                # Store OMP Runtime in new attr
                pdata["OMPRuntime"] = dataTy("OMPRuntime", 1, sumup)
        # Gen other
        for config in result:
            proj_out = result[config]
            for proj in proj_out:
                pdata = proj_out[proj].prof_data
                member = ["Kernel", "H2DTransfer", "D2HTransfer", "UpdatePtr", "OMPRuntime"]
                sumup = proj_out[proj].prof_time
                for m in member:
                    sumup -= pdata[m].Value
                if sumup < 0:
                    sumup = 0
                d = dataTy("Other", 1, sumup)
                pdata["Other"] = d
        # Avg times and store into pdata
        for config in result:
            proj_out = result[config]
            for proj in proj_out:
                pdata = proj_out[proj].prof_data
                AvgTime = ResultHelper.getAvg(proj_out[proj].times)
                pdata["Times"] = dataTy("Times", 1, AvgTime)
    def merge_result(r1, r2):
        # r1 is the master
        for config in r1:
            pass
    def invalid(result):
        projs = ResultHelper.getProjs(result)
        configs = ResultHelper.getConfigs(result)
        proj_count = len(projs)
        config_count = len(configs)
        if config_count == 0:
            print("invalid")
            return True
        if proj_count == 0:
            print("invalid")
            return True
        return False

    def getNormFactors(result, metrics, norm):
        projs = ResultHelper.getProjs(result)
        configs = ResultHelper.getConfigs(result)
        factors = {p: 1 for p in projs}
        # Norm to first
        if norm == True:
            config = configs[0]
            for p in result[config]:
                sum = 0
                for m in metrics:
                    sum += result[config][p].prof_data[m].Value
                factors[p] = 100/sum
        return factors
