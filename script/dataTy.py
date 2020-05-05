#!/usr/bin/env python3
class dataTy:
    def __init__(self):
        self.Name = ""
        self.Count = 0
        self.val = 0
    def __init__(self, name, count : int, val : float):
        self.Name = name
        if not isinstance(count, int):
            print("dataTy count is not int")
        if not isinstance(val, float):
            print("dataTy val is not float")
        self.Count = count
        self.Value = val

# Per config, per proj
class Output:
    def __init__(self):
        self.CE = False
        self.RE = False
        self.TL = False
        self.Failed=False
        # output
        self.stdouts=""
        self.stderrs=""
        # list of execution time
        self.times = []
        # list of profiling result(dataTy for omp) dict
        self.prof_datas = []
        self.prof_times = []
        # averaged profiling result
        self.prof_data = {}
        self.prof_time = 0
        # stderr of profiling mode
        # TODO
        self.logs = []

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
    def getAvg(str_list):
        if len(str_list) < 1:
            return 0
        n = 0
        sum = 0
        for s in str_list:
            f = float(s)
            sum += f
            n += 1
        return sum/n
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
            proj_out = result[config]
            for proj in proj_out:
                if ResultHelper.hasError(proj_out[proj]) == True:
                    del proj_out[proj]
                    print(config + "-" + proj + " has error, removed")
        # Avg all profiling result
        for config in result:
            proj_out = result[config]
            for proj in proj_out:
                output = proj_out[proj]
                the_sum = 0
                # avg time
                for t in output.prof_times:
                    the_sum += t
                output.prof_time += the_sum / len(output.prof_times)

                # avg data
                names = list(output.prof_datas[0].keys())
                name_count = len(output.prof_datas)
                for name in names:
                    the_sum = 0
                    for pdata in output.prof_datas:
                        the_sum += pdata[name].Value
                    val = the_sum / name_count
                    count = output.prof_datas[0][name].Count
                    output.prof_data[name] = dataTy(name, count, val)

        # Substract runtime with others
        for config in result:
            proj_out = result[config]
            for proj in proj_out:
                pdata = proj_out[proj].prof_data
                member = ["Kernel", "H2DTransfer", "D2HTransfer", "UpdatePtr"]
                get = pdata.get("Runtime")
                if get == None:
                    del proj_out[proj]
                    print(config + "-" + proj + " does not has key - Runtime, removed")
                    
                sumup = get.Value
                for m in member:
                    sumup -= pdata[m].Value
                if sumup < 0:
                    sumup = 0
                pdata["Runtime"].Value = sumup
        # Gen other
        for config in result:
            proj_out = result[config]
            for proj in proj_out:
                pdata = proj_out[proj].prof_data
                member = ["Kernel", "H2DTransfer", "D2HTransfer", "UpdatePtr", "Runtime"]
                sumup = proj_out[proj].prof_time
                for m in member:
                    sumup -= pdata[m].Value
                if sumup < 0:
                    sumup = 0
                d = dataTy("Other", 1, sumup)
                pdata["Other"] = d
