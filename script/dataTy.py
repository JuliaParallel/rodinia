#!/usr/bin/env python3
class dataTy:
    def __init__(self):
        self.Name = ""
        self.Count = 0
        self.val = 0
    def __init__(self, name, count, val):
        self.Name = name
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
        # Perf
        self.times = []
        # dataTy dir
        self.data = {}
        # stderr of profiling mode
        self.log = ""
        #self.break_down = {}
        self.prof_time = 0
        self.is_large = False

class ResultHelper:
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

    def preprocessing(result):
        # Remove data if the result is error
        for config in result:
            proj_out = result[config]
            for proj in proj_out:
                if ResultHelper.hasError(proj_out[proj]) == True:
                    del proj_out[proj]
                    print(config + "-" + proj + " has error, removed")
        # Substract runtime with others
        for config in result:
            proj_out = result[config]
            for proj in proj_out:
                data = proj_out[proj].data
                member = ["Kernel", "H2DTransfer", "D2HTransfer", "UpdatePtr"]
                get = data.get("Runtime")
                sumup = float(get.Value)
                for m in member:
                    sumup -= float(data[m].Value)
                if sumup < 0:
                    sumup = 0
                data["Runtime"].Value = str(sumup)
        # Gen other
        for config in result:
            proj_out = result[config]
            for proj in proj_out:
                data = proj_out[proj].data
                member = ["Kernel", "H2DTransfer", "D2HTransfer", "UpdatePtr", "Runtime"]
                sumup = proj_out[proj].prof_time
                for m in member:
                    sumup -= float(data[m].Value)
                if sumup < 0:
                    sumup = 0
                d = dataTy("Other", 1, str(sumup))
                data["Other"] = d
