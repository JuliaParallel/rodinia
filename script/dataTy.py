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
        # Perf
        self.times = []

        # dataTy dir
        self.data = {}
        # stderr of profiling mode
        self.log = ""
        #self.break_down = {}
        self.prof_time = 0
        self.is_large = False
