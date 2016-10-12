class Schedule:
    def __init__(self):
        self.processes = []

    def addProcess(self, process):
        self.processes.append(process)

class Process:
    def __init__(self):
        self.columns = []

    class MovingAverageProcess:
        def __init__(self):
            self.window = None



