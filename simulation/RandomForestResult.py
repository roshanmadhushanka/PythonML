from file import FileHandler
from datetime import datetime

class RandomForestResult:
    def __init__(self, job_no=0, mse=0.0, mae=0.0, ntrees =0, depth=0):
        self.time_stamp = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        self.job_no = job_no
        self.mse = mse
        self.mae = mae
        self.ntrees = ntrees
        self.depth = depth

    def save_to_file(self):
        result = {'time_stamp': self.time_stamp, 'job_no': self.job_no, 'mse': self.mse, 'mae':self.mae, 'ntrees': self.ntrees, 'depth': self.depth}
        FileHandler.write_json("result/rfr_result " + str(self.job_no) + ".txt", result)

    def load_from_file(self, file_name):
        return FileHandler.read_json(file_name)

    def get_result(self):
        return {'time_stamp': self.time_stamp, 'job_no': self.job_no, 'mse': self.mse, 'mae':self.mae}



