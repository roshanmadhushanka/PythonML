class Detail:
    def __init__(self, name=None, id=None, algorithm=None, rmse=None, mae=None):
        self._name = name
        self._id = id
        self._algorithm = algorithm
        self._rmse = rmse
        self._mae = mae


    def getModelDetails(self):
        return {"name": self._name, "id": self._id, "algorithm": self._algorithm, "rmse": self._rmse, "mae": self._mae}



