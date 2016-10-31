import pandas as pd
import h2o
import numpy as np
from sklearn_pandas import DataFrameMapper

def pandasToH2O(panda_frame):
    h2o.init()
    parsed_frame = h2o.H2OFrame(panda_frame)
    parsed_frame.set_names(list(panda_frame.columns))
    return parsed_frame

def pandasToSkLearn(panda_frame, training_features, response_feature):
    df_mapper = DataFrameMapper([(training_features, None), (response_feature, None)])
    parsed_frame = df_mapper.fit_transform(panda_frame)
    return parsed_frame

def h2oToNumpyArray(h2o_frame):
    h2o_frame = h2o_frame.get_frame_data()
    return np.array(map(float, h2o_frame.split("\n")[1:-1]))

def h2oToList(h2o_frame):
    h2o_frame = h2o_frame.get_frame_data()
    return h2o_frame.split("\n")[1:-1]