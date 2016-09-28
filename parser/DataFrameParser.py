import pandas as pd
import h2o
from sklearn_pandas import DataFrameMapper

def pandasToH2O(panda_frame=pd.DataFrame()):
    h2o.init()
    parsed_frame = h2o.H2OFrame(panda_frame)
    parsed_frame.set_names(list(panda_frame.columns))
    return parsed_frame

def pandasToSkLearn(panda_frame, training_features, response_feature):
    df_mapper = DataFrameMapper([(training_features, None), (response_feature, None)])
    parsed_frame = df_mapper.fit_transform(panda_frame)
    return parsed_frame