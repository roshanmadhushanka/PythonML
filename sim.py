# Data set preprocessor

from featureeng import Math, Select, DataSetSpecific, Progress
import pandas as pd
import numpy as np
import time


training_frame = pd.read_csv("train.csv")
Math.entropy(training_frame['Sensor2'], 250)





