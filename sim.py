# Data set preprocessor

from featureeng import Math, Select, DataSetSpecific, Progress
import pandas as pd
import numpy as np
import time

a = np.arange(6)
b = np.arange(6).reshape((2, 3))

print a
print b.reshape(-1, 1)
print a.reshape(1, -1)





