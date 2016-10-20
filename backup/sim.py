import pandas as pd
import numpy as np

from featureeng import Select

testing_frame = pd.read_csv("test.csv")
indices = Select.indices_seperate(feature_name="UnitNumber", data_frame=testing_frame)
filtered_frame = pd.DataFrame(columns=testing_frame.columns)

# Add last index to the indices
indices = np.insert(indices, len(indices), len(testing_frame['UnitNumber']) - 1, axis=0)
for index in indices:
    filtered_frame.loc[len(filtered_frame)] = testing_frame.loc[index]

del filtered_frame['UnitNumber']
del filtered_frame['Time']

filtered_frame.to_csv("Siddhi.csv", index=False)


