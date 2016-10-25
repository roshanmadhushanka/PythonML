import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# load data
from sklearn.linear_model import LogisticRegression
from sklearn_pandas import DataFrameMapper

p_data = pd.read_csv('Training.csv')

response_column = 'RUL'
training_columns = list(p_data.columns)
training_columns.remove('RUL')
training_columns.remove('UnitNumber')
training_columns.remove('Time')

# Set mapper
df_mapper = DataFrameMapper([(training_columns, None), (response_column, None)])

# Train data - pandas to sklearn
train = df_mapper.fit_transform(p_data)

column_count = len(train[0, :])
# train
xx = train[:, 0:column_count-1]
# response
yy = train[:, column_count-1]

model = RandomForestRegressor(max_depth=20, n_estimators=50)
rfe = RFE(model, 3)
fit = rfe.fit(xx, yy)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_
