import pandas as pd
from dataprocessor import Process
from sklearn.cross_validation import cross_val_score
from sklearn_pandas import DataFrameMapper
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor

def predictionEngine(models, test_data):
    predictions = []
    for model in models:
        predictions.append(model.predict(test_data))

    for i in range







# Load data
p_train = pd.read_csv('Training.csv')
p_test = pd.read_csv('Testing.csv')

all_columns = list(p_train.columns)
removing_columns = ['UnitNumber', 'Time', 'RUL', 'Setting1', 'Setting2', 'Setting3']
selected_columns = [x for x in all_columns if x not in removing_columns]

# Filter training dataset
p_noise_filtered = Process.filterData(panda_frame=p_train, columns=[], removal_method="iqr", threshold=3)

removing_columns = ['UnitNumber', 'Time', 'RUL']
training_columns = [x for x in all_columns if x not in removing_columns]
response_column = 'RUL'

# Set mapper
df_mapper = DataFrameMapper([(training_columns, None), (response_column, None)])

# Pandas to sklearn
train = df_mapper.fit_transform(p_noise_filtered)
test = df_mapper.fit_transform(p_test)

# [row : column]
column_count = len(train[0, :])

# train
x = train[:, 0:column_count-1] # Features
y = train[:, column_count-1]   # Response

# test
tX = test[:, 0:column_count-1] # Test
tY = test[:, column_count-1]   # Ground truth

# # Define Model
# rf = RandomForestRegressor(max_depth=20, n_estimators=50)
#
# scores = cross_val_score(rf, x, y, cv=10, scoring='mean_squared_error')

index = list(p_train.index)
kFold = KFold(len(index), n_folds=10)

models = []
for train, test in kFold:
    rf = RandomForestRegressor(max_depth=20, n_estimators=50)
    pk_train = df_mapper.fit_transform(p_train.drop(test))
    x = pk_train[:, 0:column_count - 1]  # Features
    y = pk_train[:, column_count - 1]  # Response
    rf.fit(x,y)
    models.append(rf)

for model in models:
    result = model.predict(tX)
    print result
    print "-------------------"






'''
Refrences
---------
Cross validation scoring : http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

[-1175.3514944  -1174.44067313 -1376.59696607 -1097.38498768  -784.56872687
 -2409.30859195 -4031.26087206 -1137.77811166 -2940.57022154 -2426.16439006]

 [-1241.07997456 -1141.25818674 -1543.38391009  -846.61538072 -1006.27055745
 -1961.33341445 -4401.36870443 -1143.33320041 -2830.0667297  -2634.04319578]

'''