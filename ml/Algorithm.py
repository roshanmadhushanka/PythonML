import pandas as pd
import numpy as np
import h2o
from h2o.estimators import H2OAutoEncoderEstimator

class AutoEncoder:
    def __init__(self, layers=[1], activation='Rectifier', epochs=100, anomaly_remove_function='iqr', validation_ratio=0.1):
        h2o.init()
        self.layers = layers
        self.activation = activation
        self.epochs = epochs
        self.anomaly_remove_function = anomaly_remove_function
        self.validation_ratio = validation_ratio

    def filterData(self, series=pd.DataFrame(), unwanted_columns = []):
        # Setup autoencoder model
        anomaly_model = H2OAutoEncoderEstimator(
            activation=self.activation,
            hidden=self.layers,
            l1=1e-4,
            epochs=self.epochs,
        )

        # Split data frame
        pValidate = series.sample(frac=self.validation_ratio, random_state=200)
        pTrain = series.drop(pValidate.index)

        # Convert pandas to h2o frame - for anomaly detection
        hValidate = h2o.H2OFrame(pValidate)
        hValidate.set_names(list(pValidate.columns))

        hTrain = h2o.H2OFrame(pTrain)
        hTrain.set_names(list(pTrain.columns))

        # Select columns
        train_columns = [x for x in list(series.columns) if x not in unwanted_columns]

        # Train model
        anomaly_model.train(x=train_columns, training_frame=hTrain, validation_frame=hValidate)

        # Get reconstruction error
        reconstruction_error = anomaly_model.anomaly(test_data=hTrain, per_feature=False)
        error_str = reconstruction_error.get_frame_data()
        err_list = map(float, error_str.split("\n")[1:-1])
        err_list = np.array(err_list)

        if self.anomaly_remove_function == 'iqr':
            print ""

    def iqrFilter(self, series=pd.DataFrame(), reconstruction_error_array=np.array(), outlier_type='extreme'):
        ratio = 3.0
        if outlier_type == 'extreme':
            ratio = 3.0
        elif outlier_type == 'mild':
            ratio = 1.5

        q25 = np.percentile(reconstruction_error_array, 25)
        q75 = np.percentile(reconstruction_error_array, 75)
        iqr = q75 - q25

        filtered_train = pd.DataFrame()
        count = 0
        for i in range(len(series)):
            if abs(reconstruction_error_array[i] - q75) < 2 * iqr:
                df1 = series.iloc[i, :]
                filtered_train = filtered_train.append(df1, ignore_index=True)
                count += 1


        print filtered_train







