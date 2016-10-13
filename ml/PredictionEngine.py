class AnomalyFiltered:
    def __init__(self, anomaly_function="iqr", auto_encoder_layers=[1], auto_encoder=None):
        print ""