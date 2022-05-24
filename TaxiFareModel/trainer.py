# imports
import numpy as np

# sklearn
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# own functions
from data import get_data, clean_data
from encoders import DistanceTransformer, TimeFeaturesEncoder

# mlflow
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient

from ml_flow_test import EXPERIMENT_NAME

class Trainer():

    def __init__(self, X, y, exp_name):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = exp_name

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), StandardScaler())
        pipe_distance = make_pipeline(DistanceTransformer(), StandardScaler())

        pipe_preproc = ColumnTransformer([
            ('time', pipe_time, ['pickup_datetime']),
            ('distance', pipe_distance, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude'])
        ])

        self.pipe = Pipeline([
            ('preproc', pipe_preproc),
            ('linear_model', LinearRegression())
        ])

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipe.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipe.predict(X_test)
        rmse = np.sqrt(((y_pred - y_test)**2).mean())
        self.mlflow_log_param("model", "linear")
        self.mlflow_log_metric("rmse", rmse)
        return rmse

    @memoized_property
    def mlflow_client(self):
        MLFLOW_URI = "https://mlflow.lewagon.ai/"
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    print(X_train)
    # train
    exp_name = "[DE] [Berlin] [MikkelValdemar] taxifare + 0"
    trainer = Trainer(X_train, y_train, exp_name)
    trainer.run()
    # evaluate
    rmse = trainer.evaluate(X_val, y_val)
    print(rmse)
