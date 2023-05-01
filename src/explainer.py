import pandas as pd
import timeit
from sklearn.base import BaseEstimator
#from tensorflow.keras.models import Model as TFModel
from data import Data
from model import ModelWrapper


class CFCOG:
    def __init__(self, data: Data, model_wrapper: ModelWrapper, configs: dict):
        #super().__init__(data, model_wrapper, configs) No parent class for now
        self.data = data
        self.model_wrapper = model_wrapper

    def optimize(self, desired_output):
        # Implement your genetic search optimization function here
        pass

    def explain(self, instance, desired_output, total_CFs, hyperparameters):

        self.population_size = 10*total_CFs
        self.start_time = timeit.default_timer()
        features_to_vary = self.setup(hyperparameters['actionable'], hyperparameters['permitted_range'], instance, hyperparameters['feature_weights'], hyperparameters['constraints'])

        # Call the optimize function and return the counterfactual explanation
        self.optimize(desired_output)
        # Return the explanation
        pass