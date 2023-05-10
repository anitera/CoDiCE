import pandas as pd
import timeit
from sklearn.base import BaseEstimator
#from tensorflow.keras.models import Model as TFModel
from data import Data
from model import ModelWrapper


class CFCOG:
    def __init__(self, model_wrapper: ModelWrapper, feature_manager, config):
        #super().__init__(data, model_wrapper, configs) No parent class for now
        self.model_wrapper = model_wrapper
        self.feature_manager = feature_manager

    def optimize(self, desired_output):
        # Implement your genetic search optimization function here
        input = self.feature_manager.transform(self.instance)

        # Gen population
        # Run inference on population
        # Calculate fitness
        # Select parents
        # Crossover
        # Mutation
        # Repeat

        output = self.model_wrapper.predict(input) 
        pass

    def setup(self):
        self.population_size = 10*total_CFs

    def explain(self, instance, desired_output, total_CFs, optim_config):
        self.instance = instance
        self.total_CFs = total_CFs

        self.setup()
        self.start_time = timeit.default_timer()
        #features_to_vary = self.setup(hyperparameters['actionable'], hyperparameters['permitted_range'], instance, hyperparameters['feature_weights'], hyperparameters['constraints'])

        # Call the optimize function and return the counterfactual explanation
        self.optimize(desired_output)
        # Return the explanation
        pass