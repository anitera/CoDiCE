from .model_interface import ModelInterface
from trustce.ceinstance import CEInstance
from joblib import load
import pickle
import numpy as np
import warnings


class BaseModel(ModelInterface):

    def __init__(self, model_config, model):
        """Init method. The aim of this method is to pass ready model that have its own predict functions

        """
        self.model = model
        self.model_type = model_config.get('model_type')
        self.model_name = model_config.get('name')
        self.model_state = model_config.get('state')
        self.model_path = model_config.get('path')

    def load_model(self):
        if self.model_path != '':
            with open(self.model_path, 'rb') as filehandle:
                self.model = pickle.load(filehandle)

    def train(self, model_config):
        # Implement the train method for sklearn model
        raise NotImplementedError
    
    def fit(self, X, y):
        # Implement the fit method for sklearn model
        pass

    def sanity_check(self):
        return super().sanity_check()
    
    def evaluate(self, X, y):
        # Implement the evaluate method for sklearn model
        pass

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, x):
        return self.model.predict_proba(x)
    
    def predict_instance(self, x: CEInstance):
        """
        Predict instance
        """
        # Suppress specific warning
        warnings.filterwarnings(action='ignore', category=UserWarning)
        return self.predict(x.to_numpy_array().reshape(1, -1))
    
    def predict_proba_instance(self, x: CEInstance):
        """
        Predict instance
        """
        warnings.filterwarnings(action='ignore', category=UserWarning)
        instance = x.to_numpy_array().reshape(1, -1)
        full_proba = self.predict_proba(instance)
        return full_proba[0]