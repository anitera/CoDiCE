from sklearn.base import BaseEstimator
import pickle
import numpy as np
from ceinstance import CEInstance
#from tensorflow.keras.models import Model as TFModel

class ExplainableModel:
    """
    Read model config, load model and perform inference
    """
    def __init__(self, model_config):
        self.config = model_config
        if (self.config.state == "pretrained"):
            self.model = self.load_model()
        else:
            self.model = self.train(model_config)

        self.sanity_check()

    def load_model(self):
        """
        Load model from pickle file
        """
        with open(self.config.model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def sanity_check(self):
        """
        Sanity check for model. Generate fake input and perform inference
        """
        if self.config.state == "pretrained":
            print("Sanity check for model")
            input_shape = self.get_base_estimator_input_shape()
            fake_input = np.random.rand(input_shape[1])
            fake_input = fake_input.reshape(1, -1)
            self.predict(fake_input)

    def get_base_estimator_input_shape(self):
        """
        Get input shape of base estimator
        """
        if self.config.model_type == "sklearn":
            return self.model.steps[0][1].input_shape
        else:
            raise Exception("Model type {} not supported".format(self.config.model_type))

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def predict_instance(self, x: CEInstance):
        """
        Predict instance
        """
        return self.predict(x.to_numpy_array().reshape(1, -1))
        
    def predict_proba_instance(self, x: CEInstance):
        """
        Predict instance
        """
        return self.predict_proba(x.to_numpy_array().reshape(1, -1))