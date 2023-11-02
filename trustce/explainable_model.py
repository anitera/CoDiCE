from sklearn.base import BaseEstimator
import pickle
from joblib import load
import numpy as np
from trustce.ceinstance import CEInstance
import warnings
from sklearn.exceptions import DataConversionWarning
#from tensorflow.keras.models import Model as TFModel

class ExplainableModel:
    """
    Read model config, load model and perform inference
    """
    def __init__(self, model_config):
        self.config = model_config
        self.model_type = self.config["model_type"]
        self.model_backend = self.config["model_backend"]
        if (self.config["state"] == "pretrained"):
            self.model = self.load_model()
        else:
            self.model = self.train(model_config)

        #self.sanity_check() # with pyrtorch it is impossible to get input size of the model?

    def load_model(self):
        """
        Load model from pickle file
        """
        with open(self.config["path"], 'rb') as f:
            if self.model_backend == "sklearn":
                try:
                    import sklearn
                    self.model = load(f)
                except:
                    raise Exception("Sklearn not installed or model can't be loaded")
                try:
                    self.sanity_check()
                except:
                    raise Exception("Model sanity check failed")
            #elif self.model_type == "tensorflow":
            #    try:
            #        import tensorflow as tf
            #        model = tf.keras.models.load_model(f)
            #    except:
            #        raise Exception("Tensorflow not installed or model can't be loaded")
            elif self.model_backend == "pytorch":
                try:
                    import torch
                    self.model = torch.load(f)
                except:
                    raise Exception("Pytorch not installed or model can't be loaded")
                try:
                    # with pyrtorch it is impossible to get input size of the model?
                    self.sanity_check()
                except:
                    raise Exception("Model sanity check failed")
            #elif self.model_type == "gpgomea":
            #    try:
            #        import gpgomea
            #        model = gpgomea.load(f)
            #    except:
            #        raise Exception("GPGOMEA not installed or model can't be loaded")
            else:
                raise Exception("Model type {} not supported".format(self.model_type))
        return self.model

    def sanity_check(self):
        """
        Sanity check for model. Generate fake input and perform inference
        """
        if self.config["state"] == "pretrained":
            print("Sanity check for model")
            input_shape = self.get_base_estimator_input_shape()
            print("Model input shape is ", input_shape)
            fake_input = np.random.rand(input_shape)
            fake_input = fake_input.reshape(1, -1)
            self.predict(fake_input)

    def get_base_estimator_input_shape(self):
        """
        Get input shape of base estimator
        """
        if self.config["model_backend"] == "sklearn":
            # get the shape of the input of the first step of the pipeline
            # Determine the type of the model
            model_type = type(self.model).__name__
            
            # Infer input shape based on model type
            if hasattr(self.model, 'coef_'):
                # Linear models (e.g., LinearRegression, LogisticRegression)
                return self.model.coef_.shape[1]
            
            elif hasattr(self.model, 'tree_'):
                # Decision Trees
                return self.model.tree_.n_features
            
            elif hasattr(self.model, 'coefs_'):
                # Neural Networks (MLPClassifier or MLPRegressor)
                return self.model.coefs_[0].shape[0]
            
            elif hasattr(self.model, 'support_vectors_'):
                # Support Vector Machines (SVC, SVR)
                return self.model.support_vectors_.shape[1]
            
            elif hasattr(self.model, 'cluster_centers_'):
                # KMeans
                return self.model.cluster_centers_.shape[1]
            
            else:
                raise ValueError(f"Cannot infer input shape for model type: {model_type}")
        elif self.config["model_backend"] == "pytorch":
            # jit store the model and extract the shape
            return True#self.model.input_shape
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
        # Suppress specific warning
        warnings.filterwarnings(action='ignore', category=UserWarning)
        return self.predict(x.to_numpy_array().reshape(1, -1))[0]
        
    def predict_proba_instance(self, x: CEInstance):
        """
        Predict instance
        """
        warnings.filterwarnings(action='ignore', category=UserWarning)
        return self.predict_proba(x.to_numpy_array().reshape(1, -1))[0]
    
    def train(self, model_config):
        """
        Train the basic model
        """
        return None