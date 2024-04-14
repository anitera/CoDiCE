from sklearn.base import BaseEstimator
import pickle
from joblib import load
import numpy as np
from codice.ceinstance import CEInstance
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
