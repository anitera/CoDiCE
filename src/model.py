from sklearn.base import BaseEstimator
#from tensorflow.keras.models import Model as TFModel

class ModelWrapper:
    def __init__(self, model: BaseEstimator): #Union[BaseEstimator, TFModel]): # for now let's support sklearn model
        self.model = model

    def predict(self, X):
        return self.model.predict(X)