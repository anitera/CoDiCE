from abc import ABC, abstractmethod
# Making model interface to support different kind of models


class ModelInterface(ABC):
    @abstractmethod
    def __init__(self, model_config):
        self.config = model_config

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def evaluate(self, X, y):
        pass

    @abstractmethod
    def load_model(self, filepath):
        pass

    @abstractmethod
    def sanity_check(self):
        pass