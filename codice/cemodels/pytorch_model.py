from .model_interface import ModelInterface

class PyTorchModel(ModelInterface):
    def __init__(self, model_config):
        super().__init__(model_config)

    def fit(self, X, y):
        # Implement the fit method for PyTorch model
        pass

    def predict(self, X):
        # Implement the predict method for PyTorch model
        pass

    def evaluate(self, X, y):
        # Implement the evaluate method for PyTorch model
        pass

    def load_model(self, filepath):
        # Implement the load_model method for PyTorch model
        pass

    def sanity_check(self):
        return super().sanity_check()