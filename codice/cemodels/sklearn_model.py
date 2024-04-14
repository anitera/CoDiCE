from .model_interface import ModelInterface
from codice.ceinstance import CEInstance
from joblib import load
import pickle
import numpy as np
import warnings

class SklearnModel(ModelInterface):
    def __init__(self, model_config):
        super().__init__(model_config)
        # Parse specific configuration parameters
        self.model_type = model_config.get('model_type')
        self.model_name = model_config.get('name')
        self.model_state = model_config.get('state')
        self.model_path = model_config.get('path')
        self.model_categorical_encoding = model_config.get('categorical_features_encoding')
        self.model_continuous_encoding = model_config.get('continuous_features_normalization')

        if self.model_state == "pretrained":
            self.model = self.load_model(self.model_path)
        else: self.model = self.train(model_config)

        self.sanity_check()

    def train(self, model_config):
        # Implement the train method for sklearn model
        raise NotImplementedError
    
    def fit(self, X, y):
        # Implement the fit method for sklearn model
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
        print(x.to_numpy_array())
        return self.predict(x.to_numpy_array().reshape(1, -1))[0]
        
    def predict_proba_instance(self, x: CEInstance):
        """
        Predict instance
        """
        warnings.filterwarnings(action='ignore', category=UserWarning)
        print(x.to_numpy_array())
        return self.predict_proba(x.to_numpy_array().reshape(1, -1))[0]

    def evaluate(self, X, y):
        # Implement the evaluate method for sklearn model
        pass

    def load_model(self, filepath):
        # Implement the load_model method for sklearn model
        with open(self.config["path"], 'rb') as f:
            try:
                import sklearn
                model = load(f)
                return model
            except Exception as e:
                print(f"Error loading model: {e}")

    def get_base_estimator_input_shape(self):
        """
        Get input shape of base estimator
        """
        
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

    def sanity_check(self):
        if self.config["state"] == "pretrained":
            print("Sanity check for model")
            input_shape = self.get_base_estimator_input_shape()
            print("Model input shape is ", input_shape)
            fake_input = np.random.rand(input_shape)
            fake_input = fake_input.reshape(1, -1)
            self.predict(fake_input)
            print("Sanity check prediciton ", self.predict(fake_input))