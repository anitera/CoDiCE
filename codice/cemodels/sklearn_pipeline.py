from sklearn.pipeline import Pipeline
from .model_interface import ModelInterface
from codice.ceinstance import CEInstance
from joblib import load
import numpy as np
import warnings

class SklearnPipeline(ModelInterface):
    def __init__(self, model_config):
        super().__init__(model_config)
        # Configuration parameters
        self.model_type = model_config.get('model_type')
        self.model_name = model_config.get('name')
        self.model_state = model_config.get('state')
        self.model_path = model_config.get('path')
        self.model_categorical_encoding = model_config.get('categorical_features_encoding')
        self.model_continuous_encoding = model_config.get('continuous_features_normalization')

        if self.model_state == "pretrained":
            self.model = self.load_model(self.model_path)
        else:
            self.model = self.train(model_config)

        #self.sanity_check()

    def train(self, model_config):
        # Implement the train method for Pipeline model
        raise NotImplementedError
    
    def fit(self, X, y):
        # Implement the fit method for Pipeline model
        pass

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, x):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(x)
        else:
            raise NotImplementedError("This model does not support probability predictions.")

    def predict_instance(self, x: CEInstance):
        warnings.filterwarnings(action='ignore', category=UserWarning)
        return self.predict(x.to_numpy_array().reshape(1, -1))[0]
        
    def predict_proba_instance(self, x: CEInstance):
        warnings.filterwarnings(action='ignore', category=UserWarning)
        return self.predict_proba(x.to_numpy_array().reshape(1, -1))[0]

    def evaluate(self, X, y):
        # Implement the evaluate method for Pipeline model
        pass

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            try:
                model = load(f)
                return model
            except Exception as e:
                print(f"Error loading model: {e}")

    def get_base_estimator_input_shape(self):
        """
        For a Pipeline, this method should ideally return the input shape
        required by the first step of the pipeline, but typically it's more
        relevant to get the input shape for the final estimator in the pipeline.
        """
        final_estimator = self.model.steps[-1][1]
        if hasattr(final_estimator, 'coef_'):
            return final_estimator.coef_.shape[1]
        elif hasattr(final_estimator, 'tree_'):
            return final_estimator.tree_.n_features
        elif hasattr(final_estimator, 'coefs_'):
            return final_estimator.coefs_[0].shape[0]
        elif hasattr(final_estimator, 'support_vectors_'):
            return final_estimator.support_vectors_.shape[1]
        elif hasattr(final_estimator, 'cluster_centers_'):
            return final_estimator.cluster_centers_.shape[1]
        else:
            raise ValueError("Cannot infer input shape for the final estimator in the pipeline.")

    def sanity_check(self):
        if self.model_state == "pretrained":
            print("Sanity check for pipeline model")
            try:
                input_shape = self.get_base_estimator_input_shape()
                print("Model input shape is ", input_shape)
                fake_input = np.random.rand(1, input_shape)  # Adjusted for a single sample
                pred = self.predict(fake_input)
                print("Sanity check prediction: ", pred)
            except ValueError as e:
                print("Sanity check failed:", e)