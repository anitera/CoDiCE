from .model_interface import ModelInterface
from codice.ceinstance import CEInstance
import tensorflow as tf
import numpy as np
import warnings

class TensorflowModel(ModelInterface):
    def __init__(self, model_config, model, preprocessor=None):
        super().__init__(model_config)
        self.model = model
        self.model_type = model_config.get('model_type')
        self.model_name = model_config.get('name')
        self.model_state = model_config.get('state')
        self.model_path = model_config.get('path')
        self.model_categorical_encoding = model_config.get('categorical_features_encoding')
        self.model_continuous_encoding = model_config.get('continuous_features_normalization')
        self.preprocessor = preprocessor

        if self.model_state == "pretrained":
            self.model = self.load_model(self.model_path)
        elif self.model_state == "parameter":
            self.model = model
        else:
            self.model = self.train(model_config)

        self.sanity_check()

    def train(self, model_config):
        # Implement the train method for TensorFlow model
        raise NotImplementedError

    def fit(self, X, y, **kwargs):
        # Fit the model to the data. 'kwargs' can include any additional arguments TensorFlow's 'fit' method accepts.
        return self.model.fit(X, y, **kwargs)

    def predict(self, X):
        # TensorFlow models return a numpy array directly
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)
        if not isinstance(X, tf.Tensor):
            X = tf.convert_to_tensor(X, dtype=tf.float32)
        predictions = self.model.predict(X)
        class_labels = (predictions > 0.5).astype(int)
        return class_labels
    
    def predict_proba(self, X):
        # Predict probabilities for both classes
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)
        if not isinstance(X, tf.Tensor):
            X = tf.convert_to_tensor(X, dtype=tf.float32)
        probabilities = self.model.predict(X)
        # Since it's binary classification, probabilities of the negative class are 1 minus positive class probabilities
        return np.hstack((1 - probabilities, probabilities))

    def predict_instance(self, x: CEInstance):
        # Convert CEInstance to numpy array, predict class label, and return
        warnings.filterwarnings(action='ignore', category=UserWarning)
        array_x = x.to_numpy_array().reshape(1, -1)
        class_label = self.predict(array_x)[0]  # Adjusted to directly use predict
        return class_label
    
    def predict_proba_instance(self, x: CEInstance):
        # Convert CEInstance to numpy array, predict probabilities, and return
        warnings.filterwarnings(action='ignore', category=UserWarning)
        array_x = x.to_numpy_array().reshape(1, -1)
        probabilities = self.predict_proba(array_x)[0]  # Using predict_proba
        return probabilities

    def evaluate(self, X, y, **kwargs):
        # Evaluate the model on the provided data. 'kwargs' can include any additional arguments TensorFlow's 'evaluate' method accepts.
        return self.model.evaluate(X, y, **kwargs)

    def load_model(self, filepath):
        try:
            # TensorFlow/Keras models are typically saved in the HDF5 format or SavedModel format.
            model = tf.keras.models.load_model(filepath)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def get_base_estimator_input_shape(self):
        # Extract the input shape required by the first layer of the model
        return self.model.layers[0].input_shape

    def sanity_check(self):
        if self.model_state == "pretrained":
            print("Sanity check for model")
            input_shape = self.get_base_estimator_input_shape()[1:]  # Omit the batch dimension
            print("Model input shape is", input_shape)
            fake_input = np.random.rand(*input_shape).reshape(1, *input_shape)
            prediction = self.predict(fake_input)
            print("Sanity check prediction", prediction)