import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trustce.cemodels.explainable_model import ExplainableModel
from trustce.config import Config

class TestModel(unittest.TestCase):
    def setUp(self):
        self.config = Config("config/conf.yaml").get_config_value("model")

    def test_reading_sklearn_model(self):
        self.config["model_path"] = "models/homeloan_logistic_model.pkl"
        self.config["model_type"] = "sklearn"
        self.model = ExplainableModel(self.config)

    #def test_reading_pytorch_model(self):
    #    self.config["model_path"] = "models/homeloan_pytorch_model.pth"
    #    self.config["model_type"] = "pytorch"
    #    self.model = ExplainableModel(self.config)

    #def test_reading_tensorflow_model(self):
    #    self.config["model_path"] = "models/tensorflow_model.pkl"
    #    self.config["model_type"] = "tensorflow"
    #    self.model = ExplainableModel(self.config)

    #def test_reading_gpgomea_model(self):
    #    self.config["model_path"] = "models/gpgomea_model.pkl"
    #    self.config["model_type"] = "gpgomea"
    #    self.model = ExplainableModel(self.config)

if __name__ == "__main__":
    unittest.main()