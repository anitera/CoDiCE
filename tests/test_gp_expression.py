import unittest
import json

from trustce.cemodels.gp_model import GeneticProgrammingModel
from trustce.dataset import Dataset
from trustce import load_datasets
from trustce.cemodels.explainable_model import ExplainableModel

from trustce.ceinstance.instance_sampler import CEInstanceSampler
from trustce.config import Config
from trustce.transformer import Transformer
from trustce.ceinstance.instance_factory import InstanceFactory

class TestGeneticProgrammingModel(unittest.TestCase):

    def setUp(self):
        # read config yml file from config folder
        self.config = Config("config/conf_cartpole_gp.yaml")
        with open("config/constraints_conf_gp.json", 'r') as file:
            self.constraints = json.load(file)
        print(self.config)

        self.target_instance_json = "input_instance/instance_gp.json"
    

    def test_load_model(self):
        # Testing the load_model method
        gp_model = GeneticProgrammingModel(self.config.get_config_value("model"))
        self.assertIsNotNone(gp_model.models)

    def test_predict(self):
        self.data = Dataset(self.config.get_config_value("dataset"), "Req2_Promise")
        self.normalization_transformer = Transformer(self.data, self.config)
        self.instance_factory = InstanceFactory(self.data)
        self.sampler = CEInstanceSampler(self.config, self.normalization_transformer, self.instance_factory)
        # Testing the predict method
        gp_model = GeneticProgrammingModel(self.config.get_config_value("model"))
        
        with open(self.target_instance_json, 'r') as file:
            target_instance_json = file.read() #json.load(file)

        target_instance = self.instance_factory.create_instance_from_json(target_instance_json)


        prediction = gp_model.predict(target_instance)
        # You can check for specific expected output here
        print("Prediciton is ", prediction)
        self.assertIsNotNone(prediction)

    # Add more tests for other methods like evaluate, parse_expressions, etc.

if __name__ == '__main__':
    unittest.main()