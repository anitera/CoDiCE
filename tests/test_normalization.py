import unittest
import pandas as pd
import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trustce.cfsearch import CFsearch

from trustce.dataset import Dataset
from trustce.cemodels.explainable_model import ExplainableModel

from trustce.ceinstance.instance_sampler import CEInstanceSampler
from trustce.config import Config
from trustce.transformer import Transformer
from trustce.ceinstance.instance_factory import InstanceFactory


class TestCFSearch(unittest.TestCase):
    def setUp(self):
        # read config yml file from config folder
        self.config = Config("config/conf.yaml")
        with open("config/constraints_conf.json", 'r') as file:
            self.constraints = json.load(file)
        print(self.config)

        self.target_instance_json = "datasets/instance.json"
        

    def test_cf_search(self):
        self.data = Dataset(self.config.get_config_value("dataset"), "Loan_Status")
        self.normalization_transformer = Transformer(self.data, self.config.get_config_value("feature_manager"))
        self.instance_factory = InstanceFactory(self.data)
        self.sampler = CEInstanceSampler(self.config, self.normalization_transformer, self.instance_factory)

        self.model = ExplainableModel(self.config.get_config_value("model"))

        self.search = CFsearch(self.normalization_transformer, self.model, self.sampler, algorithm="genetic", distance_continuous="weighted_l1", distance_categorical="weighted_l1", loss_type="hinge_loss", sparsity_hp=0.2, coherence_hp=0.2, diversity_hp=0.2)

        with open(self.target_instance_json, 'r') as file:
            target_instance_json = file.read() #json.load(file)

        target_instance = self.instance_factory.create_instance_from_json(target_instance_json)

        counterfacturals = self.search.find_counterfactuals(target_instance, 1, "opposite", 10)

        self.search.evaluate_counterfactuals(target_instance, counterfacturals)
        # Visualise the values of counterfactuals and original instance only in jupyter notebook
        self.search.visualize_as_dataframe(target_instance, counterfacturals)
        self.search.store_counterfactuals(self.config.get_config_value("output_folder"), "first_test")
        self.search.store_evaluations(self.config.get_config_value("output_folder"), "first_eval")
        
        print("Counterfactuals:")
        print(counterfacturals[0].get_values_dict())
        print("Original instance:")
        print(target_instance.get_values_dict())

if __name__ == "__main__":
    unittest.main()