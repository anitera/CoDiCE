import unittest
import pandas as pd
import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from codice.cfsearch import CFsearch

from codice.dataset import Dataset
from codice import load_datasets
from codice.cemodels.sklearn_model import SklearnModel

from codice.ceinstance.instance_sampler import CEInstanceSampler
from codice.config import Config
from codice.transformer import Transformer
from codice.ceinstance.instance_factory import InstanceFactory


class TestCFSearch(unittest.TestCase):
    def setUp(self):
        # read config yml file from config folder
        self.config = Config("config/conf_homeloan.yaml")
        with open("config/constraints_conf_no.json", 'r') as file:
            self.constraints = json.load(file)
        print(self.config)

        self.target_instance_json = "input_instance/instance.json"
        

    def test_cf_search(self):
        load_datasets.download("homeloan")
        self.data = Dataset(self.config.get_config_value("dataset"), "Loan_Status")
        self.normalization_transformer = Transformer(self.data, self.config)
        self.instance_factory = InstanceFactory(self.data)

        #custom_rules = {"coapplicantRule": sample_rule_based_functions}

        self.sampler = CEInstanceSampler(self.config, self.normalization_transformer, self.instance_factory) #custom_rules=custom_rules)

        self.model = SklearnModel(self.config.get_config_value("model"))
        config_for_cfsearch = self.config.get_config_value("cfsearch")
        self.search = CFsearch(self.normalization_transformer, self.model, self.sampler, 
                               config=self.config,
                               optimizer_name=config_for_cfsearch["optimizer"], 
                               distance_continuous=config_for_cfsearch["continuous_distance"], 
                               distance_categorical=config_for_cfsearch["categorical_distance"], 
                               loss_type=config_for_cfsearch["loss_type"], 
                               sparsity_penalty=config_for_cfsearch["sparsity_penalty"]["type"],
                               alpha=config_for_cfsearch["sparsity_penalty"]["alpha"],
                               beta=config_for_cfsearch["sparsity_penalty"]["beta"], 
                               coherence=config_for_cfsearch["coherence"],
                               objective_function_weights=config_for_cfsearch["objective_function_weights"])

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