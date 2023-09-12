import unittest
import yaml
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.optim import CFsearch
from src.optim import GeneticOptimizer

from src.dataset import Dataset
from src.explainable_model import ExplainableModel

from src.ceinstance import CEInstance
from src.ceinstance.instance_sampler import CEInstanceSampler
from src.cefeature import CatCEFeature, NumCEFeature, CEFeatureType
from src.config import Config


class TestCFSearch(unittest.TestCase):
    def setUp(self):
        # read config yml file from config folder
        self.config = Config("config/conf.yaml")
        #self.config = yaml.safe_load(open("config/conf.yaml"))
        print(self.config)
        

    def test_cf_search(self):
        self.data = Dataset(self.config.get_config_value("dataset"), "Loan_Status")
        self.model = ExplainableModel(self.config.get_config_value("model"), self.data)
        self.search = CFsearch(self.data, self.model, algorithm="genetic", distance_continuous="weighted_l1", distance_categorical="weighted_l1", loss_type="hinge_loss", sparsity_hp=0.2, coherence_hp=0.2, diversity_hp=0.2)

if __name__ == "__main__":
    unittest.main()