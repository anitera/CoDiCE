import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from codice.dataset import Dataset
from codice.config import Config

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.config = Config("config/conf.yaml").get_config_value("dataset")

    def test_read_dataset(self):
        self.dataset = Dataset(self.config, "Loan_Status")
        

if __name__ == "__main__":
    unittest.main()