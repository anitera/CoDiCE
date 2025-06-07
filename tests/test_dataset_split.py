import unittest
from codice.dataset import Dataset
from codice.config import Config

class TestDatasetSplit(unittest.TestCase):
    def setUp(self):
        self.config = Config("config/conf.yaml").get_config_value("dataset")

    def test_split(self):
        dataset = Dataset(self.config, "Loan_Status")
        x_train, x_val, y_train, y_val = dataset.split_dataset("Loan_Status")
        # Ensure the splits have the expected lengths
        total = len(dataset.get_data())
        self.assertEqual(len(x_train) + len(x_val), total)
        self.assertEqual(len(y_train) + len(y_val), total)

if __name__ == "__main__":
    unittest.main()
