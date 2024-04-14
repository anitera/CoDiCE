import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from codice.transformer import Transformer, FeatureTransformer
from codice.config import Config


class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.dataset = Mock()
        self.dataset.data = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                          'feature2': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']})
        self.dataset.continuous_features_list = ['feature1']
        self.dataset.categorical_features_list = ['feature2']

        self.config = Mock()

        self.config.get_config_value.return_value = {
            "continuous_distance": {
                "type": "diffusion",
                "diffusion_params": {
                    "diffusion_normalization": True,
                    "k_neighbors": 5,
                    "alpha": 1.0
                    }
            },
            "continuous_features_normalization": "minmax",
            "categorical_features_encoding": "onehot"
            }
  
        self.transformer = Transformer(self.dataset, self.config)

    def test_get_cont_features_transformers(self):
        self.assertEqual(len(self.transformer.get_cont_features_transformers(self.dataset, self.config)), 1)

    def test_get_cat_features_transformers(self):
        self.assertEqual(len(self.transformer.get_cat_features_transformers(self.dataset, self.config)), 1)

    def test_get_cont_transformers_length(self):
        self.assertEqual(self.transformer.get_cont_transformers_length(), 1)

    def test_get_cat_transformers_length(self):
        self.assertEqual(self.transformer.get_cat_transformers_length(), 1)

    def test_normalize_instance(self):
        instance = Mock()
        instance.features = {'feature1': Mock(), 'feature2': Mock()}
        instance.features['feature1'].value = 1
        instance.features['feature2'].value = 'a'
        self.transformer.normalize_instance(instance)
        self.assertTrue(instance.normalized)

    def test_denormalize_instance(self):
        instance = Mock()
        instance.features = {'feature1': Mock(), 'feature2': Mock()}
        instance.features['feature1'].value = 1.0
        instance.features['feature2'].value = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.transformer.denormalize_instance(instance)
        self.assertFalse(instance.normalized)

    def test_get_normed_numerical(self):
        instance = Mock()
        instance.features = {'feature1': Mock(), 'feature2': Mock()}
        instance.features['feature1'].value = [1]
        instance.features['feature2'].value = ['a']
        normed_values = self.transformer.get_normed_numerical(instance)
        self.assertEqual(len(normed_values), 1)

    def test_get_mads(self):
        mads = self.transformer._get_mads(self.dataset)
        self.assertEqual(len(mads), 1)

class TestFeatureTransformer(unittest.TestCase):
    def setUp(self):
        self.feature_name = 'test_feature'

    def test_calculate_statistics(self):
        # Mocking the dataset
        self.dataset = Mock()
        self.dataset.data = {self.feature_name: pd.Series([1, 2, 3, 4, 5])}
        self.dataset.continuous_features_list = [self.feature_name]

        # Mocking the config
        self.config = Mock()
        self.config.get_config_value.return_value = {"continuous_features_normalization": "minmax"}

        # Creating the FeatureTransformer with the mocked dataset and config
        self.transformer = FeatureTransformer(self.dataset, self.config, self.feature_name)

        # Running the test
        stats = self.transformer.calculate_statistics(self.dataset)
        self.assertEqual(stats['min'], 1)
        self.assertEqual(stats['max'], 5)
        self.assertEqual(stats['mean'], 3)
        self.assertAlmostEqual(stats['std'], 1.5811388300841898) # np std differ from pandas std
        self.assertEqual(stats['median'], 3)

    def test_get_from_dataset(self):
        self.dataset = Mock()
        self.dataset.data = {self.feature_name: pd.Series([1, 2, 3, 4, 5])}
        self.dataset.continuous_features_list = [self.feature_name]

        # Mocking the config
        self.config = Mock()
        self.config.get_config_value.return_value = {"continuous_features_normalization": "minmax"}

        # Creating the FeatureTransformer with the mocked dataset and config
        self.transformer = FeatureTransformer(self.dataset, self.config, self.feature_name)
        range = self.transformer.get_from_dataset(self.dataset)
        self.assertEqual(range, [1, 5])

    def test_get_feature_choice(self):
        self.dataset = Mock()
        self.dataset.data = pd.DataFrame({self.feature_name: ['a', 'b', 'c', 'a', 'b']})
        self.dataset.continuous_features_list = ["another_feature"]
        self.dataset.categorical_features_list = [self.feature_name]

        # Mocking the config
        self.config = Mock()
        self.config.get_config_value.return_value = {"continuous_features_normalization": "minmax",
                                                     "categorical_features_encoding": "onehot"}

        self.transformer = FeatureTransformer(self.dataset, self.config, self.feature_name)
        choices = self.transformer.get_feature_choice(self.dataset)
        self.assertEqual(set(choices), set(['a', 'b', 'c']))

    def test_label_enc_dec(self):
        self.dataset = Mock()
        self.dataset.data = pd.DataFrame({self.feature_name: ['a', 'b', 'c', 'a', 'b']})
        self.dataset.continuous_features_list = ["another_feature"]
        self.dataset.categorical_features_list = [self.feature_name]

        # Mocking the config
        self.config = Mock()
        self.config.get_config_value.return_value = {"continuous_features_normalization": "minmax",
                                                     "categorical_features_encoding": "onehot"}

        self.transformer = FeatureTransformer(self.dataset, self.config, self.feature_name)
        self.dataset.data = pd.DataFrame({self.feature_name: ['a', 'b', 'c', 'a', 'b']})

        self.transformer._initialize_label_encoder(self.dataset)
        encoded = self.transformer._label_enc('b')
        decoded = self.transformer._label_dec(encoded)
        self.assertEqual(decoded, 'b')

    def test_ordinal_enc_dec(self):
        self.dataset = Mock()
        self.dataset.data = pd.DataFrame({self.feature_name: ['a', 'b', 'c', 'a', 'b']})
        self.dataset.continuous_features_list = ["another_feature"]
        self.dataset.categorical_features_list = [self.feature_name]

        # Mocking the config
        self.config = Mock()
        self.config.get_config_value.return_value = {"continuous_features_normalization": "minmax",
                                                     "categorical_features_encoding": "ordinal"}

        self.transformer = FeatureTransformer(self.dataset, self.config, self.feature_name)
        self.dataset.data = pd.DataFrame({self.feature_name: ['a', 'b', 'c', 'a', 'b']})
        encoded = self.transformer._ordinal_enc('b')
        decoded = self.transformer._ordinal_dec(encoded)
        self.assertEqual(decoded, 'b')


if __name__ == '__main__':
    unittest.main()