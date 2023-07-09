import unittest
from collections import defaultdict
import json
from src.instance import Instance
from src.feature import CatFeature, NumFeature, FeatureType

# Your Instance class here

class TestInstance(unittest.TestCase):
    def setUp(self):
        self.instance = Instance('{}')

    def test_empty_instance(self):
        self.assertEqual(self.instance.features, {})

    def test_schema_from_lists(self):
        Instance.schema_from_lists(['color'], ['height'])
        self.assertEqual(Instance.instance_schema['color'], CatFeature)
        self.assertEqual(Instance.instance_schema['height'], NumFeature)

    def test_non_empty_instance(self):
        Instance.schema_from_lists(['color'], ['height'])
        instance = Instance('{"color": "blue", "height": 5}')
        # Test 'color' feature
        self.assertIsInstance(instance.features['color'], CatFeature)
        self.assertEqual(instance.features['color'].name, 'color')
        self.assertEqual(instance.features['color'].value, 'blue')
        self.assertEqual(instance.features['color'].ftype, FeatureType.CATEGORICAL)
        self.assertFalse(instance.features['color'].sampled)
        # Test 'height' feature
        self.assertIsInstance(instance.features['height'], NumFeature)
        self.assertEqual(instance.features['height'].name, 'height')
        self.assertEqual(instance.features['height'].value, 5)
        self.assertEqual(instance.features['height'].ftype, FeatureType.NUMERIC)
        self.assertFalse(instance.features['height'].sampled)

    def test_iter(self):
        Instance.schema_from_lists(['color'], ['height'])
        instance = Instance('{"color": "blue", "height": 5}')
        self.assertEqual([f.value for f in instance], ["blue", 5])

if __name__ == "__main__":
    unittest.main()