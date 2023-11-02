import unittest
import os
import sys
from collections import defaultdict
import json
from trustce.ceinstance import CEInstance
from trustce.cefeature import CatCEFeature, NumCEFeature, CEFeatureType

# Your Instance class here
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestInstance(unittest.TestCase):
    def setUp(self):
        self.instance = CEInstance('{}')

    def test_empty_instance(self):
        self.assertEqual(self.instance.features, {})

    def test_schema_from_lists(self):
        CEInstance.schema_from_lists(['color'], ['height'])
        self.assertEqual(CEInstance.instance_schema['color'], CatCEFeature)
        self.assertEqual(CEInstance.instance_schema['height'], NumCEFeature)

    def test_non_empty_instance(self):
        CEInstance.schema_from_lists(['color'], ['height'])
        instance = CEInstance('{"color": "blue", "height": 5}')
        # Test 'color' feature
        self.assertIsInstance(instance.features['color'], CatCEFeature)
        self.assertEqual(instance.features['color'].name, 'color')
        self.assertEqual(instance.features['color'].value, 'blue')
        self.assertEqual(instance.features['color'].ftype, CEFeatureType.CATEGORICAL)
        self.assertFalse(instance.features['color'].sampled)
        # Test 'height' feature
        self.assertIsInstance(instance.features['height'], NumCEFeature)
        self.assertEqual(instance.features['height'].name, 'height')
        self.assertEqual(instance.features['height'].value, 5)
        self.assertEqual(instance.features['height'].ftype, CEFeatureType.NUMERIC)
        self.assertFalse(instance.features['height'].sampled)

    def test_iter(self):
        CEInstance.schema_from_lists(['color'], ['height'])
        instance = CEInstance('{"color": "blue", "height": 5}')
        self.assertEqual([f.value for f in instance], ["blue", 5])

if __name__ == "__main__":
    unittest.main()