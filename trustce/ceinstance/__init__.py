import json
import numpy as np
from collections import defaultdict
from trustce.cefeature import CEFeatureType

class CEInstance():
    """
    Represents a single instance of a dataset.
    Instance_schema is a dictionary that maps feature names to their types.
    """
    # def __init__(self, json_string) -> None:
    #     self.features = {}
    #     json_dict = json.loads(json_string)
    #     for k, v in json_dict.items():
    #         print(k, v)
    #         self.features[k] = CEInstance.instance_schema[k](k,v)

    def __init__(self, instance_schema, values_dict=None) -> None:
        self.features = defaultdict()
        self.normalized = True
        if values_dict is not None:
            for fname, ftype in instance_schema.items():
                self.features[fname] = ftype(fname, values_dict.get(fname, ftype.default_value)) 


    def to_numpy_array(self):
        return np.array([feature.value for feature in self.features.values()])

    def __iter__(self):
        return iter(self.features.values())
    
    def get_values_dict(self):
        return {feature.name: feature.value for feature in self.features.values()}
    
    def get_list_of_features_values(self):
        return [feature.value for feature in self.features.values()] 
    
    def get_list_of_features_names(self):
        return list(self.features.keys())
    
    def get_numerical_features_values(self):
        return [feature.value for feature in self.features.values() if feature.ftype == CEFeatureType.NUMERIC]
    
    def get_categorical_features_values(self):
        return [feature.value for feature in self.features.values() if feature.ftype == CEFeatureType.CATEGORICAL]

class InstanceChain(object):
    """
    Represents a chain of instances.
    Requires to track original feature to its counterfactual.
    """
    def __init__(self, root_instance: CEInstance) -> None:
        self.root = root_instance
        self.old_instance = None
        self.new_instance = root_instance
    

    def upate(self, instance: CEInstance):
        self.old_instance = self.new_instance
        self.new_instance = instance