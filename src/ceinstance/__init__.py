import json
import numpy as np
from collections import defaultdict

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

    def __init__(self, instance_schema, **kwargs) -> None:
        self.featuers = defaultdict()
        for fname, ftype in instance_schema.items():
            self.feature[fname] = ftype(fname, kwargs.get(fname, ftype.default_value)) 


    def to_numpy_array(self):
        return np.array([feature.value for feature in self.features.values()])

    def __iter__(self):
        return iter(self.features.values())

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