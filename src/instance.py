import json
from collections import defaultdict
from .cefeature import CatCEFeature, NumCEFeature


class Instance():
    """
    Represents a single instance of a dataset.
    Instance_schema is a dictionary that maps feature names to their types.
    """
    instance_schema = defaultdict()
    def __init__(self, json_string) -> None:
        self.features = {}
        json_dict = json.loads(json_string)
        for k, v in json_dict.items():
            print(k, v)
            self.features[k] = Instance.instance_schema[k](k,v)


    @staticmethod
    def schema_from_lists(cat_list, cont_list):
        for cat in cat_list:
            Instance.instance_schema[cat] = CatCEFeature

        for cont in cont_list:
            Instance.instance_schema[cont] = NumCEFeature

    @staticmethod
    def create_empty_instance():
        return Instance("{}")

    def __iter__(self):
        return iter(self.features.values())

class InstanceChain(object):
    """
    Represents a chain of instances.
    Requires to track original feature to its counterfactual.
    """
    def __init__(self, root_instance: Instance) -> None:
        self.root = root_instance
        self.old_instance = None
        self.new_instance = root_instance
    

    def upate(self, instance: Instance):
        self.old_instance = self.new_instance
        self.new_instance = instance