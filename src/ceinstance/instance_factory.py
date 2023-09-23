from . import CEInstance
from collections import defaultdict
from src.cefeature import CatCEFeature, NumCEFeature
import json


class InstanceFactory(object):
    """
    Creates instances from json strings.
    """
    def __init__(self, dataset) -> None:
        self.instance_schema = defaultdict()
        self._schema_from_lists(dataset.cat_list, dataset.cont_list)

    def _schema_from_lists(self, cat_list, cont_list):
        for cat in cat_list:
            self.instance_schema[cat] = CatCEFeature

        for cont in cont_list:
            self.instance_schema[cont] = NumCEFeature

    def create_instance_from_json(self, json_values: str):
        dict_values = json.loads(json_values)
        return self.create_instance(dict_values)

    def create_instance(self, dict_values: dict):
        return CEInstance(self.instance_schema, dict_values)

    def create_empty_instance(self):
        return CEInstance(self.instance_schema)
