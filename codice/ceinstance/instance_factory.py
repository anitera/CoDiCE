from . import CEInstance
from collections import defaultdict
from codice.cefeature import CatCEFeature, NumCEFeature
import json


class InstanceFactory(object):
    """
    Creates instances from json strings.
    """
    def __init__(self, dataset) -> None:
        """Build schema preserving the original feature ordering used for model training.
        Dataset lists provide features in the expected order (categorical
        first, followed by continuous features). If we add continuous features
        first the resulting CEInstance will have a different order than the
        dataset and scikit‑learn pipeline, causing prediction errors.  The
        schema is therefore created exactly in the order of the provided lists."""
        self.instance_schema = defaultdict()
        self._schema_from_lists(dataset.categorical_features_list, dataset.continuous_features_list)

    def _schema_from_lists(self, cat_list, cont_list):
        """Populate ``instance_schema`` with the provided feature lists.

        ``cat_list`` is expected to contain categorical feature names and
        ``cont_list`` continuous feature names.  Features are added in this
        exact order so that ``CEInstance.to_numpy_array`` matches the layout of
        the underlying dataset used by the scikit‑learn pipeline.
        """
        for cat in cat_list:
            self.instance_schema[cat] = CatCEFeature

        for cont in cont_list:
            self.instance_schema[cont] = NumCEFeature

    def create_instance_from_json(self, json_values: str):
        dict_values = json.loads(json_values)
        return self.create_instance(dict_values)

    def create_instance(self, dict_values: dict):
        return CEInstance(instance_schema=self.instance_schema, values_dict=dict_values)

    def create_empty_instance(self):
        return CEInstance(self.instance_schema)

    def create_instance_from_df_row(self, row):
        return self.create_instance(row.to_dict())
