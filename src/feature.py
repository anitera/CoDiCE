from enum import Enum
from typing import Any

class FeatureType(Enum):
    NUMERIC = 1
    CATEGORICAL = 2
    BINARY = 3

class Feature(object):
    def __init__(self, name: str, value: Any, ftype: FeatureType) -> None:
        self.name = name
        self.ftype = ftype
        self.value = value
        self.sampled = False

class CatFeature(Feature):
    def __init__(self, name: str, value: Any) -> None:
        super().__init__(name, value,  FeatureType.CATEGORICAL)

class NumFeature(Feature):
    def __init__(self, name: str, value: Any) -> None:
        super().__init__(name, value, FeatureType.NUMERIC)
