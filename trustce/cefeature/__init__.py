from enum import Enum
from typing import Any

class CEFeatureType(Enum):
    NUMERIC = 1
    CATEGORICAL = 2
    BINARY = 3

class CEFeature(object):
    default_value = None
    def __init__(self, name: str, value: Any, ftype: CEFeatureType) -> None:
        self.name = name
        self.ftype = ftype
        self.value = value
        self.sampled = False

class CatCEFeature(CEFeature):
    default_value = 0
    def __init__(self, name: str, value: Any) -> None:
        super().__init__(name, value,  CEFeatureType.CATEGORICAL)

class NumCEFeature(CEFeature):
    default_value = 0.0
    def __init__(self, name: str, value: Any) -> None:
        super().__init__(name, value, CEFeatureType.NUMERIC)
