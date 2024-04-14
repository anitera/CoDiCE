
from abc import ABC, abstractmethod
from typing import List, Tuple

class OptimizerInterface(ABC):
    @abstractmethod
    def find_counterfactual(self, query_instance, number_cf, desired_output, maxiterations):
        """
        Abstract method to find counterfactuals
        """
        pass

class OptimizerFactory:
    @staticmethod
    def get_optimizer(optimizer_name, *args, **kwargs):
        if optimizer_name == "genetic":
            from .genetic_optimizer import GeneticOptimizer
            return GeneticOptimizer(*args, **kwargs)
