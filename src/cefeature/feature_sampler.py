import numpy as np
from . import CEFeature
from src.ceinstance import CEInstance
from typing import List, Any

class ICEFeatureSampler(object):
    def __init__(self, feature_name, feature_range):
        self.feature_name = feature_name
        self.feature_range = feature_range
        #TODO add validation

    def _uniform_sample(self):
        assert len(self.feature_range) == 2, "Feature range must be a tuple of length 2" 
        return np.random.uniform(self.feature_range[0], self.feature_range[1])

    def _choice_sample(self):
        return np.random.choice(self.feature_range)

    def _sample(self, value):
        raise NotImplementedError
    
    def _dependant_sample(self, instance: CEInstance, mf_result):
        raise NotImplementedError
    
    def sample(self, instance: CEInstance):
        return {self.feature_name: self._sample(instance.features[self.feature_name].value)}

    def dependant_sample(self, instance: CEInstance, mf_result):
        return {self.feature_name: self._dependant_sample(instance, mf_result)}

    def validate(self, value):
        assert value >= self.feature_range[0] and value <= self.feature_range[1], "Value {} not in range {}".format(value, self.feature_range)

class UniformSampler(ICEFeatureSampler):
    def __init__(self, feature_name, feature_range):
        super().__init__(feature_name, feature_range)

    def _sample(self, instance):
        return self._uniform_sample()
    
class ImmutableSampler(ICEFeatureSampler):
    def __init__(self, feature_name, feature_range):
        super().__init__(feature_name, feature_range)

    def _sample(self, value):
        return value

class ChoiceSampler(ICEFeatureSampler):
    """For categorical features"""
    def __init__(self, feature_name, feature_range):
        super().__init__(feature_name, feature_range)

    def _sample(self, instance):
        return self._choice_sample()

class MonotonicSampler(ICEFeatureSampler):
    def __init__(self, feature_name, feature_range, sign):
        super().__init__(feature_name, feature_range)
        self._set_sign(sign)

    def _set_sign(self, sign):
        assert sign in ["increasing", "decreasing"], "Sign must be either increasing or decreasing"
        self.sign = sign

    def _define_sign(self, mf_value, old_value):
        if mf_value > old_value:
            return "increasing" 
        else:
            return "decreasing"

    def _increasing_sample(self, feature_value):
        assert self.sign == "increasing", "Config must be increasing for increasing sample"
        return np.random.uniform(feature_value, self.feature_range[1])

    def _decreasing_sample(self, feature_value):
        assert self.sign == "decreasing", "Config must be decreasing for decreasing sample"
        return np.random.uniform(self.feature_range[0], feature_value)

    def _sample(self, value):
        return self._increasing_sample(value) if self.sign == "increasing" else self._decreasing_sample(value)

    def _dependant_sample(self, instance, mf_result):
        fname, fvalue = next(iter(mf_result.items()))
        self.sign = self._define_sign(fvalue, instance.features[fname].value)
        return self._sample(instance.features[self.feature_name].value)

class DependentSampler(ICEFeatureSampler):
    def __init__(self, feature_name, feature_range, main_feature):
        super().__init__(feature_name, feature_range)
        self.main_feature = main_feature

    def sample(self, instance):
        pass

class ICEDependencyType(object):
    def __init__(self):
        pass

    def sample(self, feature_sampler: ICEFeatureSampler, mf_value, current_instance):
        pass

class CausalDependency(ICEDependencyType):
    """
    default random sampling from feature range
    """
    def __init__(self) -> None:
        pass

    def sample(self, feature_sampler: ICEFeatureSampler, mf_value, current_instance):
        return feature_sampler.sample(current_instance)

class MonotonicDependency(ICEDependencyType):
    def __init__(self) -> None:
        pass


    def sample(self, feature_sampler: ICEFeatureSampler, mf_value, current_instance):
        sign = self._define_sign(mf_value, current_instance.root[self.main_feature])
        feature_sampler.set_sign(sign)
        val = feature_sampler.sample(current_instance)
        return val

class RuleDependency(ICEDependencyType):
    # init with a function that takes mf_value and current_instance and returns a value
    def __init__(self, rule_func) -> None:  
        self._rule = rule_func

    def sample(self, feature_sampler: ICEFeatureSampler, mf_value, current_instance):
        val = self._rule(mf_value, current_instance)
        feature_sampler.validate(val)
        return val

class RuleSampler(ICEFeatureSampler):
    def __init__(self, feature_name, feature_range, rule_func):
        super().__init__(feature_name, feature_range)
        self._rule = rule_func

    def _sample(self, instance):
        return self._rule(instance.root[self.feature_name], instance)


class DependencySampler(ICEFeatureSampler):
    def __init__(self, main_feature: ICEFeatureSampler, dep_features: List[ICEFeatureSampler]):
        super().__init__(main_feature.feature_name, main_feature.feature_range)
        self.main_feature =  main_feature
        self.depndent_features = dep_features

    def sample(self, instance: CEInstance):
        result = self.main_feature.sample(instance)
        for df in self.depndent_features:
            df_val = df.dependant_sample(instance, result)
            result.update(df_val)

        return result