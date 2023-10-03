import numpy as np
from . import CEFeature
from typing import List

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

    def sample(self, instance):
        raise NotImplementedError

    def validate(self, value):
        assert value >= self.feature_range[0] and value <= self.feature_range[1], "Value {} not in range {}".format(value, self.feature_range)

class UniformSampler(ICEFeatureSampler):
    def __init__(self, feature_name, feature_range):
        super().__init__(feature_name, feature_range)

    def sample(self, instance):
        return self._uniform_sample()
    
class ImmutableSampler(ICEFeatureSampler):
    def __init__(self, feature_name, feature_range):
        super().__init__(feature_name, feature_range)

    def sample(self, instance):
        return instance.root[self.feature.name]

class ChoiceSampler(ICEFeatureSampler):
    """For categorical features"""
    def __init__(self, feature: CEFeature):
        super().__init__(feature.feature_range)

    def sample(self, instance):
        return self._choice_sample()

class MonotonicSampler(ICEFeatureSampler):
    def __init__(self, feature_name, feature_range, sign):
        super().__init__(feature_name, feature_range)
        self._set_sign(sign)

    def _set_sign(self, sign):
        assert sign in ["increasing", "decreasing"], "Sign must be either increasing or decreasing"
        self.sign = sign

    def _increasing_sample(self, feature_value):
        assert self.sign == "increasing", "Config must be increasing for increasing sample"
        return np.random.uniform(feature_value, self.feature_range[1])

    def _decreasing_sample(self, feature_value):
        assert self.sign == "decreasing", "Config must be decreasing for decreasing sample"
        return np.random.uniform(self.feature_range[0], feature_value)

    def sample(self, instance):
        return self._increasing_sample(instance[self.feature_name]) if self.sign == 1 else self._decreasing_sample(instance[self.feature_name])

class DependentSampler(ICEFeatureSampler):
    def __init__(self, main_feature: str):
        super().__init__()
        self.main_feature = main_feature

    def sample(self, instance):
        pass

    def validate(self, value):
        pass

class ICEDependencyType(object):
    def __init__(self):
        pass

    def sample(self, feature_sampler: ICEFeatureSampler, mf_value, current_instance):
        pass

class CausalDependency(ICEDependencyType):
    def __init__(self) -> None:
        pass

    def sample(self, feature_sampler: ICEFeatureSampler, mf_value, current_instance):
        return feature_sampler.sample(current_instance)

class MonotonicDependency(ICEDependencyType):
    def __init__(self) -> None:
        pass

    def _define_sign(self, mf_value, old_value):
        if mf_value > old_value:
            return 1
        elif mf_value < old_value:
            return -1
        else:
            return 0

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


class DependencySampler(ICEFeatureSampler):
    def __init__(self, main_feature: str, dep_features: List[str], rel_type):
        super().__init__()
        self.main_feature =  main_feature
        self.depndent_features = dep_features
        self.rel_type = rel_type

    def sample(self, instance):
        mf_val = self.main_feature.sample(instance.root)
        result = [mf_val]
        for df in self.depndent_features:
            df_val = self.rel_type.sample(df, mf_val, instance.root)
            result.append(df_val)

        return result


