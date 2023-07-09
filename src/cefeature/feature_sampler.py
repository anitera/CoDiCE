from src.cefeature.feature_sampler import ICEFeatureSampler
from . import CEFeature
from typing import List

class ICEFeatureSampler(object):
    def __init__(self):
        pass

    def sample(self, instance):
        raise NotImplementedError

class MonotonicSampler(ICEFeatureSampler):
    def __init__(self, feature: CEFeature, sign: int):
        super().__init__()
        self.feature = feature
        self.sign = sign

    def _set_sign(self, sign):
        self.sign = sign

    def sample(self, instance):
        pass

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
    def __init__(self, rule_func: function) -> None:
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


