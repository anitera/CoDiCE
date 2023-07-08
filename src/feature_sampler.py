
class IFeatureSampler(object):
    def __init__(self):
        pass



#1. range
#2. monotonicity
#3. const
#4. dependency


class MainFeature():
    def __init__(self) -> None:
        pass

class DependentFeature():
    def __init__(self) -> None:
        pass

class CausalDependency():
    def __init__(self) -> None:
        pass

class MonotonicDependency():
    def __init__(self) -> None:
        pass

class RuleDependency():
    def __init__(self) -> None:
        pass


class DependencySampler(IFeatureSampler):

    def __init__(self):
        super().__init__()
        self.main_feature = 
        self.depndent_features = 
        self.rel_type = 

    def sample(self, instance):
        mf = self.main_feature.sample(instance.root)
        result = [mf]
        for df in self.depndent_features:
            df = self.rel_type.sample(self.depndent_feature, mf, instance.root)
            result.append(df)

        return result


