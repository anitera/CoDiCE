from .instance import Instance
from .feature_sampler import IFeatureSampler
from collections import OrderedDict

class InstanceSampler(object):
    def __init__(self, config, dataset_metadata):
        self.feature_samplers = OrderedDict(IFeatureSampler) #TODO make ordered


    def sample(self, old_instance: Instance):
        new_instance = Instance.create_empty_instance()
        for feature_name, feature_sampler in self.feature_samplers.items(): # Assume samplers are ordered
            sampled_features = feature_sampler.sample(old_instance)
            for new_feature in sampled_features:
                new_instance.features[new_feature.name] = new_feature #TODO check if feature was not sampled

        return new_instance
        

