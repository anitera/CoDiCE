from . import CEInstance
from ..cefeature.feature_sampler import ICEFeatureSampler
from collections import OrderedDict, defaultdict
from src.cefeature.feature_sampler import *
import json

class CEInstanceSampler(object):
    def __init__(self, config, transformers, instance_factory, normalization=True):
        self.config = config
        self.transformers = transformers
        self.normalization = normalization
        # initialize dictionary of feature_samples with ICEFeatureSampler
        self.feature_samplers = OrderedDict()
        self.constraints = self.read_constraints(config)     
        self.instance_factory = instance_factory

    def read_constraints(self, config):
        constraints = {}
        with open(config.get_config_value("constraints_file"), 'r') as file:
            all_constraints = json.load(file)

        all_constraints = all_constraints["features"]

        self._sort_constraints(all_constraints)

        for feature_name, constraint in all_constraints.items():
            print(f"Feature: {feature_name}")
            feature_range = self._get_feature_range(feature_name)
            print(f"Range: {feature_range}")
            print(f"Constraint Type: {constraint['type']}")

            if constraint['type'] == 'dependency':
                print(f"Dependency Type: {constraint['dependencyType']}")
                print(f"Root: {feature_name}")
                child_feature = constraint['child']
                child_range = self._get_feature_range(child_feature)
                print(f"Child: {child_feature}")
                print(f"Child Range: {child_range}")
                if constraint['dependencyType'] == 'causal':
                    try:
                        child_sampler = self.feature_samplers[child_feature] # just use the sampler that was already created
                    except KeyError:
                        raise ValueError(f"Sampler {child_feature} not found")
                elif constraint['dependencyType'] == 'monotonic_dependency':
                    child_sampler = MonotonicSampler(child_feature, child_range, 0)
                elif constraint['dependencyType'] == 'rule':
                    print(f"Rule: {constraint['rule']}")
                    child_sampler = RuleSampler(child_feature, child_range, constraint['rule'])
                else:
                    raise ValueError(f"Dependency type {constraint['dependencyType']} not supported")

                try:
                    root_sampler = self.feature_samplers[feature_name]
                except  KeyError:
                    raise ValueError(f"Root sampler {feature_name} not found")

                dep_sampler = DependencySampler(root_sampler, [child_sampler])
                self.feature_samplers[feature_name] = dep_sampler 
            # Additional logic based on constraint type
            elif constraint['type'] == 'monotonic':
                print(f"Direction: {constraint['direction']}")
                sampler = MonotonicSampler(feature_name, feature_range, constraint['direction'])
                self.feature_samplers[feature_name] = sampler
            elif constraint['type'] == 'immutable':
                immutable_sample = ImmutableSampler(feature_name, feature_range)
            # ... Add further conditions as necessary
            else:
                raise ValueError(f"Constraint type {constraint['type']} not supported")
            print("-----")
            constraints[feature_name] = constraint
        return constraints

    def _get_feature_range(self, feature_name):
        # check if feature name is in continuous or categorical features transformers
        if feature_name in self.transformers.continuous_features_transformers:
            if self.normalization:
                feature_range = self.transformers.continuous_features_transformers[feature_name].normalized_range
            else:
                feature_range = self.transformers.continuous_features_transformers[feature_name].original_range
        elif feature_name in self.transformers.categorical_features_transformers:
            feature_range = self.transformers.categorical_features_transformers[feature_name].normalized_range


        return feature_range

    def create_feature_samplers(self, transformers, normalization=True):
        """Probably this is old function that is not used anymore"""
        for feature_name, feature_transformer in transformers.continuous_features_transformers.items():
            # We iterate over dependent features first. If we have a dependency, we check the dependency type
            # We initialize the root feature and dependent feature
            # Then we iterate over left features and check the sample type
            if self.config["constraints"][feature_name] == "dependency":
                if normalization:
                    self.feature_samplers[feature_name] = ICEFeatureSampler(feature_name, feature_transformer.normalized_range)
                else:
                    self.feature_samplers[feature_name] = ICEFeatureSampler(feature_name, feature_transformer.original_range)
        for feature_name, feature_transformer in transformers.categorical_features_transformers.items():
            self.feature_samplers[feature_name] = ICEFeatureSampler(feature_transformer)


    # def sample(self, old_instance: CEInstance):
    #     new_instance = CEInstance.create_empty_instance()
    #     for feature_name, feature_sampler in self.feature_samplers.items(): # Assume samplers are ordered
    #         sampled_features = feature_sampler.sample(old_instance)
    #         for new_feature in sampled_features:
    #             new_instance.features[new_feature.name] = new_feature #TODO check if feature was not sampled

    #     return new_instance

    def sample(self, old_instance: CEInstance):
        sampled_features = defaultdict()
        for target_fname, target_sampler in self.feature_samplers.items():
            sampled_values = target_sampler.sample(old_instance)
            for fname, fvalue in sampled_values.items():
                if fname in sampled_features:
                    raise ValueError(f"Feature {fname} already sampled")
                sampled_features[fname] = fvalue

        return self.instance_factory.create_instance(sampled_features)

    def _sort_constraints(self, constraints):
        sorted_dict = sorted(constraints.items(), key=lambda x: x[1]['type'] != 'dependency')
        constraints = dict(sorted_dict)

        
        
        

