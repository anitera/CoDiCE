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

        for feature_name, constraint in all_constraints["features"].items():
            print(f"Feature: {feature_name}")
            # check if feature name is in continuous or categorical features transformers
            if feature_name in self.transformers.continuous_features_transformers:
                if self.normalization:
                    feature_range = self.transformers.continuous_features_transformers[feature_name].normalized_range
                else:
                    feature_range = self.transformers.continuous_features_transformers[feature_name].original_range
            elif feature_name in self.transformers.categorical_features_transformers:
                feature_range = self.transformers.categorical_features_transformers[feature_name].normalized_range
            print(f"Range: {feature_range}")
            print(f"Constraint Type: {constraint['type']}")

            if constraint['type'] == 'dependency':
                if constraint['dependencyType'] == 'causal':
                    print(f"Dependency Type: {constraint['dependencyType']}")
                    print(f"Root: {constraint['root']}")
                    print(f"Child: {constraint['child']}")
                    causal_dep_type = CausalDependency()
                    dep_sampler = DependencySampler(constraint['root'], constraint['child'], causal_dep_type)
                    self.feature_samplers[feature_name] = dep_sampler
                elif constraint['dependencyType'] == 'monotonic_dependency':
                    print(f"Dependency Type: {constraint['dependencyType']}")
                    print(f"Root: {constraint['root']}")
                    print(f"Child: {constraint['child']}")
                    monotonic_dep_type = MonotonicDependency()
                    dep_sampler = DependencySampler(constraint['root'], constraint['child'], monotonic_dep_type)
                    self.feature_samplers[feature_name] = dep_sampler
                elif constraint['dependencyType'] == 'rule':
                    print(f"Dependency Type: {constraint['dependencyType']}")
                    print(f"Root: {constraint['root']}")
                    print(f"Child: {constraint['child']}")
                    print(f"Rule: {constraint['rule']}")
                    rule_dep_type = RuleDependency(constraint['rule'])
                    dep_sampler = DependencySampler(constraint['root'], constraint['child'], rule_dep_type)
                    self.feature_samplers[feature_name] = dep_sampler 
            # Additional logic based on constraint type
            if constraint['type'] == 'monotonic':
                print(f"Direction: {constraint['direction']}")
                sampler = MonotonicSampler(feature_name, feature_range, constraint['direction'])
                self.feature_samplers[feature_name] = sampler
            elif constraint['type'] == 'immutable':
                immutable_sample = ImmutableSampler()
            # ... Add further conditions as necessary
            print("-----")
            constraints[feature_name] = constraint
        return constraints

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

        
        

