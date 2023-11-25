from . import CEInstance
from ..cefeature.feature_sampler import ICEFeatureSampler
from collections import OrderedDict, defaultdict
from trustce.cefeature.feature_sampler import *
from trustce.cefeature import CatCEFeature, NumCEFeature
import json
import inspect
import logging

class CEInstanceSampler(object):
    def __init__(self, config, transformers, instance_factory, normalization=True, custom_rules=None):
        self.config = config
        self.transformers = transformers
        self.normalization = normalization
        # initialize dictionary of feature_samples with ICEFeatureSampler
        self.feature_samplers = OrderedDict()
        self.instance_factory = instance_factory
        if custom_rules is not None:
            if not isinstance(custom_rules, dict):
                raise ValueError("Custom rules must be a dictionary")
            else:
                self.custom_rules = custom_rules

        self.constraints = self.read_constraints(config)     

    def read_constraints(self, config):
        constraints = {}
        with open(config.get_config_value("constraints_file"), 'r') as file:
            all_constraints = json.load(file)

        all_constraints = all_constraints["features"]

        self._sort_constraints(all_constraints) # sort constraints so that dependencies are created last

        no_constraints_features  = set(self.instance_factory.instance_schema.keys()) - set(all_constraints.keys())

        for feature_name in no_constraints_features:
            self.feature_samplers[feature_name] = self._create_default_samplers(feature_name)

        for feature_name, constraint in all_constraints.items():
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
                    child_sampler = MonotonicSampler(child_feature, child_range, "increasing")
                elif constraint['dependencyType'] == 'rule':
                    print(f"Rule: {constraint['rule']}")
                    try:
                        rule_function = self.custom_rules[constraint['rule']]
                    except KeyError:
                        raise ValueError(f"Rule {constraint['rule']} not found")

                    child_sampler = RuleSampler(child_feature, child_range, rule_function)
                else:
                    raise ValueError(f"Dependency type {constraint['dependencyType']} not supported")

                del self.feature_samplers[child_feature]

                dep_sampler = self._create_default_samplers(feature_name, is_dependency=True)
                dep_sampler.add_dependent_feature(child_sampler)
                self.feature_samplers[feature_name] = dep_sampler 
            else:
                self.feature_samplers[feature_name] = self._create_sampler_from_constraint(feature_name, constraint)

            # Additional logic based on constraint type
            constraints[feature_name] = constraint
        return constraints

    def _create_default_samplers(self, feature_name, is_dependency=False):
        feature_range = self._get_feature_range(feature_name)
        print(f"Feature: {feature_name}")
        print(f"Range: {feature_range}")

        if inspect.isclass(self.instance_factory.instance_schema[feature_name]):
            if issubclass(self.instance_factory.instance_schema[feature_name], NumCEFeature):
                # Do something if the feature is a numerical feature
                if is_dependency:
                    sampler= UniformDependencySampler(feature_name, feature_range)
                else:
                    sampler = UniformSampler(feature_name, feature_range)
            elif issubclass(self.instance_factory.instance_schema[feature_name], CatCEFeature):
                # Do something if the feature is a categorical feature
                if is_dependency:
                    sampler = ChoiceDependencySampler(feature_name, feature_range)
                else:
                    sampler = ChoiceSampler(feature_name, feature_range)
            else:
                # Do something else if the feature is not a numerical or categorical feature
                raise ValueError(f"Feature type {type(self.instance_factory.instance_schema[feature_name])} not supported")
        
        return sampler

    def _create_sampler_from_constraint(self, feature_name, constraint):
        feature_range = self._get_feature_range(feature_name)

        print(f"Feature: {feature_name}")
        print(f"Range: {feature_range}")
        print(f"Constraint Type: {constraint['type']}")

        if constraint['type'] == 'monotonic':
            print(f"Direction: {constraint['direction']}")
            sampler = MonotonicSampler(feature_name, feature_range, constraint['direction'])
        elif constraint['type'] == 'immutable':
            sampler = ImmutableSampler(feature_name, feature_range)
        # ... Add further conditions as necessary
        else:
            raise ValueError(f"Constraint type {constraint['type']} not supported")

        return sampler


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

        
        
        

