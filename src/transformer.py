import numpy as np

class Transformer(object):
    def __init__(self, dataset, config):
        self.continuous_features_transformers = self.get_cont_features_transformers(dataset, config)
        self.categorical_features_transformers = self.get_cat_features_transformers(dataset, config)

    def get_cont_features_transformers(self, dataset, config):
        return {feature_name: FeatureTransformer(dataset, config, feature_name) for feature_name in dataset.continuous_features_list}

    def get_cat_features_transformers(self, dataset, config):
        return {feature_name: FeatureTransformer(dataset, config, feature_name) for feature_name in dataset.categorical_features_list}

class FeatureTransformer(object):
    """
    Store original range of data, normalized range of data and normalization type
    """
    def __init__(self, dataset, config, feature_name):
        self.feature_name = feature_name
        if self.feature_name in dataset.continuous_features_list:
            self.min, self.max, self.mean, self.std, self.median = self.calculate_statistics(dataset)
            self.original_range = self.get_from_dataset(dataset)
            self.normalized_range = self.get_normalized_range(config, dataset)
        elif self.feature_name in dataset.categorical_features_list:
            self.original_range = self.get_feature_choice(dataset)
            self.normalized_range = self.apply_encoding(config, dataset)
        else:
            raise ValueError("Feature name is not in continuous or categorical features list")

    def calculate_statistics(self, dataset):
        return dataset.data[self.feature_name].agg([min, max, np.mean, np.std, np.median])

    def get_from_dataset(self, dataset):
        aggregated_values = dataset.data[self.feature_name].agg([min, max])
        return [aggregated_values['min'], aggregated_values['max']]
    
    def get_feature_choice(self, dataset):
        return dataset.data[self.feature_name].unique()
    
    def apply_encoding(self, config, dataset):
        if config["categorical_features"]["encoding"] == "onehot":
            return [0,1]
        elif config["categorical_features"]["encoding"] == "ordinal":
            number_categories = len(dataset.data[self.feature_name].unique())
            return [0, number_categories-1]
        elif config["categorical_features"]["encoding"] == "frequency":
            raise NotImplementedError
        else:
            raise ValueError("Encoding type is not supported")
    
    def get_normalized_range(self, config, dataset):
        if config["continuous_features"]["normalization"] == "minmax":
            return [0,1]
        elif config["continuous_features"]["normalization"] == "standard":
            # Calculate std for dataset[feature_name]
            standartised = (dataset.data[self.feature_name] - self.mean) / self.std
            return [standartised.min(), standartised.max()]
        else:
            raise ValueError("Normalization type is not supported")

    def normalize_cont_value(self, config, original_value):
        if config.continuous_features["normalization"] == "minmax":
            return (original_value - self.min) / (self.max - self.min)
        elif config.continuous_features["normalization"] == "standart":
            return (original_value - self.mean) / self.std
    
    def denormalize_cont_value(self, config, normalized_value):
        if config.continuous_features["normalization"] == "minmax":
            return normalized_value * (self.max - self.min) + self.min
        elif config.continuous_features["normalization"] == "standart":
            return normalized_value * self.std + self.mean
    
    def normalize_cat_value(self, config, original_value):
        """The ordinal encoding looks weird, todo check it later"""
        if config.categorical_features["encoding"] == "onehot":
            return 1
        elif config.categorical_features["encoding"] == "ordinal":
            return original_value
        elif config.categorical_features["encoding"] == "frequency":
            raise NotImplementedError
        
    def denormalize_cat_value(self, config, normalized_value):
        if config.categorical_features["encoding"] == "onehot":
            return 1
        elif config.categorical_features["encoding"] == "ordinal":
            return normalized_value
        elif config.categorical_features["encoding"] == "frequency":
            raise NotImplementedError
