import numpy as np

class Transformer(object):
    def __init__(self, dataset, config):
        self.continuous_features_transformers = self.get_cont_features_transformers(dataset, config)
        self.categorical_features_transformers = self.get_cat_features_transformers(dataset, config)

        self.diffusion_map = self._get_diffusion_map(dataset, config)
        self.mads = self._get_mads(dataset, config)

    def normalize_instance(self, instance):
        """Return normalize instance"""
        for feature_name, feature in instance.features.items():
            if feature_name in self.continuous_features_transformers:
                feature.value = self.continuous_features_transformers[feature_name].normalize_cont_value(feature.value) #TODO check for over-assignment
            elif feature_name in self.categorical_features_transformers:
                feature.value = self.categorical_features_transformers[feature_name].normalize_cat_value(feature.value)
            else:
                raise ValueError("Feature name is not in continuous or categorical features list")
        

    def get_cont_features_transformers(self, dataset, config):
        return {feature_name: FeatureTransformer(dataset, config, feature_name) for feature_name in dataset.continuous_features_list}

    def get_cat_features_transformers(self, dataset, config):
        return {feature_name: FeatureTransformer(dataset, config, feature_name) for feature_name in dataset.categorical_features_list}
    
    def get_cont_transformers_length(self):
        return len(self.continuous_features_transformers)
    
    def get_cat_transformers_length(self):
        return len(self.categorical_features_transformers)

    def _get_diffusion_map(self, dataset, config):
        """#TODO"""
        return -1

    def _get_mads(self, dataset, ocnfig):
        """#TODO"""
        return -1

class FeatureTransformer(object):
    """
    Store original range of data, normalized range of data and normalization type
    """
    def __init__(self, dataset, config, feature_name):
        self.feature_name = feature_name
        if self.feature_name in dataset.continuous_features_list:
            self.min, self.max, self.mean, self.std, self.median = self.calculate_statistics(dataset)
            self.norm_type = config["continuous_features"]["normalization"]
            if self.norm_type == "minmax":
                self.normalize_cont_value = lambda x: (x - self.min) / (self.max - self.min)
                self.denormalize_cont_value = lambda x: x * (self.max - self.min) + self.min
            elif self.norm_type == "standard":
                self.normalize_cont_value = lambda x: (x - self.mean) / self.std
                self.denormalize_cont_value = lambda x: x * self.std + self.mean

            self.original_range = self.get_from_dataset(dataset)
            self.normalized_range = self.get_normalized_range(dataset)
        elif self.feature_name in dataset.categorical_features_list:
            self.original_range = self.get_feature_choice(dataset)
            self.enc_type = config["categorical_features"]["encoding"] #TODO this could be done better with OOP
            if self.enc_type == "onehot":
                self.normalize_cat_value = self._onehot_enc
                self.denormalize_cat_value = self._onehot_dec
            elif self.enc_type == "ordinal":
                self.normalize_cat_value = self._ordinal_enc
                self.denormalize_cat_value = self._ordinal_dec
            elif self.enc_type == "frequency":
                self.normalize_cat_value = self._freq_enc
                self.denormalize_cat_value = self._freq_dec
            else:
                raise ValueError("Encoding type is not supported")
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
        """Return feature range for categorical features"""
        if self.enc_type == "onehot":
            return [0,1]
        elif self.enc_type == "ordinal":
            number_categories = len(dataset.data[self.feature_name].unique())
            return [0, number_categories-1]
        elif self.enc_type == "frequency":
            raise NotImplementedError
        else:
            raise ValueError("Encoding type is not supported")
    
    def get_normalized_range(self, dataset):
        if self.norm_type == "minmax":
            return [0,1]
        elif self.norm_type == "standard":
            # Calculate std for dataset[feature_name]
            standartised = (dataset.data[self.feature_name] - self.mean) / self.std
            return [standartised.min(), standartised.max()]
        else:
            raise ValueError("Normalization type is not supported")

    # def normalize_cont_value(self, config, original_value):
    #     if config.continuous_features["normalization"] == "minmax":
    #         return (original_value - self.min) / (self.max - self.min)
    #     elif config.continuous_features["normalization"] == "standart":
    #         return (original_value - self.mean) / self.std

    # def denormalize_cont_value(self, config, normalized_value):
    #     if config.continuous_features["normalization"] == "minmax":
    #         return normalized_value * (self.max - self.min) + self.min
    #     elif config.continuous_features["normalization"] == "standart":
    #         return normalized_value * self.std + self.mean
    
    # def normalize_cat_value(self, config, original_value):
    #     """The ordinal encoding looks weird, todo check it later"""
    #     if config.categorical_features["encoding"] == "onehot":
    #         return 1
    #     elif config.categorical_features["encoding"] == "ordinal":
    #         return original_value
    #     elif config.categorical_features["encoding"] == "frequency":
    #         raise NotImplementedError
        
    # def denormalize_cat_value(self, config, normalized_value):
    #     if config.categorical_features["encoding"] == "onehot":
    #         return 1
    #     elif config.categorical_features["encoding"] == "ordinal":
    #         return normalized_value
    #     elif config.categorical_features["encoding"] == "frequency":
    #         raise NotImplementedError
        
    def _onehot_enc(self, value):
        return 1
    
    def _ordinal_enc(self, value):
        return value

    def _freq_enc(self, value):
        raise NotImplementedError

    def _onehot_dec(self, value):
        return 1
    
    def _ordinal_dec(self, value):
        return value

    def _freq_dec(self, value):
        raise NotImplementedError
