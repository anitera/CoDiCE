
class DatasetMetadata(object):
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

    def get_normalization_continuous(self):
        if self.config.continuous_features["normalization"] == "minmax":
            for feature in self.dataset.continuous_features_list:
                self.dataset.minmax[feature]["min"] = self.dataset.data[feature].min()
                self.dataset.minmax[feature]["max"] = self.dataset.data[feature].max()
            
        elif self.config.continuous_features["normalization"] == "standart":
            return self.dataset.get_normalization_continuous_standart()
        else:
            raise ValueError("Normalization type is not supported")
