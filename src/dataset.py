import pandas as pd

class Dataset(object):
    """
    Dataset class loads raw data and store continious and categorical features
    """
    def __init__(self, path, dataset_config):
        self.path = path
        self.config = dataset_config
        self.data = self.load_dataset()
        self.continuous_features_list, self.categorical_features_list = self.load_features_type(self.config)

        self.verify_features()

    def load_dataset(self):
        """
        Load xls file
        """
        data = pd.read_csv(self.path)
        return data

    def load_features_type(self, config):
        """
        Load features type from config file
        """
        try:
            continuous_features = config['continuous_features']
            categorical_features = config['categorical_features']
        except KeyError:
            print("No features type found in config file")
            continuous_features, categorical_features = self.infer_feature_type_from_dataset()
        return continuous_features, categorical_features

    def check_if_continious(self, feature_name):
        """
        Check if feature in data is indeed contitious
        """
        assert (self.data[feature_name].dtype == 'float64' or self.data[feature_name].dtype == 'int64'), "Feature {} is not continious".format(feature_name)

    def check_if_categorical(self, feature_name):
        """
        Check if feature in data is indeed categorical
        """
        assert (self.data[feature_name].dtype == 'object'), "Feature {} is not categorical".format(feature_name)

    def infer_feature_type_from_dataset(self):
        """
        Infer feature type from dataset
        """
        for feature_name in self.data.columns:
            if self.data[feature_name].dtype == 'object':
                self.categorical_features_list.append(feature_name)
            elif self.data[feature_name].dtype == 'float64' or self.data[feature_name].dtype == 'int64':
                self.continuous_features_list.append(feature_name)
            else:
                raise Exception("Feature {} is not continious or categorical".format(feature_name))

    def verify_features(self):
        """
        Verify if features in config file are indeed continious or categorical
        """
        for feature_name in self.config['continuous_features']:
            self.check_if_continious(feature_name)

        for feature_name in self.config['categorical_features']:
            self.check_if_categorical(feature_name)

        print("Features verified")
        print("Continious features: {}".format(self.continuous_features_list))
        print("Categorical features: {}".format(self.categorical_features_list))
        
