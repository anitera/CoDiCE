import pandas as pd
import numpy as np
import logging
import sklearn
from typing import List, Union
from dataset import Dataset

class FeatureManager(object):
    """
    Manages features and their properties. Gets dataset and configuration. 
    Knows features permitted ranges. Performs normalization and denormalization.
    Calculates meadian abolute deviation of features.
    One-hot encodes categorical features.
    #TODO init feature sampler from these class
    """
    def __init__(self, config, dataset: Dataset):
        self.config = config.feature_manager
        self.continuous_features_list = self.dataset.continuous_features_list
        self.categorical_features_list = self.dataset.categorical_features_list
        self.constraints = self.config.get_config_value('constraints')
        self.outcome_name = self.dataset.outcome_name

        #TODO move to dataset_metadata
        self.init_transformation(dataset)
        self.normalized_train_data = None
        self.encoded_cat_data = None
        self.mean_out1 = None
        self.cov1 = None
        self.mean_out0 = None
        self.cov0 = None
        self.mad = None
        self._init_cov_centroids()
        self._init_mad()


    def _get_normalization(self):
        if self.config.get_config_value('normalization') is None:
            return None
        if self.config.get_config_value('normalization') == 'minmax':
            return sklearn.preprocessing.MinMaxScaler()
        if self.config.get_config_value('normalization') == 'standard':
            return sklearn.preprocessing.StandardScaler()
        else:
            raise ValueError("Normalization type not supported")
    
    def _normalization(self, dataset):
        normalization_type = self._get_normalization()
        if normalization_type is not None:
            self.normalization = normalization_type.fit(dataset[self.continuous_features_list])
            self.transformation.append(self.normalization)

    def _get_cat_engoding(self):
        if self.config.get_config_value('cat_encoding') is None:
            return None
        if self.config.get_config_value('cat_encoding') == 'onehot':
            return sklearn.preprocessing.OneHotEncoder()
        else:
            raise ValueError("Categorical encoding type not supported")

    def _ohe(self, dataset):
        ohe_type = self._get_cat_engoding()
        if ohe_type is not None:
            self.ohe = ohe_type.fit(dataset[self.categorical_features_list])

    def init_transformation(self, dataset):
        self.transformation = []
        self._normalization(dataset)
        self._ohe(dataset)

    def calc_minmax_for_continuous_features(self, dataset: Dataset):
        """
        Calculates min and max for continuous features
        """
        minmax = {}
        for feature in self.continuous_features_list:
            minmax[feature] = {}
            minmax[feature]['min'] = self.dataset[feature].min()
            minmax[feature]['max'] = self.dataset[feature].max()
        return minmax

    def calc_median_absolute_deviation(self):
        """
        Calculates median absolute deviation for continuous features
        """
        mad = {}
        for feature in self.continuous_features_list:
            mad[feature] = self.normalized_train_data[feature].mad()
        return mad

    def get_valid_mads(self, normalized=False, display_warnings=False, return_mads=True):
        """Computes Median Absolute Deviation of features. If they are <=0, returns a practical value instead"""
        mads = self.get_mads(normalized=normalized)
        for feature in mads:
            if mads[feature] <= 0: #TODO what if permitted range is different from observed range?
                mads[feature] = 1.0
                if display_warnings:
                    logging.warning(" MAD for feature %s is 0, so replacing it with 1.0 to avoid error.", feature)


    def get_mads(self, normalized, dataset: Dataset):
        """Computes Median Absolute Deviation of features."""
        mads = {}
        if normalized is False:
            for feature in self.continuous_feature_names:
                mads[feature] = np.median(
                    abs(self.dataset[feature].values - np.median(self.dataset[feature].values)))
        else:
            normalized_train_df = self.normalize_data(self.dataset)
            for feature in self.continuous_feature_names:
                mads[feature] = np.median(
                    abs(normalized_train_df[feature].values - np.median(normalized_train_df[feature].values)))
        return mads

    def normalize_data_cont(self):
        self.normalized_train_data = (self.training_data[self.continuous_features_list]-self.training_data[self.continuous_features_list].min())/(self.training_data[self.continuous_features_list].max()-self.training_data[self.continuous_features_list].min())

    def normalize_instance(self, df):
        result = (df[self.continuous_features_list]-self.training_data[self.continuous_features_list].min())/(self.training_data[self.continuous_features_list].max()-self.training_data[self.continuous_features_list].min())
        return result
    
    def de_normalize_data(self, df):
        result = df[self.continuous_features_list]*(self.training_data[self.continuous_features_list].max()-self.training_data[self.continuous_features_list].min()) + self.training_data[self.continuous_features_list].min()
        return result
    
    def cat_encodings(self):
        #self.encoded_cat_data
        return

    def _init_cov_centroids(self, n_classes=2):
        self.mean_out1 = self.normalized_train_data.loc[self.training_data[self.outcome_name]==1, self.continuous_features_list].mean()
        self.cov1 = np.cov(self.normalized_train_data.loc[self.training_data[self.outcome_name]==1].values.T, bias=True)
        try:
            self.inv_cov_cl1 = np.linalg.inv(self.cov1)
            print("Matrix for class 1 is non-singular")
        except np.linalg.LinAlgError:
            print("Matrix is singular")
            self.inv_cov_cl1 = False

        self.mean_out0 = self.normalized_train_data.loc[self.training_data[self.outcome_name]==0, self.continuous_features_list].mean()
        self.cov0 = np.cov(self.normalized_train_data.loc[self.training_data[self.outcome_name]==0].values.T, bias=True)
        try:
            self.inv_cov_cl0 = np.linalg.inv(self.cov0)
            print("Matrix is non-singular")
        except np.linalg.LinAlgError:
            print("Matrix is singular")
            self.inv_cov_cl0 = False

    def get_mads(self, normalized=True):
        """Computes Median Absolute Deviation of features."""
        if normalized is False:
            return #self.mad.copy() #no idea what is normalized here
        else:
            mads = {}
            for feature in self.continuous_feature_names:
                if feature in self.mad:
                    mads[feature] = (self.mad[feature])/(self.training_data[feature].max() - self.training_data[feature].min())
            return mads

    def get_valid_mads(self, normalized=False, display_warnings=False, return_mads=True):
        """Computes Median Absolute Deviation of features. If they are <=0, returns a practical value instead"""
        mads = self.get_mads(normalized=normalized)
        for feature in self.continuous_feature_names:
            if feature in mads:
                if mads[feature] <= 0:
                    mads[feature] = 1.0
                    if display_warnings:
                        logging.warning(" MAD for feature %s is 0, so replacing it with 1.0 to avoid error.", feature)
            else:
                mads[feature] = 1.0
                if display_warnings:
                    logging.info(" MAD is not given for feature %s, so using 1.0 as MAD instead.", feature)

        if return_mads:
            return mads

    def transform(self, x):
        """
        #TODO
        """
        for t in self.transform:
            x = t.transform(x[self.continuous_features_list])

        if self.ohe is not None:
            ohe_features = self.ohe.transform(x[self.categorical_features_list])
            self.ohe_dims = ohe_features.shape[1]

        x = x[self.continuous_features_list]
        x = np.concatenate((x, ohe_features), axis=1)

        return x

    def inverse_transform(self, x):
        """
        #TODO
        """
        for t in self.transform[::-1]:
            x = t.inverse_transform(x[self.continuous_features_list])

        if self.ohe is not None:
            ohe_features = self.ohe.inverse_transform(x[-self.ohe_dims:])

        x = x[:-1]
        x = np.concatenate((x, ohe_features), axis=1)
        return x
        