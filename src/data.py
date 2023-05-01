import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import OneHotEncoder
from typing import List, Union

class Data():
    def __init__(self, dataframe, continuous_features: List[str], categorical_features: List[str], outcome_name: str, constraints: dict, normalized=False, encoded=False):
        self.training_data = dataframe
        self.continuous_features_list = continuous_features
        self.categorical_features_list = categorical_features
        self.outcome_name = outcome_name
        self.constraints = constraints
        if not normalized:
            self.normalize_data_cont()
        if not encoded:
            self.cat_encodings()
        # ToDO: normalization, categorical feature encoding

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

    def create_ohe_params(self):
        if len(self.categorical_feature_names) > 0:
            # simulating sklearn's one-hot-encoding
            # continuous features on the left
            self.ohe_encoded_feature_names = [
                feature for feature in self.continuous_feature_names]
            for feature_name in self.categorical_feature_names:
                for category in sorted(self.categorical_levels[feature_name]):
                    self.ohe_encoded_feature_names.append(
                        feature_name+'_'+category)
        else:
            # one-hot-encoded data is same as original data if there is no categorical features.
            self.ohe_encoded_feature_names = [feat for feat in self.feature_names]

        # base dataframe for doing one-hot-encoding
        # ohe_encoded_feature_names and ohe_base_df are created (and stored as data class's parameters)
        # when get_data_params_for_gradient_dice() is called from gradient-based DiCE explainers
        self.ohe_base_df = self.prepare_df_for_ohe_encoding()