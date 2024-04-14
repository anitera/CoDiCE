import unittest
import pandas as pd
import os
import sys
import json
import pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from trustce.cfsearch import CFsearch

from trustce.dataset import Dataset
from trustce import load_datasets
from trustce.cemodels.sklearn_model import SklearnModel
from trustce.cemodels.sklearn_pipeline import SklearnPipeline
from trustce.cemodels.base_model import BaseModel

from trustce.ceinstance.instance_sampler import CEInstanceSampler
from trustce.config import Config
from trustce.transformer import Transformer
from trustce.ceinstance.instance_factory import InstanceFactory
import numpy as np


class TestCFSearch(unittest.TestCase):
    def setUp(self):
        # read config yml file from config folder
        self.config = Config("config/conf_german.yaml")
        with open("config/constraints_german.json", 'r') as file:
            self.constraints = json.load(file)
        print(self.config)

        #self.target_instance_json = "input_instance/instance.json"
        self.prepare_data()
        
    def prepare_data(self):
        input=pd.read_csv('datasets/german.csv',sep=',')
        #input.rename(columns={'Datetime':'DateTime','oudoor_temperature':'outdoor_temperature'},inplace=True)
        #input=self.preprocess_dates_string(input, 'DateTime')
        target_name = "default"
        input[target_name] = input[target_name].astype('int')
        continuous_features = self.config.get_config_value("dataset")["continuous_features"]
        categorical_features = self.config.get_config_value("dataset")["categorical_features"]
        input = self.remove_missing_values(input)
        input = input.dropna()
        input = input[continuous_features + categorical_features + [target_name]]
        input.to_csv("datasets/german_ordered_features.csv", index=False)
        self.full_input = input

        # Print datatypes of all columns
        print(input.dtypes)
  
        X=input.copy().drop([target_name], axis=1)
        y=input.copy()[target_name]
        get_index_number_of_cont_features = [X.columns.get_loc(c) for c in continuous_features if c in X]
        get_index_number_of_cat_features = [X.columns.get_loc(c) for c in categorical_features if c in X]
        print(get_index_number_of_cont_features)
        print(get_index_number_of_cat_features)
        # Converting categorical columns to numerical values
        #for col in categorical_features_list:
        #    for i in range(X[col].nunique()):
        #        X.loc[X[col] == X[col].unique()[i], col] = i
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # Create transformers for continuous and categorical features
        continuous_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Create the ColumnTransformer to apply the transformations to the correct columns in the dataframe
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', continuous_transformer, get_index_number_of_cont_features),
                ('cat', categorical_transformer, get_index_number_of_cat_features)
            ])

        # Create the full pipeline
        self.model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000))
        ])

        self.model_pipeline.fit(X_train, y_train)
        # Check the accuracy of the model
        # Test prediction on one test instance transformed to array
        X_test_instance = X_test.iloc[0].to_numpy().reshape(1, -1)
        print("Instance to predict: ", X_test_instance)
        print("Prediction on test instance: ", self.model_pipeline.predict(X_test_instance))

        print("Accuracy on training set: ", self.model_pipeline.score(X_train, y_train))
        print("Accuracy on test set: ", self.model_pipeline.score(X_test, y_test))
        # Save the model to models/homeloan_logistic_model.pkl
        with open("models/german_logistic_model.pkl", 'wb') as file:
            pickle.dump(self.model_pipeline, file)

    def remove_missing_values(self, df):
        '''
        Remove the rows with missing value in the dataframe.
        '''
        # df = pd.DataFrame(df.to_dict())
        for col in df.columns:
            if '?' in list(df[col].unique()):
                ### Replace the missing value by the most common.
                df[col][df[col] == '?'] = df[col].value_counts().index[0] 
        return df



    def test_cf_search(self):
        #load_datasets.download("homeloan")
        import time
        self.data = Dataset(self.config.get_config_value("dataset"), "default")
        self.normalization_transformer = Transformer(self.data, self.config)
        self.instance_factory = InstanceFactory(self.data)

        custom_rules = {"coapplicantRule": sample_rule_based_functions}

        self.sampler = CEInstanceSampler(self.config, self.normalization_transformer, self.instance_factory, custom_rules=custom_rules)

        self.model = BaseModel(self.config.get_config_value("model"), self.model_pipeline)
        config_for_cfsearch = self.config.get_config_value("cfsearch")
        self.search = CFsearch(self.normalization_transformer, self.model, self.sampler, 
                               config=self.config,
                               optimizer_name=config_for_cfsearch["optimizer"], 
                               distance_continuous=config_for_cfsearch["continuous_distance"], 
                               distance_categorical=config_for_cfsearch["categorical_distance"], 
                               loss_type=config_for_cfsearch["loss_type"],
                               sparsity=config_for_cfsearch["sparsity"],
                               coherence=config_for_cfsearch["coherence"],
                               objective_function_weights=config_for_cfsearch["objective_function_weights"])

        test_indices = pd.read_csv("datasets/german_test_indices.csv", header=None).to_numpy().squeeze(1)
        print(test_indices)
        #mask = full_csv.index.isin(test_indices)
        test_df = self.full_input.iloc[test_indices]
        print(test_df.head())
        time_taken = []

        all_counterfactuals = test_df[:100].copy()
        for idx, row_instance in test_df[:100].iterrows():
            
            target_instance = self.instance_factory.create_instance_from_df_row(row_instance)
            start_time = time.time()
            counterfactuals = self.search.find_counterfactuals(target_instance, 1, "opposite", 100)
            end_time = time.time()
            all_counterfactuals.loc[idx] = pd.Series(counterfactuals[0].get_values_dict())
            time_taken.append(end_time - start_time)
 
        print(f"Average time taken per cf search: {np.mean(time_taken)} seconds")
        test_df[:100].to_csv("results/original100_german_train_weighted.csv", index=False)
        all_counterfactuals.to_csv("results/counterfactuals100_german_train_weighted.csv", index=False) 
        # If you also want to save the timing information to a file
        with open('results/german100_timing_info_weighted.txt', 'w') as f:
            for time in time_taken:
                f.write(f"{time}\n")       
"""
    def test_cf_search_for_train(self):
        self.data = Dataset(self.config.get_config_value("dataset"), "Loan_Status")
        self.normalization_transformer = Transformer(self.data, self.config)
        self.instance_factory = InstanceFactory(self.data)

        #custom_rules = {"coapplicantRule": sample_rule_based_functions}

        self.sampler = CEInstanceSampler(self.config, self.normalization_transformer, self.instance_factory)

        self.model = BaseModel(self.config.get_config_value("model"), self.model_pipeline)
        config_for_cfsearch = self.config.get_config_value("cfsearch")
        self.search = CFsearch(self.normalization_transformer, self.model, self.sampler, 
                               config=self.config,
                               optimizer_name=config_for_cfsearch["optimizer"], 
                               distance_continuous=config_for_cfsearch["continuous_distance"], 
                               distance_categorical=config_for_cfsearch["categorical_distance"], 
                               loss_type=config_for_cfsearch["loss_type"],
                               sparsity=config_for_cfsearch["sparsity"], 
                               coherence=config_for_cfsearch["coherence"],
                               objective_function_weights=config_for_cfsearch["objective_function_weights"])

        #Get 100 instances with prediction 1 and 100 with prediction 0 from X_train

        all_counterfactuals = self._combined_instances.copy()
        for idx, row_instance in self._combined_instances.iterrows():
            target_instance = self.instance_factory.create_instance_from_df_row(row_instance)
            counterfactuals = self.search.find_counterfactuals(target_instance, 1, "opposite", 100)
            all_counterfactuals.loc[idx] = pd.Series(counterfactuals[0].get_values_dict())

        
        self._combined_instances.to_csv("results/original_homeloan_train.csv", index=False)
        all_counterfactuals.to_csv("results/counterfactuals_homeloan_train_diff_1.csv", index=False)    
"""

def sample_rule_based_functions(target_val):
    return target_val+3

if __name__ == "__main__":
    unittest.main()