import unittest
import pandas as pd
import os
import sys
import json
import pickle
import numpy as np
import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.preprocessing import StandardScaler
from gplearn.genetic import SymbolicRegressor

from trustce.cfsearch import CFsearch

from trustce.dataset import Dataset
from trustce import load_datasets
from trustce.cemodels.base_model import BaseModel

from trustce.ceinstance.instance_sampler import CEInstanceSampler
from trustce.config import Config
from trustce.transformer import Transformer
from trustce.ceinstance.instance_factory import InstanceFactory

class GPModelPipeline:
    def __init__(self, X_scaler, y_scaler, estimator):
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler
        self.estimator = estimator

    def fit(self, X, y):
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y[:, np.newaxis])
        self.estimator.fit(X_scaled, y_scaled)

    def predict(self, X):
        X_scaled = self.X_scaler.transform(X)
        y_scaled = self.estimator.predict(X_scaled)
        y = self.y_scaler.inverse_transform(y_scaled[:, np.newaxis])
        return y.squeeze()

class TestCFSearch(unittest.TestCase):
    def setUp(self):
        # read config yml file from config folder
        self.config = Config("config/conf_energy_summer.yaml")
        with open("config/constraints_conf_energy.json", 'r') as file:
            self.constraints = json.load(file)
        print(self.config)

        self.target_instance_json = "input_instance/instance_energy_summer.json"
        # Load the model from the file
        self.prepare_data()
            
    def preprocess_dates_string(self, dataframe, date_column_name):
        '''
        function to preprocess date strings from the df, (changes time 0:00 to format 00:00)

        params
        dataframe: a pandas dataframe object containing the column with the dates
        date_column_name: The column name of the column with dates to be modified

        returns:
        the updated dataframe with corrected format date values

        '''
        for i in range(len(dataframe)):
            if len(dataframe.iloc[i][0].split(' ')[-1].split(':')[0])<2:
                #replace old date strings with the modified ones
                dataframe.at[i, date_column_name] = dataframe.iloc[i][0].split(' ')[0]+ " 0"+dataframe.iloc[i][0].split(' ')[-1]
        return dataframe
    
    def train_test_split(self, X, y):
        '''
        function to split data into training and testing sets
        '''
        n=len(X)
        X_train, X_test = X[0:int(n*0.85)],X[int(n*0.85):]
        y_train, y_test = y[0:int(n*0.85)],y[int(n*0.85):]

        return X_train, X_test, y_train, y_test
        
    def prepare_data(self):
        input=pd.read_csv('datasets/summer_data.csv',sep=',')
        input.rename(columns={'Datetime':'DateTime','oudoor_temperature':'outdoor_temperature'},inplace=True)
        input=self.preprocess_dates_string(input, 'DateTime')

        #get X and y
        print(input.columns)
        X=input.copy().drop(['DateTime','active_electricity'], axis=1)
        y=input.copy()['active_electricity']

        #split data
        X_train, X_test, y_train, y_test = self.train_test_split(X, y)

        target_feature= 'active_electricity'
        gp_scaled = SymbolicRegressor(population_size=68,
                           generations=75, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.0001,random_state=0,metric='rmse',feature_names=X_train.columns)

        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        # create the custom pipeline object
        self.gp_model_pipeline = GPModelPipeline(X_scaler,y_scaler, gp_scaled)

        # fit the model
        self.gp_model_pipeline.fit(X_train, np.array(y_train))
        

    def test_cf_search(self):
  
        self.data = Dataset(self.config.get_config_value("dataset"), "active_electricity")
        self.normalization_transformer = Transformer(self.data, self.config)
        self.instance_factory = InstanceFactory(self.data)
        self.sampler = CEInstanceSampler(self.config, self.normalization_transformer, self.instance_factory)

        self.model = BaseModel(self.config.get_config_value("model"), self.gp_model_pipeline)
        config_for_cfsearch = self.config.get_config_value("cfsearch")
        self.search = CFsearch(self.normalization_transformer, self.model, self.sampler, 
                               config=self.config,
                               optimizer_name=config_for_cfsearch["optimizer"], 
                               distance_continuous=config_for_cfsearch["continuous_distance"], 
                               distance_categorical=config_for_cfsearch["categorical_distance"], 
                               loss_type=config_for_cfsearch["loss_type"], 
                               coherence=config_for_cfsearch["coherence"],
                               objective_function_weights=config_for_cfsearch["objective_function_weights"])

        with open(self.target_instance_json, 'r') as file:
            target_instance_json = file.read() #json.load(file)

        target_instance = self.instance_factory.create_instance_from_json(target_instance_json)
        actual_output = self.model.predict_instance(target_instance)
        # 5% decreased output range
        target_output_upper_bound = actual_output * 0.95
        target_output_lower_bound = actual_output * 0.9

        counterfacturals = self.search.find_counterfactuals(target_instance, 1, [target_output_upper_bound, target_output_lower_bound], 100)


        self.search.evaluate_counterfactuals(target_instance, counterfacturals)
        # Visualise the values of counterfactuals and original instance only in jupyter notebook
        self.search.visualize_as_dataframe(target_instance, counterfacturals)
        self.search.store_counterfactuals(self.config.get_config_value("output_folder"), "energy_test1")
        self.search.store_evaluations(self.config.get_config_value("output_folder"), "energy_eval1")      


def sample_rule_based_functions(target_val):
    return target_val+3

if __name__ == "__main__":
    unittest.main()