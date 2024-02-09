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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from trustce.cfsearch import CFsearch

from trustce.dataset import Dataset
from trustce import load_datasets
from trustce.cemodels.base_model import BaseModel

from trustce.ceinstance.instance_sampler import CEInstanceSampler
from trustce.config import Config
from trustce.transformer import Transformer
from trustce.ceinstance.instance_factory import InstanceFactory

class RuleModelPipeline:
        
    def predict(self, X):
        """ this is model rule [1 if (x>-1.03 and z>1.3) or (z>0.75 and x>0) else 0 for x,z in zip(df["X"], df["Z"])]
        [1 if (z<-1.4 and x<1.17) or (z<-0.85 and x<0) else old_value for x,z, old_value in zip(df["X"], df["Z"], df["weird_predictor"])]"""

        predictor = np.zeros(X.shape[0])
        # First rule
        for i in range(X.shape[0]):
            x, y, z = X[i, 0], X[i,1], X[i, 2]
            if (x>-0.9 and y>2.7) or (x>-2.8 and y<-5.5) or (x>4.2):
                predictor[i] = 1

        return predictor
    
    def predict_proba(self, X):
        """Make prediction for two classes, where first dimension is probability of predicting class 0 and second dimentions is probability of predicting class 1"""

        predictor = np.zeros((X.shape[0], 2))
        # First rule
        for i in range(X.shape[0]):
            x, y, z = X[i, 0], X[i,1], X[i, 2]
            if (x>-0.9 and y>2.7) or (x>-2.8 and y<-5.5) or (x>4.2):
                predictor[i, 1] = 1

        predictor[:, 0] = 1 - predictor[:, 1]

        return predictor

class TestCFSearch(unittest.TestCase):
    def setUp(self):
        # read config yml file from config folder
        self.config = Config("config/conf_swiss_roll.yaml")
        with open("config/constraints_conf_swiss_roll.json", 'r') as file:
            self.constraints = json.load(file)
        print(self.config)

        self.target_instance_json = "input_instance/instance_swiss_roll.json"
        # Load the model from the file
        #self.svm_model = SVC(kernel='rbf', C=1, gamma='auto', probability=True)
        #self.random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        #self.logistic_model = LogisticRegression()
        self.rule_model = RuleModelPipeline()
        self.prepare_data()

    def prepare_data(self):
        input=pd.read_csv('datasets/swiss_roll.csv',sep=',')
        #input.rename(columns={'Datetime':'DateTime','oudoor_temperature':'outdoor_temperature'},inplace=True)
        #input=self.preprocess_dates_string(input, 'DateTime')

        #get X and y
        print(input.columns)
        input["output"] = input["output"].astype(int)
        X=input.copy().drop(['output'], axis=1)
        y=input.copy()['output']
        #self.svm_model.fit(X, y)
        #self.random_forest_model.fit(X, y)
        #self.logistic_model.fit(X, y)

    def default_serializer(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):  # You can add more numpy types if needed
            return int(obj) if isinstance(obj, (np.int64, np.int32)) else float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    def store_candidates(self, array_candidates_values, file_indicator=""):
        """Store candidates values to file"""
        with open(self.config.get_config_value("output_folder") + "/candidates_values_" + file_indicator + ".json", 'w') as file:
            json.dump(array_candidates_values, file, indent=4, default=self.default_serializer)

    def store_fitness(self, fitnes_history, file_indicator=""):
        """Store fitness history to json file"""
        # Modify array to be able to store it to file
        fitnes = []
        for i in range(len(fitnes_history)):
            fitnes.append(fitnes_history[i][0])
        with open(self.config.get_config_value("output_folder") + "/fitness_history_" + file_indicator + ".json", 'w') as file:
            json.dump(fitnes, file, indent=4, default=self.default_serializer)
            print("fitness_history stored to file in folder: ", self.config.get_config_value("output_folder"))

    def store_loss(self, loss_history, file_indicator=""):
        """Store loss history to json file"""
        with open(self.config.get_config_value("output_folder") + "/loss_history_" + file_indicator + ".json", 'w') as file:
            json.dump(loss_history, file, indent=4, default=self.default_serializer)
            print("loss_history stored to file in folder: ", self.config.get_config_value("output_folder"))

    def store_distance(self, distance_history, file_indicator=""):
        """Store distance history to json file"""
        with open(self.config.get_config_value("output_folder") + "/distance_history_" + file_indicator + ".json", 'w') as file:
            json.dump(distance_history, file, indent=4, default=self.default_serializer)
            print("distance_history stored to file in folder: ", self.config.get_config_value("output_folder"))
        

    def test_cf_search(self):
  
        self.data = Dataset(self.config.get_config_value("dataset"), "active_electricity")
        # Figure out how normalization works
        self.normalization_transformer = Transformer(self.data, self.config)
        self.instance_factory = InstanceFactory(self.data)
        self.sampler = CEInstanceSampler(self.config, self.normalization_transformer, self.instance_factory)

        self.model = BaseModel(self.config.get_config_value("model"), self.rule_model)
        config_for_cfsearch = self.config.get_config_value("cfsearch")
        self.search = CFsearch(self.normalization_transformer, self.model, self.sampler, 
                               config=self.config,
                               optimizer_name=config_for_cfsearch["optimizer"], 
                               distance_continuous=config_for_cfsearch["continuous_distance"], 
                               distance_categorical=config_for_cfsearch["categorical_distance"], 
                               loss_type=config_for_cfsearch["loss_type"], 
                               sparsity_penalty=config_for_cfsearch["sparsity_penalty"]["type"],
                               alpha=config_for_cfsearch["sparsity_penalty"]["alpha"],
                               beta=config_for_cfsearch["sparsity_penalty"]["beta"], 
                               coherence=config_for_cfsearch["coherence"],
                               objective_function_weights=config_for_cfsearch["objective_function_weights"])

        with open(self.target_instance_json, 'r') as file:
            target_instance_json = file.read() #json.load(file)

        target_instance = self.instance_factory.create_instance_from_json(target_instance_json)
        actual_output = self.model.predict_instance(target_instance)

        counterfacturals = self.search.find_counterfactuals(target_instance, 1, "opposite", 50)

        print("counterfactuals", counterfacturals[0].get_values_dict())

        array_candidates_values, fitnes_history, loss_history, distance_history = self.search.draw_trace_search()

        # Store candidates and fitness to files
        self.store_candidates(array_candidates_values, file_indicator="swiss_4_euc")
        self.store_fitness(fitnes_history, file_indicator="swiss_4_euc")
        self.store_loss(loss_history, file_indicator="swiss_4_euc")
        self.store_distance(distance_history, file_indicator="swiss_4_euc")

        self.search.evaluate_counterfactuals(target_instance, counterfacturals)
        # Visualise the values of counterfactuals and original instance only in jupyter notebook
        self.search.visualize_as_dataframe(target_instance, counterfacturals)
        self.search.store_counterfactuals(self.config.get_config_value("output_folder"), "swiss_4_euc")
        self.search.store_evaluations(self.config.get_config_value("output_folder"), "swiss_4_euc")      


def sample_rule_based_functions(target_val):
    return target_val+3

if __name__ == "__main__":
    unittest.main()