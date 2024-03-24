import unittest
import pandas as pd
import numpy as np
import os
import sys
import pickle
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from trustce.cfsearch import CFsearch

from trustce.dataset import Dataset
from trustce import load_datasets
from trustce.cemodels.sklearn_model import SklearnModel
from trustce.cemodels.base_model import BaseModel

from trustce.ceinstance.instance_sampler import CEInstanceSampler
from trustce.config import Config
from trustce.transformer import Transformer
from trustce.ceinstance.instance_factory import InstanceFactory

class TestCFSearch(unittest.TestCase):
    def setUp(self):
        # read config yml file from config folder
        self.config = Config("config/conf_bc.yaml")
        with open("config/constraints_conf_bc.json", 'r') as file:
            self.constraints = json.load(file)
        print(self.config)

        #self.target_instance_json = "input_instance/instance_diabetes.json"
        self.prepare_data()
        
    def prepare_data(self):
        input=pd.read_csv('datasets/breast_cancer.csv',sep=',')
        #input.rename(columns={'Datetime':'DateTime','oudoor_temperature':'outdoor_temperature'},inplace=True)
        #input=self.preprocess_dates_string(input, 'DateTime')
        target_name = 'diagnosis'
        continuous_features = self.config.get_config_value("dataset")["continuous_features"]
        categorical_features = self.config.get_config_value("dataset")["categorical_features"]

        del input['id']
        del input['Unnamed: 32']        

        #input[target_name] = input[target_name].apply(lambda x: "Y" if x==1 else "N")

        feature_names = [col for col in input.columns if col != target_name]

        input = self.remove_missing_values(input)
        input = input.dropna()
        input = input[continuous_features + categorical_features + [target_name]]

        possible_outcomes = list(input[target_name].unique())
        print("Possible outcomes: ", possible_outcomes)

        self.full_input = input
        #get X and y
        print(input.columns)
        X=input.copy().drop([target_name], axis=1)
        y=input.copy()[target_name]
        #scaler = MinMaxScaler()
        #X_normalized = scaler.fit_transform(X)

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        self._x_train = X_train

        # Optimized parameters
        C_optimized = 0.23357214690901212
        class_weight_optimized = 'balanced'
        solver_optimized = 'liblinear'
        self.model_pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('model', LogisticRegression(max_iter=1000))
        ])
        self.model_pipeline.fit(X_train, y_train)
        # Check the accuracy of the model
        print("Accuracy on training set: ", self.model_pipeline.score(X_train, y_train))
        print("Accuracy on test set: ", self.model_pipeline.score(X_test, y_test))
        # Store the model to file
        with open("models/breast_cancer_model.pkl", 'wb') as file:
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

    def store_fitness(self, fitness_history, file_indicator=""):
        """Store fitness history to json file"""
        # Modify array to be able to store it to file
        fitness = []
        for i in range(len(fitness_history)):
            fitness.append(fitness_history[i])
        with open(self.config.get_config_value("output_folder") + "/fitness_history_" + file_indicator + ".json", 'w') as file:
            json.dump(fitness, file, indent=4, default=self.default_serializer)
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
        #load_datasets.download("homeloan")
        from time import time
        self.data = Dataset(self.config.get_config_value("dataset"), "diagnosis")
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

        #with open(self.target_instance_json, 'r') as file:
        #    target_instance_json = file.read() #json.load(file)

        #target_instance = self.instance_factory.create_instance_from_json(target_instance_json)

        #counterfacturals = self.search.find_counterfactuals(target_instance, 1, "opposite", 100)
        test_indices = pd.read_csv("datasets/breast_cancer_test_indices.csv", header=None).to_numpy().squeeze(1)
        print(test_indices)
        #mask = full_csv.index.isin(test_indices)
        test_df = self.full_input.iloc[test_indices]
        print(test_df.head())
        time_taken = []

        all_counterfactuals = test_df.copy()
        for idx, row_instance in test_df.iterrows():
            
            target_instance = self.instance_factory.create_instance_from_df_row(row_instance)
            start_time = time()
            counterfactuals = self.search.find_counterfactuals(target_instance, 1, "opposite", 100)
            end_time = time()
            all_counterfactuals.loc[idx] = pd.Series(counterfactuals[0].get_values_dict())
            time_taken.append(end_time - start_time)
 
        print(f"Average time taken per cf search: {np.mean(time_taken)} seconds")
        test_df.to_csv("results/original_breast_cancer_diffusion_ablation.csv", index=False)
        all_counterfactuals.to_csv("results/counterfactuals_breast_cancer_diffusion_ablation_coherence15.csv", index=False) 
        # If you also want to save the timing information to a file
        with open('results/breast_cancer_timing_info_ablation_coh.txt', 'w') as f:
            for time in time_taken:
                f.write(f"{time}\n")     

    """
    def test_cf_search_for_train(self):
        self.data = Dataset(self.config.get_config_value("dataset"), "Class variable")
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
                               sparsity_penalty=config_for_cfsearch["sparsity_penalty"]["type"],
                               alpha=config_for_cfsearch["sparsity_penalty"]["alpha"],
                               beta=config_for_cfsearch["sparsity_penalty"]["beta"], 
                               coherence=config_for_cfsearch["coherence"],
                               objective_function_weights=config_for_cfsearch["objective_function_weights"])

        all_counterfactuals = self._x_train.copy()
        for idx, row_instance in self._x_train.iterrows():
            target_instance = self.instance_factory.create_instance_from_df_row(row_instance)
            counterfactuals = self.search.find_counterfactuals(target_instance, 1, "opposite", 100)
            all_counterfactuals.loc[idx] = pd.Series(counterfactuals[0].get_values_dict())

        
        self._x_train.to_csv("results/original_diabetes_train.csv", index=False)
        all_counterfactuals.to_csv("results/counterfactuals_diabetes_train_weighted.csv", index=False)
    """    


def sample_rule_based_functions(target_val):
    return target_val+3

if __name__ == "__main__":
    unittest.main()