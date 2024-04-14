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
from codice.cfsearch import CFsearch

from codice.dataset import Dataset
from codice import load_datasets
from codice.cemodels.sklearn_model import SklearnModel
from codice.cemodels.sklearn_pipeline import SklearnPipeline
from codice.cemodels.base_model import BaseModel

from codice.ceinstance.instance_sampler import CEInstanceSampler
from codice.config import Config
from codice.transformer import Transformer
from codice.ceinstance.instance_factory import InstanceFactory


class TestCFSearch(unittest.TestCase):
    def setUp(self):
        # read config yml file from config folder
        self.config = Config("config/conf_homeloan_coherence.yaml")
        with open("config/constraints_homeloan_ch.json", 'r') as file:
            self.constraints = json.load(file)
        print(self.config)

        self.target_instance_json = "input_instance/instance.json"
        self.prepare_data()
        
    def prepare_data(self):
        input=pd.read_csv('datasets/homeloan_clean.csv',sep=',')
        #input.rename(columns={'Datetime':'DateTime','oudoor_temperature':'outdoor_temperature'},inplace=True)
        #input=self.preprocess_dates_string(input, 'DateTime')
        input["Loan_Status"] = input["Loan_Status"].astype('int')
        #get X and y
        print(input.columns)

        #scaler = MinMaxScaler()
        #X_normalized = scaler.fit_transform(X)
        continuous_features_list = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
        categorical_features_list = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        # restructure the dataframme so that categorical features are first and continuous features are last
        input = pd.concat([input[categorical_features_list], input[continuous_features_list], input["Loan_Status"]], axis=1)
        # Store dataframe to csv
        input.to_csv("datasets/homeloan_oredered_features.csv", index=False)
        # Get 100 instances with prediction 1 and 100 with prediction 0 from input
        pred1_instances = input[input["Loan_Status"] == 1].head(100)
        pred0_instances = input[input["Loan_Status"] == 0].head(100)
        self._combined_instances = pd.concat([pred1_instances, pred0_instances])
        # Drop the target column from the input
        self._combined_instances = self._combined_instances.drop(["Loan_Status"], axis=1)
        X=input.copy().drop(["Loan_Status"], axis=1)
        y=input.copy()["Loan_Status"]
        get_index_number_of_cont_features = [X.columns.get_loc(c) for c in continuous_features_list if c in X]
        get_index_number_of_cat_features = [X.columns.get_loc(c) for c in categorical_features_list if c in X]
        print(get_index_number_of_cont_features)
        print(get_index_number_of_cat_features)
        # Converting categorical columns to numerical values
        #for col in categorical_features_list:
        #    for i in range(X[col].nunique()):
        #        X.loc[X[col] == X[col].unique()[i], col] = i
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        self._x_train = X_train
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
        print("Accuracy on training set: ", self.model_pipeline.score(X_train, y_train))
        print("Accuracy on test set: ", self.model_pipeline.score(X_test, y_test))
        # Save the model to models/homeloan_logistic_model.pkl
        with open("models/homeloan_logistic_model.pkl", 'wb') as file:
            pickle.dump(self.model_pipeline, file)



    def test_cf_search(self):
        #load_datasets.download("homeloan")
        self.data = Dataset(self.config.get_config_value("dataset"), "Loan_Status")
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

        with open(self.target_instance_json, 'r') as file:
            target_instance_json = file.read() #json.load(file)

        target_instance = self.instance_factory.create_instance_from_json(target_instance_json)

        counterfacturals = self.search.find_counterfactuals(target_instance, 1, "opposite", 100)


        self.search.evaluate_counterfactuals(target_instance, counterfacturals)
        # Visualise the values of counterfactuals and original instance only in jupyter notebook
        self.search.visualize_as_dataframe(target_instance, counterfacturals)
        self.search.store_counterfactuals(self.config.get_config_value("output_folder"), "first_test_weighted")
        self.search.store_evaluations(self.config.get_config_value("output_folder"), "first_eval_weighted")  

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


def sample_rule_based_functions(target_val):
    return target_val+3

if __name__ == "__main__":
    unittest.main()