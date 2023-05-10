import configparser
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from data import Data
from model import ModelWrapper
from explainer import CFCOG
from config import Config
from dataset import Dataset
from explainable_model import ExplainableModel
from feature_manager import FeatureManager

import argparse

def load_dataset():
    # Read the configuration file
    config_dict = {}
    file_path = '/home/rita/TRUST_AI/CF_Cog_biased/CFCOG/conf.txt'
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            contents = file.readlines()
            for line in contents:
                # Remove leading/trailing whitespaces and newlines
                line = line.strip()
                # Skip comments starting with '#' or lines without '='
                if line.startswith('#') or '=' not in line:
                    continue
                # Split line into parameter name and value
                parameter_name, parameter_value = line.split('=', 1)
                # Add parameter name and value to dictionary
                config_dict[parameter_name.strip()] = parameter_value.strip()

    print(config_dict)

    train = pd.read_csv("/home/rita/TRUST_AI/datasets/homeloan_train.xls")
    continuous_features_list = ['ApplicantIncome','CoapplicantIncome','LoanAmount', 'Loan_Amount_Term']
    categorical_features_list = train.columns.difference(continuous_features_list)
    categorical_features_list = categorical_features_list.drop("Loan_Status")

    train["Gender"].fillna(train["Gender"].mode()[0],inplace=True)
    train["Married"].fillna(train["Married"].mode()[0],inplace=True)
    train['Dependents'].fillna(train["Dependents"].mode()[0],inplace=True)
    train["Self_Employed"].fillna(train["Self_Employed"].mode()[0],inplace=True)
    train["Credit_History"].fillna(train["Credit_History"].mode()[0],inplace=True)
    train = train.dropna(subset=['LoanAmount', 'Loan_Amount_Term'])

    train.loc[train["Loan_Status"]=="Y", "Loan_Status"] = 1
    train.loc[train["Loan_Status"]=="N", "Loan_Status"] = 0
    train["Loan_Status"] = train["Loan_Status"].astype('int')

    return train, continuous_features_list, categorical_features_list, config_dict

def split_dataset(train, outcome_column_name):
    X=train.drop(outcome_column_name,1)
    y=train[[outcome_column_name]]

    x_train,x_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=1)

    return x_train, x_val, y_train, y_val

def buil_pipeline(continuous_features_list, categorical_features_list):

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    transformations = ColumnTransformer(transformers=[
        ('num', numeric_transformer, continuous_features_list),
        ('cat', categorical_transformer, categorical_features_list)])

    clf = Pipeline(steps=[('preprocessor', transformations),
                      ('classifier', LogisticRegression())])
    return clf


def train_model(clf, x_train, x_val, y_train, y_val):

    logistic_model = clf.fit(x_train, y_train.values.ravel())
    y_pred = logistic_model.predict(x_val)
    val_accuracy = accuracy_score(y_pred, y_val)*100
    print(val_accuracy)

    return logistic_model

def test_case():
    """
    Train val model
    """
    x_train, x_val, y_train, y_val, continuous_features_list, categorical_features_list, config_dict = load_dataset()

    clf = buil_pipeline(continuous_features_list, categorical_features_list)

    logistic_model = train_model(clf, x_train, x_val, y_train, y_val)


def read_arguments():
    parser = argparse.ArgumentParser(description='CFCOG')
    parser.add_argument('--config', type=str, default='config/conf.yaml', help='Path to the configuration file')
    parser.add_argument('--output', type=str, default='Loan_Status', help='Output to explain')
    parser.add_argument('--instance', type=str, default='0', help='Instance to explain')
    parser.add_argument('--constraints', type=str, default='constraints.txt', help='Path to the constraints file')
    parser.add_argument('--output_dir', type=str, default='results', help='Path to the output directory')
    parser.add_argument('--output_file', type=str, default='explanation.txt', help='Path to the output file')
    parser.add_argument('--verbose', type=bool, default=False, help='Verbose mode')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode')
    args = parser.parse_args()

    return args

def main():
    # Load config file

    args = read_arguments()

    config = Config(args.config)

    # Load dataset
    train, continuous_features_list, categorical_features_list, config_dict = load_dataset()
    x_train, x_val, y_train, y_val = split_dataset(train, "Loan_Status")
    # Create Data, ModelWrapper, and CounterfactualExplainer objects
    dataset = Dataset(args.dataset)

    # Load/train model
    clf = buil_pipeline(continuous_features_list, categorical_features_list)
    logistic_model = train_model(clf, x_train, x_val, y_train, y_val)
    #model = ExplainableModel(logistic_model, args.config)

    # Create feature manager
    data = Data(dataframe=train, continuous_features=continuous_features_list, categorical_features=categorical_features_list, outcome_name="Loan_Status", constraints=config_dict)

    # Create explainer
    model_wrapper = ModelWrapper(model=logistic_model)  # Pass the model object
    explainer = CFCOG(data, model_wrapper, config_dict)

    # Explain instance
    instance = train.sample(1)
    desired_output = 1 - instance["Loan_Status"].values[0]  # Flip the outcome
    instance = instance.drop("Loan_Status", axis=1)
    # Call the explain method with the instance and desired output
    explanation = explainer.explain(instance, desired_output)


if __name__ == "__main__":
    main()