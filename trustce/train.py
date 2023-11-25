from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import configparser
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from trustce.cemodels.explainable_model import ExplainableModel

def load_dataset(confpath, datasetpath):
    # Read the configuration file
    config_dict = {}
    file_path = confpath
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

    train = pd.read_csv(datasetpath)
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

def get_transformations(continuous_features_list, categorical_features_list):

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    transformations = ColumnTransformer(transformers=[
        ('num', numeric_transformer, continuous_features_list),
        ('cat', categorical_transformer, categorical_features_list)])

    return transformations


def train_model(transformations, x_train, x_val, y_train, y_val):
    x_train = transformations.fit_transform(x_train)
    x_val = transformations.transform(x_val)
    model = LogisticRegression()

    logistic_model = model.fit(x_train, y_train.values.ravel())
    y_pred = logistic_model.predict(x_val)
    val_accuracy = accuracy_score(y_pred, y_val)*100
    print(val_accuracy)

    return logistic_model

def test_case():
    """
    Train val model
    """
    x_train, x_val, y_train, y_val, continuous_features_list, categorical_features_list, config_dict = load_dataset()

    clf = get_transformations(continuous_features_list, categorical_features_list)

    logistic_model = train_model(clf, x_train, x_val, y_train, y_val)