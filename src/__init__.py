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


if __name__ == "__main__":
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

    train["Gender"].fillna(train["Gender"].mode()[0],inplace=True)
    train["Married"].fillna(train["Married"].mode()[0],inplace=True)
    train['Dependents'].fillna(train["Dependents"].mode()[0],inplace=True)
    train["Self_Employed"].fillna(train["Self_Employed"].mode()[0],inplace=True)
    train["Credit_History"].fillna(train["Credit_History"].mode()[0],inplace=True)
    train = train.dropna(subset=['LoanAmount', 'Loan_Amount_Term'])

    train.loc[train["Loan_Status"]=="Y", "Loan_Status"] = 1
    train.loc[train["Loan_Status"]=="N", "Loan_Status"] = 0
    train["Loan_Status"] = train["Loan_Status"].astype('int')

    X=train.drop("Loan_Status",1)
    y=train[["Loan_Status"]]

    x_train,x_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=1)

    categorical_features_list = x_train.columns.difference(continuous_features_list)

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    transformations = ColumnTransformer(transformers=[
        ('num', numeric_transformer, continuous_features_list),
        ('cat', categorical_transformer, categorical_features_list)])

    clf = Pipeline(steps=[('preprocessor', transformations),
                      ('classifier', LogisticRegression())])

    logistic_model = clf.fit(x_train, y_train.values.ravel())
    y_pred = logistic_model.predict(x_val)
    val_accuracy = accuracy_score(y_pred, y_val)*100
    print(val_accuracy)

    # Create Data, ModelWrapper, and CounterfactualExplainer objects
    data = Data(dataframe=train, continuous_features=continuous_features_list, categorical_features=categorical_features_list, outcome_name="Loan_Status", constraints=config_dict)
    model_wrapper = ModelWrapper(model=logistic_model)  # Pass the model object
    explainer = CFCOG(data, model_wrapper, config_dict)

    # Call the explain method with the instance and desired output
    explanation = explainer.explain(instance, desired_output)