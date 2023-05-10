import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset(object):
    """
    Dataset class loads raw data and store continious and categorical features
    """
    def __init__(self, path, outcome_column_name, dataset_config):
        self.path = path
        self.outcome_column_name = outcome_column_name
        self.config = dataset_config
        self.data = self.load_dataset()
        self.continuous_features_list, self.categorical_features_list = self.load_features_type(self.config)

        self.verify_features()
        self.preprocess_dataset()

    def load_dataset(self):
        """
        Load xls file
        """
        data = pd.read_csv(self.path)
        return data

    def load_features_type(self, config):
        """
        Load features type from config file
        """
        try:
            continuous_features = config['continuous_features']
            categorical_features = config['categorical_features']
        except KeyError:
            print("No features type found in config file")
            continuous_features, categorical_features = self.infer_feature_type_from_dataset()
        return continuous_features, categorical_features

    def check_if_continious(self, feature_name):
        """
        Check if feature in data is indeed contitious
        """
        assert (self.data[feature_name].dtype == 'float64' or self.data[feature_name].dtype == 'int64'), "Feature {} is not continious".format(feature_name)

    def check_if_categorical(self, feature_name):
        """
        Check if feature in data is indeed categorical
        """
        assert (self.data[feature_name].dtype == 'object'), "Feature {} is not categorical".format(feature_name)

    def infer_feature_type_from_dataset(self):
        """
        Infer feature type from dataset
        """
        for feature_name in self.data.columns:
            if self.data[feature_name].dtype == 'object':
                self.categorical_features_list.append(feature_name)
            elif self.data[feature_name].dtype == 'float64' or self.data[feature_name].dtype == 'int64':
                self.continuous_features_list.append(feature_name)
            else:
                raise Exception("Feature {} is not continious or categorical".format(feature_name))

    def verify_features(self):
        """
        Verify if features in config file are indeed continious or categorical
        """
        for feature_name in self.config['continuous_features']:
            self.check_if_continious(feature_name)

        for feature_name in self.config['categorical_features']:
            self.check_if_categorical(feature_name)

        print("Features verified")
        print("Continious features: {}".format(self.continuous_features_list))
        print("Categorical features: {}".format(self.categorical_features_list))

    def preprocess_dataset(self):
        """
        Preprocess dataset
        """
        if self.outcome_column_name == "Loan_Status":
            self.data.loc[self.data["Loan_Status"]=="Y", "Loan_Status"] = 1
            self.data.loc[self.data["Loan_Status"]=="N", "Loan_Status"] = 0
            self.data["Loan_Status"] = self.data["Loan_Status"].astype('int')

            self.data["Gender"].fillna(self.data["Gender"].mode()[0], inplace=True)
            self.data["Married"].fillna(self.data["Married"].mode()[0], inplace=True)
            self.data['Dependents'].fillna(self.data["Dependents"].mode()[0], inplace=True)
            self.data["Self_Employed"].fillna(self.data["Self_Employed"].mode()[0], inplace=True)
            self.data["Credit_History"].fillna(self.data["Credit_History"].mode()[0], inplace=True)
            self.data = self.data.dropna(subset=['LoanAmount', 'Loan_Amount_Term'])

        self.data = self.data.dropna(subset=self.config['continuous_features'])
        self.data = self.data.dropna(subset=self.config['categorical_features'])

        self.data = self.data.reset_index(drop=True)

        self.data = self.data.drop_duplicates()

        self.data = self.data.reset_index(drop=True)

        print("Dataset preprocessed")


    def split_dataset(train, outcome_column_name):
        X=train.drop(outcome_column_name,1)
        y=train[[outcome_column_name]]

        x_train,x_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=1)

        return x_train, x_val, y_train, y_val
        
