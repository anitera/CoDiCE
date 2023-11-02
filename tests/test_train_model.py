import sys
import os

# Specify the source directory path
#source_dir = './src'

# Add the source directory to the Python path
#sys.path.insert(0, source_dir)

from trustce.train import get_transformations, train_model, load_dataset, split_dataset

if __name__ == "__main__":
     # Load dataset
    train, continuous_features_list, categorical_features_list, config_dict = load_dataset("./config/conf.yaml", "./datasets/homeloan_train.xls")
    x_train, x_val, y_train, y_val = split_dataset(train, "Loan_Status")

    # Load/train model
    t = get_transformations(continuous_features_list, categorical_features_list)
    logistic_model = train_model(t, x_train, x_val, y_train, y_val)