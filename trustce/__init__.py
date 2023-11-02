#from explainer import CFCOG
#from config import Config
#from dataset import Dataset
#from feature_manager import FeatureManager

import argparse




def read_arguments():
    parser = argparse.ArgumentParser(description='CFCOG')
    parser.add_argument('--config', type=str, default='config/conf.yaml', help='Path to the configuration file')
    parser.add_argument('--dataset', type=str, default='config/dataset.yaml', help='Path to the dataset config file')
    parser.add_argument('--output', type=str, default='Loan_Status', help='Output to explain')
    parser.add_argument('--instance', type=str, default='0', help='Instance to explain')
    parser.add_argument('--constraints', type=str, default='constraints.txt', help='Path to the constraints file')
    parser.add_argument('--output_dir', type=str, default='results', help='Path to the output directory')
    parser.add_argument('--output_file', type=str, default='explanation.txt', help='Path to the output file')
    parser.add_argument('--verbose', type=bool, default=False, help='Verbose mode')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode')
    args = parser.parse_args()

    return args

'''
def main():
    # Load config file

    args = read_arguments()

    config = Config(args.config)

    # Load dataset
    train, continuous_features_list, categorical_features_list, config_dict = load_dataset()
    x_train, x_val, y_train, y_val = split_dataset(train, "Loan_Status")
    # Create Data, ModelWrapper, and CounterfactualExplainer objects
    dataset = Dataset(config, outcome_column_name="Loan_Status")

    # Load/train model
    t = get_transformations(continuous_features_list, categorical_features_list)
    logistic_model = train_model(t, x_train, x_val, y_train, y_val)

    # Create feature manager
    feat_maanger = FeatureManager(config, dataset)

    # Create explainer
    model_wrapper = ModelWrapper(model=logistic_model)  # Pass the model object
    explainer = CFCOG(model_wrapper, feat_maanger)

    # Explain instance
    instance = train.sample(1)
    desired_output = 1 - instance["Loan_Status"].values[0]  # Flip the outcome
    instance = instance.drop("Loan_Status", axis=1)
    # Call the explain method with the instance and desired output
    explanation = explainer.explain(instance, desired_output)


#if __name__ == "__main__":
    #main()'''