import unittest
import numpy as np
import pandas as pd
import os
import sys
import pickle
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import mahalanobis

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from codice.ceutils.diffusion import STDiffusionMap

# Assume model has a predict function that takes DataFrame input and returns predictions
# Transformer and model should be defined or imported here


class TestCFSearch(unittest.TestCase):
    def setUp(self):
        # Example usage:
        self.test_identifier = 'breast_cancer_weighted'
    
    def test_evaluation(self):
        # Read data
        df_train = pd.read_csv('datasets/breast_cancer.csv')
        del df_train['id']
        del df_train['Unnamed: 32'] 
        # Drop Class variable
        target_name = "diagnosis"
        df_train = df_train.drop(target_name, axis=1)
        df_original_instances = pd.read_csv('results/original_breast_cancer_train_weighted.csv')
        df_counterfactuals = pd.read_csv('results/counterfactuals_breast_cancer_train_weighted.csv')
        df_original_instances = df_original_instances[:100]
        df_counterfactuals = df_counterfactuals[:100]
        #slicing_df = pd.read_csv('results/original_adult_logistic_weighted.csv')
        # Fiind instances from slicing_df in df_original_instances
        #matching_indices = df_original_instances.index.isin(slicing_df.index)
        #df_original_instances = df_original_instances[matching_indices]
        #df_counterfactuals = df_counterfactuals[matching_indices]
        #print(len(slicing_df), len(df_counterfactuals))

        #df_counterfactuals_na_indexes = df_counterfactuals[df_counterfactuals.isna().any(axis=1)].index
        #df_original_instances = df_original_instances.drop(df_counterfactuals_na_indexes)
        #df_counterfactuals = df_counterfactuals.drop(df_counterfactuals_na_indexes)

        print(len(df_original_instances), len(df_counterfactuals))

        if target_name in df_counterfactuals.columns:
            df_counterfactuals = df_counterfactuals.drop(target_name, axis=1)

        if target_name in df_original_instances.columns:
            df_original_instances = df_original_instances.drop(target_name, axis=1)

        categorical_feature_names = []
        continuous_feature_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean','smoothness_mean', 'compactness_mean', 'concavity_mean','concave points_mean', 'symmetry_mean', 'fractal_dimension_mean','radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se','compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se','fractal_dimension_se', 'radius_worst', 'texture_worst','perimeter_worst', 'area_worst', 'smoothness_worst','compactness_worst', 'concavity_worst', 'concave points_worst','symmetry_worst', 'fractal_dimension_worst']
        model_path = 'models/breast_cancer_model.pkl' 
        df_counterfactuals = df_counterfactuals[continuous_feature_names + categorical_feature_names]
        df_original_instances = df_original_instances[continuous_feature_names + categorical_feature_names]
        with open(model_path, 'rb') as filehandle:
            model = pickle.load(filehandle)
        st_diff_map = STDiffusionMap(n_neighbors=10, alpha=1.0)
        # Make array of continuous features
        cont_matrix = df_train[continuous_feature_names].to_numpy()
        st_diff_map.fit(cont_matrix)
        evaluations_df = evaluate_counterfactuals(df_train, model, df_original_instances, df_counterfactuals, continuous_feature_names, categorical_feature_names, st_diff_map=st_diff_map)
        print(evaluations_df)
        # Save evaluations
        evaluations_df.to_csv(f'results/evaluation100_{self.test_identifier}.csv', index=False)

def evaluate_counterfactuals(df_train, model, df_original, df_counterfactuals, continuous_features, categorical_features, st_diff_map=None):
    # Placeholder for your custom functions
    def calculate_sparsity_continuous(original, counterfactual):
        changes = original[continuous_features] != counterfactual[continuous_features]
        sparsity_count = len(changes.sum())/len(continuous_features)
        return sparsity_count

    def calculate_sparsity_categorical(original, counterfactual):
        if len(categorical_features) == 0:
            return 0
        changes = original[categorical_features] != counterfactual[categorical_features]
        sparsity_count = len(changes.sum())/len(categorical_features)
        return sparsity_count
    
    def calculate_custom_diffusion_distance(original, counterfactual):
        # Transform both instances into the diffusion space
        original_transformed = st_diff_map.transform(np.array([original[continuous_features]]))
        counterfactual_transformed = st_diff_map.transform(np.array([counterfactual[continuous_features]]))
        
        # Compute and return the Euclidean distance in the diffusion space
        return euclidean_distances(original_transformed, counterfactual_transformed)[0][0]
    
    def calculate_coherence_penalty(original, counterfactual, required_label):
        # Implement your custom coherence penalty calculation here
        marginal_signs = {}
        # Get the direction of prediction change for each feature
        for feature_name in continuous_features:
            marginal_signs[feature_name] = get_only_marginal_prediction_sign(original, counterfactual, feature_name, required_label)
        for feature_name in categorical_features:
            marginal_signs[feature_name] = get_only_marginal_prediction_sign(original, counterfactual, feature_name, required_label)
        # Calculate how many minuses are in marginal_signs
        coherence_counterfactual_score = sum(1 for key, value in marginal_signs.items() if value != -1)/len(marginal_signs)
        uncoherent_suggestions = [key for key, value in marginal_signs.items() if value == -1]
        coherence_penalty = 1 - coherence_counterfactual_score
        return coherence_penalty, uncoherent_suggestions
    
    def get_only_marginal_prediction_sign(original_instance, counterfactual_instance, feature_name, required_label):
        # Get the direction of prediction change for a single feature
        original_instance_df = pd.DataFrame([original_instance])
        control_instance = original_instance.copy()
        control_instance[feature_name] = counterfactual_instance[feature_name]
        control_instance_df = pd.DataFrame([control_instance])
        original_instance_pred = model.predict_proba(original_instance_df)[0]
        control_instance_pred = model.predict_proba(control_instance_df)[0]
        probability_sign = np.sign(control_instance_pred - original_instance_pred)
        # Convert required_label to integer if it's a float
        if isinstance(required_label, (list, np.ndarray)) and len(required_label) == 1:
            required_label = int(required_label[0])
        return probability_sign[required_label]
    
    def calculate_mahalanobis_distance(x, y, VI):
        """
        Calculate the Mahalanobis distance between two vectors, x and y, using the inverse of the covariance matrix VI.
        """
        return mahalanobis(x, y, VI)

    
    evaluations = []
    for index, original_row in df_original.iterrows():
        counterfactual_row = df_counterfactuals.loc[index]
        
        # Convert rows to DataFrame for model prediction
        original_instance_df = pd.DataFrame([original_row])
        counterfactual_instance_df = pd.DataFrame([counterfactual_row])
        
        # Validity check - comparing predictions
        original_pred = model.predict(original_instance_df)
        counterfactual_pred = model.predict(counterfactual_instance_df)
        validity = original_pred != counterfactual_pred

        if validity:
            desired_prediction = counterfactual_pred
        else:
            desired_prediction = 1 - original_pred
        
        # Custom diffusion distance
        diffusion_distance = calculate_custom_diffusion_distance(original_row, counterfactual_row)
        
        # L1 and L2 distances standardized for continuous features divided by number of continuous features
        # Find mean and std for cont features from df_train
        mean_cont = df_train[continuous_features].mean()
        std_cont = df_train[continuous_features].std()
        standardised_original = (original_row[continuous_features] - mean_cont) / std_cont
        standardised_counterfactual = (counterfactual_row[continuous_features] - mean_cont) / std_cont
        l1_distance_continuous = np.sum(np.abs(standardised_original - standardised_counterfactual)) / len(continuous_features)
        l2_distance_continuous = np.sqrt(np.sum((standardised_original - standardised_counterfactual) ** 2)) / len(continuous_features)
        #l1_distance_continuous = np.sum(np.abs(original_row[continuous_features] - counterfactual_row[continuous_features]))
        #l2_distance_continuous = np.sqrt(np.sum((original_row[continuous_features] - counterfactual_row[continuous_features]) ** 2))
        
        # L1 and L2 distances for categorical features - assuming one-hot encoding or similar
        l1_distance_categorical = np.sum(original_row[categorical_features] != counterfactual_row[categorical_features])/len(categorical_features)
        l2_distance_categorical = np.sqrt(np.sum((original_row[categorical_features] != counterfactual_row[categorical_features]) ** 2))/len(categorical_features)
        
        # Sparsity for continuous and categorical features
        sparsity_continuous = calculate_sparsity_continuous(original_instance_df, counterfactual_instance_df)
        sparsity_categorical = calculate_sparsity_categorical(original_instance_df, counterfactual_instance_df)
        
        # Custom coherence penalty
        coherence_penalty, incoherent_features = calculate_coherence_penalty(original_row, counterfactual_row, desired_prediction)
        
        standardised_original = np.array((original_row[continuous_features] - mean_cont) / std_cont)
        standardised_counterfactual = np.array((counterfactual_row[continuous_features] - mean_cont) / std_cont)
        cov_matrix = np.cov(df_train[continuous_features].values, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        # Calculate Mahalanobis distance
        mahalanobis_distance = calculate_mahalanobis_distance(standardised_original, standardised_counterfactual, inv_cov_matrix)
        print("Mahalanobis distance: ", mahalanobis_distance)
        
        
        evaluations.append({
            'index': index,
            'validity': validity[0],
            'diffusion_distance': diffusion_distance,
            'l1_distance_continuous': l1_distance_continuous,
            'l2_distance_continuous': l2_distance_continuous,
            'l1_distance_categorical': l1_distance_categorical,
            'l2_distance_categorical': l2_distance_categorical,
            'mahalanobis_distance': mahalanobis_distance,
            'sparsity_continuous': sparsity_continuous,
            'sparsity_categorical': sparsity_categorical,
            'coherence_penalty': coherence_penalty,
            'incoherent_features': incoherent_features
        })
    
    return pd.DataFrame(evaluations)

