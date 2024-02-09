import numpy as np
import copy
import os
import json
from trustce.ceoptimizers.optimizer_interface import OptimizerInterface, OptimizerFactory


class CFsearch:
    def __init__(self, transformer, model, sampler, config, optimizer_name="genetic", distance_continuous="weighted_l1", 
                 distance_categorical="weighted_l1", loss_type="hinge_loss",
                 sparsity_penalty="elastic_net", alpha=0.5, beta=0.5, coherence=False,
                 objective_function_weights = [0.5, 0.5, 0.5]):
        self.transformer = transformer
        self.model = model
        self.optimizer_name = optimizer_name
        self.config = config
        self.distance_continuous = distance_continuous
        self.distance_categorical = distance_categorical
        self.loss_type = loss_type
        self.sparsity_penalty = sparsity_penalty
        self.alpha = alpha
        self.beta = beta
        self.coherence = coherence
        self.objective_function_weights = objective_function_weights
        self.counterfactuals = []
        
        # TODO make instance sampler parameter, louse coupling config per class, instance sampler should be outside
        self.instance_sampler = sampler # give as parameter instance_sampler
        self.objective_initialization()
        self.initialize_optimizer()

    def initialize_optimizer(self):
        """
        Initialize optimizer using OptimizerFactory
        """
        self.optimizer = OptimizerFactory.get_optimizer(self.optimizer_name, self.model, self.transformer, self.instance_sampler, self.distance_continuous, self.distance_categorical, self.loss_type, self.sparsity_penalty, self.alpha, self.beta, self.coherence, self.objective_function_weights, self.diffusion_map, self.mads)
        return

    def store_counterfactuals(self, output_folder, indexname):
        """Store counterfactuals in json file. TODO unnormaliza data"""
        # Ensure that the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # Getting the number of counterfactuals
        number_cf = len(self.counterfactual_instances)
        # Unnormalize counterfactuals
        for i in range(number_cf):
            if self.counterfactual_instances[i].normalized:
                self.transformer.denormalize_instance(self.counterfactual_instances[i])
    
        # making filenames for every counterfactual
        for i in range(number_cf):
            filename = indexname + "_" + str(i) + ".json"
            json_path = os.path.join(output_folder, filename)
            # Store counterfactuals in json file
            print("Store counterfactuals to ", json_path)
            with open(json_path, 'w') as file:
                counterfactual_values = self.counterfactual_instances[i].get_values_dict()
                json.dump(counterfactual_values, file, indent=4, default=self.default_serializer)
        #TODO
        return
    
    def store_evaluations(self, output_folder, indexname):
        """Store evaluations in json file"""
        # Ensure that the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # making filenames for every counterfactual
        for i in range(len(self.evaluations)):
            filename = indexname + "_eval_" + str(i) + ".json"
            json_path = os.path.join(output_folder, filename)
            # Store counterfactuals in json file
            print("Store counterfactuals evaluation to ", json_path)
            with open(json_path, 'w') as file:
                json.dump(self.evaluations[i], file, indent=4, default=self.default_serializer)
        #TODO
        return
    
    def default_serializer(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):  # You can add more numpy types if needed
            return int(obj) if isinstance(obj, (np.int64, np.int32)) else float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


    def objective_initialization(self):
        self.diffusion_map = None
        if self.distance_continuous["type"] == "diffusion":
            self.diffusion_map = self.transformer.diffusion_map
            self.k = self.transformer.k_neighbors
            self.alpha = self.transformer.alpha
        self.mads = self.transformer.mads
        return self.objective_function_weights

    
    def find_counterfactuals(self, query_instance, number_cf, desired_class, maxiterations=100):
        """Find counterfactuals by generating them through genetic optimizer"""
        self.original_instance = query_instance
        self.query_instance = query_instance
        self.transformer.normalize_instance(self.query_instance)
        self.original_instance_prediciton = self.model.predict_instance(self.query_instance)
        if desired_class == "opposite" and self.model.model_type == "classification":
            self.desired_output = 1 - self.original_instance_prediciton
        elif self.model.model_type == "regression":
            self.desired_output = [desired_class[0], desired_class[1]]
        else:
            self.desired_output = desired_class
        
        self.counterfactual_instances, self.best_candidates, self.fitnes_history, self.loss_history, self.distance_history = self.optimizer.find_counterfactuals(self.query_instance, number_cf, self.desired_output, maxiterations)
        self.new_outcome = []
        # Check validity
        # Checking if we have valid counterfactuals
        for cf in self.counterfactual_instances:
            self.store_outcome(cf)
            if self.check_validity(cf):
                print("Valid counterfactuals were found: ", cf.get_values_dict())
            else:
                print("Not all conterfactuals are valid, but the closest instances reported")
        return self.counterfactual_instances
        '''
        # Rounding
        print("Visualizing before rounding")
        # Checking validity all logic for one cf
        for cf in self.counterfactual_instances:
            validity = self.check_validity(cf)
        self.visualize_as_dataframe(self.original_instance, self.counterfactual_instances)
        rounding = True
        rounded_counterfactuals = []
        if rounding == True:
            for cf in self.counterfactual_instances:
                rounded_cf = self.round_modified_values(query_instance, cf, decimal_places=2)
                if self.is_outcome_same(cf, rounded_cf):
                    rounded_counterfactuals.append(rounded_cf)
                else:
                    rounded_counterfactuals.append(cf)
            print("Visualizing after rounding")
            self.visualize_as_dataframe(self.original_instance, rounded_counterfactuals)
            return rounded_counterfactuals
        else:
            return self.counterfactual_instances
        '''

    def draw_trace_search(self):
        """Draw the trace of search"""
        array_candidates_values = []
        for candidate in self.best_candidates:
            array_candidates_values.append(candidate.get_list_of_features_values())
        return array_candidates_values, self.fitnes_history, self.loss_history, self.distance_history
    
    def round_modified_values(self, original_instance, counterfactual_instance, decimal_places=0):
        """Round only the modified values in the counterfactual instance."""
        rounded_cf = copy(counterfactual_instance)
        for feature in counterfactual_instance.features:
            if counterfactual_instance.features[feature].value != original_instance.features[feature].value:
                rounded_cf.features[feature].value = round(counterfactual_instance.features[feature].value, decimal_places)
        return rounded_cf

    def is_outcome_same(self, original_cf, rounded_cf):
        """Check if the outcome is the same for the original and rounded counterfactuals."""
        original_outcome = self.model.predict_instance(original_cf)
        rounded_outcome = self.model.predict_instance(rounded_cf)
        return original_outcome == rounded_outcome

    
    def evaluate_counterfactuals(self, original_instance, counterfactual_instances):
        # compute validity
        # compute sparsity
        # compute coherence
        self.evaluations = []
        self.original_instance = original_instance
        self.original_instance_prediciton = self.model.predict_instance(original_instance)
        self.counterfactual_instances = counterfactual_instances
        if self.counterfactual_instances == []:
            print("No counterfactuals found, nothing to evaluate.")
            return
        for i, counterfactual_instance in enumerate(self.counterfactual_instances):
            if not counterfactual_instance.normalized:
                counterfactual_instance = self.transformer.normalize_instance(counterfactual_instance)
            distance_continuous = self.distance_counterfactual_continuous(original_instance, counterfactual_instance)
            distance_categorical = self.distance_counterfactual_categorical(original_instance, counterfactual_instance)
            sparsity_cont = self.sparsity_continuous(original_instance, counterfactual_instance) 
            sparsity_cat = self.sparsity_categorical(original_instance, counterfactual_instance)
            validity = self.check_validity(counterfactual_instance)
            coherence_score, incoherent_features = self.check_coherence(original_instance, counterfactual_instance)
            print("CF instance: ", counterfactual_instance.get_values_dict())
            print("Distance continuous: ", distance_continuous)
            print("Distance categorical: ", distance_categorical)
            print("Sparsity continuous: ", sparsity_cont)
            print("Sparsity categorical: ", sparsity_cat)
            print("Validity: ", validity)
            print("Coherence: ", coherence_score, " incoherent features are ", incoherent_features)
            self.evaluations.append({"distance_continuous": distance_continuous, 
                                     "distance_categorical": distance_categorical, 
                                     "sparsity_cont": sparsity_cont, 
                                     "sparsity_cat": sparsity_cat, 
                                     "coherence_score": coherence_score,
                                     "incoherent_features": incoherent_features,
                                     "validity": validity,
                                     "new_outcome": self.new_outcome[i]})
        return distance_continuous, distance_categorical, sparsity_cont, sparsity_cat, validity
    
    def distance_counterfactual_categorical(self, original_instance, counterfactual_instance):
        """Calculate distance function for categorical features"""
        distance_categorical = 0          
        if self.distance_categorical == "hamming":
            # apply weights depending if it's categorical or continuous feature weighted with inverse MAD from train data
            # TODO: add weights for categorical and continious features
            for feature_name in self.transformer.categorical_features_transformers:
                distance_categorical += original_instance.features[feature_name].value != counterfactual_instance.features[feature_name].value
        return distance_categorical
    
    def distance_counterfactual_continuous(self, original_instance, counterfactual_instance):
        """Calculate distance function for continuous features"""
        if self.distance_continuous["type"] == "weighted_l1":
            distance_continuous = 0
            # apply weights depending if it's categorical or continuous feature weighted with inverse MAD from train data
            # TODO: add weights for categorical and continious features
            for feature_name in self.transformer.continuous_features_transformers:
                distance_continuous += abs(original_instance.features[feature_name].value - counterfactual_instance.features[feature_name].value)/self.mads[feature_name]
        elif self.distance_continuous["type"] == "diffusion":
            from scipy.spatial import distance

            distance_continuous = 0
            # Get only continuous features
            # transform the original point to diffusion space
            is_norm = self.distance_continuous["diffusion_params"]["diffusion_normalization"]
            if is_norm:
                point = self.transformer.get_normed_numerical(original_instance)
            else:   
                point = original_instance.get_numerical_features_values()
            point = np.array(point).reshape(1, -1)

            # transform the counterfactual point to diffusion space
            if is_norm:
                counterfactual = self.transformer.get_normed_numerical(counterfactual_instance)
            else:
                counterfactual = counterfactual_instance.get_numerical_features_values()
            counterfactual = np.array(counterfactual).reshape(1, -1)
            

            point_transformed = self.transformer.diffusion_map.transform(point)
            counterfactual_transformed = self.transformer.diffusion_map.transform(counterfactual)
            # Calculate the Euclidean distance between the point and the counterfactuals
            diff_distance = np.linalg.norm(counterfactual_transformed - point_transformed)
            distance_continuous = diff_distance
            
        return distance_continuous
    
    def project_point_to_diffusion_space(self, new_point):
        distances = np.sqrt(np.sum((self.transformer.normlaize_cont_dataset_numpy - new_point)**2, axis=1))
        local_scale_new_point = np.sort(distances)[self.k]
        affinity = np.exp(-distances ** 2 / (self.local_scale * local_scale_new_point))
        affinity /= affinity.sum()
        projection = (self.eigenvectors * (self.eigenvalues**self.alpha)).T @ affinity
        return projection
    
    def sparsity_continuous(self, original_instance, counterfactual_instance):
        """Calculate sparsity function for continuous features"""
        sparsity_continuous = 0
        for feature_name in self.transformer.continuous_features_transformers:
            if abs(original_instance.features[feature_name].value - counterfactual_instance.features[feature_name].value) > 0:
                sparsity_continuous += 1
        return sparsity_continuous
    
    def sparsity_categorical(self, original_instance, counterfactual_instance):
        """Calculate sparsity function for categorical features"""
        sparsity_categorical = 0
        for feature_name in self.transformer.categorical_features_transformers:
            if original_instance.features[feature_name].value != counterfactual_instance.features[feature_name].value:
                sparsity_categorical += 1
        return sparsity_categorical
    
    def store_outcome(self, counterfactual_instance):
        """Store the outcome of the counterfactual instance"""
        self.new_outcome.append(self.model.predict_instance(counterfactual_instance))
        return
    
    def check_validity(self, counterfactual_instance):
        """Check if counterfactual instance is valid"""
        # check if counterfactual instance is valid
        if self.model.model_type == "classification":
            counterfactual_prediction = self.model.predict_instance(counterfactual_instance)
            if counterfactual_prediction == self.desired_output:
                return True
            else:
                return False
        elif self.model.model_type == "regression":
            counterfactual_prediction = self.model.predict_instance(counterfactual_instance)
            if self.desired_output[0] <= counterfactual_prediction <= self.desired_output[1]:
                return True
            else:
                return False
        else:
            return False

    def check_coherence(self, original_instance, counterfactual_instance):
        """Simplified version"""    
        control_instance = copy.deepcopy(original_instance)
        marginal_signs = {}
        required_label = self.desired_output
        # Get the direction of prediction change for each feature
        for feature_name in self.transformer.continuous_features_transformers:
            marginal_signs[feature_name] = self.get_only_marginal_prediction_sign(original_instance, counterfactual_instance, feature_name, required_label)
        for feature_name in self.transformer.categorical_features_transformers:
            marginal_signs[feature_name] = self.get_only_marginal_prediction_sign(original_instance, counterfactual_instance, feature_name, required_label)
        # Calculate how many minuses are in marginal_signs
        coherence_counterfactual_score = sum(1 for key, value in marginal_signs.items() if value == 1)/len(marginal_signs)
        uncoherent_suggestions = [key for key, value in marginal_signs.items() if value == -1]
        return coherence_counterfactual_score, uncoherent_suggestions

    def check_coherence_old_version(self, original_instance, counterfactual_instance):
        """Check the direction of prediction change for each feature and their paris"""
        control_instance = copy.deepcopy(original_instance)
        marginal_signs = {}
        required_label = self.desired_output
        # Get the direction of prediction change for each feature
        for feature_name in self.transformer.continuous_features_transformers:
            marginal_signs[feature_name] = self.get_marginal_sign_for_feature(original_instance, counterfactual_instance, feature_name, required_label)
        # Get the direction of prediction change for each pair of features with changed ones
        joint_signs = {}
        for feature_name in self.transformer.continuous_features_transformers:
            original_instance_value = original_instance.features[feature_name].value
            counterfactual_instance_value = counterfactual_instance.features[feature_name].value
            if original_instance_value != counterfactual_instance_value:
                for feature_name2 in self.transformer.continuous_features_transformers:
                    if feature_name != feature_name2:
                        # Ensure the tuple is always in sorted order
                        key = tuple(sorted((feature_name, feature_name2)))
                        if key not in joint_signs:
                            joint_signs[key] = self.get_joint_sign_for_feature_pair(original_instance, counterfactual_instance, feature_name, feature_name2, required_label)
        # Compare marginal and joint signs and print which are different
        common_keys_same_value, common_keys_diff_value = self.compare_common_keys(marginal_signs, joint_signs)
        if common_keys_same_value:
            print("Common keys with the same value:", common_keys_same_value)
        else:
            print("No common keys with the same value found.")

        if common_keys_diff_value:
            print("Common keys with different values:", common_keys_diff_value)
        else:
            print("No common keys with different values found.")
        
        return
    
    def compare_common_keys(self, dict1, dict2):
        common_keys_same_value = {}
        common_keys_diff_value = {}
        for key1, value1 in dict1.items():
            for key2, value2 in dict2.items():
                if key1 in key2:  # This checks if the string key1 is one of the elements in the tuple key2
                    if value1 == value2:
                        common_keys_same_value[key1] = value1
                    else:
                        common_keys_diff_value[key1] = (value1, value2)
        return common_keys_same_value, common_keys_diff_value
    
    def get_only_marginal_prediction_sign(self, original_instance, counterfactual_instance, feature_name, required_label):
        """This implementation is checking if current change leads to increase or decrease of the prediction"""
        control_instance = copy.deepcopy(original_instance)
        original_instance_value = original_instance.features[feature_name].value
        counterfactual_instance_value = counterfactual_instance.features[feature_name].value
        print("Feature {} changed its value from {} to {}".format(feature_name, original_instance.features[feature_name].value, counterfactual_instance.features[feature_name].value))
        # Current prediction of original instance
        # Let's change only counterfactual value of the feature
        control_instance.features[feature_name].value = counterfactual_instance_value
        # If it is classification
        if self.model.model_type == "classification":
            original_prediction = self.model.predict_proba_instance(original_instance)
            control_prediction = self.model.predict_proba_instance(control_instance)
            probability_sign = np.sign(control_prediction - original_prediction)
            # I modified required_label to required_label[0] because it was a list of one element
            # Debugging print statements
            print(f"probability_sign: {probability_sign}, type: {type(probability_sign)}")
            print(f"required_label: {required_label}, type: {type(required_label)}")

            # Convert required_label to integer if it's a float
            if isinstance(required_label, (list, np.ndarray)) and len(required_label) == 1:
                required_label = int(required_label[0])

            # Further debugging
            print(f"Modified required_label: {required_label}, type: {type(required_label)}")

            return probability_sign[required_label]
        else:
            original_prediction = self.model.predict_instance(original_instance)
            control_prediction = self.model.predict_instance(control_instance)
            probability_sign = np.sign(control_prediction - original_prediction)
            return probability_sign
    
    
    def get_marginal_sign_for_feature(self, original_instance, counterfactual_instance, feature_name, required_label):
        control_instance = copy.deepcopy(original_instance)
        original_instance_value = original_instance.features[feature_name].value
        counterfactual_instance_value = counterfactual_instance.features[feature_name].value
        print("Feature {} changed its value from {} to {}".format(feature_name, original_instance.features[feature_name].value, counterfactual_instance.features[feature_name].value))
        # Current prediction of original instance
        original_prediction = self.model.predict_proba_instance(original_instance)
        # Let's change only counterfactual value of the feature
        control_instance.features[feature_name].value = counterfactual_instance.features[feature_name].value
        # Current prediction of control instance
        control_prediction = self.model.predict_proba_instance(control_instance)
        probability_sign = np.sign(control_prediction - original_prediction)
        # If value changed in the same direction as prediction, then sign is positive
        if probability_sign[required_label] > 0 and counterfactual_instance_value > original_instance_value:
            marginal_sign = 1
        elif probability_sign[required_label] > 0 and counterfactual_instance_value < original_instance_value:
            marginal_sign = -1
        elif probability_sign[required_label] < 0 and counterfactual_instance_value > original_instance_value:
            marginal_sign = -1
        elif probability_sign[required_label] < 0 and counterfactual_instance_value < original_instance_value :
            marginal_sign = 1
        return marginal_sign
    
    def get_joint_sign_for_feature_pair(self, original_instance, counterfactual_instance, feature_name1, feature_name2, required_label):
        control_instance = copy.deepcopy(original_instance)
        original_instance_value1 = original_instance.features[feature_name1].value
        original_instance_value2 = original_instance.features[feature_name2].value
        counterfactual_instance_value1 = counterfactual_instance.features[feature_name1].value
        counterfactual_instance_value2 = counterfactual_instance.features[feature_name2].value
        # Current prediction of original instance
        original_prediction = self.model.predict_proba_instance(original_instance)
        # Let's change only counterfactual value of the feature
        control_instance.features[feature_name1].value = counterfactual_instance.features[feature_name1].value
        control_instance.features[feature_name2].value = counterfactual_instance.features[feature_name2].value
        # Current prediction of control instance
        control_prediction = self.model.predict_proba_instance(control_instance)
        probability_sign = np.sign(control_prediction - original_prediction)
        # If value changed in the same direction as prediction, then sign is positive
        if probability_sign[required_label] > 0:
            return 1
        else:
            return -1
        #if probability_sign[required_label] > 0 and counterfactual_instance_value1 > original_instance_value1 and counterfactual_instance_value2 > original_instance_value2:
        #    joint_sign = 1
        #elif probability_sign[required_label] > 0 and counterfactual_instance_value1 < original_instance_value1 and counterfactual_instance_value2 < original_instance_value2:
        #    joint_sign = -1
        #elif probability_sign[required_label] < 0 and counterfactual_instance_value1 > original_instance_value1 and counterfactual_instance_value2 > original_instance_value2:
        #    joint_sign = -1
        #elif probability_sign[required_label] < 0 and counterfactual_instance_value1 < original_instance_value1 and counterfactual_instance_value2 < original_instance_value2:
        #    joint_sign = 1
    
    def visualize_counterfactuals(self):
        return
    
    def visualize_as_dataframe(self, target_instance, counterfactuals, display_sparse_df=True, show_only_changes=True):
        from IPython.display import display
        import pandas as pd

        # original instance
        print('Query instance (original outcome : %i)' % self.original_instance_prediciton)
        if self.query_instance.normalized:
            self.transformer.denormalize_instance(self.query_instance)
        display(pd.DataFrame([self.query_instance.get_values_dict()]))  # works only in Jupyter notebook
        self._visualize_internal(target_instance, counterfactuals, show_only_changes=show_only_changes,
                                 is_notebook_console=True)
        
    def _visualize_internal(self, target_instance, counterfactuals, show_only_changes=True, is_notebook_console=False):
        if counterfactuals is not None and len(counterfactuals) > 0:
            print('\nCounterfactual set (new outcome: {0})'.format(self.new_outcome)) # if more than 1 cf won't work
            self._dump_output(content=counterfactuals, show_only_changes=show_only_changes,
                                is_notebook_console=is_notebook_console)
        else:
            print('\nNo counterfactuals found!')

    def _dump_output(self, content, show_only_changes=True, is_notebook_console=False):
        import pandas as pd
        if is_notebook_console:
            self.display_df(content, show_only_changes=show_only_changes)
        else:
            assert isinstance(content, pd.DataFrame), "Expecting a pandas dataframe"
            self.print_list(content.values.tolist(),
                            show_only_changes=show_only_changes)

    def print_list(self, li, show_only_changes):
        if show_only_changes is False:
            for ix in range(len(li)):
                print(li[ix])
        else:
            newli = copy.deepcopy(li)
            org = self.test_instance_df.values.tolist()[0]
            for ix in range(len(newli)):
                for jx in range(len(newli[ix])):
                    if newli[ix][jx] == org[jx]:
                        newli[ix][jx] = '-'
                print(newli[ix])

    def display_df(self, df, show_only_changes):
        from IPython.display import display
        import pandas as pd
        # We need to display denormalized feature
    
        if show_only_changes is False:
            newdf = [cf_instance.get_list_of_features_values() for cf_instance in df]
            display(pd.DataFrame(newdf, columns=self.original_instance.get_list_of_features_names()))  # works only in Jupyter notebook
        else:
            if df[0].normalized:
                newdf = []
                for cf_instance in df:
                    self.transformer.denormalize_instance(cf_instance)
                    get_list_of_features_values = cf_instance.get_list_of_features_values()
                    newdf.append(get_list_of_features_values)
                #newdf = [self.transformer.denormalize_instance(cf_instance).get_list_of_features_values() for cf_instance in df]
            else:
                newdf = [cf_instance.get_list_of_features_values() for cf_instance in df]
            #org = self.test_instance_df.values.tolist()[0]
            if self.original_instance.normalized:
                self.transformer.denormalize_instance(self.original_instance) 
            org = self.original_instance.get_list_of_features_values()
            for ix in range(len(df)):
                for jx in range(len(org)):
                    if newdf[ix][jx] == org[jx]:
                        newdf[ix][jx] = '-'
                    else:
                        newdf[ix][jx] = str(newdf[ix][jx])
            display(pd.DataFrame(newdf, columns=self.original_instance.get_list_of_features_names()))#, index=df.index))