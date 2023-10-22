import numpy as np
from copy import copy
import os
import json
from src.ceoptimizers.genetic_opt import GeneticOptimizer

class CFsearch:
    def __init__(self, transformer, model, sampler, algorithm="genetic", distance_continuous="weighted_l1", distance_categorical="weighted_l1", loss_type="hinge_loss", sparsity_hp=0.2, coherence_hp=0.2, diversity_hp=0.2):
        self.transformer = transformer
        self.model = model
        self.algorithm = algorithm
        self.distance_continuous = distance_continuous
        self.distance_categorical = distance_categorical
        self.loss_type = loss_type
        self.counterfactuals = []
        
        # TODO make instance sampler parameter, louse coupling config per class, instance sampler should be outside
        self.instance_sampler = sampler # give as parameter instance_sampler
        self.objective_initialization(sparsity_hp, coherence_hp, diversity_hp)

    def store_counterfactuals(self, output_folder, indexname):
        """Store counterfactuals in json file"""
        # Ensure that the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # Getting the number of counterfactuals
        number_cf = len(self.counterfactuals)
        # making filenames for every counterfactual
        for i in range(number_cf):
            filename = indexname + "_" + str(i) + ".json"
            json_path = os.path.join(output_folder, filename)
            # Store counterfactuals in json file
            with open(json_path, 'w') as file:
                json.dump(self.counterfactuals[i].get_values_dict(), file, indent=4)
        #TODO
        return
    
    def store_evaluations(self, output_folder, indexname):
        """Store evaluations in json file"""
        # Ensure that the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # Getting the number of counterfactuals
        number_cf = len(self.counterfactuals)
        # making filenames for every counterfactual
        for i in range(number_cf):
            filename = indexname + "_eval_" + str(i) + ".json"
            json_path = os.path.join(output_folder, filename)
            # Store counterfactuals in json file
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


    def objective_initialization(self, sparsity_hp, coherence_hp, diversity_hp):
        self.diffusion_map = self.transformer.diffusion_map
        self.mads = self.transformer.mads
        self.hyperparameters = [sparsity_hp, coherence_hp, diversity_hp]
        return self.hyperparameters

    
    def find_counterfactuals(self, query_instance, number_cf, desired_class, maxiterations=100):
        """Find counterfactuals by generating them through genetic algorithm"""
        self.original_instance = query_instance
        self.query_instance = query_instance
        self.transformer.normalize_instance(self.query_instance)
        original_prediction = self.model.predict_instance(self.query_instance)
        if desired_class == "opposite" and self.model.model_type == "classification":
            self.desired_output = 1 - original_prediction
        elif self.model.model_type == "regression":
            self.desired_output = [desired_class[0], desired_class[1]]
        else:
            self.desired_output = desired_class
        if self.algorithm == "genetic":
            # there might genetic related parameters, like population size, mutation rate etc.
            optimizer = GeneticOptimizer(self.model, self.transformer, self.instance_sampler, self.distance_continuous, self.distance_categorical, self.loss_type, self.hyperparameters, self.diffusion_map, self.mads)
            self.counterfactuals = optimizer.find_counterfactuals(self.query_instance, number_cf, self.desired_output, maxiterations)
        return self.counterfactuals
    
    def evaluate_counterfactuals(self, original_instance, counterfactual_instances):
        # compute validity
        # compute sparsity
        # compute coherence
        self.evaluations = []
        self.original_instance = original_instance
        self.original_instance_prediciton = self.model.predict_instance(original_instance)
        self.counterfactual_instances = counterfactual_instances
        for counterfactual_instance in self.counterfactual_instances:
            distance_continuous = self.distance_counterfactual_continuous(original_instance, counterfactual_instance)
            distance_categorical = self.distance_counterfactual_categorical(original_instance, counterfactual_instance)
            sparsity_cont = self.sparsity_continuous(original_instance, counterfactual_instance) 
            sparsity_cat = self.sparsity_categorical(original_instance, counterfactual_instance)
            validity = self.check_validity(counterfactual_instance)
            print("CF instance: ", counterfactual_instance.get_values_dict())
            print("Distance continuous: ", distance_continuous)
            print("Distance categorical: ", distance_categorical)
            print("Sparsity continuous: ", sparsity_cont)
            print("Sparsity categorical: ", sparsity_cat)
            print("Validity: ", validity)
            self.evaluations.append({"distance_continuous": distance_continuous, 
                                     "distance_categorical": distance_categorical, 
                                     "sparsity_cont": sparsity_cont, 
                                     "sparsity_cat": sparsity_cat, 
                                     "validity": validity,
                                     "new_outcome": self.new_outcome})
        return distance_continuous, distance_categorical, sparsity_cont, sparsity_cat, validity
    
    def distance_counterfactual_categorical(self, original_instance, counterfactual_instance):
        """Calculate distance function for categorical features"""          
        if self.distance_categorical == "weighted_l1":
            distance_categorical = 0
            # apply weights depending if it's categorical or continuous feature weighted with inverse MAD from train data
            # TODO: add weights for categorical and continious features
            for feature_name in self.transformer.categorical_features_transformers:
                distance_categorical += original_instance.features[feature_name].value != counterfactual_instance.features[feature_name].value
        return distance_categorical
    
    def distance_counterfactual_continuous(self, original_instance, counterfactual_instance):
        """Calculate distance function for continuous features"""
        if self.distance_continuous == "weighted_l1":
            distance_continuous = 0
            # apply weights depending if it's categorical or continuous feature weighted with inverse MAD from train data
            # TODO: add weights for categorical and continious features
            for feature_name in self.transformer.continuous_features_transformers:
                distance_continuous += abs(original_instance.features[feature_name].value - counterfactual_instance.features[feature_name].value)
            
        return distance_continuous
    
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
    
    def check_validity(self, counterfactual_instance):
        """Check if counterfactual instance is valid"""
        # check if counterfactual instance is valid
        if self.model.model_type == "classification":
            counterfactual_prediction = self.model.predict_instance(counterfactual_instance)
            if counterfactual_prediction == self.desired_output:
                self.new_outcome = counterfactual_prediction[0]
                return True
            else:
                self.new_outcome = counterfactual_prediction
                return False
        elif self.model.model_type == "regression":
            counterfactual_prediction = self.model.predict_instance(counterfactual_instance)
            if self.desired_output[0] <= counterfactual_prediction <= self.desired_output[1]:
                self.new_outcome = counterfactual_prediction[0]
                return True
            else:
                self.new_outcome = counterfactual_prediction[0]
                return False
        else:
            return False
    
    def visualize_counterfactuals(self):
        return
    
    def visualize_as_dataframe(self, display_sparse_df=True, show_only_changes=False):
        from IPython.display import display
        import pandas as pd

        # original instance
        print('Query instance (original outcome : %i)' % self.original_instance_prediciton)
        display(pd.DataFrame([self.query_instance.get_values_dict()]))  # works only in Jupyter notebook
        self._visualize_internal(show_only_changes=show_only_changes,
                                 is_notebook_console=True)
        
    def _visualize_internal(self, show_only_changes=False, is_notebook_console=False):
        if self.counterfactual_instances is not None and len(self.counterfactual_instances) > 0:
            print('\nCounterfactual set (new outcome: {0})'.format(self.new_outcome)) # if more than 1 cf won't work
            self._dump_output(content=self.counterfactual_instances, show_only_changes=show_only_changes,
                                is_notebook_console=is_notebook_console)
        else:
            print('\nNo counterfactuals found!')

    def _dump_output(self, content, show_only_changes=False, is_notebook_console=False):
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
        if show_only_changes is False:
            display(df)  # works only in Jupyter notebook
        else:
            newdf = [cf_instance.get_list_of_features_values() for cf_instance in df]
            #org = self.test_instance_df.values.tolist()[0]
            org = self.original_instance.get_list_of_features_values()
            for ix in range(len(df)):
                for jx in range(len(org)):
                    if newdf[ix][jx] == org[jx]:
                        newdf[ix][jx] = '-'
                    else:
                        newdf[ix][jx] = str(newdf[ix][jx])
            display(pd.DataFrame(newdf, columns=self.original_instance.get_list_of_features_names()))#, index=df.index))