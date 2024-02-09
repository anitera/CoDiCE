import random
import numpy as np
from copy import copy, deepcopy
from scipy.spatial import distance_matrix
from scipy.linalg import eigh
import scipy.sparse.linalg as spsl
from trustce.ceinstance.instance_sampler import ImmutableSampler, PermittedRangeSampler

# TODO: make Optimizer parent class and implement genetic optimizer as child class
class GeneticOptimizer():
    def __init__(self, model, transformer, instance_sampler, distance_continuous="weighted_l1", distance_categorical="weighted_l1", loss_type="hinge_loss", sparsity_penalty="elastic_net", alpha=0.5, beta=0.5, coherence=False, hyperparameters=[0.2, 0.2, 0.2], diffusion_map=None, mads=None):
        """Initialize data metaparmeters and model"""
        self.model = model
        self.transformer = transformer
        self.distance_continuous = distance_continuous
        self.distance_categorical = distance_categorical
        self.loss_type = loss_type
        self.diffusion_map = diffusion_map
        self.sparsity_penalty = sparsity_penalty
        self.alpha = alpha
        self.beta = beta
        self.coherence = coherence
        self.mads = mads
        self.hyperparameters = hyperparameters
        self.instance_sampler = instance_sampler

    def fitness_function(self, counterfactual_instance, query_instance, desired_output):
        """Calculate fitness function which consist of loss function and distance function"""
        # calculate loss function depending on task classification or regression
        loss = 0
        if self.loss_type == "hinge":
            # if prediction is flipped set the term to be one, if not account for probabilities
            prediction = self.model.predict_instance(counterfactual_instance)
            # if desired output is not array
            if prediction == desired_output:
                loss = 0
            else:
                # If desired_output is array
                #full_prediction = self.model.predict_proba_instance(counterfactual_instance)
                #counterfactual_prediction = full_prediction[desired_output[0]]
                counterfactual_prediction = self.model.predict_instance(counterfactual_instance)
                # TODO: take into consideration that predicitons might be not normalized
                loss = max(0, 1 - desired_output * counterfactual_prediction)
        elif self.loss_type == "MSE":
            original_prediction = self.model.predict_instance(query_instance)
            counterfactual_prediction = self.model.predict_instance(counterfactual_instance)
            loss_lower = (desired_output[0] - counterfactual_prediction)**2
            loss_upper = (desired_output[1] - counterfactual_prediction)**2
            loss = min(loss_lower, loss_upper)
        elif self.loss_type == "MAE":
            original_prediction = self.model.predict_instance(query_instance)
            counterfactual_prediction = self.model.predict_instance(counterfactual_instance)
            loss_lower = abs(desired_output[0] - counterfactual_prediction)
            loss_upper = abs(desired_output[1] - counterfactual_prediction)
            loss = min(loss_lower, loss_upper)
        elif self.loss_type == "RMSE":
            original_prediction = self.model.predict_instance(query_instance)
            counterfactual_prediction = self.model.predict_instance(counterfactual_instance)
            loss_lower = (desired_output[0] - counterfactual_prediction)**2
            loss_lower = loss_lower**0.5
            loss_upper = (desired_output[1] - counterfactual_prediction)**2
            loss_upper = loss_upper**0.5
            loss = min(loss_lower, loss_upper)
        # calculate distance function depending on distance type
        distance_continuous = 0
        if self.distance_continuous["type"] == "weighted_l1":
            # apply weights depending if it's categorical or continuous feature weighted with inverse MAD from train data
            # TODO: add weights for categorical and continious features
            for feature_name in self.transformer.continuous_features_transformers:
                distance_continuous += abs(query_instance.features[feature_name].value - counterfactual_instance.features[feature_name].value)/self.mads[feature_name]
            if len(self.transformer.continuous_features_transformers) > 0:
                distance_continuous = distance_continuous/self.transformer.get_cont_transformers_length()
        elif self.distance_continuous["type"] == "diffusion":
            from scipy.spatial import distance
            import numpy as np

            distance_continuous = 0
            # Get only continuous features
            # transform the original point to diffusion space
            is_norm = self.distance_continuous["diffusion_params"]["diffusion_normalization"]
            if is_norm:
                point = self.transformer.get_normed_numerical(query_instance)
            else:   
                point = query_instance.get_numerical_features_values()
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

        elif self.distance_continuous["type"] == "pydiffmap":
            distance_continuous = self.pydiffmap_distance(query_instance, counterfactual_instance)

        elif self.distance_continuous["type"] == "comparison":
            import numpy as np

            is_norm = self.distance_continuous["diffusion_params"]["diffusion_normalization"]
            if is_norm:
                point = self.transformer.get_normed_numerical(query_instance)
            else:   
                point = query_instance.get_numerical_features_values()

            # A test to confirm if distance makes sense
            # One side of class
            comparison_point_closest = [0.75, 0.04, -1.52]
            pydiff_cl1_closest = self.pydiffmap_distance(query_instance, comparison_point_closest)
            custom_cl1_closest = self.recalculate_diffusion_map(point, np.array(comparison_point_closest).reshape(1, -1))
            print("pydiffmap distance for closest point: ", pydiff_cl1_closest)
            print("custom distance for closest point: ", custom_cl1_closest)
            comparison_point_further = [0.04, -0.04, -2.11]
            pydiff_cl1_further = self.pydiffmap_distance(query_instance, comparison_point_further)
            custom_cl1_further = self.recalculate_diffusion_map(point, np.array(comparison_point_further).reshape(1, -1))
            print("pydiffmap distance for further point: ", pydiff_cl1_further)
            print("custom distance for further point: ", custom_cl1_further)
            comparison_point_the_most = [-1.09, 0.05, -1.02]
            pydiff_cl1_the_most = self.pydiffmap_distance(query_instance, comparison_point_the_most)
            custom_cl1_the_most = self.recalculate_diffusion_map(point, np.array(comparison_point_the_most).reshape(1, -1))
            print("pydiffmap distance for the most point: ", pydiff_cl1_the_most)
            print("custom distance for the most point: ", custom_cl1_the_most)

            #Other side of class
            comparison_point_closest2 = [-0.86, 0.39, 1.58]
            pydiff_cl2_closest = self.pydiffmap_distance(query_instance, comparison_point_closest2)
            custom_cl2_closest = self.recalculate_diffusion_map(point, np.array(comparison_point_closest2).reshape(1, -1))
            print("pydiffmap distance for closest point: ", pydiff_cl2_closest)
            print("custom distance for closest point: ", custom_cl2_closest)

            comparison_point_further2 = [0.04, 0.19, 1.89]
            pydiff_cl2_further = self.pydiffmap_distance(query_instance, comparison_point_further2)
            custom_cl2_further = self.recalculate_diffusion_map(point, np.array(comparison_point_further2).reshape(1, -1))
            print("pydiffmap distance for further point: ", pydiff_cl2_further)
            print("custom distance for further point: ", custom_cl2_further)
            comparison_point_the_most2 = [1.12, 0.20, 1.15]
            pydiff_cl2_the_most = self.pydiffmap_distance(query_instance, comparison_point_the_most2)
            custom_cl2_the_most = self.recalculate_diffusion_map(point, np.array(comparison_point_the_most2).reshape(1, -1))
            print("pydiffmap distance for the most point: ", pydiff_cl2_the_most)
            print("custom distance for the most point: ", custom_cl2_the_most)
            print("End of experiment")
            print("-----------------------------------")


        distance_categorical = 0
        if self.distance_categorical == "hamming":
            # apply weights depending if it's categorical or continuous feature weighted with inverse MAD from train data
            # TODO: add weights for categorical and continious features
            for feature_name in self.transformer.categorical_features_transformers:
                distance_categorical += query_instance.features[feature_name].value != counterfactual_instance.features[feature_name].value
            if len(self.transformer.categorical_features_transformers) > 0:
                distance_continuous = distance_continuous/self.transformer.get_cat_transformers_length()
        distance_combined = distance_continuous + distance_categorical
        elastic_net_penalty_approx = 0
        if self.sparsity_penalty == "elastic_net":
            epsilon = 1e-5
            l1_penalty_approx = 0
            l2_penalty_approx = 0
            # check if changed features are sparse

            base_prediction = self.model.predict_instance(counterfactual_instance)

            for feature_name in self.transformer.continuous_features_transformers:
                original_feature_value = counterfactual_instance.features[feature_name].value
                # Continuous feature perturbation
                counterfactual_instance.features[feature_name].value += epsilon

                perturbed_prediction = self.model.predict_instance(counterfactual_instance)
                
                # Approximate derivative (importance)
                derivative = abs(perturbed_prediction - base_prediction) / (epsilon if feature_name in self.transformer.continuous_features_transformers else 1)
                l1_penalty_approx += derivative
                l2_penalty_approx += derivative ** 2

    
            for feature_name in self.transformer.categorical_features_transformers:
                original_feature_value = counterfactual_instance.features[feature_name].value
                # Categorical feature perturbation - switch category
                # Check the length of the category list
                num_categories = len(self.transformer.categorical_features_transformers[feature_name].original_range)
                counterfactual_instance.features[feature_name].value = (counterfactual_instance.features[feature_name].value + 1) % num_categories

                perturbed_prediction = self.model.predict_instance(counterfactual_instance)
                
                # Approximate derivative (importance)
                derivative = abs(perturbed_prediction - base_prediction) / (epsilon if feature_name in self.transformer.continuous_features_transformers else 1)
                l1_penalty_approx += derivative
                l2_penalty_approx += derivative ** 2
                
                # Reset feature value
                counterfactual_instance.features[feature_name].value = original_feature_value

            elastic_net_penalty_approx = self.beta * (self.alpha * l1_penalty_approx + (1 - self.alpha) * l2_penalty_approx)


        if self.coherence:
            coherence = 0
            # check if changed features are coherent with prediction direciton
        # calculate fitness function
        # TODO: normalize distance and loss
        fitness = 10*self.hyperparameters[0]*loss + self.hyperparameters[1]*distance_combined + self.hyperparameters[2]*coherence + elastic_net_penalty_approx
        return fitness, loss, distance_combined

    
    def pydiffmap_distance(self, query_instance, counterfactual_instance):
        """Calculate distance using pydiffmap"""
        point = query_instance.get_numerical_features_values()
        point = np.array(point).reshape(1, -1)
        original_point = self.transformer.diffusion_map.transform(point)

        #counterfactual = counterfactual_instance.get_numerical_features_values()
        counterfactual = np.array(counterfactual_instance).reshape(1, -1)
        counterfactual_point = self.transformer.diffusion_map.transform(counterfactual)


        # Calculate the Euclidean distance between the point and the counterfactuals
        diff_distance = np.linalg.norm(counterfactual_point - original_point)
        return diff_distance
    
    def recalculate_diffusion_map(self, original_point, new_point):
        """Recalculate diffusion map with new point"""
        original_point_projection = self.transformer.diffusion_map.transform(original_point)
        new_point_projection = self.transformer.diffusion_map.transform(new_point)
        diff_distance = np.linalg.norm(new_point_projection - original_point_projection)
        return diff_distance
    
    def project_point_to_diffusion_space(self, new_point):
        distances = np.sqrt(np.sum((self.transformer.normlaize_cont_dataset_numpy - new_point)**2, axis=1))
        local_scale_new_point = np.sort(distances)[self.k]
        affinity = np.exp(-distances ** 2 / (self.local_scale * local_scale_new_point))
        affinity /= affinity.sum()
        projection = (self.eigenvectors * (self.eigenvalues**self.alpha)).T @ affinity
        return projection

    def generate_population(self, query_instance, population_size):
        """Initialize the populationg following sampling strategy"""
        # initialize population
        population = []
        counterfactual_instance = deepcopy(query_instance)

        # generate population
        for i in range(population_size):
            new_instance = self.instance_sampler.sample(counterfactual_instance)
            population.append(new_instance)
            #for feature_name in self.feature_names:
            #    if self.feature_types_dict[feature_name] == "continuous":
            #        #counterfactual_instance[feature_name] = random.uniform(self.data.minmax[feature_name][0], self.data.minmax[feature_name][1])
            #        # Sample according to sampler strategy
            #    else:
            #        #counterfactual_instance[feature_name] = random.choice(self.data.categorical_features_dict[feature_name])
        return population

    def one_point_crossover(self, parent1, parent2):
        """Perform one point crossover. TODO: crossover needs feature selection and OOP aaproach"""
         # Ensure parents are of the same type and have the same schema
        assert parent1.features.keys() == parent2.features.keys()

        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)

        #child1 = copy(parent1)
        #child2 = copy(parent2)

        # Get ordered feature names
        feature_names = list(parent1.features.keys())
        feature_names = [feature_name for feature_name in feature_names if not isinstance(self.instance_sampler.feature_samplers[feature_name], ImmutableSampler)]
        rand_key = random.choice(feature_names)

        # Swap values after the random key
        for key in feature_names:
            if key > rand_key:
                child1.features[key], child2.features[key] = child2.features[key], child1.features[key]
        
        return child1, child2


          
    def mutate(self, instance):
        """Perform mutation"""
        # Create a copy of the instance to mutate
        #mutated_instance = copy(instance)
        mutated_instance = deepcopy(instance)

        # Get ordered feature names
        feature_names = list(instance.features.keys())

        # Get feature names with no immutability constraints
        feature_names = [feature_name for feature_name in feature_names if not isinstance(self.instance_sampler.feature_samplers[feature_name], ImmutableSampler)]
        # Select random feature for mutation
        mutation_key = random.choice(feature_names)
        if isinstance(self.instance_sampler.feature_samplers[mutation_key], PermittedRangeSampler):
            if mutation_key in self.transformer.continuous_features_transformers:
                # Apply the mutation function to the selected feature's value
                original_value = mutated_instance.features[mutation_key].value
                if self.cont_mutation_function(original_value) in self.instance_sampler.feature_samplers[mutation_key].permitted_range:
                    mutated_instance.features[mutation_key].value = self.cont_mutation_function(original_value)
                else:
                    mutated_instance.features[mutation_key].value = random.choice(self.instance_sampler.feature_samplers[mutation_key].permitted_range)
        else:
            if mutation_key in self.transformer.continuous_features_transformers:
                # Apply the mutation function to the selected feature's value
                original_value = mutated_instance.features[mutation_key].value
                mutated_instance.features[mutation_key].value = self.cont_mutation_function(original_value)
            elif mutation_key in self.transformer.categorical_features_transformers:
                # Apply the mutation function to the selected feature's value
                original_value = mutated_instance.features[mutation_key].value
                mutated_instance.features[mutation_key].value = self.cat_mutation_function(original_value, mutation_key)

        return mutated_instance
    
    def cont_mutation_function(self, value):
        """Mutation function for continuous features"""
        mutation_range = 0.5
        mutation_value = value + random.uniform(-mutation_range, mutation_range)
        return mutation_value

    def cat_mutation_function(self, fvalue, fname):
        """Mutation function for categorical features"""
        range = self.transformer.categorical_features_transformers[fname].normalized_range
        # random choice from frange excluding current value
        while True:
            mutation_value = random.randint(range[0], range[1])
            if mutation_value != fvalue:
                break

        return mutation_value

    def evaluate_population(self, population, query_instance, desired_output):
        """Evaluate the population by calculating fitness function"""
        # evaluate population
        fitness = []
        loss = []
        distance_combined = []
        for i in range(len(population)):
            fitness_curr, loss_curr, distance_curr = self.fitness_function(population[i], query_instance, desired_output)
            fitness.append(fitness_curr)
            loss.append(loss_curr)
            distance_combined.append(distance_curr)
        return fitness, loss, distance_combined

    def sort_population(self, population, fitness_list, loss_list, distance_combined_list):
        """Sort the population by fitness function"""
        # sort population according to fitness list
        paired_population = zip(population, fitness_list, loss_list, distance_combined_list)      
        sorted_population = sorted(paired_population, key=lambda x: x[1])
        sorted_population, sorted_fitness_values, sorted_loss, sorted_distance = zip(*sorted_population)
    
        return list(sorted_population), list(sorted_fitness_values), list(sorted_loss), list(sorted_distance)

    def select_population(self, population, population_size):
        """Select the population by truncation"""
        # select population
        population = population[:population_size]
        return population
    
    def check_prediction_one(self, counterfactual_instance, desired_output):
        """Check if counterfactual instance is valid"""
        # check if counterfactual instance is valid
        if self.model.model_type == "classification":
            counterfactual_prediction = self.model.predict_instance(counterfactual_instance)
            if counterfactual_prediction == desired_output:
                return True
            else:
                return False
        elif self.model.model_type == "regression":
            counterfactual_prediction = self.model.predict_instance(counterfactual_instance)
            if desired_output[0] <= counterfactual_prediction <= desired_output[1]:
                return True
            else:
                return False
        else:
            return False
        
    def check_prediction_all(self, population, desired_output):
        """Check if all counterfactual instances are valid"""
        # check if counterfactual instance is valid
        if self.model.model_type == "classification":
            for i in range(len(population)):
                counterfactual_prediction = self.model.predict_instance(population[i])
                if counterfactual_prediction != desired_output:
                    return False
            return True
        elif self.model.model_type == "regression":
            for i in range(len(population)):
                counterfactual_prediction = self.model.predict_instance(population[i])
                if not desired_output[0] <= counterfactual_prediction <= desired_output[1]:
                    return False
            return True
        else:
            return False
        
    def calculate_population_diversity(self, population):
        # Example: Calculate diversity based on feature variance
        feature_values = [individual.get_values_dict() for individual in population]
        # Assuming the features are numerical
        feature_matrix = np.array([[values[feature] for feature in values] for values in feature_values])
        diversity = np.var(feature_matrix)
        return diversity

    def adjust_mutation_rate(self, current_rate, diversity, min_rate, max_rate):
        threshold_diversity = 0.5
        if diversity < threshold_diversity:
            return min(current_rate * 1.1, max_rate)  # Increase mutation rate
        else:
            return max(current_rate * 0.9, min_rate)  # Decrease mutation rate
        
    

    def find_counterfactuals(self, query_instance, number_cf, desired_output, maxiterations):
        """Find counterfactuals by generating them through genetic algorithm"""
        # population size might be parameter or depend on number cf required
        population_size = 40*number_cf
        # prepare data instance to format and transform categorical features
        # Normalization is happening one level above
        #self.transformer.normalize_instance(query_instance)
        # find predictive value of original instance
        # set desired class to be opposite of original instance

        self.counterfactuals = []
        iterations = 0
        fitness_history = []
        loss_history = []
        diff_distance_history = []
        best_candidates_history = []
        t = 1e-4
        stop_count = 0
        self.population = self.generate_population(query_instance, population_size)
        fitness_list, loss, distance_combined = self.evaluate_population(self.population, query_instance, desired_output)
        
        # Sorting didn't work properly
        self.population, fitness_list, loss, distance_combined = self.sort_population(self.population, fitness_list, loss, distance_combined)
        fitness_history.append(fitness_list[0])
        loss_history.append(loss[0])
        diff_distance_history.append(distance_combined[0])
        best_candidates_history.append(self.population[0]) 

        min_mutation_rate, max_mutation_rate = 0.1, 0.9
        mutation_rate = 0.5

        # until the stopping criteria is reached
        while iterations < maxiterations and len(self.counterfactuals) < number_cf:
            # if fitness is not improving for 5 generation break
            if len(fitness_history) > 1 and abs(fitness_history[-2] - fitness_history[-1]) <= t and self.check_prediction_all(self.population[:number_cf], desired_output):
                stop_count += 1
            if stop_count > 50:
                break
            # predict and compare to desired output
            # Select 50% of the best individuals, crossover the rest
            top_individuals = self.select_population(self.population, population_size//2)
            # crossover
            children = []
            for i in range(len(top_individuals)//2):
                child1, child2 = self.one_point_crossover(top_individuals[i], top_individuals[len(top_individuals)-1-i])
                children.append(child1)
                children.append(child2)
            self.population = top_individuals + children
            # For debugging
            # mutate with probability of mutation Maybe decrease mutation within convergence
            # In your main GA loop
            diversity = self.calculate_population_diversity(self.population)
            mutation_rate = self.adjust_mutation_rate(mutation_rate, diversity, min_mutation_rate, max_mutation_rate)

            for i in range(len(self.population)):
                # If the mutation probability is greater than a random number, mutate
                if random.random() < mutation_rate:
                    self.population[i] = self.mutate(self.population[i])
            fitness_list, loss, distance_combined = self.evaluate_population(self.population, query_instance, desired_output)
            self.population, fitness_list, loss, distance_combined = self.sort_population(self.population, fitness_list, loss, distance_combined)
            fitness_history.append(fitness_list[0])
            loss_history.append(loss[0])
            diff_distance_history.append(distance_combined[0])
            best_candidates_history.append(self.population[0])

            iterations += 1

        self.counterfactuals = self.population[:number_cf]

        return self.counterfactuals, best_candidates_history, fitness_history, loss_history, diff_distance_history