import random
from copy import copy

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
        if self.loss_type == "hinge_loss":
            original_prediction = self.model.predict_instance(query_instance)
            counterfactual_prediction = self.model.predict_instance(counterfactual_instance)
            # TODO: take into consideration that predicitons might be not normalized
            # TODO: add loss function for regression
            loss = max(0, 1 - original_prediction * counterfactual_prediction)
        # calculate distance function depending on distance type
        distance_continuous = 0
        if self.distance_continuous["type"] == "weighted_l1":
            # apply weights depending if it's categorical or continuous feature weighted with inverse MAD from train data
            # TODO: add weights for categorical and continious features
            for feature_name in self.transformer.continuous_features_transformers:
                distance_continuous += abs(query_instance.features[feature_name].value - counterfactual_instance.features[feature_name].value)/self.mads[feature_name]
            if len(self.transformer.continuous_features_transformers) > 0:
                distance_continuous = distance_continuous/self.transformer.get_cont_transformers_length()
        elif self.distance_continuous["type"] == "diffusion_map":
            pass

        distance_categorical = 0
        if self.distance_categorical == "hamming":
            # apply weights depending if it's categorical or continuous feature weighted with inverse MAD from train data
            # TODO: add weights for categorical and continious features
            for feature_name in self.transformer.categorical_features_transformers:
                distance_categorical += query_instance.features[feature_name].value != counterfactual_instance.features[feature_name].value
            if len(self.transformer.categorical_features_transformers) > 0:
                distance_continuous = distance_continuous/self.transformer.get_cat_transformers_length()
        distance = distance_continuous + distance_categorical

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
                num_categories = len(self.transformer.categorical_features_transformers[feature_name].categories)
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
        fitness = self.hyperparameters[0]*loss + self.hyperparameters[1]*distance + self.hyperparameters[2]*coherence + elastic_net_penalty_approx
        return fitness

    def generate_population(self, query_instance, population_size):
        """Initialize the populationg following sampling strategy"""
        # initialize population
        population = []
        counterfactual_instance = copy(query_instance)

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

        child1 = copy(parent1)
        child2 = copy(parent2)

        # Get ordered feature names
        feature_names = list(parent1.features.keys())
        rand_key = random.choice(feature_names)

        # Swap values after the random key
        for key in feature_names:
            if key > rand_key:
                child1.features[key], child2.features[key] = child2.features[key], child1.features[key]
        
        return child1, child2


          
    def mutate(self, instance):
        """Perform mutation"""
        # Create a copy of the instance to mutate
        mutated_instance = copy(instance)

        # Get ordered feature names
        feature_names = list(instance.features.keys())

        # Select random feature for mutation
        mutation_key = random.choice(feature_names)

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
        for i in range(len(population)):
            fitness.append(self.fitness_function(population[i], query_instance, desired_output))
        return fitness

    def sort_population(self, population, fitness_list):
        """Sort the population by fitness function"""
        # sort population according to fitness list
        paired_population = zip(population, fitness_list)      
        sorted_population = sorted(paired_population, key=lambda x: x[1])
        sorted_population, sorted_fitness_values = zip(*sorted_population)
    
        return list(sorted_population), list(sorted_fitness_values)

    def select_population(self, population, population_size):
        """Select the population by truncation"""
        # select population
        population = population[:population_size]
        return population
    
    def check_prediction_one(self, counterfactual_instance, desired_output):
        """Check if counterfactual instance is valid"""
        # check if counterfactual instance is valid
        if self.model.model_type == "classification":
            counterfactual_prediction = self.model.predict(counterfactual_instance)
            if counterfactual_prediction == desired_output:
                return True
            else:
                return False
        elif self.model.model_type == "regression":
            counterfactual_prediction = self.model.predict(counterfactual_instance)
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
        
    

    def find_counterfactuals(self, query_instance, number_cf, desired_output, maxiterations):
        """Find counterfactuals by generating them through genetic algorithm"""
        # population size might be parameter or depend on number cf required
        population_size = 10*number_cf
        # prepare data instance to format and transform categorical features
        query_original = copy(query_instance)
        # Normalization is happening one level above
        #self.transformer.normalize_instance(query_instance)
        # find predictive value of original instance
        # set desired class to be opposite of original instance

        self.counterfactuals = []
        iterations = 0
        fitness_history = []
        best_candidates_history = []
        t = 1e-2
        stop_count = 0
        self.population = self.generate_population(query_instance, population_size)
        fitness_list = self.evaluate_population(self.population, query_instance, desired_output)
        self.sorted_population, sorted_fitness = self.sort_population(self.population, fitness_list)
        fitness_history.append(sorted_fitness[0])
        best_candidates_history.append(self.sorted_population[0])

        # until the stopping criteria is reached
        while iterations < maxiterations and len(self.counterfactuals) < number_cf:
            # if fitness is not improving for 5 generation break
            if len(fitness_history) > 1 and abs(fitness_history[-2] - fitness_history[-1]) <= t and self.check_prediction_all(self.population[:number_cf], desired_output):
                stop_count += 1
            if stop_count > 4:
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
            # mutate with probability of mutation Maybe decrease mutation within convergence
            for i in range(len(children)):
                # If the mutation probability is greater than a random number, mutate
                if random.random() < 0.1:
                    children[i] = self.mutate(children[i])
            # concatenate children and top individuals
            self.population = top_individuals + children
            fitness_list = self.evaluate_population(self.population, query_instance, desired_output)
            self.sorted_population, sorted_fitness = self.sort_population(self.population, fitness_list)
            fitness_history.append(sorted_fitness[0])
            best_candidates_history.append(self.sorted_population[0])
            iterations += 1

        
        self.counterfactuals = self.population[:number_cf]
        return self.counterfactuals