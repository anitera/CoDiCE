import numpy as np
import random
from src.ceinstance.instance_sampler import CEInstanceSampler

class CFsearch:
    def __init__(self, data, model, feature_sampler, algorithm="genetic", distance_continuous="weighted_l1", distance_categorical="weighted_l1", loss_type="hinge_loss", sparsity_hp=0.2, coherence_hp=0.2, diversity_hp=0.2):
        self.data = data
        self.model = model
        self.algorithm = algorithm
        self.distance_continuous = distance_continuous
        self.distance_categorical = distance_categorical
        self.loss_type = loss_type
        # TODO make instance sampler parameter, louse coupling config per class, instance sampler should be outside
        self.instance_sampler = feature_sampler # give as parameter instance_sampler
        self.objective_initialization(sparsity_hp, coherence_hp, diversity_hp)


    def objective_initialization(self, sparsity_hp, coherence_hp, diversity_hp):
        self.diffusion_map = self.data.diffusion_map
        self.mads = self.data.mads
        self.hyperparameters = [sparsity_hp, coherence_hp, diversity_hp]
        return self.hyperparameters

    
    def find_counterfactuals(self, query_instance, number_cf, desired_class, maxiterations=100):
        if self.algorithm == "genetic":
            # there might genetic related parameters, like population size, mutation rate etc.
            optimizer = GeneticOptimizer(self.data, self.model, self.distance_continuous, self.distance_categorical, self.loss_type, self.hyperparameters, self.diffusion_map, self.mads)
            self.counterfactuals = optimizer.find_counterfactuals(query_instance, number_cf, desired_class, maxiterations)
        return self.counterfactuals
    
    def evaluate_counterfactuals(self):
        # compute validity
        # compute sparsity
        return
    
    def visualize_counterfactuals(self):
        return


class GeneticOptimizer():
    def __init__(self, data, model, distance_continuous="weighted_l1", distance_categorical="weighted_l1", loss_type="hinge_loss", hyperparameters=[0.2, 0.2, 0.2], diffusion_map=None, mads=None):
        """Initialize data metaparmeters and model"""
        self.data = data
        self.model = model
        self.distance_continuous = distance_continuous
        self.distance_categorical = distance_categorical
        self.loss_type = loss_type
        self.diffusion_map = diffusion_map
        self.mads = mads
        self.hyperparameters = hyperparameters
        self.continuous_features_list = data.continuous_features_list
        self.categorical_features_list = data.categorical_features_list
        self.config = data.config
        self.outcome_column_name = data.outcome_column_name
        self.feature_names = self.continuous_features_list + self.categorical_features_list
        self.feature_types = ["continuous"] * len(self.continuous_features_list) + ["categorical"] * len(self.categorical_features_list)
        self.feature_types_dict = {k: v for k, v in zip(self.feature_names, self.feature_types)}

    def fitness_function(self, counterfactual_instance, query_instance, desired_output, hyperparameters):
        """Calculate fitness function which consist of loss function and distance function"""
        # calculate loss function depending on task classification or regression
        if self.loss_type == "hinge_loss":
            original_prediction = self.model.predict(query_instance)
            counterfactual_prediction = self.model.predict(counterfactual_instance)
            loss = max(0, 1 - original_prediction * counterfactual_prediction)
        # calculate distance function depending on distance type
        if self.distance_continuous == "weighted_l1":
            distance_continuous = 0
            # apply weights depending if it's categorical or continuous feature weighted with inverse MAD from train data
            # TODO: add weights for categorical and continious features
            for feature_name in self.feature_names:
                if self.feature_types_dict[feature_name] == "continuous":
                    distance_continuous += abs(query_instance[feature_name] - counterfactual_instance[feature_name])
            
        if self.distance_categorical == "weighted_l1":
            distance_categorical = 0
            # apply weights depending if it's categorical or continuous feature weighted with inverse MAD from train data
            # TODO: add weights for categorical and continious features
            for feature_name in self.feature_names:
                if self.feature_types_dict[feature_name] == "categorical":
                    distance_categorical += query_instance[feature_name] != counterfactual_instance[feature_name]
        distance = distance_continuous + distance_categorical
        # calculate fitness function
        fitness = hyperparameters[0]*loss + hyperparameters[1]*distance
        return fitness

    def generate_population(self, query_instance, population_size):
        """Initialize the populationg following sampling strategy"""
        # initialize population
        population = []
        counterfactual_instance = query_instance.copy()

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
        """Perform one point crossover"""
        arr1 = np.array(list(parent1.values()))
        arr2 = np.array(list(parent2.values()))
        # choose random point
        crossover_point = random.randint(1, len(arr1))
        child1_arr = np.concatenate((arr1[:crossover_point], arr2[crossover_point:]))
        child2_arr = np.concatenate((arr2[:crossover_point], arr1[crossover_point:]))
        child1 = {k: v for k, v in zip(self.feature_names, child1_arr)}
        child2 = {k: v for k, v in zip(self.feature_names, child2_arr)}
        return child1, child2

          
    def mutate(self, instance):
        """Perform mutation"""
        # choose random feature
        feature_name = random.choice(self.feature_names)
        # mutate feature
        if self.feature_types_dict[feature_name] == "continuous":
            pass
            # mutate continuous feature following sampler strategy
            #instance[feature_name] = random.uniform(self.data.minmax[feature_name][0], self.data.minmax[feature_name][1])
        else:
            pass
            # mutate categorical feature following sampler strategy
            #instance[feature_name] = random.choice(self.data.categorical_features_dict[feature_name])
        return instance

    def evaluate_population(self, population, query_instance, desired_output):
        """Evaluate the population by calculating fitness function"""
        # evaluate population
        for i in range(len(population)):
            population[i]["fitness"] = self.fitness_function(population[i], query_instance, desired_output)
        return population

    def sort_population(self, population):
        """Sort the population by fitness function"""
        # sort population
        population = sorted(population, key=lambda k: k["fitness"])
        return population

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
                counterfactual_prediction = self.model.predict(population[i])
                if counterfactual_prediction != desired_output:
                    return False
            return True
        elif self.model.model_type == "regression":
            for i in range(len(population)):
                counterfactual_prediction = self.model.predict(population[i])
                if not desired_output[0] <= counterfactual_prediction <= desired_output[1]:
                    return False
            return True
        else:
            return False
        
    

    def find_counterfactuals(self, query_instance, number_cf, desired_class, maxiterations):
        """Find counterfactuals by generating them through genetic algorithm"""
        # population size might be parameter or depend on number cf required
        population_size = 10*number_cf
        # prepare data instance to format and transform categorical features
        query_instance = self.data.prepare_query_instance(query_instance)
        query_original = query_instance.copy()
        # find predictive value of original instance
        original_prediction = self.model.predict(query_instance)
        # set desired class to be opposite of original instance
        if desired_class == "opposite" and self.model.model_type == "classification":
            desired_output = 1 - original_prediction
        elif self.model.model_type == "regression":
            desired_output = [desired_class[0], desired_class[1]]
        else:
            desired_output = desired_class

        self.counterfactuals = []
        iterations = 0
        fitness_history = []
        best_candidates_history = []
        t = 1e-2
        stop_count = 0
        self.population = self.generate_population(query_instance, population_size)
        self.population = self.evaluate_population(self.population, query_instance, desired_output)
        self.population = self.sort_population(self.population)
        fitness_history.append(self.population[0]["fitness"])
        best_candidates_history.append(self.population[0])
        print("Parent generation")
        print("Fitness: ", self.population[0]["fitness"])

        # until the stopping criteria is reached
        while iterations < maxiterations or len(self.counterfactuals) < number_cf:
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
            # mutate with probability of mutation
            for i in range(len(children)):
                children[i] = self.mutate(children[i])
            # concatenate children and top individuals
            self.population = top_individuals + children
            self.population = self.evaluate_population(self.population, desired_output)
            self.population = self.sort_population(self.population)
            fitness_history.append(self.population[0]["fitness"])
            best_candidates_history.append(self.population[0])
            print("Generation: ", iterations)
            print("Fitness: ", self.population[0]["fitness"])
            iterations += 1

        self.counterfactuals.append(self.population[:number_cf])
        return self.counterfactuals