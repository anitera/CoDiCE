class GeneticOptimizer:
    def __init__(self, data, model):
        """Initialize data metaparmeters and model"""
        self.data = data
        self.model = model
        self.continuous_features_list = data.continuous_features_list
        self.categorical_features_list = data.categorical_features_list
        self.config = data.config
        self.outcome_column_name = data.outcome_column_name
        self.feature_names = self.continuous_features_list + self.categorical_features_list
        self.feature_types = ["continuous"] * len(self.continuous_features_list) + ["categorical"] * len(self.categorical_features_list)
        self.feature_types_dict = {k: v for k, v in zip(self.feature_names, self.feature_types)}

    def fitness_function(self, query_instance, counterfactual_instance, hyperparameters):
        """Calculate fitness function which consist of loss function and distance function"""
        # calculate loss function depending on task classification or regression
        loss_type = "hinge_loss"
        if loss_type == "hinge_loss":
            original_prediction = self.model.predict(query_instance)
            counterfactual_prediction = self.model.predict(counterfactual_instance)
            loss = max(0, 1 - original_prediction * counterfactual_prediction)
        # calculate distance function
        distance_type = "weighted_l1"
        if distance_type == "weighted_l1":
            distance_continuous = 0
            distance_categorical = 0
            # apply weights depending if it's categorical or continuous feature weighted with inverse MAD from train data
            # TODO: add weights for categorical and continious features
            for feature_name in self.feature_names:
                if self.feature_types_dict[feature_name] == "continuous":
                    distance_continuous += abs(query_instance[feature_name] - counterfactual_instance[feature_name])
                else:
                    distance_categorical += query_instance[feature_name] != counterfactual_instance[feature_name]
            distance = distance_continuous + distance_categorical
        # calculate fitness function
        fitness = hyperparameters[0]*loss + hyperparameters[1]*distance
        return fitness

    def generate_population(self, query_instance, population_size):
        """Initialize the populationg following sampling strategy"""
        # initialize population
        population = []
        # generate population
        for i in range(population_size):
            counterfactual_instance = query_instance.copy()
            for feature_name in self.feature_names:
                if self.feature_types_dict[feature_name] == "continuous":
                    #counterfactual_instance[feature_name] = random.uniform(self.data.minmax[feature_name][0], self.data.minmax[feature_name][1])
                    # Sample according to sampler strategy
                else:
                    #counterfactual_instance[feature_name] = random.choice(self.data.categorical_features_dict[feature_name])
            population.append(counterfactual_instance)
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
            # mutate continuous feature following sampler strategy
            #instance[feature_name] = random.uniform(self.data.minmax[feature_name][0], self.data.minmax[feature_name][1])
        else:
            # mutate categorical feature following sampler strategy
            #instance[feature_name] = random.choice(self.data.categorical_features_dict[feature_name])
        return instance

    def evaluate_population(self, population):
        """Evaluate the population by calculating fitness function"""
        # evaluate population
        for i in range(len(population)):
            population[i]["fitness"] = self.fitness_function(population[i])
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
    

    def find_counterfactuals(self, query_instance, population_size, maxiterations):
        """Find counterfactuals by generating them through genetic algorithm"""
        # until the stopping criteria is reached
        iterations = 0
        while iterations < maxiterations:
            self.population = self.generate_population(query_instance, population_size)
            self.population = self.evaluate_population(self.population)
            self.population = self.sort_population(self.population)
            self.population = self.select_population(self.population, population_size)
            self.population = self.mutate_population(self.population, maxiterations)
            self.population = self.evaluate_population(self.population)
        return self.population