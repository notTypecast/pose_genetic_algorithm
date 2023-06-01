from random import random
from GeneticAlgorithm.Population import Population

class GeneticAlgorithm:
    """
    Defines a genetic algorithm
    """
    def __init__(self, population, fitness_func, crossover_prob=0.8, mutation_prob=0.01, max_epochs = 1000, epoch_threshold=5, minimum_improvement=0.01, early_stopping=True):
        """
        Initializes the genetic algorithm
        :param population: Population object
        :param fitness_func: fitness function; must be callable that accepts Individual object and returns an int/float between 0 and 1
        :param crossover_prob: probability of each individual to be chosen for crossover
        :param mutation_prob: probability of each bit of each individual to be mutated (flipped)
        :param max_epochs: stop if this number of epochs has been reached (set to 0 to disable)
        :param epoch_threshold: stop if fittest individual does not improve for this many epochs (set 0 to disable)
        :param minimum_improvement: minimum improvement for epoch threshold (set to 0 to reset with any improvement, however small)
        :param early_stopping: determines whether early stopping is used at all; if set to False, the algorithm will always run an epoch whenever called
        """
        if type(population) is not Population:
            raise TypeError("population must be a Population object")
        
        if not callable(fitness_func):
            raise TypeError("fitness_func must be callable")
        
        if type(crossover_prob) is not float:
            raise TypeError("crossover_prob must be a float")
        
        if type(mutation_prob) is not float:
            raise TypeError("mutation_prob must be a float")
        
        if not 0 <= crossover_prob <= 1:
            raise ValueError("crossover_prob must be in [0, 1]")
        
        if not 0 <= mutation_prob <= 1:
            raise ValueError("mutation_prob must be in [0, 1]")
        
        self.population = population
        self.fitness_func = fitness_func
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_epochs = max_epochs
        self.epoch_threshold = epoch_threshold
        self.minimum_improvement = minimum_improvement
        self.early_stopping = early_stopping

        # fitness values for the current fittest individual
        self.current_max_fitness = 0
        self.max_fitness_index = -1

        self.epochs = 0 # current epoch
        self.last_max_fitness = 0 # largest fitness value ever reached
        self.no_improvement_epochs = 0 # epochs there hasn't been improvement for
        self.stop = False # whether algorithm has stopped due to some criterion

    def run_epoch(self):
        """
        Runs one epoch of the genetic algorithm
        This consists of:
        1. Selection: selecting which chromosomes will be used as parents for the next generation;
            this is done using roulette wheel selection based on fitness
        2. Crossover: crossing over certain chromosomes out of the ones selected to create offspring;
            this is done using one-point crossover
        3. Mutation: mutating chromosomes with a given probability;
            elitism is implemented for the fittest individual
        """
        if self.early_stopping and self.stop:
            return False

        # Selection
        cumulative_prob = self.population.get_cumulative_probabilities(self.fitness_func)

        temp_population = []
        # for each individual, select 1 individual using the cumulative probability
        for _ in range(len(self.population)):
            for i, prob in enumerate(cumulative_prob):
                if random() < prob:
                    temp_population.append(self.population[i])
                    break

        # Crossover
        crossover_indices = []
        for i in range(len(temp_population)):
            if random() < self.crossover_prob:
                crossover_indices.append(i)

        # crossover each chosen individual with the next one
        # if an odd number of individuals is picked, the last one is discarded due to len()-1
        for i in range(0, len(crossover_indices) - 1, 2):
            temp_population[i].crossover(temp_population[i+1])

        # find fittest individual for elitism
        max_fitness = self.fitness_func(temp_population[0])
        max_fitness_index = 0
        
        for i, individual in enumerate(temp_population[1:]):
            fitness = self.fitness_func(individual)
            if fitness > max_fitness:
                max_fitness = fitness
                max_fitness_index = i

        # save max fitness and individual
        self.current_max_fitness = max_fitness
        self.max_fitness_index = max_fitness_index

        # mutate all individuals except fittest
        for i, individual in enumerate(temp_population):
            if i != max_fitness_index:
                individual.mutate(self.mutation_prob)

        # there is no need to replace the old population with the new one
        # since all operators are in-place, the population is already replaced

        # stop if no improvement threshold has been reached
        if max_fitness > (1 + self.minimum_improvement)*self.last_max_fitness:
            self.last_max_fitness = max_fitness
            self.no_improvement_epochs = 0
        else:
            self.no_improvement_epochs += 1
            
            if self.epoch_threshold > 0 and self.no_improvement_epochs == self.epoch_threshold:
                self.stop = True

        self.epochs += 1
        if (self.epochs == self.max_epochs):
            self.stop = True

        return True
    
    def get_fittest(self):
        """
        Returns current individual with maximum fitness, and its fitness value
        """
        return self.population[self.max_fitness_index], self.current_max_fitness
