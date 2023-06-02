from random import random, choice
from copy import deepcopy
from math import ceil
from GeneticAlgorithm.Population import Population

class GeneticAlgorithm:
    """
    Defines a genetic algorithm
    """

    def __init__(self, 
                 population, 
                 fitness_func, 
                 crossover_prob=0.8, 
                 mutation_prob=0.01, 
                 selection="tournament", 
                 tournament_size=3, 
                 max_epochs = 1000, 
                 epoch_threshold=10, 
                 minimum_improvement=0.01, 
                 early_stopping=True
                 ):
        """
        Initializes the genetic algorithm
        :param population: Population object
        :param fitness_func: fitness function; must be callable that accepts Individual object and returns an int/float between 0 and 1
        :param crossover_prob: probability of each individual to be chosen for crossover
        :param mutation_prob: probability of each bit of each individual to be mutated (flipped)
        :param selection: selection process to use; can be one of "tournament", "roulette_fitness"
        :param tournament_size: size of each tournament of tournament selection; higher values increase selection pressure
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
        
        self.SELECTION = {"tournament": self._tournament_selection, "roulette_fitness": self._roulette_fitness_selection}
        
        if selection not in self.SELECTION.keys():
            raise ValueError(f"selection must be one of: {GeneticAlgorithm.SELECTION.keys()}")
        
        self.population = population
        self.fitness_func = fitness_func
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.selection = selection
        self.tournament_size = tournament_size
        self.max_epochs = max_epochs
        self.epoch_threshold = epoch_threshold
        self.minimum_improvement = minimum_improvement
        self.early_stopping = early_stopping

        # fitness values for the current fittest individual
        self.fittest = None
        self.max_fitness = 0

        self.epochs = 0 # current epoch
        self.no_improvement_epochs = 0 # epochs there hasn't been improvement for
        self.stop = False # whether algorithm has stopped due to some criterion

    def _tournament_selection(self):
        """
        Performs tournament selection with tournament size = self.tournament_size, returning a list of selected individuals
        """
        temp_population = []
        for _ in range(len(self.population)):
            selected = []
            for _ in range(self.tournament_size):
                selected.append(choice(self.population))

            fittest, max_fitness = selected[0], self.fitness_func(selected[0])
            for individual in selected[1:]:
                fitness = self.fitness_func(individual)
                if fitness > max_fitness:
                    fittest, max_fitness = individual, fitness

            # if individual was already picked, we need to add a copy instead
            if fittest in temp_population:
                temp_population.append(deepcopy(fittest))
                continue
            
            temp_population.append(fittest)

        return temp_population
    
    def _roulette_fitness_selection(self):
        """
        Performs roulette wheel selection based on fitness, returning a list of selected individuals
        TODO: algorithm doesn't converge with this, figure out why
        TODO: make all fitnesses positive by adding polarization
        """
        cumulative_prob = self.population.get_cumulative_probabilities(self.fitness_func)

        temp_population = []
        # for each individual, select 1 individual using the cumulative probability
        for _ in range(len(self.population)):
            p = random()
            for i, prob in enumerate(cumulative_prob):
                if p < prob:
                    if self.population[i] in temp_population:
                        temp_population.append(deepcopy(self.population[i]))
                        break

                    temp_population.append(self.population[i])
                    break

        return temp_population
    
    def _one_point_crossover(self, temp_population):
        """
        Performs in-place one-point crossover with probability self.crossover_prob
        :param temp_population: a list of the selected individuals to crossover
        """
        # for each pair, decide if there should be crossover
        # len()-1 deals with odd population numbers
        crossover_indices = []
        for i in range(0, len(temp_population) - 1, 2):
            if random() < self.crossover_prob:
                crossover_indices.append(i)

        for i in crossover_indices:
            temp_population[i].crossover(temp_population[i+1])

    def _find_fittest(self):
        """
        Returns the current maximum fitness and the index of the fittest individual in the population
        """
        max_fitness = 0
        max_fitness_index = 0
        
        for i, individual in enumerate(self.population):
            fitness = self.fitness_func(individual)
            if fitness > max_fitness:
                max_fitness = fitness
                max_fitness_index = i

        return max_fitness, max_fitness_index

    def run_epoch(self):
        """
        Runs one epoch of the genetic algorithm
        This consists of:
        1. Selection: selecting which chromosomes will be used as parents for the next generation;
            this can be done using roulette wheel selection based on fitness, or tournament selection
        2. Crossover: crossing over certain chromosomes out of the ones selected to create offspring;
            this is done using one-point crossover
        3. Mutation: mutating chromosomes with a given probability;
            elitism is implemented for the fittest individual
        """
        if self.early_stopping and self.stop:
            return False

        # Selection
        temp_population = self.SELECTION[self.selection]()
        
        # Crossover
        self._one_point_crossover(temp_population)

        # replace population with the newly created one
        self.population.replace(temp_population)

        # find fittest individual for elitism
        _, max_fitness_index = self._find_fittest()
    
        # mutate all individuals except fittest
        for i, individual in enumerate(self.population):
            if i != max_fitness_index:
                individual.mutate(self.mutation_prob)

        # find fittest again and save
        current_max_fitness, max_fitness_index = self._find_fittest()

        # update existing fittest if better solution is found
        prev_max_fitness = self.max_fitness
        if current_max_fitness > self.max_fitness:
            self.fittest = deepcopy(self.population[max_fitness_index])
            self.max_fitness = current_max_fitness

        # there is no need to replace the old population with the new one
        # since all operators are in-place, the population is already replaced

        # stop if no improvement threshold has been reached
        if self.max_fitness > (1 + self.minimum_improvement)*prev_max_fitness:
            self.no_improvement_epochs = 0
        else:
            self.no_improvement_epochs += 1
            
            if self.epoch_threshold > 0 and self.no_improvement_epochs == self.epoch_threshold:
                self.stop = True

        self.epochs += 1
        if self.epochs == self.max_epochs:
            self.stop = True
        
        return True
    
    def get_fittest(self):
        """
        Returns current individual with maximum fitness, and its fitness value
        """

        return self.fittest, self.max_fitness
