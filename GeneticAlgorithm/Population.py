from GeneticAlgorithm.Individual import Individual

class Population:
    """
    Defines a population of individuals
    Initial population is generated randomly
    """
    def __init__(self, size, individual_num_genes, gene_sizes):
        """
        Initializes a population of individuals
        :param size: Number of individuals in the population
        :param individual_num_genes: Number of genes in each individual
        :param gene_sizes: Size of each gene; can be integer (all genes of same size) or list of size individual_genes (each gene has its own size)
        """
        if type(size) is not int:
            raise TypeError("size must be an integer")
        elif size < 1:
            raise ValueError("size must be a positive integer")

        self.individuals = []
        for _ in range(size):
            self.individuals.append(Individual(individual_num_genes, gene_sizes))

    def __len__(self):
        return len(self.individuals)
    
    def __iter__(self):
        return iter(self.individuals)
    
    def __getitem__(self, key):
        return self.individuals[key]
    
    def replace(self, new_population):
        """
        Replaces the old population with a new one
        :param new_population: a list of Individuals of the new population; must be of the same size as old list
        """
        if len(new_population) != len(self.individuals):
            raise ValueError("incorrect population size")
        
        self.individuals = new_population

    def get_cumulative_probabilities(self, fitness_func):
        """
        Returns a list of cumulative probabilities for the individuals of the population
        :param fitness_func: the fitness function to use
        """
        cumulative_prob = []
        total_fitness = 0

        for individual in self.individuals:
            fitness = fitness_func(individual)
            cumulative_prob.append(fitness)
            total_fitness += fitness

        for i, fitness in enumerate(cumulative_prob):
            cumulative_prob[i] = fitness/total_fitness
            if i != 0:
                cumulative_prob[i] += cumulative_prob[i - 1]
        
        return cumulative_prob
    