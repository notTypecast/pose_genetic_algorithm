import csv
from numpy import dot, array
from numpy.linalg import norm
from GeneticAlgorithm import *
import utils

with open("data/normalization-extrema.csv", "r") as csvf:
    reader = csv.reader(csvf, delimiter=";")

    minima = [int(val) for val in next(reader)[1:]]
    maxima = [int(val) for val in next(reader)[1:]]

class_means_shifted = {}
with open("data/class-means.csv", "r") as csvf:
    reader = csv.reader(csvf, delimiter=";")

    for row in reader:
        class_means_shifted[row[0]] = [float(val) - minima[i] for i, val in enumerate(row[1:])]

class_means_normalized = {}
with open("data/class-means-normalized.csv", "r") as csvf:
    reader = csv.reader(csvf, delimiter=";")

    for row in reader:
        class_means_normalized[row[0]] = [float(val) for val in row[1:]]

def F(individual, binary=True):
    v = utils.decode_individual(individual, m) if binary else individual

    return (utils.cos_similarity(v, class_means_normalized["sitting"]) + c*(1 - sum(utils.cos_similarity(v, class_means_normalized[class_]) for class_ in class_means_normalized if class_ != "sitting")/4))/(1 + c)

def F2(individual, binary=True):
    v = utils.decode_individual(individual, m) if binary else individual
    #v = utils.shifted_denormalize_vector(v, minima, maxima)

    return 1 - norm(array(v) - array(class_means_normalized["sitting"]))

def F3(individual, binary=True):
    v = utils.decode_individual(individual, m) if binary else individual

    return 1 - (norm(array(v) - array(class_means_normalized["sitting"])) + c*(1 - sum([norm(array(v) - array(class_means_normalized[class_])) for class_ in class_means_normalized if class_ != "sitting"])/4))/(1 + c)

FITNESS_FUNC = F2
c = 0.1 # constant for fitness function
m = 10 # bits used to represent each gene
POP_SIZE = 100
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.01

if __name__ == "__main__":
    pop = Population(POP_SIZE, 12, m)
    GA = GeneticAlgorithm(pop, FITNESS_FUNC, crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB)

    running = True
    while running:
        running = GA.run_epoch()

    fittest_individual, max_fitness = GA.get_fittest()
    fittest_individual_vector = [round(gene, 3) for gene in utils.decode_individual(fittest_individual, m)]
    print("Fittest invividual (binary): ", fittest_individual)
    print("Fittest individual: ", fittest_individual_vector)
    print("Fittest individual (denormalized): ", utils.denormalize_vector(fittest_individual_vector, minima, maxima))
    print("Actual sitting values: ", )
    print("Fitness: {:.4f}".format(max_fitness))
    print("Accuracy: {:.2f}%".format(100*max_fitness / FITNESS_FUNC(class_means_normalized["sitting"], binary=False)))
    print("Epochs: ", GA.epochs)
