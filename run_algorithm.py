import csv
from numpy import dot
from numpy.linalg import norm
from GeneticAlgorithm import *

c = 0 # constant for fitness function
m = 10 # bits used to represent each gene
POP_SIZE = 20
CROSSOVER_PROB = 0.6
MUTATION_PROB = 0.0

class_means = {}
with open("data/class-means.csv", "r") as csvf:
    reader = csv.reader(csvf, delimiter=";")

    for row in reader:
        class_means[row[0]] = [float(val) for val in row[1:]]

def decode_individual(individual):
    v = []
    for gene in individual:
        quantized_mid = int(gene, 2)/2**m + 1/2**(m+1)
        v.append(quantized_mid)

    return v

def cos_similarity(A, B):
    return dot(A, B) / (norm(A)*norm(B))

def F(individual, binary=True):
    v = decode_individual(individual) if binary else individual

    return (cos_similarity(v, class_means["sitting"]) + c*(1 - sum(cos_similarity(v, class_) for class_ in class_means.values() if class_ != "sitting")/4))/(1 + c)    

if __name__ == "__main__":
    pop = Population(POP_SIZE, 12, m)
    GA = GeneticAlgorithm(pop, F, crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB)

    running = True
    while running:
        running = GA.run_epoch()

    fittest_individual, max_fitness = GA.get_fittest()
    print("Fittest invividual (binary): ", fittest_individual)
    print("Fittest individual: ", [round(gene, 3) for gene in decode_individual(fittest_individual)])
    print("Fitness: {:.4f}".format(max_fitness))
    print("Accuracy: {:.2f}%".format(100*max_fitness / F(class_means["sitting"], binary=False)))
    print("Epochs: ", GA.epochs)

    """
    for class_ in class_means:
        print(class_, F(class_means[class_], binary=False))

    print("Test", F([0.366]*12, binary=False))
    """
