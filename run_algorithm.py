import csv
from numpy import dot, array
from numpy.linalg import norm
from matplotlib import pyplot as plt
from glob import glob
from os import mkdir
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

TOTAL_EXECUTIONS = 1
DRAW_CURVE = False
FITNESS_FUNC = F2
c = 0.1 # constant for fitness function
m = 10 # bits used to represent each gene
POP_SIZE = 20
CROSSOVER_PROB = 0.6
MUTATION_PROB = 0.01

if __name__ == "__main__":
    if TOTAL_EXECUTIONS > 1:
        mean_accuracy = 0
        mean_epochs = 0
    
    if DRAW_CURVE:
        accuracy_per_epoch = [[] for _ in range(TOTAL_EXECUTIONS)]

    target_fitness = FITNESS_FUNC(class_means_normalized["sitting"], binary=False)

    for i in range(TOTAL_EXECUTIONS):
        pop = Population(POP_SIZE, 12, m)
        GA = GeneticAlgorithm(pop, FITNESS_FUNC, crossover_prob=CROSSOVER_PROB, mutation_prob=MUTATION_PROB)

        running = True
        while running:
            if DRAW_CURVE:
                _, current_max_fitness = GA.get_fittest()
                accuracy_per_epoch[i].append(100*current_max_fitness/target_fitness)

            running = GA.run_epoch()

        fittest_individual, max_fitness = GA.get_fittest()
        accuracy = max_fitness / target_fitness

        if TOTAL_EXECUTIONS == 1:
            fittest_individual_vector = [round(gene, 3) for gene in utils.decode_individual(fittest_individual, m)]
            print("Fittest indvividual (binary): ", fittest_individual)
            print("Fittest individual: ", fittest_individual_vector)
            print("Fittest individual (denormalized): ", utils.denormalize_vector(fittest_individual_vector, minima, maxima))
            print("Fitness: {:.4f}".format(max_fitness))
            print("Accuracy: {:.2f}%".format(100*accuracy))
            print("Epochs: ", GA.epochs)
        else:
            print("Execution {}, accuracy {:.2f}% in {} epochs".format(i+1, 100*accuracy, GA.epochs))
            mean_accuracy += accuracy
            mean_epochs += GA.epochs

    if TOTAL_EXECUTIONS > 1:
        mean_accuracy /= TOTAL_EXECUTIONS
        mean_epochs /= TOTAL_EXECUTIONS

        print(f"Total executions: {TOTAL_EXECUTIONS}")
        print(f"Population size: {POP_SIZE}")
        print(f"Crossover probability: {CROSSOVER_PROB}")
        print(f"Mutation probability: {MUTATION_PROB}")
        print("Mean maximum achieved accuracy: {:.2f}%".format(100*mean_accuracy))
        print(f"Mean epochs: {mean_epochs}")

    if DRAW_CURVE:
        if TOTAL_EXECUTIONS > 1:
            mean_accuracy_per_epoch = []

            for i in range(int(sum(len(l) for l in accuracy_per_epoch)/TOTAL_EXECUTIONS)):
                s = 0
                total = 0
                for j in range(TOTAL_EXECUTIONS):
                    if len(accuracy_per_epoch[j]) > i:
                        s += accuracy_per_epoch[j][i]
                        total += 1
                mean_accuracy_per_epoch.append(s/total)
        else:
            mean_accuracy_per_epoch = accuracy_per_epoch[0]
        
        plt.plot(mean_accuracy_per_epoch)
        plt.ylabel("Mean Accuracy (%)")
        plt.xlabel("Epoch")
        plt.title(f"Population size: {POP_SIZE}\nCrossover probability: {CROSSOVER_PROB}\nMutation probability: {MUTATION_PROB}")
        plt.tight_layout()

        if "plots" not in glob("*"):
            mkdir("plots")

        plt.savefig(f"plots/meanAccuracy-epoch_{POP_SIZE}_{CROSSOVER_PROB}_{MUTATION_PROB}.png")
