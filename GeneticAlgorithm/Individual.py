from random import randint, random

class Individual:
    """
    Defines an individual of a population
    """
    def __init__(self, num_genes, gene_sizes):
        """
        Initializes an individual
        :param num_genes: Number of genes in the individual
        :param gene_sizes: Size of each gene; can be integer (all genes of same size) or list of size individual_genes (each gene has its own size)
        """
        if type(num_genes) is not int:
            raise TypeError("genes must be an integer")
        elif num_genes < 1:
            raise ValueError("genes must be a positive integer")
        
        if type(gene_sizes) is int:
            if gene_sizes < 1:
                raise ValueError("gene_sizes must be a positive integer")
        elif type(gene_sizes) is list:
            if len(gene_sizes) != num_genes:
                raise ValueError("gene_sizes must be of the same length as genes")
        else:
            raise TypeError("gene_sizes must be an integer or a list")

        self.genes = []

        for i in range(num_genes):
            if type(gene_sizes) is list:
                self.genes.append(Individual.random_bitstring(gene_sizes[i]))
            else:
                self.genes.append(Individual.random_bitstring(gene_sizes))

    def __repr__(self):
        return ("".join(self.genes))

    def __iter__(self):
        return iter(self.genes)

    def __len__(self):
        """
        Returns the full length of the chromosome
        """
        return sum(len(gene) for gene in self.genes)

    def crossover(self, other):
        """
        Crossover operator for individuals, performing in-place crossover with other individual
        :param other: Individual object to crossover with
        """
        crossover_point = randint(1, len(self) - 2)

        # keep cumulative length of genes to count through full chromosome
        cumulative_length = 0
        for i, gene in enumerate(self.genes):
            cumulative_length += len(gene)
            # check if crossover point has been passed (therefore is in this gene)
            if cumulative_length > crossover_point:
                # find crossover point on this gene
                local_crossover_point = crossover_point - cumulative_length + len(gene)
                # crossover the genes
                crossover_gene = other.genes[i]
                new_gene_a = gene[:local_crossover_point] + crossover_gene[local_crossover_point:]
                new_gene_b = crossover_gene[:local_crossover_point] + gene[local_crossover_point:]

                # replace genes with crossed over ones
                self.genes[i] = new_gene_a
                other.genes[i] = new_gene_b

                # swap all genes after this one
                for j in range(i+1, len(self.genes)):
                    self.genes[j], other.genes[j] = other.genes[j], self.genes[j]

                break

    def mutate(self, mutation_prob):
        """
        Performs mutations on each bit of the chromosome with a given probability
        :param mutation_prob: the probability to flip a bit
        """
        for gene in self.genes:
            new_gene = ""
            
            for i in range(len(gene)):
                if random() < mutation_prob:
                    new_gene += "0" if gene[i] == "1" else "1"
                else:
                    new_gene += gene[i]

            self.genes[i] = new_gene

    @staticmethod
    def random_bitstring(size):
        """
        Returns a random bitstring of size size
        :param size: size of bitstring
        """
        bitstr = ""
        for i in range(size):
            bitstr += str(randint(0, 1))

        return bitstr
