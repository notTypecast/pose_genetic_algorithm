from numpy import dot
from numpy.linalg import norm

def decode_individual(individual, m):
    v = []
    for gene in individual:
        quantized_mid = int(gene, 2)/2**m + 1/2**(m+1)
        v.append(quantized_mid)

    return v

def cos_similarity(A, B):
    return dot(A, B) / (norm(A)*norm(B))

def shifted_denormalize_vector(v, minima, maxima):
    return [int(v[i]*(maxima[i] - minima[i])) for i in range(len(v))]

def denormalize_vector(v, minima, maxima):
    return [int(v[i]*(maxima[i] - minima[i]) + minima[i]) for i in range(len(v))]

def one_max(individual):
    return repr(individual).count("1")+1
