import numpy as np

def counts_to_vector(counts, shots):
    keys = [format(i, '04b') for i in range(16)]
    vec = np.array([counts.get(k,0) for k in keys])
    return vec / shots

def normalize(vec):
    vec = np.maximum(vec, 0)
    return vec / np.sum(vec)