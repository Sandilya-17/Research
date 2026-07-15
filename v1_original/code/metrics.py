import numpy as np

def kl(p,q):
    return np.sum(p * np.log((p+1e-10)/(q+1e-10)))

def fidelity(p,q):
    return (np.sum(np.sqrt(p*q)))**2

def mse(p,q):
    return ((p-q)**2).mean()