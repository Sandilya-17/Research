import numpy as np
import random

SEED = 42
SHOTS = 2048

def set_seed():
    np.random.seed(SEED)
    random.seed(SEED)