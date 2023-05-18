import torch
import random
import numpy as np

def reseed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)