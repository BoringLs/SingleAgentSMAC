import torch
import numpy as np


def to_one_hot(x, max_size):
    out = torch.zeros([max_size])
    out[x] = 1
    return out.reshape(1, -1)
