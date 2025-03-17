import torch
from torch import nn
from lib.hyper_parameters import hyper_parameters as HP
import numpy as np


class Core(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.fc1 = nn.Linear(
            HP.embedded_entity_size + HP.embedded_spatial_size, HP.original_128
        )
        self.fc2 = nn.Linear(HP.original_128, HP.original_256)
        self.fc3 = nn.Linear(HP.original_256, HP.embedded_state_size)
        self.activate = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.activate(self.fc1(x))
        x = self.activate(self.fc2(x))
        out = self.activate(self.fc3(x))
        return out
