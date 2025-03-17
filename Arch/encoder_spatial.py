import numpy as np
import torch
from torch import nn
from lib.hyper_parameters import hyper_parameters as HP


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            out_channel,
            out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.activate = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        x = self.activate(self.conv1(x))
        x = self.activate(self.conv2(x))
        x = x + residual
        return x


class Spatial_encoder(nn.Module):
    def __init__(self, args, n_resblock=2, device="cpu"):
        super().__init__()
        self.device = device

        # input: (batch, 2, 32, 32)
        self.proj = torch.nn.Conv2d(
            HP.map_channel,
            HP.original_16,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.ds1 = torch.nn.Conv2d(
            HP.original_16,
            HP.original_32,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.ds2 = torch.nn.Conv2d(
            HP.original_32,
            HP.original_64,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.resblocks = nn.ModuleList(
            [ResBlock(HP.original_64, HP.original_64) for _ in range(n_resblock)]
        )

        self.fc = nn.Linear(HP.original_64 * 8 * 8, HP.embedded_spatial_size)

        self.activate = nn.ReLU(inplace=False)

    def preprocess(self, state):
        unit_map = torch.zeros(32, 32, 4)
        mystate = state[: HP.entity_num * 4]
        enemy_state = state[HP.entity_num * 4 : HP.entity_num * 4 + HP.enemy_num * 3]

        for i in range(HP.entity_num):
            entity_state = mystate[i * 4 : i * 4 + 4]
            blood = entity_state[0]
            relative_x = entity_state[2]
            relative_y = entity_state[3]
            true_x = round((relative_x * 28) + 16)
            true_y = round((relative_y * 28) + 16)
            unit_map[true_x, true_y, 0] = 1
            unit_map[true_x, true_y, 1] = torch.tensor(blood)

        for i in range(HP.enemy_num):
            entity_state = enemy_state[i * 3 : i * 3 + 3]
            blood = entity_state[0]
            relative_x = entity_state[1]
            relative_y = entity_state[2]
            true_x = round((relative_x * 28) + 16)
            true_y = round((relative_y * 28) + 16)
            unit_map[true_x, true_y, 2] = 1
            unit_map[true_x, true_y, 3] = torch.tensor(blood)

        # batch_size, map_channel, mapx, mapy
        unit_map = unit_map.permute(2, 0, 1).unsqueeze(0)
        return unit_map

    def forward(self, prep):

        batch_size = prep.shape[0]
        x = prep.to(self.device)

        out = self.proj(x)
        out = self.ds1(out)
        out = self.ds2(out)

        for resblock in self.resblocks:
            out = resblock(out)

        out = out.reshape(batch_size, -1)
        embedded_spatial = self.activate(self.fc(out))

        return embedded_spatial
