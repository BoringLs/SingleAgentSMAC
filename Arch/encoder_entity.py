import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from lib.transformer import Transformer
from lib.hyper_parameters import hyper_parameters as HP


class Entity_encoder(nn.Module):
    def __init__(self, args, device="cpu"):
        super().__init__()
        self.args = args
        self.bias_value = HP.bias_value

        self.embed = nn.Linear(HP.entity_size, HP.original_64)

        self.transformer = Transformer(
            d_model=HP.original_64,
            d_inner=HP.original_256,
            n_head=2,
            d_k=HP.original_32,
            d_v=HP.original_32,
            n_layers=2,
            droput=0.1,
        )

        self.conv1 = nn.Conv1d(
            HP.original_64,
            HP.original_64,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.fc = nn.Linear(HP.original_64, HP.embedded_entity_size)
        self.activation = nn.ReLU(inplace=False)

        self.device = device

    def preprocess(self, obs):
        # obs is list, each item is a array of shape (80,)
        # so obs can be turn to a tensor of shape (batch,8, 80)
        entity_list = []
        alive_entity_ids = []
        for id, unit_info in enumerate(obs):
            unit_feature = []
            if unit_info[-1] != 0:
                alive_entity_ids.append(id)

                unit_feature.append(torch.tensor(unit_info))

                entity_tensor = torch.cat(unit_feature, dim=0)
                entity_list.append(entity_tensor)

        # all_entity_tensor = torch.cat(entity_list, dim=0)
        all_entity_tensor = torch.stack(entity_list)

        if all_entity_tensor.shape[0] < HP.max_entity:
            pad_len = HP.max_entity - all_entity_tensor.shape[0]
            pad = torch.zeros([pad_len, HP.entity_size])
            pad[:, :] = self.bias_value
            all_entity_tensor = torch.cat([all_entity_tensor, pad], dim=0)

        all_entity_tensor = all_entity_tensor.unsqueeze(0)
        # all_entity_tensor:[1, max_entity_num, entity_size]
        return all_entity_tensor, alive_entity_ids

    def forward(self, prep):
        x = prep.to(self.device)

        batch_size = x.shape[0]
        entity_or_bias = torch.mean(x, dim=2, keepdim=False) > self.bias_value
        entity_num = torch.sum(entity_or_bias, dim=1, keepdim=True)
        # mask.shape = [batch_size,max_entity_size]
        mask = (
            torch.arange(0, HP.max_entity).float().to(self.device).repeat(batch_size, 1)
            < entity_num
        )

        trans_mask = mask.unsqueeze(1).repeat(1, HP.max_entity, 1)
        trans_mask = trans_mask * mask.unsqueeze(2)

        out = self.embed(x)
        out = self.transformer(out, trans_mask)

        entity_embedding = self.activation(self.conv1(out.transpose(1, 2))).transpose(
            1, 2
        )

        masked_out = out * mask.unsqueeze(2)
        embedd_entity = torch.sum(masked_out, dim=1) / entity_num
        embedd_entity = self.fc(embedd_entity)

        return entity_embedding, embedd_entity
