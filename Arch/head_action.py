import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from lib.hyper_parameters import hyper_parameters as HP
from lib.functions import *
from lib.const import *


class ActionTypeHead(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.fc0 = nn.Linear(
            HP.embedded_state_size + HP.embedded_entity_size,
            HP.autoregressive_embedding_size,
        )
        self.fc1 = nn.Linear(HP.autoregressive_embedding_size, HP.original_128)
        self.fc2 = nn.Linear(HP.original_128, HP.original_64)
        self.fc3 = nn.Linear(HP.original_64, HP.action_num)

        self.fc4 = nn.Linear(HP.action_num, HP.autoregressive_embedding_size)

        self.softmax = nn.Softmax(dim=1)
        self.activate = nn.ReLU()

        self.device = device

    def forward(self, autoregressive_embedding, entity_state, action_type_mask):
        x = entity_state
        x = self.fc0(x) + autoregressive_embedding

        x = self.activate(self.fc1(x))
        x = self.activate(self.fc2(x))
        action_type_logits = self.fc3(x)

        action_type_mask = action_type_mask.to(self.device)
        masked_action_logits = action_type_logits + (~action_type_mask.bool()) * (-1e9)
        action_prob = self.softmax(masked_action_logits)

        dist = Categorical(action_prob)
        action_type = dist.sample()
        log_prob = dist.log_prob(action_type)

        action_type_one_hot = to_one_hot(action_type, HP.action_num).to(self.device)
        action_type_one_hot = action_type_one_hot.squeeze(-2)

        autoregressive_embedding = (
            self.fc4(action_type_one_hot) + autoregressive_embedding
        )

        return action_type, log_prob, autoregressive_embedding

    def evaluate(
        self, autoregressive_embedding, entity_state, action_type_mask, action_type
    ):
        x = entity_state
        x = self.fc0(x) + autoregressive_embedding

        x = self.activate(self.fc1(x))
        x = self.activate(self.fc2(x))
        action_type_logits = self.fc3(x)

        action_type_mask = action_type_mask.to(self.device)
        masked_action_logits = action_type_logits + (~action_type_mask.bool()) * (-1e9)
        action_prob = self.softmax(masked_action_logits)

        dist = Categorical(action_prob)
        entropy = dist.entropy()
        log_prob = dist.log_prob(action_type.to(self.device))

        action_type_one_hot = to_one_hot(action_type, HP.action_num).to(self.device)
        action_type_one_hot = action_type_one_hot.squeeze(-2)

        autoregressive_embedding = (
            self.fc4(action_type_one_hot) + autoregressive_embedding
        )

        return entropy, log_prob, autoregressive_embedding
