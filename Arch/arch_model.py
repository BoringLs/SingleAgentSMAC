import torch
from torch import nn
from lib.hyper_parameters import hyper_parameters as HP
from Arch.core import Core
from Arch.encoder_entity import Entity_encoder
from Arch.encoder_spatial import Spatial_encoder
from Arch.head_action import ActionTypeHead


from lib.functions import *
from lib.const import *


class ArchModel(nn.Module):
    def __init__(self, args, device="cpu"):
        super().__init__()
        self.device = device

        self.entity_encoder = Entity_encoder(args=args, device=device)
        self.spatial_encoder = Spatial_encoder(args=args, device=device)
        self.action_type_head = ActionTypeHead(device=device)
        self.core = Core(device)

        self.critic = nn.Sequential(
            nn.Linear(HP.embedded_state_size, HP.original_64),
            nn.ReLU(),
            nn.Linear(HP.original_64, HP.original_32),
            nn.ReLU(),
            nn.Linear(HP.original_32, 1),
        )

        self.ar_fc = nn.Linear(HP.embedded_state_size, HP.autoregressive_embedding_size)

        self.action_type_head = ActionTypeHead(device)

    def gen_mask(self, action_type_mask):

        action_type_mask = torch.tensor(action_type_mask).to(self.device)
        action_type_mask = action_type_mask.unsqueeze(0)
        return action_type_mask

    def preprocess(self, obs, state, action_mask):
        entity_prep, entity_ids = self.entity_encoder.preprocess(obs)
        spatial_prep = self.spatial_encoder.preprocess(state)
        all_masks = self.gen_mask(action_mask)

        return entity_prep, spatial_prep, all_masks, entity_ids

    def forward(self, prep):
        with torch.no_grad():
            entity_prep, spatial_prep, action_type_masks, entity_ids = prep
            entity_embedding, embedded_entity = self.entity_encoder(entity_prep)
            embedded_spatial = self.spatial_encoder(spatial_prep)

            state = torch.cat([embedded_entity, embedded_spatial], dim=1)
            state = self.core(state)

            autoregressive_embedding = self.ar_fc(state)

            state_value = self.critic(state).cpu()

            total_actions = []
            total_action_type_logits = []

            for i, entity_id in enumerate(entity_ids):
                state_entity = torch.cat((state, entity_embedding[:, i]), dim=-1)
                action_type_mask = action_type_masks[:, entity_id]
                (
                    action_type,
                    action_type_logits,
                    autoregressive_embedding,
                ) = self.action_type_head(
                    autoregressive_embedding,
                    state_entity,
                    action_type_mask,
                )
                action = {
                    "unit_id": entity_id,
                    "action_type": action_type.detach().cpu(),
                }
                total_actions.append(action)
                total_action_type_logits.append(action_type_logits.detach().cpu())

            batch_action_type_logits = (
                torch.sum(torch.stack(total_action_type_logits), dim=0).detach().cpu()
            )

        # actions = [0] * HP.entity_size
        # for item in total_actions:
        #     actions[item["unit_id"]] = item["action_type"]

        actions = total_actions

        del total_action_type_logits
        del (
            state,
            autoregressive_embedding,
            entity_embedding,
        )

        return (
            actions,
            batch_action_type_logits,
            state_value,
        )

    def evaluate(self, preps, actions):
        entity_prep, spatial_prep, all_masks, entity_ids = preps

        entity_embedding, embedded_entity = self.entity_encoder(entity_prep)
        embedded_spatial = self.spatial_encoder(spatial_prep)

        state = torch.cat([embedded_entity, embedded_spatial], dim=1)
        state = self.core(state)

        autoregressive_embedding = self.ar_fc(state)
        state_value = self.critic(state)

        action_type_masks = all_masks
        total_action_type_logits = []
        total_action_type_entropy = []

        for i in range(HP.max_entity):
            state_entity = torch.cat((state, entity_embedding[:, i]), dim=-1)
            action_type_mask = action_type_masks[:, i]
            action_type = actions[:, i]

            exist_action_mask = torch.sum(action_type_mask.float(), dim=1) > 0
            exist_action_mask = exist_action_mask.to(self.device)
            if not exist_action_mask.any():
                break

            (
                action_type_entropy,
                action_type_logits,
                autoregressive_embedding,
            ) = self.action_type_head.evaluate(
                autoregressive_embedding,
                state_entity,
                action_type_mask,
                action_type,
            )

            total_action_type_logits.append(action_type_logits * exist_action_mask)
            total_action_type_entropy.append(action_type_entropy * exist_action_mask)

        batch_action_type_logits = torch.sum(
            torch.stack(total_action_type_logits), dim=0
        )
        batch_action_type_entropy = torch.sum(
            torch.stack(total_action_type_entropy), dim=0
        )

        return (
            batch_action_type_logits,
            batch_action_type_entropy,
            state_value.squeeze(-1),
        )
