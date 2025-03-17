import os, sys

sys.path.append("..")

import torch
import copy
import pickle
from tensorboardX import SummaryWriter
import time

from torch import nn
from torch.nn import functional as F
from lib.hyper_parameters import hyper_parameters as HP
from Arch.arch_model import ArchModel


class ReplayBuffer(object):
    def __init__(self):
        super(ReplayBuffer, self).__init__()
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.values = []

        self.buffer_size = 0
        self.seperates = []

    def store_transition(self, trajectories):
        for trajectory in trajectories:
            self.actions.append(trajectory["action"])
            self.states.append(trajectory["state"])
            self.log_probs.append(trajectory["log_prob"])
            self.rewards.append(trajectory["reward"])
            self.values.append(trajectory["value"])

        self.buffer_size += len(trajectories)
        self.seperates.append(self.buffer_size)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.values[:]
        del self.seperates[:]
        self.buffer_size = 0


class Learner:
    def __init__(
        self,
        model,
        args,
    ):
        self.device = model.device
        self.model = model
        self.buffer = ReplayBuffer()

        self.algo = PPO(model=self.model, args=args)
        self.buffer = ReplayBuffer()

        self.writer = SummaryWriter(
            log_dir=args.save_path + "\\logs\\" + args.starttime,
            comment=args.starttime,
        )

        self.update_count = 0
        self.learning_step = args.learning_step
        self.game_num = 0

    def send_traj(self, trajectories, episode_reward, n_game):
        self.game_num += 1
        self.buffer.store_transition(trajectories)
        self.writer.add_scalar("episode_reward", episode_reward, global_step=n_game)

    def update(self):
        if self.game_num % self.learning_step == 0:
            print("----------train-----------")
            loss, action_loss, entropy_loss, value_loss = self.algo.update(self.buffer)
            self.buffer.clear()
            self.writer.add_scalar("loss", loss, global_step=self.update_count)
            self.writer.add_scalar(
                "action_loss", action_loss, global_step=self.update_count
            )
            self.writer.add_scalar(
                "entropy_loss", entropy_loss, global_step=self.update_count
            )
            self.writer.add_scalar(
                "value_loss", value_loss, global_step=self.update_count
            )

            self.update_count += 1


class PPO:
    def __init__(self, model, args):
        self.model = model

        self.batch_size = args.batch_size

        self.lr = args.lr
        self.beta = args.beta
        self.gamma = args.gamma
        self.lmbda = args.lmbda
        self.k_epoch = args.k_epoch
        self.eps_clip = args.eps_clip

        self.actor_coef = args.actor_coef
        self.critic_coef = args.critic_coef
        self.entropy_coef = args.entropy_coef

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=self.beta,
            eps=1e-5,
        )
        self.criterion = nn.MSELoss()

        self.device = self.model.device

    def get_gaes(self, seperates, rewards, v_preds):

        total_gaes = []
        last_sep = 0
        v_target = []

        for sep in seperates:
            # deltas means TD error, deltas = reward + gamma * Vt+1 - Vt
            sep_r = rewards[last_sep:sep]
            sep_v = v_preds[last_sep:sep]

            sep_v_next = v_preds[last_sep + 1 : sep] + [0]

            deltas = [
                r_t + self.gamma * v_next - v
                for r_t, v_next, v in zip(sep_r, sep_v_next, sep_v)
            ]
            v_target += [
                r_t + self.gamma * v_next for r_t, v_next in zip(sep_r, sep_v_next)
            ]

            gaes = copy.deepcopy(deltas)
            for t in reversed(range(len(gaes) - 1)):
                gaes[t] = deltas[t] + self.gamma * self.lmbda * gaes[t + 1]

            total_gaes += gaes
            last_sep = sep
        return total_gaes, v_target

    def get_state_batch(self, preps, begin, end):
        entity_preps = []
        spatial_preps = []
        action_type_masks = []
        entity_ids = []

        for (
            entity_prep,
            spatial_prep,
            action_type_mask,
            entity_id,
        ) in preps[begin:end]:
            entity_preps.append(entity_prep)
            spatial_preps.append(spatial_prep)
            action_type_masks.append(action_type_mask)
            entity_ids.append(entity_id)

        entity_preps = torch.cat(entity_preps, dim=0)
        spatial_preps = torch.cat(spatial_preps, dim=0)
        action_type_masks = torch.cat(action_type_masks, dim=0)
        masks = action_type_masks
        return entity_preps, spatial_preps, masks, entity_ids

    def get_action_batch(self, actions, begin, end):
        action_types = []
        for action in actions[begin:end]:
            action_types.append(torch.tensor(action))

        action_types = torch.stack(action_types).to(self.device)
        return action_types

    def get_logit_batch(self, logits, begin, end):
        logit_batch = []
        for logit in logits[begin:end]:
            logit_batch.append(logit.to(self.device))

        logit_batch = torch.cat(logit_batch, dim=0)
        return logit_batch

    def get_adv_batch(self, advs):
        # batch adv norm
        advs = torch.tensor(advs).to(self.device)
        # advs = (advs - advs.mean()) / (advs.std() + 1e-9)
        return advs

    def get_vtarget_batch(self, vtarget):
        vtarget = torch.tensor(vtarget).to(self.device)
        return vtarget

    def update(self, replay_buffer):
        print(">>>>>len:", replay_buffer.buffer_size)

        old_state_values = torch.stack(replay_buffer.values).squeeze(-1).squeeze(-1)
        values_list = old_state_values.detach().cpu().numpy().tolist()

        advantages, v_targets = self.get_gaes(
            replay_buffer.seperates,
            replay_buffer.rewards,
            values_list,
        )

        buffer_size = replay_buffer.buffer_size
        batch_num = (buffer_size - 1) // self.batch_size + 1

        old_states = []
        old_actions = []
        old_log_probs = []
        adv_batches = []
        v_targets_batches = []

        for batch_id in range(batch_num):
            begin, end = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            old_states.append(self.get_state_batch(replay_buffer.states, begin, end))
            old_actions.append(self.get_action_batch(replay_buffer.actions, begin, end))
            old_log_probs.append(
                self.get_logit_batch(replay_buffer.log_probs, begin, end)
            )
            adv_batches.append(self.get_adv_batch(advantages[begin:end]))
            v_targets_batches.append(self.get_vtarget_batch(v_targets[begin:end]))

        if len(adv_batches[-1]) == 1:
            batch_num -= 1

        loss = []
        loss_action = []
        loss_entropy = []
        loss_value = []
        for epoch in range(self.k_epoch):
            print("epoch:", epoch)
            # torch.cuda.empty_cache()
            for batch_id in range(batch_num):
                print("batch:", batch_id)
                # torch.autograd.set_detect_anomaly(True)
                log_probs, dist_entropys, state_values = self.model.evaluate(
                    old_states[batch_id],
                    old_actions[batch_id],
                )

                ratios = torch.exp(log_probs - old_log_probs[batch_id])
                surr1 = ratios * adv_batches[batch_id]
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * adv_batches[batch_id]
                )
                action_loss = -torch.min(surr1, surr2).float().mean() * self.actor_coef

                entropy_loss = -torch.mean(dist_entropys) * self.entropy_coef

                target_values = v_targets_batches[batch_id]

                value_loss = (
                    self.criterion(state_values, target_values) * self.critic_coef
                )

                total_loss = action_loss + entropy_loss + value_loss

                print("value loss=", value_loss)
                print("entropy loss=", entropy_loss)
                print("action loss=", action_loss)
                print("total loss=", total_loss)

                # torch.autograd.set_detect_anomaly(True)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                loss.append(total_loss.item())
                loss_action.append(action_loss.item())
                loss_entropy.append(entropy_loss.item())
                loss_value.append(value_loss.item())

                del (
                    value_loss,
                    entropy_loss,
                    action_loss,
                    total_loss,
                    log_probs,
                    dist_entropys,
                    state_values,
                )

        del (
            old_states,
            old_actions,
            old_log_probs,
            v_targets_batches,
            adv_batches,
        )

        loss_mean = sum(loss) / len(loss)
        action_loss_mean = sum(loss_action) / len(loss_action)
        entropy_loss_mean = sum(loss_entropy) / len(loss_entropy)
        value_loss_mean = sum(loss_value) / len(loss_value)

        return loss_mean, action_loss_mean, entropy_loss_mean, value_loss_mean
