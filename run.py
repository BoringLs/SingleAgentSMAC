from Learn.Actor import Actor
from Learn.Learner import Learner
import os
import argparse
import torch
from Arch.arch_model import ArchModel
from lib.const import *
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Run:
    def __init__(self, args):

        self.model = ArchModel(args, device=device)
        self.model.to(device)

        self.Actor = Actor(self.model, args)
        self.Learner = Learner(self.model, args)

        self.Actor.set_learner(self.Learner)

    def run(self):
        self.Actor.run()


if __name__ == "__main__":
    starttime = time.strftime("%Y-%m-%d_%H-%M")
    data_save_path = "D:\\SC2\\sc2proj\\data_save"

    parser = argparse.ArgumentParser(description="SC2")
    # Actor
    parser.add_argument("--map_name", default="8m", type=str)
    parser.add_argument("--max_game_num", default=100000, type=int)

    # Learner
    parser.add_argument("--learning_step", default=50, type=int)
    parser.add_argument("--starttime", default=starttime, type=str)
    parser.add_argument("--save_path", default=data_save_path, type=str)

    # PPO
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--beta", default=[0.9, 0.999], type=list)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--lmbda", default=0.99, type=float)
    parser.add_argument("--k_epoch", default=3, type=int)
    parser.add_argument("--eps_clip", default=0.2, type=float)
    parser.add_argument("--actor_coef", default=1, type=float)
    parser.add_argument("--critic_coef", default=0.5, type=float)
    parser.add_argument("--entropy_coef", default=0.01, type=float)

    args = parser.parse_args()
    runner = Run(args)
    runner.run()
