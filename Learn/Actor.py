from ai.myAgent import Agent

from smac.env import StarCraft2Env
# from smac_hard.env import StarCraft2Env
from lib.hyper_parameters import hyper_parameters as HP


class Actor:
    def __init__(self, model, args):
        self.device = model.device

        self.args = args
        self.Agent = Agent(model)
        self.learner = None
        # self.starttime = starttime
        # self.save_path = data_save_path + "/logs/" + starttime + "/"
        self.max_game_num = args.max_game_num
        self.map_name = args.map_name

    def set_learner(self, learner):
        self.learner = learner

    def run(self):
        env = StarCraft2Env(map_name=self.map_name)
        env_info = env.get_env_info()

        n_actions = env_info["n_actions"]
        n_agents = env_info["n_agents"]

        for n_game in range(self.max_game_num):
            trajectories = []

            env.reset()
            terminated = False
            episode_reward = 0

            while not terminated:
                obs = env.get_obs()
                state = env.get_state()

                # gen action mask
                action_mask = []
                for agent_id in range(n_agents):
                    avail_actions = env.get_avail_agent_actions(agent_id)
                    action_mask.append(avail_actions)

                myactions, logits, value, prep = self.Agent.step(
                    obs, state, action_mask
                )

                # gen actions
                actions = [0] * n_agents
                for item in myactions:
                    actions[item["unit_id"]] = item["action_type"].item()

                reward, terminated, _ = env.step(actions)
                episode_reward += reward

                trajectories.append(
                    {
                        "state": prep,
                        "action": actions,
                        "log_prob": logits,
                        "reward": reward,
                        "value": value,
                    }
                )

            # print("Game end. Send traj")
            print("Current episode reward: ", episode_reward)
            self.learner.send_traj(trajectories, episode_reward, n_game)
            self.learner.update()

        print("Finise all games")
