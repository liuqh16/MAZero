import gymnasium as gym
from typing import List
import numpy as np
from core.game import Game
from .mappo_smac.StarCraft2_Env import StarCraft2Env


class SMACWrapper(Game):

    def __init__(self, env: StarCraft2Env, save_video=False):
        """SMAC Wrapper

        Parameters
        ----------
        env: StarCraft2Env
            StarCraft2Env instance
        discount: float
            discount of env
        """
        self.env = env
        self.n_agents = env.n_agents
        self.obs_size = self.env.get_obs_size()[0]
        self.action_space_size = env.n_actions
        '''
        n_action defines all potential actions for a single agent, includes:
            0: no operation (valid only when dead)
            1: stop
            2: move north
            3: move south
            4: move east
            5: move west
            6~: specific enemy_id to attack
        So n_action = 6 + n_enemies
        '''
        self.save_video = save_video
        setattr(self.env, "observation_space",
                gym.spaces.Box(-np.inf, np.inf, shape=(self.obs_size, 1, 1), dtype=np.float64))
        setattr(self.env, "action_space",
                gym.spaces.Discrete(self.action_space_size))

    def legal_actions(self) -> List[List[int]]:
        return self.env.get_avail_actions()

    def get_max_episode_steps(self) -> int:
        return self.env.episode_limit

    def step(self, action: List[int]):
        local_obs, global_state, rewards, dones, infos, available_actions = self.env.step(action)
        observation = np.asarray(local_obs)[:, :, None, None]
        reward = np.mean(rewards)
        done = np.all(dones)
        info = {
            "battle_won": infos[0]["won"]
        }
        return observation, reward, done, info

    def reset(self, **kwargs):
        local_obs, global_state, available_actions = self.env.reset()
        observation = np.asarray(local_obs)[:, :, None, None]
        return observation

    def close(self):
        if self.save_video:
            self.env.save_replay()
        self.env.close()
        return
