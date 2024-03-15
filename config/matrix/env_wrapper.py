from typing import List
import numpy as np
from core.game import Game
from .matgame import MatgameEnv


class MatrixWrapper(Game):

    def __init__(self, env: MatgameEnv):
        """Matrix Game Wrapper

        Parameters
        ----------
        env: MultiAgentEnv
            MultiAgentEnv instance
        discount: float
            discount of env
        """
        self.env = env
        self.n_agents = env.n_agents
        self.obs_size = self.env.observation_space.shape[0]
        self.action_space_size = env.n_actions

    def get_max_episode_steps(self) -> int:
        return self.env.episode_limit

    def step(self, action: List[int]):
        observation, reward, done, info = self.env.step(action)
        observation = np.repeat(observation[None, :, None, None], self.n_agents, axis=0)
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset()
        observation = np.repeat(observation[None, :, None, None], self.n_agents, axis=0)
        return observation

    def close(self):
        self.env.close()
