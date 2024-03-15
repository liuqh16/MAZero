import gymnasium as gym
import gymnasium.spaces
import numpy as np


class MatgameEnv(gym.Env):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """

    def __init__(
            self,
            map_name='matgame1',
            seed=None
    ):
        self.n_agents = 2

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0

        self.p_step = 0
        self.rew_gather = []
        self.is_print_once = False

        if map_name == 'matgame1':
            self.n_actions = 3
            self.matrix_table = np.array([
                [8., -12., -12.],
                [-12., 0., 0.],
                [-12., 0., 0.]
            ])
            self.episode_limit = 10
        elif map_name == 'matgame2':
            self.n_actions = 3
            self.matrix_table = np.array([
                [0., 0., -12.],
                [0., 0., -12.],
                [-12., -12., 8.]
            ])
            self.episode_limit = 10
        elif map_name == 'matgame3':
            self.n_actions = 3
            self.matrix_table = np.array([
                [10., -10, -10],
                [-10, 9, 0],
                [-10, 0, 9]
            ])
            self.episode_limit = 10
        elif map_name == 'matgame4':
            self.n_actions = 3
            self.matrix_table = np.array([
                [2, 6, 10],
                [2, 4, 9],
                [1, 2, 3]
            ])
            self.episode_limit = 10
        else:
            raise NotImplementedError

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.n_agents, 1, 1), dtype=np.float64)
        self.action_space = gym.spaces.Discrete(self.n_actions)

    def step(self, actions):
        """Returns reward, terminated, info."""
        self._total_steps += 1
        self._episode_steps += 1
        info = {}

        state = np.array([self._episode_steps for _ in range(self.n_agents)])

        reward = self.matrix_table[actions[0]][actions[1]]

        terminated = False

        if self._episode_steps >= self.episode_limit:
            terminated = True

        if terminated:
            self._episode_count += 1

        return state, reward, terminated, info

    def reset(self):
        """Returns initial observations and states."""
        self._episode_steps = 0
        state = np.array([self._episode_steps for _ in range(self.n_agents)])
        return state
