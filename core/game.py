from abc import ABC, abstractmethod
import copy
from typing import List, Tuple

import ray
import numpy as np

from core.utils import str_to_arr, concat_with_zero_padding


class Game(ABC):
    def __init__(self, env):
        self.env = env
        self.n_agents = env.n_agents                    # type: int
        self.obs_size = env.obs_size                    # type: int
        self.action_space_size = env.action_space_size  # type: int

    def legal_actions(self):
        return [[1] * self.action_space_size] * self.n_agents

    @abstractmethod
    def get_max_episode_steps(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: List[int]) -> Tuple[np.ndarray, float, bool, dict]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> np.ndarray:
        raise NotImplementedError

    def close(self, *args, **kwargs):
        self.env.close(*args, **kwargs)

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)


class GameHistory:
    """
    Store only usefull information of a full trajectory.
    """
    def __init__(self, config=None, ray_store_obs=True):
        self.ray_store_obs = ray_store_obs

        self.stacked_observations = config.stacked_observations
        self.action_space_size = config.action_space_size
        self.max_samples = config.sampled_action_times

        self.discount = config.discount

        self.cvt_string = config.cvt_string
        self.gray_scale = config.gray_scale

    def init(self, init_observations):
        """Initialize a history, stack the previous stacked_observations frames.

        Parameters
        ----------
        init_observations: list
            list of the stack observations in the previous time steps
        """
        # trajectory
        self.obs_history = []
        self.actions = []
        self.rewards = []
        self.legal_actions = []
        self.model_indices = []
        # MCTS
        self.root_values = []
        self.pred_values = []
        self.sampled_actions = []
        self.sampled_policies = []
        self.sampled_qvalues = []
        self.sampled_padding_masks = []

        assert len(init_observations) == self.stacked_observations

        for observation in init_observations:
            self.obs_history.append(copy.deepcopy(observation))

    def store_transition(self, action: np.ndarray, reward: float, next_obs: np.ndarray, legal_actions: np.ndarray = None, model_index: int = None):
        """store a transition tuple (a_t, r_t, o_{t+1})

        Parameters
        ----------
        action : np.ndarray
            Joint action a_t = (a_t^1, ..., a_t^N)
        reward : float
        next_obs : np.ndarray
            Next observations
        """
        self.actions.append(action)
        self.rewards.append(reward)
        self.obs_history.append(next_obs)
        if self.legal_actions is not None:
            self.legal_actions.append(legal_actions)
        if model_index is not None:
            self.model_indices.append(model_index)

    def store_search_stats(self, root_value: float, pred_value: float, sampled_actions: np.ndarray, sampled_policy: np.ndarray,
                           sampled_qvalues: np.ndarray):
        """store the search policy and value of the root node after MCTS

        Parameters
        ----------
        root_value : float
            search value generated via MCTS
        pred_value : float
            predicted value output from network when sampled
        """
        self.root_values.append(root_value)
        self.pred_values.append(pred_value)
        self.sampled_actions.append(sampled_actions)
        self.sampled_policies.append(sampled_policy)
        self.sampled_qvalues.append(sampled_qvalues)

    def obs(self, i, extra_len=0, padding=False):
        """To obtain an observation of correct format: o[t: t + stack frames + extra len]

        Parameters
        ----------
        i: int
            time step i
        extra_len: int
            extra len of the obs frames
        padding: bool
            True -> padding frames if (t + stack frames) are out of trajectory
        """
        if self.ray_store_obs:
            frames = ray.get(self.obs_history)[i:i + self.stacked_observations + extra_len]
        else:
            frames = self.obs_history[i:i + self.stacked_observations + extra_len]
        if padding:
            pad_len = self.stacked_observations + extra_len - len(frames)
            if pad_len > 0:
                pad_frames = [frames[-1, ...] for _ in range(pad_len)]
                frames = np.concatenate((frames, pad_frames))
        if self.cvt_string:
            frames = [str_to_arr(obs, self.gray_scale) for obs in frames]
        return frames

    def step_obs(self):
        """Return a stacked observation of correct format for model inference
        """
        index = len(self.rewards)
        frames = self.obs_history[index:index + self.stacked_observations]
        if self.cvt_string:
            frames = [str_to_arr(obs, self.gray_scale) for obs in frames]
        return frames

    def game_over(self):
        """post processing the data when a episode is terminated
        """
        assert len(self.obs_history) - self.stacked_observations == len(self.actions) == len(self.root_values)
        # obs_history should be sent into the ray memory. Otherwise, it will cost large amounts of time in copying obs.
        if self.ray_store_obs:
            self.obs_history = ray.put(np.array(self.obs_history))
        else:
            self.obs_history = np.array(self.obs_history)
        self.actions = np.array(self.actions)
        self.legal_actions = np.array(self.legal_actions)
        self.rewards = np.array(self.rewards)
        self.root_values = np.array(self.root_values)
        self.pred_values = np.array(self.pred_values)
        # padding sampled actions
        self.sampled_actions, self.sampled_padding_masks = concat_with_zero_padding(self.sampled_actions, self.max_samples)
        self.sampled_policies, _ = concat_with_zero_padding(self.sampled_policies, self.max_samples)
        self.sampled_qvalues, _ = concat_with_zero_padding(self.sampled_qvalues, self.max_samples)

    def __len__(self):
        return len(self.actions)
