from abc import ABC
import time
from typing import List, Tuple

import ray
from ray.actor import ActorHandle
import numpy as np
import torch
from gymnasium.utils import seeding
from torch.cuda.amp import autocast as autocast
from core.config import BaseConfig
from core.game import GameHistory
from core.utils import prepare_observation_lst


class BaseBuffer(ABC):

    def __init__(self):
        self.buffer = []            # type: List[GameHistory]
        self.priorities = []        # type: List[float]
        self.game_look_up = []      # type: List[Tuple[int, int]]
        self.game_model_index = []
        self.base_idx = 0
        self._eps_collected = 0
        self._total_transitions = 0

    def save_pools(self, pools):
        # save a list of game histories
        for (game, priorities) in pools:
            if len(game) > 0:
                self.save_game(game, priorities)

    def save_game(self, game: GameHistory, priorities=None):
        """Save a game history
        Parameters
        ----------
        game: Any
            a game history
        priorities: list
            the priorities corresponding to the transitions in the game history
        """
        if priorities is None:
            max_prio = self.priorities.max() if self.buffer else 1
            self.priorities = np.concatenate((self.priorities, [max_prio for _ in range(len(game))]))
        else:
            assert len(game) == len(priorities), " priorities should be of same length as the game steps"
            priorities = np.asarray(priorities).copy().reshape(-1)
            self.priorities = np.concatenate((self.priorities, priorities))

        self.buffer.append(game)
        self.game_look_up += [(self.base_idx + len(self.buffer) - 1, step_pos) for step_pos in range(len(game))]
        self.game_model_index.append(np.mean(game.model_indices))
        self._eps_collected += 1
        self._total_transitions += len(game)

    def update_priorities(self, batch_indices: List[int], batch_priorities: List[float]):
        self.priorities[batch_indices] = batch_priorities

    def _remove(self, num_excess_games):
        raise NotImplementedError
        # need to modify function update_priorities() if you want to use this function
        # delete game histories
        excess_games_steps = sum([len(game) for game in self.buffer[:num_excess_games]])
        del self.buffer[:num_excess_games]
        self.priorities = self.priorities[excess_games_steps:]
        del self.game_look_up[:excess_games_steps]
        self.base_idx += num_excess_games
        return excess_games_steps

    def episodes_collected(self):
        # number of collected episodes
        return self._eps_collected

    def transitions_collected(self):
        # number of collected transitions
        return self._total_transitions

    def buffer_size(self):
        # number of buffer size / stored transitions
        return len(self.priorities)

    def get_buffer(self, start_index=0):
        return self.buffer[start_index:]

    def get_priority_status(self):
        return {
            'mean': np.mean(self.priorities),
            'max': np.max(self.priorities),
            'min': np.min(self.priorities),
            'std': np.std(self.priorities),
        }


class ReplayBuffer(BaseBuffer):
    """Reference : DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY
    Algo. 1 and Algo. 2 in Page-3 of (https://arxiv.org/pdf/1803.00933.pdf
    """
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.alpha = config.priority_prob_alpha
        self.batch_size = config.batch_size
        self.np_random, _ = seeding.np_random(config.seed)

    def prepare_batch_context(self, batch_size, beta):
        """Prepare a batch context that contains:

        Parameters
        ----------
        batch_size: int
            batch size
        beta: float
            the parameter in PER for calculating the priority

        Returns
        -------
        tuple
            game_lst:           a list of game histories
            game_pos_lst:       transition index in game (relative index)
            indices_lst:        transition index in replay buffer
            weights_lst:        the weight concering the priority
        """
        assert beta > 0

        total = len(self.priorities)
        probs = self.priorities ** self.alpha

        probs /= probs.sum()
        # sample data
        indices_lst = self.np_random.choice(total, batch_size, p=probs, replace=False)   # type: List[int]

        weights_lst = (total * probs[indices_lst]) ** (-beta)
        weights_lst /= weights_lst.max()

        game_lst = []
        game_pos_lst = []

        for idx in indices_lst:
            game_id, game_pos = self.game_look_up[idx]
            game_id -= self.base_idx
            game = self.buffer[game_id]

            game_lst.append(game)
            game_pos_lst.append(game_pos)

        context = (game_lst, game_pos_lst, indices_lst, weights_lst)
        return context

    def can_sample(self, batch_size):
        return self.buffer_size() >= batch_size

    def prepare_game(self):
        game_id = np.argmin(self.game_model_index)
        self.game_model_index[game_id] = np.inf
        return (game_id, self.buffer[game_id])

    def update_game_history(self, update_context: Tuple[int, GameHistory]):
        game_id, game_history = update_context
        self.buffer[game_id] = game_history
        self.game_model_index[game_id] = np.mean(game_history.model_indices)


@ray.remote
class RemoteReplayBuffer(ReplayBuffer):

    def __init__(self, config: BaseConfig):
        super().__init__(config)


class PriorityRefresher(BaseBuffer):
    """Reference : SpeedyZero: Mastering Atari with Limited Data and Time

    Priority refresher periodically updates the priorities of all data points in the replay buffer
    to address the issue of unstable values, which is due to sample efficiency requirement.
    """
    def __init__(self, config: BaseConfig, replica: BaseBuffer):
        super().__init__()
        self.config = config
        self.replica = replica
        self.image_based = config.image_based
        self.stacked_observations = config.stacked_observations
        self.gamma = config.discount
        self.gae_lambda = config.gae_lambda

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = config.get_uniform_network()
        self.model.to(self.device)
        self.model.eval()
        self.value_obs_tensors = torch.empty((
            0, config.num_agents, config.stacked_observations * config.obs_shape[-1], *config.obs_shape[:2]
        )).to(self.device)
        self.rewards = np.empty((0,), dtype=np.float32)
        self.masks = np.ones((1,), dtype=np.float32)

    def save_game(self, game: GameHistory):
        """For priority refresh, we store the observations in the replay buffer on GPU to
        avoid loading them every time the priorities are re-computed.
        """
        super().save_game(game, priorities=None)

        obs_history = ray.get(game.obs_history) if game.ray_store_obs else game.obs_history
        stacked_obs_history = np.asarray([obs_history[i:i + self.stacked_observations] for i in range(len(game))])
        value_obs_lst = prepare_observation_lst(stacked_obs_history, self.image_based)
        if self.image_based:
            value_obs_tensor = torch.from_numpy(value_obs_lst).to(self.device).float() / 255.0
        else:
            value_obs_tensor = torch.from_numpy(value_obs_lst).to(self.device).float()
        self.value_obs_tensors = torch.cat([self.value_obs_tensors, value_obs_tensor])

        self.rewards = np.concatenate([self.rewards, game.rewards])
        masks = np.ones_like(game.rewards)
        masks[-1] = 0
        self.masks = np.concatenate([self.masks, masks])

    def _synchronize_buffer(self):
        """synchronize buffer from replica"""
        new_games = self.replica.get_buffer(len(self.buffer))
        for game in new_games:
            self.save_game(game)

    def update_priorities(self):
        self._synchronize_buffer()

        # split full buffer into slices of refresh_mini_size: to save the GPU memory
        m_batch = self.config.refresh_mini_size
        slices = np.ceil(self._total_transitions / m_batch).astype(np.intc)
        pred_values = []
        with torch.no_grad():
            for i in range(slices):
                beg_index = m_batch * i
                end_index = m_batch * (i + 1)
                with autocast():
                    m_output = self.model.initial_inference(self.value_obs_tensors[beg_index:end_index])
                pred_values.append(m_output.value.flatten())
        pred_values = np.concatenate(pred_values)

        # Since the goal is to stabilize the values, we:
        # (1) use TD errors as the priorities
        # new_priorities = np.zeros_like(pred_values)
        # new_priorities[:-1] = np.abs(pred_values[:-1] - (self.rewards[:-1] + pred_values[1:]))
        # new_priorities[-1] = np.abs(pred_values[-1] - self.rewards[-1])

        # (2) use episode return errors as the priorities
        returns = np.zeros((self._total_transitions + 1))
        for step in reversed(range(self._total_transitions)):
            returns[step] = returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]
        new_priorities = np.abs(returns[:-1] - pred_values)

        # (3) use GAE as the priorities
        # returns = np.zeros_like(pred_values)
        # pred_values = np.concatenate([pred_values, [0.]])
        # gae = 0
        # for step in reversed(range(self._total_transitions)):
        #     delta = self.rewards[step] + self.gamma * pred_values[step + 1] * self.masks[step + 1] - pred_values[step]
        #     gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
        #     returns[step] = gae + pred_values[step]
        # new_priorities = np.abs(returns - pred_values[:-1])

        assert self._total_transitions == new_priorities.shape[0]
        return np.arange(self._total_transitions), new_priorities


@ray.remote
class RemotePriorityRefresher(PriorityRefresher):

    def __init__(self, config: BaseConfig, replica: ActorHandle, shared_storage: ActorHandle):
        assert isinstance(replica, ActorHandle), 'Must input RemoteReplayBuffer as replica!'
        super().__init__(config, replica)
        self.shared_storage = shared_storage
        self.last_model_index = -1

    def _synchronize_buffer(self):
        """synchronize buffer from replica"""
        new_games = ray.get(self.replica.get_buffer.remote(len(self.buffer)))
        for game in new_games:
            self.save_game(game)

    def run_loop(self):
        start = False
        while True:
            # waiting for start signal
            if not start:
                start = ray.get(self.shared_storage.get_start_signal.remote())
                time.sleep(1)
                continue
            # break
            trained_steps = ray.get(self.shared_storage.get_counter.remote())
            if trained_steps >= self.config.training_steps + self.config.last_steps:
                time.sleep(10)
                break

            # periodically update priorities of all data points in the replay buffer
            if self.last_model_index // self.config.refresh_interval < trained_steps // self.config.refresh_interval:
                # obtain the latest model weights
                model_index, weights = ray.get(self.shared_storage.get_weights.remote())
                self.model.set_weights(weights)
                self.model.to(self.device)
                self.model.eval()
                self.last_model_index = model_index

                indices, new_priorities = self.update_priorities()
                self.replica.update_priorities.remote(indices, new_priorities)
            else:
                time.sleep(1)
