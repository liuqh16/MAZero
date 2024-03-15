import logging
import math
import os
import time
from typing import Tuple, List, Union

import numpy as np
import ray
from ray.actor import ActorHandle
import torch
from torch.cuda.amp import autocast as autocast
from gymnasium.utils import seeding

from core.mcts import SampledMCTS
from core.config import BaseConfig
from core.replay_buffer import ReplayBuffer
from core.storage import SharedStorage
from core.game import GameHistory
from core.utils import select_action, prepare_observation_lst, get_max_entropy, eps_greedy_action


class DataWorker(object):
    def __init__(self, rank, config: BaseConfig, replay_buffer: ReplayBuffer, shared_storage: SharedStorage):
        """Data Worker for collecting data through self-play

        Parameters
        ----------
        rank: int
            id of the worker
        replay_buffer: Any
            Replay buffer to save self-play data
        shared_storage: Any
            The share storage to control & get latest model
        """
        self.rank = rank
        self.config = config
        self.replay_buffer = replay_buffer
        self.shared_storage = shared_storage
        self.np_random, _ = seeding.np_random(config.seed * 1000 + self.rank)

        self.device = 'cuda' if (config.selfplay_on_gpu and torch.cuda.is_available()) else 'cpu'
        self.gap_step = self.config.num_unroll_steps + self.config.td_steps
        self.max_visit_entropy = get_max_entropy(self.config.action_space_size)

        # create env & logs
        self.init_envs()

        # create model from game_config
        self.model = self.config.get_uniform_network()
        self.model.to(self.device)
        self.model.eval()
        self.last_model_index = -1

        self.trajectory_pool = []
        self.pool_size = 1  # max size for buffering pool

    def init_envs(self):
        num_envs = self.config.num_pmcts

        self.ray_store_obs = False

        self.envs = [
            self.config.new_game(self.config.seed + (self.rank + 1) * i)
            for i in range(num_envs)
        ]

        self.eps_steps_lst = np.zeros(num_envs)
        self.eps_reward_lst = np.zeros(num_envs)
        self.visit_entropies_lst = np.zeros(num_envs)
        self.model_index_lst = np.zeros(num_envs)
        if self.config.case in ['smac', 'gfootball']:
            self.battle_won_lst = np.zeros(num_envs)

        init_obses = [env.reset() for env in self.envs]
        self.game_histories = [None for _ in range(num_envs)]       # type: list[GameHistory]

        # stack observation windows in boundary
        self.stack_obs_windows = [[] for _ in range(num_envs)]
        # initial stack observation: [s0, s0, s0, s0]
        for i in range(num_envs):
            self.stack_obs_windows[i] = [init_obses[i] for _ in range(self.config.stacked_observations)]
            self.game_histories[i] = GameHistory(config=self.config, ray_store_obs=self.ray_store_obs)
            self.game_histories[i].init(self.stack_obs_windows[i])

        # for priorities in self-play
        self.pred_values_lst = [[] for _ in range(num_envs)]    # pred value
        self.search_values_lst = [[] for _ in range(num_envs)]  # target value

        self.dones = np.zeros(num_envs, dtype=np.bool_)

    def put(self, data: Tuple[GameHistory, List[float]]):
        # put a game history into the pool
        self.trajectory_pool.append(data)

    def _free(self):
        # save the game histories and clear the pool
        if len(self.trajectory_pool) >= self.pool_size:
            self.replay_buffer.save_pools(self.trajectory_pool)
            del self.trajectory_pool[:]

    def _log_to_buffer(self, log_dict: dict):
        self.shared_storage.add_worker_logs(log_dict)

    def _update_model_before_step(self):
        # no update when serial
        return

    def update_model(self, model_index, weights):
        self.model.set_weights(weights)
        self.last_model_index = model_index

    def log(self, env_id, **kwargs):
        # send logs
        log_dict = {
            'eps_len': self.eps_steps_lst[env_id],
            'eps_reward': self.eps_reward_lst[env_id],
            'visit_entropy': self.visit_entropies_lst[env_id] / max(self.eps_steps_lst[env_id], 1),
            'model_index': self.model_index_lst[env_id] / max(self.eps_steps_lst[env_id], 1),
        }
        for k, v in kwargs.items():
            log_dict[k] = v
        if self.config.case in ['smac', 'gfootball']:
            log_dict['win_rate'] = self.battle_won_lst[env_id]

        self._log_to_buffer(log_dict)

    def reset_env(self, env_id):
        self.eps_steps_lst[env_id] = 0
        self.eps_reward_lst[env_id] = 0
        self.visit_entropies_lst[env_id] = 0
        self.model_index_lst[env_id] = 0
        if self.config.case in ['smac', 'gfootball']:
            self.battle_won_lst[env_id] = 0
        # new trajectory
        init_obs = self.envs[env_id].reset()
        self.stack_obs_windows[env_id] = [init_obs for _ in range(self.config.stacked_observations)]
        self.game_histories[env_id] = GameHistory(config=self.config, ray_store_obs=self.ray_store_obs)
        self.game_histories[env_id].init(self.stack_obs_windows[env_id])
        self.pred_values_lst[env_id] = []
        self.search_values_lst[env_id] = []
        self.dones[env_id] = False

    def get_priorities(self, pred_values: List[float], search_values: List[float]) -> Union[List[float], None]:
        # obtain the priorities
        if self.config.use_priority and not self.config.use_max_priority:
            priorities = np.abs(np.asarray(pred_values) - np.asarray(search_values)) + self.config.prioritized_replay_eps
        else:
            # priorities is None -> use the max priority for all newly collected data
            priorities = None

        return priorities

    def run(self, start_training: bool = False, trained_steps: int = 0) -> int:
        num_envs = self.config.num_pmcts
        episodes_collected = 0
        transitions_collected = 0

        with torch.no_grad():
            # play games until max episodes
            while episodes_collected < num_envs:

                # set temperature for distributions
                temperature = self.config.visit_softmax_temperature_fn(trained_steps)
                sampled_tau = self.config.sampled_action_temperature_fn(trained_steps)
                greedy_epsilon = self.config.eps_greedy_fn(trained_steps)

                # update model
                self._update_model_before_step()

                # stack obs for model inference
                stack_obs = [game_history.step_obs() for game_history in self.game_histories]
                stack_obs = prepare_observation_lst(stack_obs, self.config.image_based)
                if self.config.image_based:
                    stack_obs = torch.from_numpy(stack_obs).to(self.device).float() / 255.0
                else:
                    stack_obs = torch.from_numpy(stack_obs).to(self.device).float()

                with autocast():
                    network_output = self.model.initial_inference(stack_obs)
                legal_actions_lst = np.asarray([env.legal_actions() for env in self.envs])

                search_results = SampledMCTS(self.config, self.np_random).batch_search(
                    self.model, network_output, legal_actions_lst, self.device, add_noise=True, sampled_tau=sampled_tau)
                roots_values = search_results.value
                roots_sampled_visit_counts = search_results.sampled_visit_count
                roots_sampled_actions = search_results.sampled_actions
                roots_sampled_qvalues = search_results.sampled_qvalues

                for i in range(num_envs):
                    root_value = roots_values[i]
                    pred_value = network_output.value[i].item()
                    sampled_actions = roots_sampled_actions[i]
                    # use MCTS policy after starting training, otherwise use random policy before starting training
                    sampled_visit_counts = roots_sampled_visit_counts[i] if start_training else np.ones_like(roots_sampled_visit_counts[i])
                    sampled_policy = sampled_visit_counts / np.sum(sampled_visit_counts)
                    sampled_qvalues = roots_sampled_qvalues[i]

                    # sample action from policy under sampled actions respectively
                    action_pos, visit_entropy = select_action(
                        sampled_visit_counts,
                        temperature=temperature,
                        deterministic=False,
                        np_random=self.np_random
                    )
                    action = sampled_actions[action_pos]
                    action = eps_greedy_action(action, legal_actions_lst[i], greedy_epsilon)

                    next_obs, reward, done, info = self.envs[i].step(action)
                    self.dones[i] = done

                    # store data
                    self.game_histories[i].store_transition(action, reward, next_obs, legal_actions_lst[i], self.last_model_index)
                    self.game_histories[i].store_search_stats(root_value, pred_value, sampled_actions, sampled_policy, sampled_qvalues)
                    if self.config.use_priority:
                        self.pred_values_lst[i].append(pred_value)
                        self.search_values_lst[i].append(root_value)

                    # update logs
                    self.eps_steps_lst[i] += 1
                    self.eps_reward_lst[i] += reward
                    self.visit_entropies_lst[i] += visit_entropy
                    self.model_index_lst[i] += self.last_model_index
                    if self.config.case in ['smac', 'gfootball']:
                        self.battle_won_lst[i] = info['battle_won']

                    # fresh stack windows
                    del self.stack_obs_windows[i][0]
                    self.stack_obs_windows[i].append(next_obs)

                    # if is the end of the game:
                    if self.dones[i]:
                        # calculate priority
                        priorities = self.get_priorities(self.pred_values_lst[i], self.search_values_lst[i])

                        # store current trajectory
                        self.game_histories[i].game_over()
                        self.put((self.game_histories[i], priorities))
                        self._free()
                        # reset the finished env and new a env
                        episodes_collected += 1
                        transitions_collected += len(self.game_histories[i])
                        self.log(i, temperature=temperature)
                        self.log(i, greedy_epsilon=greedy_epsilon)
                        self.reset_env(i)
                    elif len(self.game_histories[i]) > self.config.max_moves:
                        # discard this trajectory
                        self.reset_env(i)

        return transitions_collected

    def close(self):
        self.replay_buffer = None
        self.shared_storage = None
        for env in self.envs:
            env.close()


@ray.remote
class RemoteDataWorker(DataWorker):

    def __init__(self, rank, config, replay_buffer: ActorHandle, shared_storage: ActorHandle):
        """Remote Data Worker for collecting data through self-play
        """
        assert isinstance(replay_buffer, ActorHandle), 'Must input RemoteReplayBuffer for RemoteDataWorker!'
        super().__init__(rank, config, replay_buffer, shared_storage)
        self.ray_store_obs = True  # put obs into ray memory

        formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(filename)s>%(funcName)s] ==> %(message)s')
        file_path = os.path.join(config.exp_path, 'logs', 'root.log')
        self.logger = logging.getLogger('root')
        handler = logging.FileHandler(file_path, mode='a')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _free(self):
        # save the game histories and clear the pool
        if len(self.trajectory_pool) >= self.pool_size:
            self.replay_buffer.save_pools.remote(self.trajectory_pool)
            del self.trajectory_pool[:]

    def _log_to_buffer(self, log_dict: dict):
        self.shared_storage.add_worker_logs.remote(log_dict)

    def _update_model_before_step(self):
        trained_steps = ray.get(self.shared_storage.get_counter.remote())
        if self.last_model_index // self.config.checkpoint_interval < trained_steps // self.config.checkpoint_interval:
            model_index, weights = ray.get(self.shared_storage.get_weights.remote())
            self.update_model(model_index, weights)

    def run_loop(self):

        start_training = False
        transitions_collected = 0

        # max transition to collect for this data worker
        max_transitions = self.config.total_transitions // self.config.data_actors

        while True:
            # ------------------ update training status ------------------
            trained_steps = ray.get(self.shared_storage.get_counter.remote())
            # (1) stop data-collecting when training finished
            if trained_steps >= self.config.training_steps + self.config.last_steps:
                time.sleep(10)
                break
            if not start_training:
                start_training = ray.get(self.shared_storage.get_start_signal.remote())
            # (2) balance training & selfplay
            if start_training and (transitions_collected / max_transitions) > (trained_steps / self.config.training_steps):
                # self-play is faster than training speed or finished
                target_trained_steps = math.ceil(transitions_collected / max_transitions * self.config.training_steps)
                self.logger.debug("(DataWorker{}) #{:<7} Wait for model updating...{}/{}".format(
                    self.rank, transitions_collected, trained_steps, target_trained_steps
                ))
                time.sleep(10)
                continue
            # -------------------------------------------------------------

            transitions_collected += self.run(start_training, trained_steps)

        self.close()
