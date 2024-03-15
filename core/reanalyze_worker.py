import math
import time
from typing import List, Tuple

import numpy as np
import ray
import torch
from torch.cuda.amp import autocast as autocast
from gymnasium.utils import seeding

from core.config import BaseConfig
from core.mcts import SampledMCTS
from core.game import GameHistory
from core.utils import prepare_observation_lst, concat_with_zero_padding, LinearSchedule


class ReanalyzeWorker(object):

    def __init__(self, rank: int, config: BaseConfig):
        """ReanalyzeWorker for reanalyzing targets
        receive the context from replay buffer and prepare training batches

        Parameters
        ----------
        rank: int
            id of the worker
        """
        self.rank = rank
        self.config = config
        self.np_random, _ = seeding.np_random(config.seed * 2000 + self.rank)
        zero_obs_shape = (config.stacked_observations, config.num_agents, *config.obs_shape)
        if self.config.image_based:
            self.zero_obs = np.zeros(zero_obs_shape, dtype=np.uint8)
        else:
            self.zero_obs = np.zeros(zero_obs_shape, dtype=np.float32)

        self.beta_schedule = LinearSchedule(config.training_steps + config.last_steps,
                                            initial_p=config.priority_prob_beta, final_p=1.0)

        self.device = 'cuda' if (config.reanalyze_on_gpu and torch.cuda.is_available()) else 'cpu'

        self.model = config.get_uniform_network()
        self.model.to(self.device)
        self.model.eval()
        self.last_model_index = -1

    def update_model(self, model_index, weights):
        self.model.set_weights(weights)
        self.last_model_index = model_index

    def _prepare_reward_value_re(
        self,
        indices: List[int],
        games: List[GameHistory],
        game_pos_lst: List[int],
        transitions_collected: int
    ):
        """prepare reward and value targets with reanalyzing

        Parameters
        ----------
        indices: list
            transition index in replay buffer
        games: list
            list of game histories
        game_pos_lst: list
            transition index in game
        transitions_collected: int
            number of collected transitions
        """
        # (1) flatten the unroll bootstrap obs(B, K, ...) into obs_lst(B * K, ...)
        value_obs_lst, rewards_lst, traj_lens, legal_actions_lst = [], [], [], []
        value_mask = []     # the value is valid or not (out of trajectory)
        td_steps_lst = []   # off-policy correction

        for idx, game, state_index in zip(indices, games, game_pos_lst):
            traj_len = len(game)
            traj_lens.append(traj_len)
            rewards_lst.append(game.rewards)

            # off-policy correction: shorter horizon of td steps
            delta_td = (transitions_collected - idx) // self.config.auto_td_steps
            td_steps = self.config.td_steps - delta_td
            td_steps = np.clip(td_steps, 1, self.config.td_steps).astype(np.intc)

            # prepare the corresponding observations for bootstrapped values
            game_obs = game.obs(state_index + td_steps, self.config.num_unroll_steps)

            # obs[T:T+S] ~ obs[T+K:T+K+S]: T - td_steps, K - num_unroll_steps, S - stacked_observations
            for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                td_steps_lst.append(td_steps)
                bootstrap_index = current_index + td_steps
                if bootstrap_index < traj_len:
                    value_mask.append(1)
                    legal_actions_lst.append(game.legal_actions[bootstrap_index])
                    beg_index = bootstrap_index - (state_index + td_steps)
                    end_index = beg_index + self.config.stacked_observations
                    obs = game_obs[beg_index:end_index]
                else:
                    value_mask.append(0)
                    legal_actions_lst.append(game.legal_actions[0])
                    obs = self.zero_obs
                value_obs_lst.append(obs)

        # (2) generate target reward & value from reanalyzing
        batch_values, batch_rewards = [], []

        value_obs_lst = prepare_observation_lst(value_obs_lst, self.config.image_based)
        if self.config.image_based:
            value_obs_tensor = torch.from_numpy(value_obs_lst).to(self.device).float() / 255.0
        else:
            value_obs_tensor = torch.from_numpy(value_obs_lst).to(self.device).float()
        with autocast():
            network_output = self.model.initial_inference(value_obs_tensor)

        # use the root values from MCTS
        if self.config.use_root_value:
            # concat the output slices after model inference
            legal_actions_lst = np.asarray(legal_actions_lst)

            search_results = SampledMCTS(self.config, self.np_random).batch_search(self.model, network_output, legal_actions_lst, self.device, True, 1.0)
            value_lst = search_results.value.flatten()
        # use the predicted values
        elif self.config.use_pred_value:
            value_lst = network_output.value.flatten()
        else:
            raise NotImplementedError

        # get last state discounted value
        value_lst = value_lst * np.power(self.config.discount, td_steps_lst)
        value_lst = value_lst * value_mask

        value_index = 0
        for traj_len, reward_lst, state_index in zip(traj_lens, rewards_lst, game_pos_lst):
            target_values = []
            target_rewards = []

            for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                bootstrap_index = current_index + td_steps_lst[value_index]
                for i, reward in enumerate(reward_lst[current_index:bootstrap_index]):
                    value_lst[value_index] += reward * self.config.discount ** i

                if current_index < traj_len:
                    target_values.append(value_lst[value_index])
                    target_rewards.append(reward_lst[current_index])
                else:
                    target_values.append(0.0)
                    target_rewards.append(0.0)
                value_index += 1

            batch_rewards.append(target_rewards)
            batch_values.append(target_values)

        batch_rewards = np.asarray(batch_rewards).reshape(self.config.batch_size, self.config.num_unroll_steps + 1)
        batch_values = np.asarray(batch_values).reshape(self.config.batch_size, self.config.num_unroll_steps + 1)
        return batch_rewards, batch_values

    def _prepare_reward_value_non_re(
        self,
        games: List[GameHistory],
        game_pos_lst: List[int],
    ):
        """prepare reward and value without reanalyzing, just return the value in self-play

        Parameters
        ----------
        games: list
            list of game histories
        game_pos_lst: list
            transition index in game
        """
        batch_values, batch_rewards = [], []

        for game, state_index in zip(games, game_pos_lst):
            target_values, target_rewards = [], []
            traj_len = len(game)
            reward_lst = game.rewards
            for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                if current_index < traj_len:
                    target_rewards.append(reward_lst[current_index])
                    bootstrap_index = current_index + self.config.td_steps
                    if bootstrap_index < traj_len:
                        if self.config.use_root_value:
                            bootstrap_value = game.root_values[bootstrap_index]
                        elif self.config.use_pred_value:
                            bootstrap_value = game.pred_values[bootstrap_index]
                        else:
                            raise NotImplementedError
                    else:
                        bootstrap_value = 0
                    for i, reward in enumerate(reward_lst[current_index:bootstrap_index]):
                        bootstrap_value += reward * self.config.discount ** i
                    target_values.append(bootstrap_value)
                else:
                    target_rewards.append(0.)
                    target_values.append(0.)
            batch_rewards.append(target_rewards)
            batch_values.append(target_values)

        batch_rewards = np.asarray(batch_rewards).reshape(self.config.batch_size, self.config.num_unroll_steps + 1)
        batch_values = np.asarray(batch_values).reshape(self.config.batch_size, self.config.num_unroll_steps + 1)
        return batch_rewards, batch_values

    def _prepare_policy_re(
        self,
        games: List[GameHistory],
        game_pos_lst: List[int]
    ):
        """prepare policy targets with reanalyzing

        Parameters
        ----------
        games: list
            list of game histories
        game_pos_lst: list
            transition index in game
        """
        # (1) flatten the unroll obs(B, K+1, ...) into obs_lst(B * (K+1), ...)
        policy_obs_lst, legal_actions_lst = [], []
        policy_mask = []    # the policy is valid or not (out of trajectory)
        # NOTE: policy mask is equal to np.concatenate(mask_lst) in `self.make_batch`

        B, K, N, A, C = (
            len(games),
            self.config.num_unroll_steps,
            self.config.num_agents,
            self.config.action_space_size,
            self.config.sampled_action_times,
        )

        for game, state_index in zip(games, game_pos_lst):
            traj_len = len(game)

            # prepare the corresponding observations for predict policy
            game_obs = game.obs(state_index, K)

            for current_index in range(state_index, state_index + K + 1):
                if current_index < traj_len:
                    policy_mask.append(True)
                    legal_actions_lst.append(game.legal_actions[current_index])
                    beg_index = current_index - state_index
                    end_index = beg_index + self.config.stacked_observations
                    obs = game_obs[beg_index:end_index]
                else:
                    policy_mask.append(False)
                    legal_actions_lst.append(game.legal_actions[0])
                    obs = self.zero_obs
                policy_obs_lst.append(obs)

        # (2) generate target policy from reanalyzing
        policy_obs_lst = prepare_observation_lst(policy_obs_lst, self.config.image_based)
        if self.config.image_based:
            policy_obs_tensor = torch.from_numpy(policy_obs_lst).to(self.device).float() / 255.0
        else:
            policy_obs_tensor = torch.from_numpy(policy_obs_lst).to(self.device).float()
        with autocast():
            network_output = self.model.initial_inference(policy_obs_tensor)

        legal_actions_lst = np.asarray(legal_actions_lst).reshape(B * (K + 1), N, A)

        search_results = SampledMCTS(self.config, self.np_random).batch_search(
            self.model, network_output, legal_actions_lst, self.device, add_noise=True, sampled_tau=1.0)

        batch_sampled_actions_re, batch_sampled_masks_re = concat_with_zero_padding(search_results.sampled_actions, C)
        batch_sampled_masks_re[~np.asarray(policy_mask)] = False

        batch_sampled_visit_counts_re, _ = concat_with_zero_padding(search_results.sampled_visit_count, C)
        batch_sampled_policies_re = batch_sampled_visit_counts_re / self.config.num_simulations
        batch_sampled_policies_re[~np.asarray(policy_mask)] = 0.
        assert batch_sampled_policies_re[~batch_sampled_masks_re].sum() == 0

        batch_sampled_imp_ratio, _ = concat_with_zero_padding(search_results.sampled_imp_ratio, C)
        batch_sampled_imp_ratio[~batch_sampled_masks_re] = 0.

        batch_sampled_qvalues_re, _ = concat_with_zero_padding(search_results.sampled_qvalues, C)
        batch_root_mcts_values_re = np.expand_dims(search_results.value, axis=-1)
        batch_root_pred_values_re = network_output.value

        return (
            batch_sampled_actions_re,
            batch_sampled_policies_re,
            batch_sampled_imp_ratio,
            batch_sampled_masks_re,
            batch_sampled_qvalues_re,
            batch_root_mcts_values_re,
            batch_root_pred_values_re
        )

    def _prepare_policy_non_re(
        self,
        games: List[GameHistory],
        game_pos_lst: List[int]
    ):
        raise NotImplementedError

    def make_batch(
        self,
        buffer_context: Tuple[List[GameHistory], List[int], List[int], List[float], List[float]],
        transitions_collected: int
    ):
        """prepare the context of a batch

        Parameters
        ----------
        buffer_context : Any
            batch context from replay buffer
        transitions_collected: int
            number of collected transitions

        """
        # (1) obtain the batch context from replay buffer
        game_lst, game_pos_lst, indices_lst, weights_lst = buffer_context

        # (2) prepare the inputs of a batch
        obs_lst, action_lst, mask_lst = [], [], []
        future_return_lst, model_index_lst = [], []
        for game, state_index in zip(game_lst, game_pos_lst):
            _obs = game.obs(state_index, self.config.num_unroll_steps, padding=True)
            _actions = game.actions[state_index:state_index + self.config.num_unroll_steps + 1].tolist()
            _mask = [1] * len(_actions)
            # padding zero-actions and add mask for invalid actions (out of trajectory)
            _actions += [[0] * self.config.num_agents for _ in range(self.config.num_unroll_steps + 1 - len(_actions))]
            _mask += [0] * (self.config.num_unroll_steps + 1 - len(_mask))
            # obtain the input observations
            obs_lst.append(_obs)
            action_lst.append(_actions)
            mask_lst.append(_mask)
            future_return_lst.append(np.sum(game.rewards[state_index:]))
            model_index_lst.append(game.model_indices[state_index])
        obs_lst = prepare_observation_lst(obs_lst, self.config.image_based)
        # inputs_shape: (B, N, (S+K)xC, W, H) | (B, K+1, N) | (B, K+1) | (B,) | (B,)
        inputs_batch = [obs_lst, action_lst, mask_lst, indices_lst, weights_lst]
        for i in range(len(inputs_batch)):
            inputs_batch[i] = np.asarray(inputs_batch[i])

        # (3) obtain the context of value targets
        if self.config.use_reanalyze_value:
            batch_rewards, batch_values = self._prepare_reward_value_re(indices_lst, game_lst, game_pos_lst, transitions_collected)
        else:
            batch_rewards, batch_values = self._prepare_reward_value_non_re(game_lst, game_pos_lst)

        B, K, N, C = (
            len(indices_lst),
            self.config.num_unroll_steps,
            self.config.num_agents,
            self.config.sampled_action_times,
        )
        # 0:re_num -> reanalyzed policy, re_num:end -> non reanalyzed policy
        re_num = math.ceil(B * self.config.revisit_policy_search_rate)

        batch_sampled_actions = np.empty((0, C, N), dtype=np.intc)
        batch_sampled_policies = np.empty((0, C))
        batch_sampled_imp_ratio = np.empty((0, C))
        batch_sampled_masks = np.empty((0, C), dtype=np.bool_)
        batch_sampled_qvalues = np.empty((0, C))
        batch_root_mcts_values = np.empty((0, 1))
        batch_root_pred_values = np.empty((0, 1))

        # (4) obtain the context of reanalyzed policy targets
        if re_num > 0:
            (
                batch_sampled_actions_re,
                batch_sampled_policies_re,
                batch_sampled_imp_ratio_re,
                batch_sampled_masks_re,
                batch_sampled_qvalues_re,
                batch_root_mcts_values_re,
                batch_root_pred_values_re
            ) = self._prepare_policy_re(game_lst[:re_num], game_pos_lst[:re_num])
            batch_sampled_actions = np.concatenate([batch_sampled_actions, batch_sampled_actions_re])
            batch_sampled_policies = np.concatenate([batch_sampled_policies, batch_sampled_policies_re])
            batch_sampled_imp_ratio = np.concatenate([batch_sampled_imp_ratio, batch_sampled_imp_ratio_re])
            batch_sampled_masks = np.concatenate([batch_sampled_masks, batch_sampled_masks_re])
            batch_sampled_qvalues = np.concatenate([batch_sampled_qvalues, batch_sampled_qvalues_re])
            batch_root_mcts_values = np.concatenate([batch_root_mcts_values, batch_root_mcts_values_re])
            batch_root_pred_values = np.concatenate([batch_root_pred_values, batch_root_pred_values_re])

        # (5) obtain the context of non-reanalyzed policy targets
        if re_num < B:
            raise NotImplementedError

        # (6) compute and normalize advantage
        batch_sampled_adv = batch_sampled_qvalues - batch_root_pred_values
        adv_copy = np.array(batch_sampled_adv)
        adv_copy[~batch_sampled_masks] = np.nan
        adv_mean = np.nanmean(adv_copy)
        adv_std = np.nanstd(adv_copy)
        batch_sampled_adv = (batch_sampled_adv - adv_mean) / (adv_std + 1e-5)
        batch_sampled_adv[~batch_sampled_masks] = 0.

        # (7) reshape policy data
        batch_sampled_actions = batch_sampled_actions.reshape(B, K + 1, C, N)
        batch_sampled_policies = batch_sampled_policies.reshape(B, K + 1, C)
        batch_sampled_imp_ratio = batch_sampled_imp_ratio.reshape(B, K + 1, C)
        batch_sampled_adv = batch_sampled_adv.reshape(B, K + 1, C)
        batch_sampled_masks = batch_sampled_masks.reshape(B, K + 1, C)

        batch_policies = (batch_sampled_actions, batch_sampled_policies,
                          batch_sampled_imp_ratio, batch_sampled_adv, batch_sampled_masks)
        targets_batch = (batch_rewards, batch_values, batch_policies)

        info = (np.mean(future_return_lst), np.mean(model_index_lst), self.last_model_index)
        # a batch contains the inputs and the targets
        batch_data = (inputs_batch, targets_batch, info)
        return batch_data

    def make_renalyze_update(self, game_id: int, game: GameHistory):
        # prepare context for policy reanalyze
        policy_obs_lst = []
        game_obs = game.obs(0, len(game))
        for state_index in range(len(game)):
            policy_obs_lst.append(game_obs[state_index:state_index + self.config.stacked_observations])

        # generate target policy from reanalyzing
        policy_obs_lst = prepare_observation_lst(policy_obs_lst, self.config.image_based)
        if self.config.image_based:
            policy_obs_tensor = torch.from_numpy(policy_obs_lst).to(self.device).float() / 255.0
        else:
            policy_obs_tensor = torch.from_numpy(policy_obs_lst).to(self.device).float()
        with autocast():
            network_output = self.model.initial_inference(policy_obs_tensor)

        legal_actions_lst = game.legal_actions

        search_results = SampledMCTS(self.config, self.np_random).batch_search(
            self.model, network_output, legal_actions_lst, self.device, add_noise=True, sampled_tau=1.0)

        C = self.config.sampled_action_times

        batch_sampled_actions, batch_sampled_masks = concat_with_zero_padding(search_results.sampled_actions, C)
        batch_sampled_visit_counts, _ = concat_with_zero_padding(search_results.sampled_visit_count, C)
        batch_sampled_policies = batch_sampled_visit_counts / self.config.num_simulations
        batch_sampled_qvalues, _ = concat_with_zero_padding(search_results.sampled_qvalues, C)
        batch_root_values = np.asarray(search_results.value)

        # update game history
        game.model_indices = np.array([self.last_model_index for _ in range(len(game))])
        game.sampled_actions = batch_sampled_actions
        game.sampled_policies = batch_sampled_policies
        game.sampled_padding_masks = batch_sampled_masks
        game.sampled_qvalues = batch_sampled_qvalues
        game.root_values = batch_root_values

        return (game_id, game)


@ray.remote
class RemoteReanalyzeWorker(ReanalyzeWorker):

    def __init__(self, rank, config, shared_storage, replay_buffer, batch_storage):
        """ReanalyzeWorker for reanalyzing targets on remote

        Parameters
        ----------
        rank: int
            id of the worker
        shared_storage: Any
            The model storage
        replay_buffer: Any
            Replay buffer
        batch_storage: Any
            The batch storage (batch queue)
        """
        super().__init__(rank, config)
        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer
        self.batch_storage = batch_storage

    def get_beta(self, trained_steps):
        return self.beta_schedule.value(trained_steps)

    def run_loop(self):
        start = False
        while True:
            # waiting for start signal
            if not start:
                start = ray.get(self.shared_storage.get_start_signal.remote())
                # request for first batch
                if start:
                    trained_steps = ray.get(self.shared_storage.get_counter.remote())
                    beta = self.beta_schedule.value(trained_steps)
                    buffer_context_handle = self.replay_buffer.prepare_batch_context.remote(self.config.batch_size, beta)
                time.sleep(0.1)
                continue

            # break
            trained_steps = ray.get(self.shared_storage.get_counter.remote())
            if trained_steps >= self.config.training_steps + self.config.last_steps:
                time.sleep(30)
                break

            # obtain the latest model weights
            if self.last_model_index // self.config.target_model_interval < trained_steps // self.config.target_model_interval:
                target_model_index, target_weights = ray.get(self.shared_storage.get_target_weights.remote())
                self.model.load_state_dict(target_weights)
                self.model.to(self.device)
                self.model.eval()
                self.last_model_index = target_model_index

            # make batch only when batch_storage is under threshold
            if self.batch_storage.get_len() < self.batch_storage.threshold:

                # obtain the context from replay buffer and prepare for next batch
                buffer_context = ray.get(buffer_context_handle)
                transitions_collected = ray.get(self.replay_buffer.transitions_collected.remote())
                beta = self.beta_schedule.value(trained_steps)
                buffer_context_handle = self.replay_buffer.prepare_batch_context.remote(self.config.batch_size, beta)

                batch_context = self.make_batch(buffer_context, transitions_collected)
                self.batch_storage.push(batch_context)
            else:
                time.sleep(1)

    def update_loop(self):
        start = False
        while True:
            # waiting for start signal
            if not start:
                start = ray.get(self.shared_storage.get_start_signal.remote())
                # request for first batch
                if start:
                    game_handle = self.replay_buffer.prepare_game.remote()
                time.sleep(0.1)
                continue

            # break
            trained_steps = ray.get(self.shared_storage.get_counter.remote())
            if trained_steps >= self.config.training_steps + self.config.last_steps:
                time.sleep(30)
                break

            # obtain the latest model weights
            if self.last_model_index // self.config.target_model_interval < trained_steps // self.config.target_model_interval:
                target_model_index, target_weights = ray.get(self.shared_storage.get_target_weights.remote())
                self.model.load_state_dict(target_weights)
                self.model.to(self.device)
                self.model.eval()
                self.last_model_index = target_model_index

            game_id, game = ray.get(game_handle)
            game_handle = self.replay_buffer.prepare_game.remote()
            update_context = self.make_renalyze_update(game_id, game)
            self.replay_buffer.update_game_history.remote(update_context)
