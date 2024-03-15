import os
import ray
import time
import torch
from typing import List

import numpy as np
from torch.cuda.amp import autocast as autocast
from gymnasium.utils import seeding

from core.config import BaseConfig, Game
from core.model import BaseNet
from core.mcts import SampledMCTS
from core.game import GameHistory
from core.utils import select_action, prepare_observation_lst


def test(
    config: BaseConfig,
    model: BaseNet,
    counter: int,
    test_episodes: int,
    envs: List[Game] = None,
    np_random: np.random.RandomState = None,
    save_video: bool = False,
):
    """evaluation test
    Parameters
    ----------
    model: any
        models for evaluation
    counter: int
        current training step counter
    test_episodes: int
        number of test episodes
        True -> use tqdm bars
    """

    print('Start evaluation for model {}.'.format(counter))

    device = 'cuda' if (config.selfplay_on_gpu and torch.cuda.is_available()) else 'cpu'
    model.to(device)
    model.eval()

    # new games
    if envs is None:
        envs = [config.new_game(seed=i, save_video=save_video) for i in range(test_episodes)]
        create_temp_envs = True
    else:
        create_temp_envs = False

    with torch.no_grad():
        max_episode_steps = envs[0].get_max_episode_steps()
        # initializations
        init_obses = [env.reset() for env in envs]
        dones = np.array([False for _ in range(test_episodes)])
        game_histories = [
            GameHistory(config=config, ray_store_obs=False) for _ in range(test_episodes)]
        for i in range(test_episodes):
            game_histories[i].init([init_obses[i] for _ in range(config.stacked_observations)])

        step = 0
        eps_steps_lst = np.zeros(test_episodes)
        eps_reward_lst = np.zeros(test_episodes)

        if config.case in ['smac', 'gfootball']:
            battle_won_lst = np.zeros(test_episodes)

        # loop
        while (not dones.all()) and step < max_episode_steps:

            stack_obs = [game_history.step_obs() for game_history in game_histories]
            stack_obs = prepare_observation_lst(stack_obs, config.image_based)
            if config.image_based:
                stack_obs = torch.from_numpy(stack_obs).to(device).float() / 255.0
            else:
                stack_obs = torch.from_numpy(stack_obs).to(device).float()

            with autocast():
                network_output = model.initial_inference(stack_obs.float())
            legal_actions_lst = np.asarray([env.legal_actions() for env in envs])

            if config.use_mcts_test:
                search_results = SampledMCTS(config, np_random).batch_search(model, network_output, legal_actions_lst, device, False, 1.0)

                roots_sampled_visit_counts = search_results.sampled_visit_count
                roots_sampled_actions = search_results.sampled_actions
            else:
                # use network output directly as the evaluation policy, instead of MCTS search
                batch_policy_logits = network_output.policy_logits
                batch_policy_probs = np.exp(batch_policy_logits - np.max(batch_policy_logits, axis=-1, keepdims=True))
                batch_policy_probs *= legal_actions_lst
                batch_policy_probs = batch_policy_probs / np.sum(batch_policy_probs, axis=-1, keepdims=True)    # type: np.ndarray

            for i in range(test_episodes):
                if dones[i]:
                    continue

                # select the argmax, not sampling
                if config.use_mcts_test:
                    action_pos, _ = select_action(
                        roots_sampled_visit_counts[i],
                        temperature=1,
                        deterministic=True,
                        np_random=np_random
                    )
                    action = roots_sampled_actions[i][action_pos]
                else:
                    action = np.argmax(batch_policy_probs[i], axis=-1)

                next_obs, reward, done, info = envs[i].step(action)
                dones[i] = done
                eps_steps_lst[i] += 1
                eps_reward_lst[i] += reward

                game_histories[i].store_transition(action, reward, next_obs)

                if config.case in ['smac', 'gfootball']:
                    battle_won_lst[i] = info['battle_won']

            step += 1

    if create_temp_envs:
        for env in envs:
            env.close()

    test_logs = {
        'test_counter': counter,
        'mean_score': eps_reward_lst.mean(),
        'std_score': eps_reward_lst.std(),
        'max_score': eps_reward_lst.max(),
        'min_score': eps_reward_lst.min(),
    }
    if config.case in ['smac', 'gfootball']:
        test_logs['win_rate'] = np.mean(battle_won_lst)

    test_msg = '#{:<10} Test Mean Score of {}: {:<10} (max: {:<10}, min:{:<10}, std: {:<10})' \
               ''.format(test_logs['test_counter'], config.env_name, test_logs["mean_score"], test_logs["max_score"], test_logs["min_score"], test_logs["std_score"])
    if 'win_rate' in test_logs:
        test_msg += ' | WinRate: {:.2f}'.format(test_logs['win_rate'])
    print(test_msg)

    return test_logs, step


class TestWorker(object):

    def __init__(self, config: BaseConfig):
        self.config = config
        self.eval_model = config.get_uniform_network()
        self.np_random, _ = seeding.np_random(config.seed * 3000)

        self.test_episodes = config.test_episodes
        self.eval_envs = [config.new_game(seed=i, save_video=False) for i in range(config.test_episodes)]

        self.device = 'cuda' if (config.selfplay_on_gpu and torch.cuda.is_available()) else 'cpu'
        self.eval_model.to(self.device)
        self.eval_model.eval()
        self.last_model_index = -1

    def update_model(self, model_index, weights):
        self.eval_model.set_weights(weights)
        self.last_model_index = model_index

    def run(self):
        """run evaluation test once
        """
        return test(self.config, self.eval_model, self.last_model_index, self.test_episodes, self.eval_envs, self.np_random)

    def close(self):
        for env in self.eval_envs:
            env.close()


@ray.remote
class RemoteTestWorker(TestWorker):

    def __init__(self, config: BaseConfig, shared_storage):
        super().__init__(config)
        self.shared_storage = shared_storage

    def run_loop(self):
        best_test_score = float('-inf')
        while True:
            trained_steps = ray.get(self.shared_storage.get_counter.remote())
            if trained_steps >= self.config.training_steps + self.config.last_steps:
                time.sleep(10)
                break
            if self.last_model_index // self.config.test_interval < trained_steps // self.config.test_interval:
                model_index, weights = ray.get(self.shared_storage.get_weights.remote())
                self.update_model(model_index, weights)

                test_log, eval_steps = self.run()

                self.shared_storage.add_test_logs.remote(test_log)

                if test_log['mean_score'] >= best_test_score:
                    best_test_score = test_log['mean_score']
                    torch.save(self.eval_model.state_dict(), self.config.model_path)

            time.sleep(30)

        self.close()
