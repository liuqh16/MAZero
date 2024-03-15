import io
import logging

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import ray
from ray.actor import ActorHandle
import seaborn as sns


train_logger = logging.getLogger('train')
test_logger = logging.getLogger('train_test')


def _log(config, step_count, train_logs, replay_buffer, shared_storage, summary_writer):

    if isinstance(replay_buffer, ActorHandle) and isinstance(shared_storage, ActorHandle):
        replay_episodes_collected, replay_transitions_collected, worker_logs, test_logs, priority_logs = ray.get([
            replay_buffer.episodes_collected.remote(), replay_buffer.transitions_collected.remote(),
            shared_storage.get_worker_logs.remote(), shared_storage.get_test_logs.remote(),
            replay_buffer.get_priority_status.remote()])
    else:
        replay_episodes_collected, replay_transitions_collected, worker_logs, test_logs, priority_logs = [
            replay_buffer.episodes_collected(), replay_buffer.transitions_collected(),
            shared_storage.get_worker_logs(), shared_storage.get_test_logs(),
            replay_buffer.get_priority_status()]

    _msg = '#{:<10d} Episodes Collected: {:<10d} Transitions Collected: {:<10d} Batch Size: {:<5d} | '.format(
        step_count, replay_episodes_collected, replay_transitions_collected, config.batch_size)
    # _msg += 'Priority(mean:{:<5.2f}, max:{:<5.2f}, min:{:<5.2f}, std:{:<5.2f}) | '.format(
    #     priority_logs['mean'], priority_logs['max'], priority_logs['min'], priority_logs['std'])
    if worker_logs is not None:
        _msg += 'NewEpisode Model(mean:{:<10d}) Reward(mean:{:<5.2f}, max:{:<5.2f}, min:{:<5.2f}, std:{:<5.2f}) | '.format(
            int(worker_logs['model_index']), worker_logs['eps_reward'], worker_logs['eps_reward_max'], worker_logs['eps_reward_min'], worker_logs['eps_reward_std'])
    for k, v in train_logs.items():
        if k == 'lr':
            _msg += '{}: {:<10.6f}'.format(k, v)
        else:
            _msg += '{}: {:<8.3f}'.format(k, v)
    train_logger.info(_msg)

    if test_logs is not None:
        test_msg = '#{:<10} Test Mean Score of {}: {:<10} (max: {:<10}, min:{:<10}, std: {:<10})' \
                   ''.format(test_logs['test_counter'], config.env_name, test_logs['mean_score'], test_logs['max_score'], test_logs['min_score'], test_logs['std_score'])
        if 'win_rate' in test_logs:
            test_msg += ' | WinRate: {:.2f}'.format(test_logs['win_rate'])
        test_logger.info(test_msg)

    train_logs['episodes_collected'] = replay_episodes_collected
    train_logs['transitions_collected'] = replay_transitions_collected

    if summary_writer is not None:
        for k, v in train_logs.items():
            summary_writer.add_scalar('train/{}'.format(k), v, step_count)

        if worker_logs is not None:
            for k, v in worker_logs.items():
                summary_writer.add_scalar('workers/{}'.format(k), v, step_count)

        if test_logs is not None:
            for k, v in test_logs.items():
                summary_writer.add_scalar('test/{}'.format(k), v, test_logs['test_counter'])
