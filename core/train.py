from collections import deque
import logging
import math
import os
import time

import numpy as np
import ray
import torch
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import torch.optim as optim

from core.config import BaseConfig
from core.model import BaseNet
from core.log import _log, train_logger
from core.test import test, TestWorker, RemoteTestWorker
from core.replay_buffer import ReplayBuffer, RemoteReplayBuffer, RemotePriorityRefresher
from core.storage import SharedStorage, RemoteShareStorage, RemoteQueueStorage
from core.selfplay_worker import DataWorker, RemoteDataWorker
from core.reanalyze_worker import ReanalyzeWorker, RemoteReanalyzeWorker
from core.utils import Timer, remote_worker_handles


def update_weights(config: BaseConfig, step_count: int, model: BaseNet, batch: tuple, optimizer: optim.Optimizer, scaler: GradScaler, device):
    """update models given a batch data
    Parameters
    ----------
    model: Any
        EfficientZero models
    batch: Any
        a batch data inlcudes [inputs_batch, targets_batch]
    scaler: Any
        scaler for torch amp
    """
    inputs_batch, targets_batch, info = batch
    obs_batch, action_batch, mask_batch, indices, weights_lst = inputs_batch
    target_reward, target_value, target_policy = targets_batch
    (
        target_sampled_actions,
        target_sampled_policies,
        target_sampled_imp_ratio,
        target_sampled_adv,
        sampled_action_mask,
    ) = target_policy
    batch_future_return, batch_model_index, target_model_index = info

    if config.image_based:
        obs_batch = torch.from_numpy(np.array(obs_batch)).to(device).float() / 255.0
    else:
        obs_batch = torch.from_numpy(np.array(obs_batch)).to(device).float()

    # do augmentations
    if config.use_augmentation:
        obs_batch = config.augmentation_transform(obs_batch)

    # use GPU tensor
    action_batch = torch.from_numpy(np.array(action_batch)).to(device).long()
    mask_batch = torch.from_numpy(np.array(mask_batch)).to(device).float()
    weights = torch.from_numpy(np.array(weights_lst)).to(device).float()

    target_reward = torch.from_numpy(np.array(target_reward)).to(device).float()
    target_value = torch.from_numpy(np.array(target_value)).to(device).float()
    # additional context for policy loss
    target_sampled_actions = torch.from_numpy(np.array(target_sampled_actions)).to(device).long()
    target_sampled_policies = torch.from_numpy(np.array(target_sampled_policies)).to(device).float()
    target_sampled_imp_ratio = torch.from_numpy(np.array(target_sampled_imp_ratio)).to(device).float()
    target_sampled_adv = torch.from_numpy(np.array(target_sampled_adv)).to(device).float()
    sampled_action_mask = torch.from_numpy(np.array(sampled_action_mask)).to(device).float()

    batch_size = obs_batch.size(0)
    obs_pad_size = config.image_channel * (config.stacked_observations + config.num_unroll_steps)
    # data shape check
    assert batch_size == config.batch_size
    assert obs_batch.shape == (batch_size, config.num_agents, obs_pad_size, *config.obs_shape[:-1])
    assert action_batch.shape == (batch_size, config.num_unroll_steps + 1, config.num_agents)
    assert mask_batch.shape == (batch_size, config.num_unroll_steps + 1)
    assert target_reward.shape == (batch_size, config.num_unroll_steps + 1)
    assert target_value.shape == (batch_size, config.num_unroll_steps + 1)

    assert target_sampled_actions.shape == (batch_size, config.num_unroll_steps + 1, config.sampled_action_times, config.num_agents)
    assert target_sampled_policies.shape == (batch_size, config.num_unroll_steps + 1, config.sampled_action_times)
    assert target_sampled_imp_ratio.shape == (batch_size, config.num_unroll_steps + 1, config.sampled_action_times)
    assert target_sampled_adv.shape == (batch_size, config.num_unroll_steps + 1, config.sampled_action_times)
    assert sampled_action_mask.shape == (batch_size, config.num_unroll_steps + 1, config.sampled_action_times)

    # transform targets to categorical representation
    target_reward_phi = config.reward_transform(target_reward)
    target_value_phi = config.value_transform(target_value)

    gradient_scale = 1 / config.num_unroll_steps

    with autocast():

        # init with the stacked observations:
        step_i, beg_index = 0, 0
        end_index = config.image_channel * config.stacked_observations
        network_output = model.initial_inference(obs_batch[:, :, beg_index:end_index])

        # calculate the new priorities for each transition
        scaled_value = config.inverse_value_transform(network_output.value).squeeze(-1)
        value_priority = np.abs(
            scaled_value.detach().cpu().numpy() - target_value[:, step_i].detach().cpu().numpy()
        ) + config.prioritized_replay_eps
        new_priority_data = (indices, value_priority)

        # loss of the first step

        sampled_actions_log_prob = (
            network_output.policy_logits.log_softmax(dim=-1)        # (batch_size, num_agents, action_space_size)
            .gather(dim=2, index=target_sampled_actions[:, step_i].transpose(1, 2))  # index: (.., sampled_times, num_agents) -> (.., num_agents, sampled_times)
        ).sum(dim=1)            # (batch_size, num_agents, sampled_times) -> (batch_size, sampled_times)

        if config.PG_type == "none":
            policy_loss = -(
                sampled_actions_log_prob
                * target_sampled_policies[:, step_i]                    # visit count
                * sampled_action_mask[:, step_i]                        # mask invalid actions
            ).sum(dim=1)
        else:
            if config.awac_lambda > 0:
                adv_weights = torch.exp(target_sampled_adv[:, step_i] / config.awac_lambda)
                '''Reference: https://github.com/Junyoungpark/Pytorch-AWAC/blob/main/src/Learner/AWAC.py#L77'''
                # adv_weights = torch.nn.functional.softmax(target_sampled_adv[:, step_i] / config.awac_lambda, dim=0) * batch_size
                '''Reference: https://github.com/hari-sikchi/AWAC/blob/master/AWAC/awac.py#L307'''
            else:
                adv_weights = target_sampled_adv[:, step_i]

            if config.adv_clip > 0:
                adv_weights = torch.clamp(adv_weights, -config.adv_clip, config.adv_clip)

            if config.PG_type == "raw":
                policy_loss = -(
                    sampled_actions_log_prob
                    * adv_weights                                       # AWAC policy loss
                    * target_sampled_imp_ratio[:, step_i]               # importance ratio
                    * sampled_action_mask[:, step_i]                    # mask invalid actions
                ).sum(dim=1)            # (batch_size, sampled_times) -> (batch_size,)
            elif config.PG_type == "sharp":
                policy_loss = -(
                    sampled_actions_log_prob
                    * adv_weights                                       # AWAC policy loss
                    * target_sampled_policies[:, step_i]                # visit count
                    * sampled_action_mask[:, step_i]                    # mask invalid actions
                ).sum(dim=1)            # (batch_size, sampled_times) -> (batch_size,)
            else:
                raise NotImplementedError

        reward_loss = torch.zeros(batch_size, device=device)
        value_loss = config.value_loss(network_output.value, target_value_phi[:, 0])
        if config.consistency_coeff > 0:
            consistency_loss = torch.zeros(batch_size, device=device)

        # unroll with the dynamics function using actual executed action
        for step_i in range(1, config.num_unroll_steps + 1):
            beg_index = config.image_channel * step_i
            end_index = config.image_channel * (step_i + config.stacked_observations)
            network_output = model.recurrent_inference(network_output.hidden_state, action_batch[:, step_i - 1])

            # loss of the unrolled steps (k=1,...,K)

            sampled_actions_log_prob = (
                network_output.policy_logits.log_softmax(dim=-1)        # (batch_size, num_agents, action_space_size)
                .gather(dim=2, index=target_sampled_actions[:, step_i].transpose(1, 2))  # index: (.., sampled_times, num_agents) -> (.., num_agents, sampled_times)
            ).sum(dim=1)            # (batch_size, num_agents, sampled_times) -> (batch_size, sampled_times)

            if config.PG_type == "none":
                policy_loss += -(
                    sampled_actions_log_prob
                    * target_sampled_policies[:, step_i]                    # visit count
                    * sampled_action_mask[:, step_i]                        # mask invalid actions
                ).sum(dim=1)
            else:
                if config.awac_lambda > 0:
                    adv_weights = torch.exp(target_sampled_adv[:, step_i] / config.awac_lambda)
                    # adv_weights = torch.nn.functional.softmax(target_sampled_adv[:, step_i] / config.awac_lambda, dim=0) * batch_size
                else:
                    adv_weights = target_sampled_adv[:, step_i]

                if config.adv_clip > 0:
                    adv_weights = torch.clamp(adv_weights, -config.adv_clip, config.adv_clip)

                if config.PG_type == "raw":
                    policy_loss += -(
                        sampled_actions_log_prob
                        * adv_weights                                       # AWAC policy loss
                        * target_sampled_imp_ratio[:, step_i]               # importance ratio
                        * sampled_action_mask[:, step_i]                    # mask invalid actions
                    ).sum(dim=1)            # (batch_size, sampled_times) -> (batch_size,)
                elif config.PG_type == "sharp":
                    policy_loss += -(
                        sampled_actions_log_prob
                        * adv_weights                                       # AWAC policy loss
                        * target_sampled_policies[:, step_i]                # visit count
                        * sampled_action_mask[:, step_i]                    # mask invalid actions
                    ).sum(dim=1)            # (batch_size, sampled_times) -> (batch_size,)
                else:
                    raise NotImplementedError

            reward_loss += config.reward_loss(network_output.reward, target_reward_phi[:, step_i - 1])          # don't mask reward loss
            value_loss += config.value_loss(network_output.value, target_value_phi[:, step_i])                  # don't mask value loss
            if config.consistency_coeff > 0:
                # obtain the oracle hidden states from representation function
                representation = model.initial_inference(obs_batch[:, :, beg_index:end_index])
                # no grad for the presentation_state branch
                dynamic_proj = model.project(network_output.hidden_state, with_grad=True)    # P2(P1(s_{t,k}))
                represet_proj = model.project(representation.hidden_state, with_grad=False)  # sg(P1(s_{t+k,0}))
                consistency_loss += config.consistency_loss(dynamic_proj, represet_proj)                        # don't mask consistency loss ???
            # Follow MuZero, set half gradient
            network_output.hidden_state.register_hook(lambda grad: grad * 0.5)

        # weighted loss with masks (some invalid states which are out of trajectory.)
        loss = (
            config.reward_loss_coeff * reward_loss
            + config.policy_loss_coeff * policy_loss
            + config.value_loss_coeff * value_loss
        )
        if config.consistency_coeff > 0:
            loss += config.consistency_coeff * consistency_loss
        total_loss = (weights * loss).mean()
        total_loss.register_hook(lambda grad: grad * gradient_scale)

    # backward
    lr = config.adjust_lr(optimizer, step_count)
    optimizer.zero_grad()
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    scaler.step(optimizer)
    scaler.update()

    # packing data for logging
    train_logs = {
        'total_loss': total_loss.item(),
        'reward_loss': (weights * reward_loss).mean().item(),
        'policy_loss': (weights * policy_loss).mean().item(),
        'value_loss': (weights * value_loss).mean().item(),
    }

    if config.consistency_coeff > 0:
        train_logs['consistency_loss'] = (weights * consistency_loss).mean().item()
    train_logs['lr'] = lr
    train_logs['batch_future_return'] = batch_future_return
    train_logs['batch_model_diff'] = step_count - batch_model_index
    train_logs['target_model_diff'] = step_count - target_model_index

    return train_logs, new_priority_data


def _train(model, target_model, replay_buffer, shared_storage, batch_storage, config: BaseConfig, summary_writer):
    """training loop
    Parameters
    ----------
    model: Any
        EfficientZero models
    target_model: Any
        EfficientZero models for reanalyzing
    replay_buffer: Any
        replay buffer
    shared_storage: Any
        model storage
    batch_storage: Any
        batch storage (queue)
    summary_writer: Any
        logging for tensorboard
    """
    # ----------------------------------------------------------------------------------
    device = 'cuda' if (config.train_on_gpu and torch.cuda.is_available()) else 'cpu'
    model = model.to(device)
    target_model = target_model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, eps=config.opti_eps,
                           weight_decay=config.weight_decay)

    scaler = GradScaler()

    model.train()
    target_model.eval()
    # ----------------------------------------------------------------------------------
    # set augmentation tools
    if config.use_augmentation:
        config.set_augmentation_transforms()

    # wait until collecting enough data to start
    while True:
        transitions_collected = ray.get(replay_buffer.transitions_collected.remote())
        train_logger.debug(f'ReplayBufferSize:{transitions_collected}/{config.start_transitions}')
        if transitions_collected >= config.start_transitions:
            break
        else:
            time.sleep(3)

    train_logger.info('Begin training...')
    # set signals for other workers
    shared_storage.set_start_signal.remote()

    step_count = 0
    # Note: the interval of the current model and the target model is between x and 2x. (x = target_model_interval)
    # recent_weights is the param of the target model
    recent_weights = model.get_weights()

    batch_start_time = time.time()
    train_start_time = batch_start_time
    batch_timecost, train_timecost = 0, 0

    # while loop
    while step_count < config.training_steps + config.last_steps:

        # obtain a batch
        batch = batch_storage.pop()
        if batch is None:
            time.sleep(0.5)
            continue
        current_time = time.time()
        batch_timecost += current_time - batch_start_time
        train_start_time = current_time
        shared_storage.incr_counter.remote()

        # update model for self-play
        if step_count % config.checkpoint_interval == 0:
            shared_storage.set_weights.remote(step_count, model.get_weights())

        # update model for reanalyzing
        if step_count % config.target_model_interval == 0:
            shared_storage.set_target_weights.remote(step_count, recent_weights)
            recent_weights = model.get_weights()

        train_logs, new_priority_data = update_weights(config, step_count, model, batch, optimizer, scaler, device)

        if config.use_priority and not config.use_priority_refresh:
            # update priority if no refresher
            indices, new_priority = new_priority_data
            replay_buffer.update_priorities.remote(indices, new_priority)

        current_time = time.time()
        train_timecost += current_time - train_start_time
        batch_start_time = current_time
        train_logs['Tp_perstep'] = batch_timecost / (step_count + 1)
        train_logs['Tu_perstep'] = train_timecost / (step_count + 1)

        if step_count % config.log_interval == 0:
            _log(config, step_count, train_logs, replay_buffer, shared_storage, summary_writer)

        # Chech queue capacity.
        if step_count >= 100 and step_count % 50 == 0:
            if batch_storage.get_len() == 0:
                train_logger.warn(f'#{step_count} Batch Queue is empty (Require more reanalyze actors Or actor fails).')
            elif batch_storage.get_len() == batch_storage.threshold:
                train_logger.warn(f'#{step_count} Batch Queue is excess (Reduce reanalyze actors).')

        step_count += 1

        # save models
        if step_count % config.save_interval == 0:
            model_path = os.path.join(config.model_dir, 'model_{}.p'.format(step_count))
            torch.save(model.state_dict(), model_path)

    shared_storage.set_weights.remote(step_count, model.get_weights())
    time.sleep(30)
    return model.get_weights()


def train(config: BaseConfig, summary_writer, model_path=None):
    """training process
    Parameters
    ----------
    summary_writer: Any
        logging for tensorboard
    model_path: str
        model path for resuming
        default: train from scratch
    """
    model = config.get_uniform_network()
    target_model = config.get_uniform_network()
    if model_path:
        train_logger.info('resume model from path: {}'.format(model_path))
        weights = torch.load(model_path)
        model.load_state_dict(weights)
        target_model.load_state_dict(weights)

    shared_storage = RemoteShareStorage.remote(model, target_model)

    # training-data flow: replay_buffer -> reanalyze_workers -> batch_storage -> _train
    replay_buffer = RemoteReplayBuffer.remote(config=config)
    batch_storage = RemoteQueueStorage(config.reanalyze_actors, math.ceil(config.reanalyze_actors * 1.5))

    # parallel tasks
    tasks = []

    availabel_gpus = ray.available_resources().get('GPU', 0)
    num_gpu_workers = 2  # train + test
    if config.selfplay_on_gpu:
        num_gpu_workers += config.data_actors
    if config.reanalyze_on_gpu:
        num_gpu_workers += (config.reanalyze_actors + config.reanalyze_update_actors)
    if config.use_priority_refresh:
        num_gpu_workers += config.refresh_actors
    num_gpus_per_worker = 1 / math.ceil(num_gpu_workers / availabel_gpus)

    # self-play workers
    data_workers = [
        RemoteDataWorker.options(
            num_gpus=num_gpus_per_worker if config.selfplay_on_gpu else 0,
        ).remote(
            rank, config, replay_buffer, shared_storage,
        ) for rank in range(config.data_actors)
    ]
    tasks += [worker.run_loop.remote() for worker in data_workers]

    # test workers
    test_worker = RemoteTestWorker.options(
        num_gpus=num_gpus_per_worker if config.selfplay_on_gpu else 0,
    ).remote(config, shared_storage)
    tasks += [test_worker.run_loop.remote()]

    # priority-refresh workers
    if config.use_priority_refresh:
        refresh_workers = [
            RemotePriorityRefresher.options(
                num_gpus=num_gpus_per_worker
            ).remote(config, replay_buffer, shared_storage) for _ in range(config.refresh_actors)
        ]
        tasks += [worker.run_loop.remote() for worker in refresh_workers]

    # reanalyze workers
    reanalyze_workers = [
        RemoteReanalyzeWorker.options(
            num_gpus=num_gpus_per_worker if config.reanalyze_on_gpu else 0,
        ).remote(
            idx, config, shared_storage, replay_buffer, batch_storage,
        ) for idx in range(config.reanalyze_actors + config.reanalyze_update_actors)
    ]
    tasks += [worker.run_loop.remote() for worker in reanalyze_workers[:config.reanalyze_actors]]
    tasks += [worker.update_loop.remote() for worker in reanalyze_workers[config.reanalyze_actors:]]

    # add handles to global variables
    global remote_worker_handles
    remote_worker_handles += data_workers
    remote_worker_handles.append(test_worker)

    # training loop
    final_weights = _train(model, target_model, replay_buffer, shared_storage, batch_storage, config, summary_writer)

    ray.wait(tasks)
    train_logger.info('Training over...')

    return model, final_weights


def train_sync_serial(config: BaseConfig, summary_writer, model_path=None):
    assert config.data_actors == 1, 'Sync training only support 1 data collector!'

    ''' initialize model '''

    model = config.get_uniform_network()
    target_model = config.get_uniform_network()
    recent_weights = model.get_weights()
    target_model.set_weights(recent_weights)
    if model_path:
        train_logger.info('resume model from path: {}'.format(model_path))
        weights = torch.load(model_path)
        model.load_state_dict(weights)
        target_model.load_state_dict(weights)

    ''' initialize workers '''

    shared_storage = SharedStorage(model, target_model)
    replay_buffer = ReplayBuffer(config)

    data_worker = DataWorker(0, config, replay_buffer, shared_storage)
    data_worker.update_model(0, model.get_weights())

    reanalyze_worker = ReanalyzeWorker(0, config)
    reanalyze_worker.update_model(0, recent_weights)

    test_worker = TestWorker(config)

    ''' initialize training utils '''

    device = 'cuda' if (config.train_on_gpu and torch.cuda.is_available()) else 'cpu'
    model = model.to(device)
    target_model = target_model.to(device)
    model.train()
    target_model.eval()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, eps=config.opti_eps,
                           weight_decay=config.weight_decay)
    scaler = GradScaler()

    ''' training loop '''

    transitions_collected = 0
    start_training = False
    step_count = 0
    timer = Timer()

    while step_count < config.training_steps + config.last_steps:

        # Run for a whole episode at a time
        if transitions_collected < config.total_transitions:
            timer.start('collect')
            data_worker.update_model(step_count, model.get_weights())
            transitions_collected += data_worker.run(start_training, step_count)
            timer.stop('collect')

        if replay_buffer.can_sample(config.batch_size):
            if not start_training:
                start_training = True
                train_logger.info('Begin training...')

            # compute training steps under current transitions_collected
            if step_count < config.training_steps:
                target_steps = int(config.training_steps * transitions_collected / config.total_transitions)
            else:
                target_steps = config.training_steps + config.last_steps

            while step_count < target_steps:

                # obtain a batch
                timer.start('prepare')
                beta = reanalyze_worker.beta_schedule.value(step_count)
                buffer_context = replay_buffer.prepare_batch_context(config.batch_size, beta)
                batch_context = reanalyze_worker.make_batch(buffer_context, transitions_collected)
                timer.stop('prepare')

                timer.start('update')
                train_logs, new_priority_data = update_weights(config, step_count, model, batch_context, optimizer, scaler, device)

                # update priority if no refresher
                if config.use_priority:
                    indices, new_priority = new_priority_data
                    replay_buffer.update_priorities(indices, new_priority)

                # update model for reanalyzing
                if step_count % config.target_model_interval == 0:
                    reanalyze_worker.update_model(step_count, recent_weights)
                    recent_weights = model.get_weights()

                # save models
                if step_count % config.save_interval == 0:
                    model_path = os.path.join(config.model_dir, 'model_{}.p'.format(step_count))
                    torch.save(model.state_dict(), model_path)
                timer.stop('update')

                # evaluation
                timer.start('eval')
                if step_count % config.test_interval == 0:
                    test_worker.update_model(step_count, model.get_weights())
                    test_log, eval_steps = test_worker.run()
                    shared_storage.add_test_logs(test_log)
                timer.stop('eval')

                train_logs['Tc_perstep'] = timer.sum('collect') / (step_count + 1)
                train_logs['Tp_perstep'] = timer.sum('prepare') / (step_count + 1)
                train_logs['Tu_perstep'] = timer.sum('update') / (step_count + 1)
                train_logs['Te_perstep'] = timer.sum('eval') / (step_count + 1)

                # logging
                if step_count % config.log_interval == 0:
                    _log(config, step_count, train_logs, replay_buffer, shared_storage, summary_writer)

                step_count += 1
        else:
            train_logger.debug(f'ReplayBufferSize:{transitions_collected}/{config.batch_size}')

    train_logger.info('Training over...')
    data_worker.close()
    test_worker.close()
    model.eval()
    test_log, eval_steps = test(config, model, step_count, config.test_episodes)
    test_msg = "#{:<10} Test Mean Score of {}: {:<10} (max: {:<10}, min:{:<10}, std: {:<10})" \
               "".format(step_count, config.env_name, test_log["mean_score"], test_log["max_score"], test_log["min_score"], test_log["std_score"])
    logging.getLogger("train_test").info(test_msg)

    return model, model.get_weights()


def train_sync_parallel(config: BaseConfig, summary_writer, model_path=None):
    assert config.data_actors == 1, 'Sync training only support 1 data collector!'

    ''' initialize model '''

    model = config.get_uniform_network()
    target_model = config.get_uniform_network()
    recent_weights = model.get_weights()
    target_model.set_weights(recent_weights)
    if model_path:
        train_logger.info('resume model from path: {}'.format(model_path))
        weights = torch.load(model_path)
        model.load_state_dict(weights)
        target_model.load_state_dict(weights)

    ''' initialize workers '''

    assert ray.is_initialized(), "Must invoke ray.init when parallel"
    availabel_gpus = ray.available_resources().get('GPU', 0)
    num_gpu_workers = 2  # train + test
    if config.selfplay_on_gpu:
        num_gpu_workers += config.data_actors
    if config.reanalyze_on_gpu:
        num_gpu_workers += config.reanalyze_actors
    num_gpus_per_worker = 1 / math.ceil(num_gpu_workers / availabel_gpus)

    # shared_storage = RemoteShareStorage.remote(model, target_model)
    # replay_buffer = RemoteReplayBuffer.remote(config)

    # data_worker = RemoteDataWorker.options(
    #     num_gpus=num_gpus_per_worker if config.selfplay_on_gpu else 0
    # ).remote(0, config, replay_buffer, shared_storage)
    # data_worker.update_model.remote(0, model.get_weights())
    # data_handle = data_worker.run.remote(False, 0)

    shared_storage = SharedStorage(model, target_model)
    replay_buffer = ReplayBuffer(config)

    data_worker = DataWorker(0, config, replay_buffer, shared_storage)
    data_worker.update_model(0, model.get_weights())

    reanalyze_workers = [
        RemoteReanalyzeWorker.options(
            num_gpus=num_gpus_per_worker if config.reanalyze_on_gpu else 0
        ).remote(
            idx, config, None, None, None,  # manually control data sync
        ) for idx in range(config.reanalyze_actors)
    ]
    for worker in reanalyze_workers:
        worker.update_model.remote(0, recent_weights)

    test_worker = RemoteTestWorker.options(num_gpus=num_gpus_per_worker).remote(config, shared_storage)
    test_worker.update_model.remote(0, model.get_weights())
    test_handle = test_worker.run.remote()

    global remote_worker_handles
    # remote_worker_handles.append(data_worker)
    remote_worker_handles.append(test_worker)

    ''' initialize training utils '''

    device = 'cuda' if (config.train_on_gpu and torch.cuda.is_available()) else 'cpu'
    model = model.to(device)
    target_model = target_model.to(device)
    model.train()
    target_model.eval()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, eps=config.opti_eps, weight_decay=config.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)
    scaler = GradScaler()

    ''' training loop '''

    transitions_collected = 0
    start_training = False
    step_count = 0
    timer = Timer()

    while step_count < config.training_steps + config.last_steps:

        # Run for a whole episode at a time
        if transitions_collected < config.total_transitions:
            timer.start('collect')
            # transitions_collected += ray.get(data_handle)
            # data_worker.update_model.remote(step_count, model.get_weights())
            # data_handle = data_worker.run.remote(start_training, step_count)
            data_worker.update_model(step_count, model.get_weights())
            transitions_collected += data_worker.run(start_training, step_count)
            timer.stop('collect')

        # if ray.get(replay_buffer.can_sample.remote(config.batch_size)):
        if replay_buffer.can_sample(config.batch_size):
            if not start_training:
                start_training = True
                train_logger.info('Begin training...')

            # compute training steps under current transitions_collected
            if step_count < config.training_steps:
                target_steps = int(config.training_steps * transitions_collected / config.total_transitions)
            else:
                target_steps = config.training_steps + config.last_steps

            # assign target batch to reanalyze-workers
            beta_lst = ray.get([reanalyze_workers[i % config.reanalyze_actors].get_beta.remote(step_count + i) for i in range(target_steps - step_count)])
            # buffer_context_deque = deque([replay_buffer.prepare_batch_context.remote(config.batch_size, beta) for beta in beta_lst])
            buffer_context_deque = deque([replay_buffer.prepare_batch_context(config.batch_size, beta) for beta in beta_lst])
            batch_context_deque = deque()
            for idle_idx, _ in zip(range(config.reanalyze_actors), beta_lst):
                buffer_context = buffer_context_deque.popleft()
                batch_context_deque.append((idle_idx, reanalyze_workers[idle_idx].make_batch.remote(buffer_context, transitions_collected)))

            while step_count < target_steps:

                # obtain a batch
                timer.start('prepare')
                idle_idx, batch_handle = batch_context_deque.popleft()
                if len(buffer_context_deque) > 0:
                    buffer_context = buffer_context_deque.popleft()
                    batch_context_deque.append((idle_idx, reanalyze_workers[idle_idx].make_batch.remote(buffer_context, transitions_collected)))
                batch_context = ray.get(batch_handle)
                timer.stop('prepare')

                timer.start('update')
                train_logs, new_priority_data = update_weights(config, step_count, model, batch_context, optimizer, scaler, device)

                # update priority if no refresher
                if config.use_priority:
                    indices, new_priority = new_priority_data
                    # replay_buffer.update_priorities.remote(indices, new_priority)
                    replay_buffer.update_priorities(indices, new_priority)

                # update model for reanalyzing
                if step_count % config.target_model_interval == 0:
                    for worker in reanalyze_workers:
                        worker.update_model.remote(step_count, recent_weights)
                    recent_weights = model.get_weights()

                # save models
                if step_count % config.save_interval == 0:
                    model_path = os.path.join(config.model_dir, 'model_{}.p'.format(step_count))
                    torch.save(model.state_dict(), model_path)
                timer.stop('update')

                # evaluation
                timer.start('eval')
                if step_count % config.test_interval == 0 and step_count > 0:
                    test_log, eval_steps = ray.get(test_handle)
                    # shared_storage.add_test_logs.remote(test_log)
                    shared_storage.add_test_logs(test_log)
                    test_worker.update_model.remote(step_count, model.get_weights())
                    test_handle = test_worker.run.remote()
                timer.stop('eval')

                train_logs['Tc_perstep'] = timer.sum('collect') / (step_count + 1)
                train_logs['Tp_perstep'] = timer.sum('prepare') / (step_count + 1)
                train_logs['Tu_perstep'] = timer.sum('update') / (step_count + 1)
                train_logs['Te_perstep'] = timer.sum('eval') / (step_count + 1)

                # logging
                if step_count % config.log_interval == 0:
                    _log(config, step_count, train_logs, replay_buffer, shared_storage, summary_writer)

                step_count += 1
        else:
            train_logger.debug(f'ReplayBufferSize:{transitions_collected}/{config.batch_size}')

    train_logger.info('Training over...')
    # data_worker.close.remote()
    data_worker.close()
    test_worker.close.remote()
    model.eval()
    test_logs, eval_steps = test(config, model, step_count, config.test_episodes)
    test_msg = '#{:<10} Test Mean Score of {}: {:<10} (max: {:<10}, min:{:<10}, std: {:<10})' \
               ''.format(test_logs['test_counter'], config.env_name, test_logs["mean_score"], test_logs["max_score"], test_logs["min_score"], test_logs["std_score"])
    if 'win_rate' in test_logs:
        test_msg += ' | WinRate: {:.2f}'.format(test_logs['win_rate'])
    logging.getLogger("train_test").info(test_msg)

    return model, model.get_weights()
