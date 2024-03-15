import importlib
import logging
import os
import sys

import ray
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb


if __name__ == "__main__":
    sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))

    from core.train import train_sync_serial
    from core.config import parse_args, BaseConfig
    from core.utils import init_logger, make_results_dir, set_seed

    args = parse_args([
        '--opr', 'train_sync', '--case', 'matrix', '--env', 'matgame3', '--exp_name', 'debug', '--seed', '1',
        '--train_on_gpu', '--data_actors', '1', '--num_pmcts', '1', '--reanalyze_actors', '1',
        '--test_interval', '200', '--target_model_interval', '100',
        '--batch_size', '256', '--num_simulations', '50', '--sampled_action_times', '5',
        '--training_steps', '2000', '--last_step', '100', '--lr', '0.02', '--lr_adjust_func', 'const', '--max_grad_norm', '10',
        '--total_transitions', '2000', '--start_transition', '256',
        '--target_value_type', 'pred-re', '--revisit_policy_search_rate', '1', '--use_off_correction',
        '--value_transform_type', 'scalar', '--use_mcts_test',
        '--use_priority', '--use_max_priority'
    ])

    # seeding random iterators
    set_seed(args.seed)

    # import corresponding configuration, neural networks and envs
    module = importlib.import_module("config.{}".format(args.case))
    game_config = getattr(module, 'GameConfig', BaseConfig)(args)

    exp_path, log_base_path = make_results_dir(game_config.exp_path, args)

    # set-up logger
    init_logger(log_base_path)
    logging.getLogger("train").info("Path: {}".format(exp_path))
    logging.getLogger("train").info("Param: {}".format(game_config.get_hparams()))

    summary_writer = SummaryWriter(exp_path, flush_secs=10)
    if args.pretrained_model_path is not None and os.path.exists(args.pretrained_model_path):
        model_path = args.pretrained_model_path
    else:
        model_path = None
    # Parallel Training under synchronization via Ray
    model, weights = train_sync_serial(game_config, summary_writer, model_path)

    if args.reanalyze_actors > 1:
        ray.shutdown()
