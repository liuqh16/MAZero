import importlib
import logging
import os
import sys

import ray
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb

from core.test import test
from core.train import train, train_sync_serial, train_sync_parallel
from core.config import parse_args, BaseConfig
from core.utils import init_logger, make_results_dir, set_seed, remote_worker_handles


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if args.opr == "train" or (args.opr == "train_sync" and args.reanalyze_actors > 1):
        ray.init(num_gpus=args.num_gpus, num_cpus=args.num_cpus,
                 object_store_memory=args.object_store_memory)

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

    try:
        if args.opr == "train":
            if args.use_wandb:
                wandb.init(
                    project=args.case, group=args.env_name, job_type=args.exp_name,
                    sync_tensorboard=True, config=game_config.get_hparams(),
                    name=f"{args.case}_{args.env_name}_{args.exp_name}_seed{args.seed}",
                )
            summary_writer = SummaryWriter(exp_path, flush_secs=10)
            if args.pretrained_model_path is not None and os.path.exists(args.pretrained_model_path):
                model_path = args.pretrained_model_path
            else:
                model_path = None
            # Parallel Training via Ray
            model, weights = train(game_config, summary_writer, model_path)
            model.set_weights(weights)
            total_steps = game_config.training_steps + game_config.last_steps

            # Test final model
            test_log, eval_steps = test(game_config, model, total_steps, game_config.test_episodes)
            test_msg = "#{:<10} Test Mean Score of {}: {:<10} (max: {:<10}, min:{:<10}, std: {:<10})" \
                       "".format(total_steps, game_config.env_name, test_log["mean_score"], test_log["max_score"], test_log["min_score"], test_log["std_score"])
            logging.getLogger("train_test").info(test_msg)
        elif args.opr == "test":
            assert os.path.exists(args.pretrained_model_path), "model not found at {}".format(args.pretrained_model_path)

            model = game_config.get_uniform_network()
            model.load_state_dict(torch.load(args.pretrained_model_path, map_location="cpu"))
            test_log, eval_steps = test(game_config, model, 0, args.test_episodes, save_video=True)
            logging.getLogger("test").info("Test Mean Score: {} (max: {}, min: {})".format(
                test_log["mean_score"], test_log["max_score"], test_log["min_score"]))
            logging.getLogger("test").info("Test Std Score: {}".format(test_log["std_score"]))
        elif args.opr == "train_sync":
            if args.use_wandb:
                wandb.init(
                    entity="heavycrab", project=args.case, group=args.env_name, job_type=args.exp_name,
                    sync_tensorboard=True, config=game_config.get_hparams(),
                    name=f"{args.case}_{args.env_name}_{args.exp_name}_seed{args.seed}",
                )
            summary_writer = SummaryWriter(exp_path, flush_secs=10)
            if args.pretrained_model_path is not None and os.path.exists(args.pretrained_model_path):
                model_path = args.pretrained_model_path
            else:
                model_path = None
            if game_config.reanalyze_actors == 1:
                model, weights = train_sync_serial(game_config, summary_writer, model_path)
            else:
                # Parallel Training under synchronization via Ray
                model, weights = train_sync_parallel(game_config, summary_writer, model_path)
        else:
            raise Exception("Please select a valid operation(--opr) to be performed")
    except (Exception, KeyboardInterrupt) as e:
        logging.getLogger("root").warning("Wait for RemoteActor.close()......")
        ray.get([worker.close.remote() for worker in remote_worker_handles])
        logging.getLogger("root").error(e, exc_info=True)
    finally:
        ray.shutdown()
