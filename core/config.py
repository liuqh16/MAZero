from abc import ABC, abstractmethod
import argparse
import datetime
import math
import os

import torch
import numpy as np

from core.game import Game
from core.model import BaseNet


def parse_args(args):
    parser = argparse.ArgumentParser(description="Base Config for MultiAgent-MuZero")

    groups = parser.add_argument_group("Prepare parameters")
    groups.add_argument("--opr", required=True, type=str, choices=["train", "test", "train_sync"])
    groups.add_argument("--case", required=True, type=str, choices=["smac", "matrix", "lunarlander", "bandit", "mujoco", "gfootball"],
                        help="It's used for switching between different domains.")
    groups.add_argument("--env_name", required=True, type=str, help="Name of the environment/scenario")
    groups.add_argument("--exp_name", required=True, type=str,
                        help="An identifier to distinguish different experiment.")
    groups.add_argument("--seed", type=int, default=0,
                        help="Random seed for numpy/torch (default: %(default)s)")
    groups.add_argument("--discount", type=float, default=0.997,
                        help="Discount of env (default: %(default)s)")
    groups.add_argument("--result_dir", default=os.path.join(os.getcwd(), "results"),
                        help="Directory Path to store results (default: %(default)s)")
    groups.add_argument("--use_wandb", action="store_true", default=False,
                        help="By default False, otherwise log data to wandb server.")

    groups = parser.add_argument_group("Ray parameters")
    groups.add_argument("--num_gpus", type=int, default=1, help="GPUs available (default: %(default)s)")
    groups.add_argument("--num_cpus", type=int, default=10, help="CPUs available (default: %(default)s)")
    groups.add_argument("--object_store_memory", type=int, default=150 * 1024 * 1024 * 1024,
                        help="Object store memory, Bytes (default: %(default)s)")

    groups = parser.add_argument_group("Selfplay parameters")
    groups.add_argument("--selfplay_on_gpu", action="store_true", default=False,
                        help="By default False, otherwise run selfplay on GPU when available.")
    groups.add_argument("--data_actors", type=int, default=1,
                        help="Selfplay data-collector actor in parallel (default: %(default)s)")
    groups.add_argument("--num_pmcts", type=int, default=1,
                        help="Number of parallel mcts in each selfplay actor (default: %(default)s)")
    groups.add_argument("--checkpoint_interval", type=int, default=100,
                        help="Interval of updating the models for self-play (default: %(default)s)")
    groups.add_argument("--total_transitions", type=int, default=int(1e5),
                        help="Total number of collected transitions (default: %(default)s)")
    groups.add_argument("--start_transitions", type=int, default=300,
                        help="Least transition numbers to start the training stage (larger than batch size) (default: %(default)s)")
    groups.add_argument("--use_priority", action="store_true", default=False,
                        help="By default False, otherwise use PER for data sampling in replaybuffer.")
    groups.add_argument("--use_max_priority", action="store_true", default=False,
                        help="By default False, otherwise give max priority for new data.")
    groups.add_argument("--use_change_temperature", action="store_true", default=False,
                        help="By default False, otherwise change temperature of visit count distributions.")
    groups.add_argument("--eps_start", type=float, default=0.,
                        help="For epsilon greedy.")
    groups.add_argument("--eps_end", type=float, default=0.,
                        help="For epsilon greedy.")
    groups.add_argument("--eps_annealing_time", type=int, default=1000,
                        help="For epsilon greedy.")

    groups = parser.add_argument_group("Priority Refresher parameters")
    groups.add_argument("--use_priority_refresh", action="store_true", default=False,
                        help="By default False, otherwise create priority refresher to update all data.")
    groups.add_argument("--refresh_actors", type=int, default=1,
                        help="Priority Refresher actor in parallel (default: %(default)s)")
    groups.add_argument("--refresh_interval", type=int, default=100,
                        help="Interval of updating the models for priority refresh (default: %(default)s)")
    groups.add_argument("--refresh_mini_size", type=int, default=256,
                        help="Split full buffer into slices of refresh_mini_size to save GPU Memory (default: %(default)s)")

    groups = parser.add_argument_group("MCTS & UCB parameters")
    groups.add_argument("--num_simulations", type=int, default=50,
                        help="Number of simulations in MCTS (default: %(default)s)")
    groups.add_argument("--pb_c_base", type=float, default=19652)
    groups.add_argument("--pb_c_init", type=float, default=1.25)
    groups.add_argument("--tree_value_stat_delta_lb", type=float, default=0.01,
                        help="Threshold in the minmax normalization of Q-values in MCTS. (default: %(default)s)")
    groups.add_argument("--root_dirichlet_alpha", type=float, default=0.3,
                        help="Dirichlet alpha of exploration noise in MCTS. (default: %(default)s)")
    groups.add_argument("--root_exploration_fraction", type=float, default=0.25,
                        help="Noisy fraction. (default: %(default)s)")
    groups.add_argument("--sampled_action_times", type=int, default=5,
                        help="Sampled times per Node (default: %(default)s)")
    groups.add_argument("--mcts_rho", type=float, default=0.75, 
                        help="Quantile rho in subtree value estimation (default: %(default)s)")
    groups.add_argument("--mcts_lambda", type=float, default=0.8,
                        help="Decay lambda in subtree value estimation (default: %(default)s)")

    groups = parser.add_argument_group("Train parameters")
    groups.add_argument("--train_on_gpu", action="store_true", default=False,
                        help="By default False, otherwise train on GPU when available")
    groups.add_argument("--training_steps", type=int, default=int(1e5),
                        help="training steps while collecting data (default: %(default)s)")
    groups.add_argument("--last_steps", type=int, default=int(2e4),
                        help="training steps without collecting data after @training_steps (default: %(default)s)")
    groups.add_argument("--batch_size", type=int, default=256,
                        help="batch size (default: %(default)s)")
    groups.add_argument("--num_unroll_steps", type=int, default=5,
                        help="number of unroll steps (default: %(default)s)")
    groups.add_argument("--max_grad_norm", type=float, default=5.0,
                        help="max norm of gradients (default: %(default)s)")
    groups.add_argument("--reward_loss_coeff", type=float, default=1.0,
                        help="coefficient of reward loss (default: %(default)s)")
    groups.add_argument("--value_loss_coeff", type=float, default=0.25,
                        help="coefficient of value loss (default: %(default)s)")
    groups.add_argument("--policy_loss_coeff", type=float, default=1.0,
                        help="coefficient of policy loss (default: %(default)s)")
    groups.add_argument("--use_consistency_loss", action="store_true", default=False,
                        help="By default False, otherwise use temporal consistency.")
    groups.add_argument("--consistency_coeff", type=float, default=2.0,
                        help="coefficient of consistency loss (default: %(default)s)")
    groups.add_argument("--awac_lambda", type=float, default=1.0,
                        help="parameter lambda in AWAC policy loss (default: %(default)s)")
    groups.add_argument("--adv_clip", type=float, default=3.0,
                        help="clip parameter in advantage (default: %(default)s)")
    groups.add_argument("--PG_type", type=str, default="none", choices=["none", "sharp", "raw"], help="type of PG loss")

    groups = parser.add_argument_group("Optimizer parameters")
    groups.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: %(default)s)')
    groups.add_argument("--lr_adjust_func", type=str, default="const", choices=["const", "linear", "cos"],
                        help="By default const(no adjust), otherwise adjust lr based on different schedule.")
    groups.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: %(default)s)')
    groups.add_argument("--weight_decay", type=float, default=0)

    groups = parser.add_argument_group("Reanalyze parameters")
    groups.add_argument("--reanalyze_on_gpu", action="store_true", default=False,
                        help="By default False, otherwise reanalyze on GPU when available")
    groups.add_argument("--reanalyze_actors", type=int, default=20,
                        help="Number of reanalyze actor in parallel (default: %(default)s)")
    groups.add_argument("--reanalyze_update_actors", type=int, default=0,
                        help="Number of reanalyze update actor in parallel (default: %(default)s)")
    groups.add_argument("--td_steps", type=int, default=5,
                        help="Number of td steps for bootstrapped value targets (default: %(default)s)")
    groups.add_argument("--target_value_type", type=str, default="root-re",
                        choices=["root", "pred", "root-re", "pred-re"],
                        help="Choose how to compute target value")
    groups.add_argument("--revisit_policy_search_rate", type=float, default=0.99,
                        help="Rate at which target policy is re-estimated (default: %(default)s)")
    groups.add_argument("--use_off_correction", action="store_true", default=False,
                        help="By default False, otherwise use off-policy correction")
    groups.add_argument("--auto_td_steps_ratio", type=float, default=0.3,
                        help="ratio of short td steps, samller td steps for older trajectories. "
                             "auto_td_steps = training_steps * auto_td_steps_ratio (default: %(default)s)")
    groups.add_argument("--target_model_interval", type=int, default=200,
                        help="Interval of updating the target models for reanalyzing (default: %(default)s)")

    groups = parser.add_argument_group("Test parameters")
    groups.add_argument("--test_interval", type=int, default=1000,
                        help="Interval of evaluation. (default: %(default)s)")
    groups.add_argument("--test_episodes", type=int, default=32,
                        help="Evaluation episode count (default: %(default)s)")
    groups.add_argument("--pretrained_model_path", type=str, default=None,
                        help="By default None, otherwise set the path to pretrained model.")
    groups.add_argument("--use_mcts_test", action="store_true", default=False,
                        help="By default False, otherwise conduct MCTS when test model.")

    groups = parser.add_argument_group("Save & Log parameters")
    groups.add_argument("--save_interval", type=int, default=10000,
                        help="Interval of models saving. (default: %(default)s)")
    groups.add_argument("--log_interval", type=int, default=100,
                        help="Interval of log printing. (default: %(default)s)")

    # Image & Augmentation parameters
    groups = parser.add_argument_group("Image & Augmentation parameters")
    groups.add_argument("--use_augmentation", action="store_true", default=False, help="use augmentation")
    groups.add_argument("--augmentation", type=str, default=["shift", "intensity"], nargs="+",
                        choices=["none", "rrc", "affine", "crop", "blur", "shift", "intensity"],
                        help="Style of augmentation")

    # Custom parameters
    groups = parser.add_argument_group("Custom parameters")
    groups.add_argument("--value_transform_type", type=str, default="vector", choices=["vector", "scalar"],
                        help="use Vectorization+CrossEntropyLoss or Scalar+MSELoss for value/reward prediction")
    groups.add_argument("--ppo_loss_proportion", type=float, default=0.5, help="proportion of ppo policy loss [0,1]. (default: %(default)s)")
    groups.add_argument("--stacked_observations", type=int, default=1, help="num of stacked observations. (default: %(default)s)")

    return parser.parse_args(args)


class DiscreteSupport(object):
    def __init__(self, min: int, max: int):
        self.min = min
        self.max = max
        self.range = np.arange(min, max + 1)
        self.size = len(self.range)


class BaseConfig(ABC):

    def __init__(self, args):

        # Prepare
        self.case = args.case
        self.env_name = args.env_name
        self.seed = args.seed
        self.discount = args.discount
        self.use_wandb = args.use_wandb
        # create experiment result path
        self.exp_path = os.path.join(
            args.result_dir, args.case, args.env_name, args.exp_name,
            'seed={}'.format(self.seed),
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        self.model_dir = os.path.join(self.exp_path, 'model')
        self.model_path = os.path.join(self.exp_path, 'model.p')
        # obtain game info
        game = self.new_game()
        self.num_agents = game.n_agents
        self.image_channel = 1
        self.obs_shape = (game.obs_size, 1, self.image_channel)
        self.action_space_size = game.action_space_size
        self.max_moves = game.get_max_episode_steps()
        self.test_max_moves = game.get_max_episode_steps()

        # Self-Play
        self.selfplay_on_gpu = args.selfplay_on_gpu and torch.cuda.is_available()
        self.data_actors = args.data_actors
        self.num_pmcts = args.num_pmcts
        self.checkpoint_interval = args.checkpoint_interval
        self.total_transitions = args.total_transitions
        self.start_transitions = args.start_transitions
        self.use_priority = args.use_priority
        self.priority_prob_alpha = 0.6
        self.priority_prob_beta = 0.4
        self.prioritized_replay_eps = 1e-6
        if not self.use_priority:
            self.priority_prob_alpha = 0.0
        self.use_max_priority = args.use_max_priority and self.use_priority
        self.use_change_temperature = args.use_change_temperature
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_annealing_time = args.eps_annealing_time

        # Priority Refresher
        self.use_priority_refresh = args.use_priority_refresh
        self.refresh_actors = args.refresh_actors
        self.refresh_interval = args.refresh_interval
        self.refresh_mini_size = args.refresh_mini_size

        # MCTS & UCB
        self.num_simulations = args.num_simulations
        self.pb_c_base = args.pb_c_base
        self.pb_c_init = args.pb_c_init
        self.tree_value_stat_delta_lb = args.tree_value_stat_delta_lb
        self.root_dirichlet_alpha = args.root_dirichlet_alpha
        self.root_exploration_fraction = args.root_exploration_fraction
        self.sampled_action_times = args.sampled_action_times
        self.mcts_rho = args.mcts_rho
        self.mcts_lambda = args.mcts_lambda

        # Training
        self.train_on_gpu = args.train_on_gpu and torch.cuda.is_available()
        self.training_steps = args.training_steps
        self.last_steps = args.last_steps
        self.batch_size = args.batch_size
        self.num_unroll_steps = args.num_unroll_steps
        self.max_grad_norm = args.max_grad_norm
        self.reward_loss_coeff = args.reward_loss_coeff
        self.value_loss_coeff = args.value_loss_coeff
        self.policy_loss_coeff = args.policy_loss_coeff
        self.consistency_coeff = args.consistency_coeff
        self.awac_lambda = args.awac_lambda
        self.adv_clip = args.adv_clip
        self.PG_type = args.PG_type

        # optimization control
        self.lr = args.lr   # type: float
        self.lr_warm_step = self.training_steps * 0.01
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = self.training_steps
        self.lr_adjust_func = args.lr_adjust_func
        if self.lr_adjust_func == "const":
            self.adjust_lr = lambda optimizer, step_count: self.lr
        elif self.lr_adjust_func == "linear":
            self.adjust_lr = self._linear
        elif self.lr_adjust_func == "cos":
            self.adjust_lr = self._cos
        else:
            raise Exception("Invalid --lr_adjust_func {} option".format(args.lr_adjust_func))
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        # Reanalyze config
        self.reanalyze_on_gpu = args.reanalyze_on_gpu and torch.cuda.is_available()
        self.reanalyze_actors = args.reanalyze_actors
        self.reanalyze_update_actors = args.reanalyze_update_actors
        self.td_steps = args.td_steps
        self.mini_infer_size = 64
        # choose target value type
        self.use_root_value = False
        self.use_pred_value = False
        self.use_reanalyze_value = False
        if args.target_value_type == "root":
            self.use_root_value = True
        elif args.target_value_type == "pred":
            self.use_pred_value = True
        elif args.target_value_type == "root-re":
            self.use_root_value = True
            self.use_reanalyze_value = True
        elif args.target_value_type == "pred-re":
            self.use_pred_value = True
            self.use_reanalyze_value = True
        else:
            raise Exception("Invalid --target_value_type {} option".format(args.target_value_type))
        self.revisit_policy_search_rate = args.revisit_policy_search_rate
        assert 0 <= self.revisit_policy_search_rate <= 1, 'Revisit policy search rate should be in [0,1]'
        self.use_off_correction = args.use_off_correction
        if not self.use_off_correction:
            self.auto_td_steps = self.training_steps
        else:
            self.auto_td_steps = self.training_steps * args.auto_td_steps_ratio
        self.target_model_interval = args.target_model_interval

        # testing arguments
        self.test_interval = args.test_interval
        self.test_episodes = args.test_episodes
        self.use_mcts_test = args.use_mcts_test

        # save & log
        self.save_interval = args.save_interval
        self.log_interval = args.log_interval

        ######################################################################

        # [Deprecated] Image & Augmentation
        self.image_based = False
        self.gray_scale = False
        self.cvt_string = False
        self.frame_skip = 1
        self.episode_life = False
        self.use_augmentation = False
        self.augmentation = args.augmentation

        # [Deprecated] PPO
        self.clip_param = 0.2
        self.gae_lambda = 0.95
        self.ppo_loss_proportion = 0

        ######################################################################

        # Network structure
        self.stacked_observations = args.stacked_observations

        self.hidden_state_size = 128
        self.fc_representation_layers = [128, 128]
        '''Define the hidden layers in the representation network'''
        self.fc_dynamic_layers = [128, 128]
        '''Define the hidden layers in the dynamic network'''
        self.fc_reward_layers = [32]
        '''Define the hidden layers in the reward head of the dynamic network'''
        self.fc_value_layers = [32]
        '''Define the hidden layers in the value head of the prediction network'''
        self.fc_policy_layers = [32]
        '''Define the hidden layers in the policy head of the prediction network'''

        # contrastive architecture
        self.proj_hid = 128
        self.proj_out = 128
        self.pred_hid = 64
        self.pred_out = 128

        # support of value/reward to represent the scalars
        self.value_support = DiscreteSupport(-300, 300)
        self.reward_support = DiscreteSupport(-300, 300)

        ######################################################################

        # custom config

        if args.value_transform_type == 'vector':
            self.use_vectorization = True
        elif args.value_transform_type == 'scalar':
            self.use_vectorization = False
            self.reward_support = DiscreteSupport(0, 0)
            self.value_support = DiscreteSupport(0, 0)
        else:
            assert False, 'key error: illegal: value_transform_type {}'.format(args.value_transform_type)

        self.start_transitions = max(self.start_transitions, self.batch_size)

    @abstractmethod
    def new_game(self, seed=None, **kwargs) -> Game:
        """ returns a new instance of the game"""
        raise NotImplementedError

    def visit_softmax_temperature_fn(self, trained_steps):
        if self.use_change_temperature:
            if trained_steps < 0.5 * (self.training_steps):
                return 1.0
            elif trained_steps < 0.75 * (self.training_steps):
                return 0.5
            else:
                return 0.25
        else:
            return 1.0

    def sampled_action_temperature_fn(self, trained_steps):
        return 1.0

    def eps_greedy_fn(self, trained_steps):
        return self.eps_start - (self.eps_start - self.eps_end) * min(trained_steps / self.eps_annealing_time, 1)

    @abstractmethod
    def get_uniform_network(self) -> BaseNet:
        raise NotImplementedError

    def set_augmentation_transforms(self):
        raise NotImplementedError

    def augmentation_transform(self, images):
        raise NotImplementedError

    def _h(self, x: torch.Tensor):
        """ Reference from Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)

        Reducing the scale using an invertible transform
        """
        epsilon = 0.001
        sign = torch.ones(x.shape).float().to(x.device)
        sign[x < 0] = -1.0
        output = sign * (torch.sqrt(torch.abs(x) + 1) - 1) + epsilon * x
        return output

    def _inv_h(self, x: torch.Tensor):
        """ Reference from Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
        """
        epsilon = 0.001
        sign = torch.ones(x.shape).float().to(x.device)
        sign[x < 0] = -1.0
        output = (((torch.sqrt(1 + 4 * epsilon * (torch.abs(x) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
        output = sign * output

        nan_part = torch.isnan(output)
        output[nan_part] = 0.
        output[torch.abs(output) < epsilon] = 0.
        return output

    def _phi(self, x: torch.Tensor, support: DiscreteSupport):
        """ Reference from MuZero: Appendix F => Network Architecture

        Transform a scalar to a categorical representation with (2 * support_size + 1) categories
        """
        min = support.min
        max = support.max
        set_size = support.size

        x = torch.clamp(x, min, max)
        floor = x.floor()
        ceil = x.ceil()
        prob = x - floor

        output = torch.zeros(*x.shape, set_size).to(x.device)
        output.scatter_(-1, (floor - min).long().unsqueeze(-1), (1 - prob).unsqueeze(-1))
        output.scatter_(-1, (ceil - min).long().unsqueeze(-1), prob.unsqueeze(-1))
        return output

    def _inv_phi(self, x: torch.Tensor, support: DiscreteSupport):
        """ Reference from MuZero: Appendix F => Network Architecture

        `x` must be a probabilty distribution on support.range
        """
        support = (
            torch.from_numpy(np.arange(support.min, support.max + 1))
            .expand(x.shape)
            .float()
            .to(x.device)
        )
        output = torch.sum(support * x, dim=-1, keepdim=True)
        return output

    def reward_transform(self, reward_scalars):
        if not self.use_vectorization:
            return reward_scalars
        return self._phi(self._h(reward_scalars), self.reward_support)

    def value_transform(self, value_scalars):
        if not self.use_vectorization:
            return value_scalars
        return self._phi(self._h(value_scalars), self.value_support)

    def inverse_reward_transform(self, rewards, use_logits=True):
        if not self.use_vectorization:
            return rewards
        if use_logits:
            rewards = torch.softmax(rewards, dim=-1)
        return self._inv_h(self._inv_phi(rewards, self.reward_support))

    def inverse_value_transform(self, values, use_logits=True):
        if not self.use_vectorization:
            return values
        if use_logits:
            values = torch.softmax(values, dim=-1)
        return self._inv_h(self._inv_phi(values, self.value_support))

    def _linear(self, optimizer, step_count) -> float:
        # adjust learning rate, step lr every lr_decay_steps
        if step_count < self.lr_warm_step:
            lr = self.lr * step_count / self.lr_warm_step
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = self.lr * self.lr_decay_rate ** ((step_count - self.lr_warm_step) // self.lr_decay_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        return lr

    def _cos(self, optimizer, step_count) -> float:
        # adjust learning rate using a cosine schedule
        total_training_step = self.training_steps + self.last_steps
        lr = self.lr / 2 * (1 + math.cos(math.pi * min(step_count / total_training_step, 1)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def reward_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.use_vectorization:
            return torch.nn.MSELoss(reduction='none')(prediction, target.unsqueeze(-1)).squeeze(-1)
        return -(torch.log_softmax(prediction, dim=-1) * target).sum(1)

    def value_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.use_vectorization:
            return torch.nn.MSELoss(reduction='none')(prediction, target.unsqueeze(-1)).squeeze(-1)
        return -(torch.log_softmax(prediction, dim=-1) * target).sum(1)

    def policy_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        # behavior cloning loss for prediction & target
        return -(torch.log_softmax(prediction, dim=-1) * target).sum(2).mean(1)

    def consistency_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        return -(
            torch.nn.functional.normalize(prediction, p=2., dim=-1, eps=1e-5)
            * torch.nn.functional.normalize(target, p=2., dim=-1, eps=1e-5)
        ).sum(1)

    def get_hparams(self):
        # get all the hyper-parameters
        hparams = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (int, float, tuple, list)):
                hparams[k] = v
        return hparams
