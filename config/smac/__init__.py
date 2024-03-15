from core.config import BaseConfig, DiscreteSupport
from .model import MAMuZeroNet
from .mappo_smac.StarCraft2_Env import StarCraft2Env
from .env_wrapper import SMACWrapper
from absl import flags
import numpy as np
FLAGS = flags.FLAGS
FLAGS(['train_sc.py'])


class GameConfig(BaseConfig):
    def __init__(self, args):
        super(GameConfig, self).__init__(args)

        self.hidden_state_size = 128
        self.fc_representation_layers = [128, 128]
        self.fc_dynamic_layers = [128, 128]
        self.fc_reward_layers = [32]
        self.fc_value_layers = [32]
        self.fc_policy_layers = [32]
        self.proj_hid = 128
        self.proj_out = 128
        self.pred_hid = 64
        self.pred_out = 128

        if self.use_vectorization:
            self.value_support = DiscreteSupport(-5, 5)
            self.reward_support = DiscreteSupport(-5, 5)

    def get_uniform_network(self):
        return MAMuZeroNet(
            self.num_agents,
            (self.stacked_observations, *self.obs_shape),
            self.action_space_size,
            self.hidden_state_size,
            self.fc_representation_layers,
            self.fc_dynamic_layers,
            self.fc_reward_layers,
            self.fc_value_layers,
            self.fc_policy_layers,
            self.reward_support.size,
            self.value_support.size,
            self.inverse_value_transform,
            self.inverse_reward_transform,
            proj_hid=self.proj_hid,
            proj_out=self.proj_out,
            pred_hid=self.pred_hid,
            pred_out=self.pred_out,
            use_feature_norm=True,
        )

    def new_game(self, seed=None, save_video=False, **kwargs):
        args = type("Config", (object,), {
            "map_name": self.env_name,
            "use_stacked_frames": False,
            "stacked_frames": 1,
            "add_local_obs": False,
            "add_move_state": False,
            "add_visible_state": False,
            "add_distance_state": False,
            "add_xy_state": False,
            "add_enemy_action_state": False,
            "add_agent_id": False,
            "use_state_agent": False,
            "use_mustalive": True,
            "add_center_xy": True,
            "use_obs_instead_of_state": False,
        })
        env = StarCraft2Env(args, seed=seed, replay_dir=self.exp_path)
        return SMACWrapper(env, save_video=save_video)
