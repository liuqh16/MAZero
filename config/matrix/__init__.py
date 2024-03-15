from core.config import BaseConfig, DiscreteSupport
from .model import MAMuZeroNet
from .matgame import MatgameEnv
from .env_wrapper import MatrixWrapper


class GameConfig(BaseConfig):
    def __init__(self, args):
        super(GameConfig, self).__init__(args)

        self.hidden_state_size = 64
        self.fc_representation_layers = [64, 64]
        self.fc_dynamic_layers = [64, 64]
        self.fc_reward_layers = [32]
        self.fc_value_layers = [32]
        self.fc_policy_layers = [32]
        self.proj_hid = 128
        self.proj_out = 128
        self.pred_hid = 64
        self.pred_out = 128

        if self.use_vectorization:
            self.value_support = DiscreteSupport(-10, 10)
            self.reward_support = DiscreteSupport(-3, 3)

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
            use_feature_norm=False
        )

    def new_game(self, seed=None, **kwargs):
        env = MatgameEnv(map_name=self.env_name, seed=seed)
        return MatrixWrapper(env)
