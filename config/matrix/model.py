from typing import List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

from core.model import BaseNet, NetworkOutput, Action, HiddenState


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def mlp(
    input_size: int,
    layer_sizes: List[int],
    output_size: int,
    use_orthogonal: bool = True,
    use_ReLU: bool = True,
    use_value_out: bool = False,
):
    """MLP layers

    Parameters
    ----------
    input_size: int
        dim of inputs
    layer_sizes: list
        dim of hidden layers
    output_size: int
        dim of outputs
    """

    active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
    init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
    gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

    def init_(m):
        return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        layers += [init_(nn.Linear(sizes[i], sizes[i + 1])), active_func, nn.LayerNorm(sizes[i + 1])]

    if use_value_out:
        layers = layers[:-2]

    return nn.Sequential(*layers)


class RepresentationNetwork(nn.Module):
    def __init__(
        self,
        observation_size: int,
        hidden_state_size: int,
        fc_representation_layers: List[int],
        use_feature_norm: bool = True,
    ):
        """Representation network: Encode the observations into hidden states

        Parameters
        ----------
        observation_size: int
            dim of observations
        hidden_state_size: int
            dim of hidden states
        fc_representation_layers: list
            hidden layers of the representation function (obs -> state)
        """
        super().__init__()
        self.use_feature_norm = use_feature_norm
        if self.use_feature_norm:
            self.feature_norm = nn.LayerNorm(observation_size)
        self.mlp = mlp(observation_size, fc_representation_layers, hidden_state_size)

    def forward(self, x):
        if self.use_feature_norm:
            x = self.feature_norm(x)
        x = self.mlp(x)
        return x


class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        concat_hidden_state_size: int,
        action_space_size: int,
        fc_dynamic_layers: List[int],
        fc_reward_layers: List[int],
        full_support_size: int,
    ):
        """Dynamics network: Predict next hidden states given current states and actions

        Parameters
        ----------
        concat_hidden_state_size: int
            dim of concat hidden states
        action_space_size: int
            action space * num_agents
        fc_dynamic_layers: list
            hidden layers of the state transition (state+action -> state)
        fc_reward_layers: list
            hidden layers of the reward prediction head (MLP head)
        full_support_size: int
            dim of reward output
        """
        super().__init__()
        # NOTE: here the concat_hidden_state_size equals to num_agents * hidden_state_size!
        self.concat_hidden_state_size = concat_hidden_state_size

        concat_size = self.concat_hidden_state_size + action_space_size
        self.fc_dynamic = mlp(concat_size, fc_dynamic_layers, concat_hidden_state_size)
        self.fc_reward = mlp(concat_size, fc_reward_layers, full_support_size, use_value_out=True)

    def forward(self, concat_hidden_state, action):
        pre_state = concat_hidden_state

        state = self.fc_dynamic(torch.cat([concat_hidden_state, action], dim=1))
        state += pre_state

        reward = self.fc_reward(torch.cat([state, action], dim=1))

        return state, reward


class PredictionNetwork(nn.Module):
    def __init__(
        self,
        num_agents: int,
        hidden_state_size: int,
        action_space_size: int,
        fc_value_layers: List[int],
        fc_policy_layers: List[int],
        full_support_size: int,
    ):
        """Prediction network: predict the value and policy given hidden states

        Parameters
        ----------
        num_agents: int
            number of agents
        hidden_state_size: int
            dim of hidden states
        action_space_size: int
            action space
        fc_value_layers: list
            hidden layers of the value prediction head (MLP head)
        fc_policy_layers: list
            hidden layers of the policy prediction head (MLP head)
        full_support_size: int
            dim of value output
        """
        super().__init__()
        self.num_agents = num_agents
        self.hidden_state_size = hidden_state_size
        self.action_space_size = action_space_size
        self.fc_value = mlp(hidden_state_size * num_agents, fc_value_layers, full_support_size, use_value_out=True)
        self.fc_policy = mlp(hidden_state_size, fc_policy_layers, action_space_size, use_value_out=True)

    def forward(self, x):
        value = self.fc_value(x)
        policy_logit = self.fc_policy(x.reshape(-1, self.hidden_state_size)).reshape(-1, self.num_agents, self.action_space_size)
        return policy_logit, value


class ProjectionNetwork(nn.Module):
    def __init__(
        self,
        porjection_in_dim: int,
        proj_hid: int = 256,
        proj_out: int = 256,
        pred_hid: int = 64,
        pred_out: int = 256,
    ):
        """Projection network: SimSiam self-supervised framework

        Parameters
        ----------
        porjection_in_dim : int
            dim of the projection input
        proj_hid : int, optional
            dim of projection hidden layer, by default 256
        proj_out : int, optional
            dim of projection output layer, by default 256
        pred_hid : int, optional
            dim of projection head (prediction) hidden layer, by default 64
        pred_out : int, optional
            dim of projection head (prediction) output layer, by default 256
        """
        super().__init__()
        self.porjection_in_dim = porjection_in_dim
        self.projection = mlp(self.porjection_in_dim, [proj_hid], proj_out)
        self.projection_norm = nn.LayerNorm(proj_out)
        self.prediction = mlp(proj_out, [pred_hid], pred_out)

    def project(self, hidden_state: torch.Tensor) -> torch.Tensor:
        proj = self.projection(hidden_state)
        proj = self.projection_norm(proj)
        return proj

    def predict(self, projection: torch.Tensor) -> torch.Tensor:
        return self.prediction(projection)


class MAMuZeroNet(BaseNet):
    def __init__(
        self,
        num_agents: int,
        observation_shape: Tuple[int, int, int],
        action_space_size: int,
        hidden_state_size: int,
        fc_representation_layers: List[int],
        fc_dynamic_layers: List[int],
        fc_reward_layers: List[int],
        fc_value_layers: List[int],
        fc_policy_layers: List[int],
        reward_support_size: int,
        value_support_size: int,
        inverse_value_transform: Any,
        inverse_reward_transform: Any,
        proj_hid: int = 256,
        proj_out: int = 256,
        pred_hid: int = 64,
        pred_out: int = 256,
        **kwargs
    ):
        """Centralized MuZero network

        Parameters
        ----------
        num_agents: int
            num of agents = n
        observation_shape : Tuple[int, int, int]
            shape of observations: [C, W, H]
        action_space_size : int
            action space for each agent
        hidden_state_size : int
            dim of hidden states
        fc_representation_layers : List[int]
            hidden layers of the representation function (obs -> state)
        fc_dynamic_layers : List[int]
            hidden layers of the state transition (state+action -> state)
        fc_reward_layers : List[int]
            hidden layers of the reward prediction head (MLP head)
        fc_value_layers : List[int]
            hidden layers of the value prediction head (MLP head)
        fc_policy_layers : List[int]
            hidden layers of the policy prediction head (MLP head)
        reward_support_size : int
            dim of reward output
        value_support_size : int
            dim of value output
        inverse_value_transform: Any
            A function that maps value supports into value scalars
        inverse_reward_transform: Any
            A function that maps reward supports into value scalars
        proj_hid : int, optional
            dim of projection hidden layer, by default 256
        proj_out : int, optional
            dim of projection output layer, by default 256
        pred_hid : int, optional
            dim of projection head (prediction) hidden layer, by default 64
        pred_out : int, optional
            dim of projection head (prediction) output layer, by default 256
        """
        super(MAMuZeroNet, self).__init__(inverse_value_transform, inverse_reward_transform)
        self.use_feature_norm = kwargs["use_feature_norm"]

        self.num_agents = num_agents
        self.obs_size = np.prod(observation_shape)
        self.action_space_size = action_space_size
        self.hidden_state_size = hidden_state_size

        self.representation_network = RepresentationNetwork(
            self.obs_size,
            hidden_state_size,
            fc_representation_layers,
            self.use_feature_norm
        )

        self.dynamics_network = DynamicsNetwork(
            self.num_agents * hidden_state_size,
            self.num_agents * action_space_size,
            fc_dynamic_layers,
            fc_reward_layers,
            reward_support_size,
        )

        self.prediction_network = PredictionNetwork(
            num_agents,
            hidden_state_size,
            action_space_size,
            fc_value_layers,
            fc_policy_layers,
            value_support_size,
        )

        self.projection_network = ProjectionNetwork(
            hidden_state_size * self.num_agents,
            proj_hid,
            proj_out,
            pred_hid,
            pred_out,
        )

    def prediction(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy_logit, value = self.prediction_network(hidden_state)
        return policy_logit, value

    def representation(self, observation: torch.Tensor) -> torch.Tensor:
        batch_size = observation.shape[0]
        hidden_state = self.representation_network(
            observation.reshape(batch_size * self.num_agents, self.obs_size)  # flatten the image-format vector obs
        ).reshape(batch_size, self.num_agents * self.hidden_state_size)
        return hidden_state

    def dynamics(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_state.shape[0]
        # the input action tensor must be deterministic index for each agent.
        action_onehot = (
            torch.zeros((batch_size * self.num_agents, self.action_space_size))
            .to(action.device)
            .float()
        )
        action_onehot.scatter_(1, action.reshape(batch_size * self.num_agents, 1).long(), 1.0)

        next_hidden_state, reward = self.dynamics_network(hidden_state, action_onehot.view(batch_size, -1))

        return next_hidden_state, reward

    def project(self, hidden_state: torch.Tensor, with_grad: bool = True) -> torch.Tensor:
        proj = self.projection_network.project(hidden_state)

        if with_grad:
            # only the branch of proj + pred can share the gradients
            proj = self.projection_network.predict(proj)
            return proj
        else:
            return proj.detach()

    def initial_inference(self, observation: torch.Tensor) -> NetworkOutput:
        # observation: (batch_size, ...)
        batch_size = observation.size(0)

        hidden_state = self.representation(observation)
        policy_logit, value = self.prediction(hidden_state)

        # if not in training, obtain the scalars of the value/reward
        if not self.training:
            value = self.inverse_value_transform(value).detach().cpu().numpy()
            policy_logit = policy_logit.detach().cpu().numpy()

        return NetworkOutput(hidden_state, np.zeros((batch_size, 1)), value, policy_logit)

    def recurrent_inference(self, hidden_state: HiddenState, action: Action) -> NetworkOutput:
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy_logit, value = self.prediction(next_hidden_state)

        # if not in training, obtain the scalars of the value/reward
        if not self.training:
            reward = self.inverse_reward_transform(reward).detach().cpu().numpy()
            value = self.inverse_value_transform(value).detach().cpu().numpy()
            policy_logit = policy_logit.detach().cpu().numpy()

        return NetworkOutput(next_hidden_state, reward, value, policy_logit)
