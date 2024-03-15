from typing import List, Union, Tuple, NamedTuple

import torch
import numpy as np
import torch.nn as nn


DefaultDataType = Union[np.ndarray, torch.Tensor]

Action = DefaultDataType
HiddenState = torch.Tensor


class NetworkOutput(NamedTuple):
    # output format of the model
    hidden_state: HiddenState
    reward: DefaultDataType
    value: DefaultDataType
    policy_logits: DefaultDataType


def concat_output_value(output_lst: List[NetworkOutput]):
    # concat the values of the model output list
    value_lst = []
    for output in output_lst:
        value_lst.append(output.value)
    value_lst = np.concatenate(value_lst)
    return value_lst


def concat_output(output_lst: List[NetworkOutput]):
    # concat the model output
    hidden_state_lst, reward_lst, value_lst, policy_logits_lst = [], [], [], []
    for output in output_lst:
        hidden_state_lst.append(output.hidden_state)
        reward_lst.append(output.reward)
        value_lst.append(output.value)
        policy_logits_lst.append(output.policy_logits)
    hidden_state_lst = np.concatenate(hidden_state_lst)
    reward_lst = np.concatenate(reward_lst)
    value_lst = np.concatenate(value_lst)
    policy_logits_lst = np.concatenate(policy_logits_lst)
    return NetworkOutput(hidden_state_lst, reward_lst, value_lst, policy_logits_lst)


class BaseNet(nn.Module):
    def __init__(self, inverse_value_transform, inverse_reward_transform):
        """Base Network

        Parameters
        ----------
        inverse_value_transform: Any
            A function that maps value supports into value scalars
        inverse_reward_transform: Any
            A function that maps reward supports into value scalars
        """
        super(BaseNet, self).__init__()
        self.inverse_value_transform = inverse_value_transform
        self.inverse_reward_transform = inverse_reward_transform

    def representation(self, observation: torch.Tensor) -> HiddenState:
        raise NotImplementedError

    def prediction(self, hidden_state: HiddenState) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def dynamics(self, hidden_state: HiddenState, action: Action) -> Tuple[HiddenState, torch.Tensor]:
        raise NotImplementedError

    def project(self, hidden_state: HiddenState, with_grad: bool = True) -> torch.Tensor:
        raise NotImplementedError

    def initial_inference(self, observation: torch.Tensor) -> NetworkOutput:
        raise NotImplementedError

    def recurrent_inference(self, hidden_state: HiddenState, action: Action) -> NetworkOutput:
        raise NotImplementedError

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)


def renormalize(tensor, first_dim=1):
    # normalize the tensor (states)
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min) / (max - min)

    return flat_tensor.view(*tensor.shape)
