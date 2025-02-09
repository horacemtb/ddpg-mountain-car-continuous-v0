import torch
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self, n_observations, n_actions, fc_size=64):
        super(QNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_observations + n_actions, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, 1)
        )

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        return self.net(x)


class PolicyNetwork(nn.Module):

    def __init__(self, n_observations, n_actions, fc_size=64):
        super(PolicyNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_observations, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, n_actions)
        )

    def forward(self, state):
        x = self.net(state)
        return torch.tanh(x)
