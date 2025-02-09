import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from memory import ReplayBuffer
from model import QNetwork, PolicyNetwork
import numpy as np


class DDPGAgent():
    def __init__(self,
                  env,
                  device,
                  fc_size=64,
                  batch_size=128,
                  gamma=0.99,
                  q_rl=0.005,
                  mu_rl=0.005,
                  tau=0.002,
                  memory_length=20000,
                  mu=np.zeros(1),
                  sigma=np.ones(1)*0.3,
                  theta=0.15,
                  dt=1e-1,
                  x0=None):

        self.env = env
        self.state_n = env.observation_space.shape[0]
        self.action_n = env.action_space.shape[0]
        self.device = device
        self.fc_size = fc_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.q_rl = q_rl
        self.mu_rl = mu_rl
        self.tau = tau
        self.memory_length = memory_length
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0

        self.q_origin_model = QNetwork(self.state_n, self.action_n, fc_size).to(device)
        self.q_target_model = QNetwork(self.state_n, self.action_n, fc_size).to(device)
        _ = self.q_target_model.requires_grad_(False)

        self.mu_origin_model = PolicyNetwork(self.state_n, self.action_n, fc_size).to(device)
        self.mu_target_model = PolicyNetwork(self.state_n, self.action_n, fc_size).to(device)
        _ = self.mu_target_model.requires_grad_(False)

        self.opt_q = optim.AdamW(self.q_origin_model.parameters(), lr=q_rl)
        self.opt_mu = optim.AdamW(self.mu_origin_model.parameters(), lr=mu_rl)

        self.memory = ReplayBuffer(memory_length, device)

        # copied from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
        self.ou_action_noise = OrnsteinUhlenbeckActionNoise(mu=mu,
                                                            sigma=sigma,
                                                            theta=theta,
                                                            dt=dt
                                                            )

    def choose_action(self, state, add_noise=True):

        with torch.no_grad():
            state = np.array(state)
            state_batch = np.expand_dims(state, axis=0)
            state_batch = torch.tensor(state_batch, dtype=torch.float).to(self.device)
            action_det = self.mu_origin_model(state_batch)
            action_det = action_det.squeeze(dim=1)

            if add_noise:
                noise = self.ou_action_noise()
                action_det += noise

            action = np.clip(action_det.cpu().numpy(), -1.0, 1.0)
            return float(action.item())

    def update_target(self, q_origin_model, q_target_model, mu_origin_model, mu_target_model):

        for var, var_target in zip(q_origin_model.parameters(), q_target_model.parameters()):
            var_target.data = self.tau * var.data + (1.0 - self.tau) * var_target.data
        for var, var_target in zip(mu_origin_model.parameters(), mu_target_model.parameters()):
            var_target.data = self.tau * var.data + (1.0 - self.tau) * var_target.data

    def learn(self, s, a, r, s_next, done):

        s = torch.tensor([s], dtype=torch.float, device=self.device)
        a = torch.tensor([[a]], dtype=torch.float, device=self.device)
        r = torch.tensor([[r]], dtype=torch.float, device=self.device)
        s_next = torch.tensor([s_next], dtype=torch.float, device=self.device) if not done else None

        self.memory.push(s, a, s_next, r)

        if len(self.memory) > self.batch_size:

            states, actions, rewards, next_states, non_final_mask = self.memory.sample(self.batch_size)

            self.opt_q.zero_grad()
            q_org = self.q_origin_model(states, actions)

            mu_tgt_next = self.mu_target_model(next_states)
            q_tgt_next = torch.zeros(self.batch_size, 1, device=self.device)
            q_tgt_next[non_final_mask] = self.q_target_model(next_states, mu_tgt_next)

            q_tgt = rewards + self.gamma * q_tgt_next
            loss_q = F.mse_loss(q_org, q_tgt)
            loss_q.backward()
            self.opt_q.step()

            self.opt_mu.zero_grad()
            mu_org = self.mu_origin_model(states)
            for p in self.q_origin_model.parameters():
                p.requires_grad = False
            q_tgt_max = self.q_origin_model(states, mu_org)
            (-q_tgt_max).mean().backward()
            self.opt_mu.step()
            for p in self.q_origin_model.parameters():
                p.requires_grad = True

            self.update_target(self.q_origin_model, self.q_target_model, self.mu_origin_model, self.mu_target_model)


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
