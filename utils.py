import torch
import numpy as np
from agent import DDPGAgent
import os


def running_mean(x, N=100):
    x = np.array(x)
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y


def save_agent(agent, folder='model_weights', filename='ddpg_agent'):

    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    torch.save({
        'actor_state_dict': agent.mu_origin_model.state_dict(),
        'critic_state_dict': agent.q_origin_model.state_dict(),
        'actor_target_state_dict': agent.mu_target_model.state_dict(),
        'critic_target_state_dict': agent.q_target_model.state_dict(),
        'actor_optimizer_state_dict': agent.opt_mu.state_dict(),
        'critic_optimizer_state_dict': agent.opt_q.state_dict(),
    }, f"{filepath}.pth")

    print(f'Agent saved to {filepath}.pth')


def load_agent(agent, filepath='model_weights/ddpg_agent.pth', device='cpu'):

    checkpoint = torch.load(filepath, map_location=torch.device(device))

    agent.mu_origin_model.load_state_dict(checkpoint['actor_state_dict'])
    agent.q_origin_model.load_state_dict(checkpoint['critic_state_dict'])
    agent.mu_target_model.load_state_dict(checkpoint['actor_target_state_dict'])
    agent.q_target_model.load_state_dict(checkpoint['critic_target_state_dict'])
    
    agent.opt_mu.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    agent.opt_q.load_state_dict(checkpoint['critic_optimizer_state_dict'])

    print("Agent's policy and critic networks loaded")