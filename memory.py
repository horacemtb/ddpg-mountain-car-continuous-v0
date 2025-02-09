import torch
import random
from collections import namedtuple, deque


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer:

    def __init__(self, memory_length, device):
        self.memory = deque([], maxlen=memory_length)
        self.device = device

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):

        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        return state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask

    def __len__(self):
        return len(self.memory)
