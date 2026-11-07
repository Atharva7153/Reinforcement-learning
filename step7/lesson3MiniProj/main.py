import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from car_env import CarEnvironment

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.net(x)


env = CarEnvironment()

action_names = ['brake', 'acc', 'maintain']

state = env.reset()

q_net = DQN()
target_net = DQN()

target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=0.01)

loss_function = nn.MSELoss()

replay_buffer = deque(maxlen=5000)
batch_size = 32
gamma = 0.9
epsilon = 0.2

def choose_action(state, epsilon):

    if random.random() < epsilon:
        return random.randint(0,3)

    with torch.no_grad():

        state_t = torch.tensor(state, dtype=torch.float32)

        q_values = q_net(state_t)

        return torch.argmax(q_values).item()