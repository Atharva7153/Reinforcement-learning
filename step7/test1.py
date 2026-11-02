import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

##NEURAL NETWORK STRUC
class DQN(nn.Module):
    def __init__(self, state_dim=2, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,action_dim)

        )

    def forward(self,x):
        return self.net(x)

action_names = ['left', 'right', 'up', 'down']

##Environmet
def step(state, action_idx):
    action = action_names[action_idx]
    
    if action == 'left':
        next_state = (state[0], state[1] - 1)
    elif action == 'right':
        next_state = (state[0], state[1] + 1)
    elif action == 'up':
        next_state = (state[0] + 1, state[1])
    else: # 'down'
        next_state = (state[0] - 1, state[1])
    # Out of bounds reset (your portal mechanic!)
    if next_state[0] < 0 or next_state[1] < 0 or next_state[0] > 2 or next_state[1] > 2:
        next_state = (0, 0)
    # Reached the Goal (2, 2)
    if next_state[0] == 2 and next_state[1] == 2:
        reward = 100.0
        done = True
    else:
        reward = -1.0
        done = False
    return next_state, reward, done


## NN HyperParameters

q_net = DQN()
target_net = DQN()

target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=0.001)
criterion = nn.MSELoss()

replay_buffer = deque(maxlen=5000)
batch_size = 32
gamma = 0.9
epsilon = 0.2

## Choose Action based on Epsilon
def choose_action(state, epsilon):

    if random.random() < epsilon:
        return random.randint(0,3)

    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32)
        q_values = q_net(state_t)

        return torch.argmax(q_values).item()


## Training
print("Traingin the Network :-")

for i in range(500):

    state = (0,0)
    done = False

    while not done:
        action = choose_action(state, epsilon)

        next_state, reward, done = step(state, action)

        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state

        if len(replay_buffer) >= batch_size:

            batch = random.sample(replay_buffer, batch_size)

            states      = torch.tensor([b[0] for b in batch], dtype=torch.float32)
            actions     = torch.tensor([b[1] for b in batch], dtype=torch.long)
            rewards     = torch.tensor([b[2] for b in batch], dtype=torch.float32)
            next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32)
            dones       = torch.tensor([b[4] for b in batch], dtype=torch.float32)

            q_values = q_net(states)
            predictions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():

                next_q = target_net(next_states).max(dim=1)[0]
                targets = rewards + gamma * next_q * (1.0 - dones)

            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if i % 10 == 0:
        target_net.load_state_dict(q_net.state_dict())


print("Training finished!\n")
# ---------------------------------------------------------
# 5. TEST: ASK THE TRAINED NETWORK WHAT TO DO AT EACH TILE!
# ---------------------------------------------------------
print("--- Trained Policy (Best Action from each Tile) ---")
grid_policy = []
for r in range(3):
    row_actions = []
    for c in range(3):
        if (r, c) == (2, 2):
            row_actions.append("GOAL")
            continue
        with torch.no_grad():
            st = torch.tensor([r, c], dtype=torch.float32)
            best_a = torch.argmax(q_net(st)).item()
            row_actions.append(action_names[best_a])
    grid_policy.append(row_actions)
for row in grid_policy:
    print(row)