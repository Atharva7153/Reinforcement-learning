
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
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.net(x)


env = CarEnvironment()

action_names = ['brake', 'acc', 'maintain']

q_net = DQN()
target_net = DQN()

target_net.load_state_dict(q_net.state_dict())
target_net.eval()

optimizer = optim.Adam(q_net.parameters(), lr=0.001)

loss_function = nn.MSELoss()

replay_buffer = deque(maxlen=5000)

batch_size = 32
gamma = 0.9
epsilon = 0.2

def choose_action(state, epsilon):

    if random.random() < epsilon:
        return random.randint(0, 2)

    with torch.no_grad():

        state_t = torch.tensor(
            state, dtype=torch.float32
        )

        q_values = q_net(state_t)

        return torch.argmax(q_values).item()



print("Training started...")

for i in range(10000):

    state = env.reset()
    done = False

    episode_losses = []

    while not done:

        action = choose_action(state, epsilon)

        reward, next_state, done = env.step(
            action_names[action]
        )

        replay_buffer.append(
            (state, action, reward, next_state, done)
        )

        state = next_state

        if len(replay_buffer) >= batch_size:

            batch = random.sample(
                replay_buffer, batch_size
            )

            states = torch.tensor(
                [b[0] for b in batch],
                dtype=torch.float32
            )

            actions = torch.tensor(
                [b[1] for b in batch],
                dtype=torch.long
            )

            rewards = torch.tensor(
                [b[2] for b in batch],
                dtype=torch.float32
            )

            next_states = torch.tensor(
                [b[3] for b in batch],
                dtype=torch.float32
            )

            dones = torch.tensor(
                [b[4] for b in batch],
                dtype=torch.float32
            )

            q_values = q_net(states)

            predictions = q_values.gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)

            with torch.no_grad():

                next_q = target_net(next_states).max(dim=1)[0]

                targets = rewards + gamma * next_q * (1 - dones)

            loss = loss_function(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_losses.append(loss.item())

    
    if i % 10 == 0:
        target_net.load_state_dict(q_net.state_dict())

    
    if i % 100 == 0:

        if episode_losses:
            avg_loss = sum(episode_losses) / len(episode_losses)
        else:
            avg_loss = 0

        print(
            f"Episode: {i} | "
            f"Average Loss: {avg_loss:.4f}"
        )

print("Training completed!")


print("\n========== TESTING TRAINED AGENT ==========\n")


epsilon = 0


for episode in range(10):

    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    print(f"\nEpisode {episode + 1}")
    print("--------------------------------")

    while not done:

        
        action = choose_action(state, epsilon)

        action_name = action_names[action]

        reward, next_state, done = env.step(action_name)

        total_reward += reward
        step_count += 1

        print(
            f"Step: {step_count} | "
            f"Velocity: {next_state[0]} | "
            f"Distance: {next_state[1]} | "
            f"Action: {action_name} | "
            f"Reward: {reward}"
        )

        state = next_state

    if env.distance <= env.buffer and env.initial_velocity == 0:
        print("\nRESULT: SUCCESS - Car stopped safely!")
    else:
        print("\nRESULT: FAILED - Car did not stop safely!")

    print("Total reward:", total_reward)
    print("Total steps:", step_count)

print("\n========== TESTING COMPLETED ==========")
