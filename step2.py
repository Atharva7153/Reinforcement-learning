import random

actions = ['left', 'right']

def environment(action):
    if action == 'left':
        return 1 if random.random() < 0.2 else 0
    else:
        return 1 if random.random() < 0.8 else 0

reward_count = {'left': 0, 'right': 0}
action_count = {'left': 0, 'right': 0}

for i in range(1000):

    action = random.choice(actions)

    reward = environment(action)

    action_count[action] += 1
    reward_count[action] += reward

print("Results after training:")

for action in actions:
    value = reward_count[action] / action_count[action]
    print(action, "Estimated value:", round(value, 3))