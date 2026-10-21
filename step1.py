
import random

actions = ['left', 'right']

def environment(action):
    if action == 'left':                        
        return 1 if random.random() < 0.2 else 0
    else:
        return 1 if random.random() < 0.8 else 0

reward_count = {'left': 0, 'right': 0}
action_count = {'left': 0, 'right': 0}

epsilon = 0.1
total_reward = 0

for i in range(10):

    
    values = {}

    for action in actions:
        if action_count[action] == 0:
            values[action] = 0
        else:
            values[action] = (
                reward_count[action] / action_count[action]
            )


    if random.random() < epsilon:
        action = random.choice(actions)
    else:
        best_value = max(values.values())

        

        
        best_actions = [
            a for a in actions
            if values[a] == best_value
        ]

        action = random.choice(best_actions)

    reward = environment(action)

    action_count[action] += 1
    reward_count[action] += reward
    total_reward += reward

print("Training completed!")
print("Total reward:", total_reward)

for action in actions:
    estimated_value = (
        reward_count[action] / action_count[action]
    )

    print(
        action,
        "Selected:", action_count[action],
        "times | Estimated value:",
        round(estimated_value, 3)
    )
