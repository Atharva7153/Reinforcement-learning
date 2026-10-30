import random

Q = {
    0 : {'left': 0, 'right': 0},
    1 : {'left': 0, 'right': 0},
    2 : {'left': 0, 'right': 0},
    3 : {'left': 0, 'right': 0},
}

actions = ['left', 'right']

def choose_action(state, Q, epsilon=0.1):

    if random.random() < 0.1:
        action = random.choice(actions)
        return action
    else:
        if Q[state]['right'] > Q[state]['left']:
            return actions[1] #returns right
        else:
            return actions[0] #returns left


def step(state, action):

    if action == 'right':
        next_state = state + 1
    else:
        next_state = state -1

    if next_state < 0:
        next_state = 0

    if next_state == 3:
        reward = 10
        done = True
    else:
        reward = -1
        done = False

    return next_state, reward, done

alpha = 0.1
gamma = 0.9
rewards = []

for i in range(200):

    done = False
    state = 0

    while not done:

        action = choose_action(state, Q)

        next_state, reward, done = step(state, action)
        rewards.append(reward)

        if done:
            best_next = 0
        else:
            best_next = max(Q[next_state].values())

        target = reward + gamma * best_next
        old_val = Q[state][action]

        Q[state][action] = old_val + (alpha * (target - old_val))

        state = next_state

        # print("State : ", state)
        # print("Done : ", done)


print("After training : ")
for key, value in Q.items():
    print(f"{key}: left = {value['left']}, right = {value['right']}")