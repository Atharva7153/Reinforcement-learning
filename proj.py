import random

actions = ['left', 'right', 'up', 'down']

Q = {
    (0,0) : {'left' : 0, 'right' : 0, 'up' : 0, 'down' : 0},
    (0,1) : {'left' : 0, 'right' : 0, 'up' : 0, 'down' : 0},
    (0,2) : {'left' : 0, 'right' : 0, 'up' : 0, 'down' : 0},
    (1,0) : {'left' : 0, 'right' : 0, 'up' : 0, 'down' : 0},
    (1,1) : {'left' : 0, 'right' : 0, 'up' : 0, 'down' : 0},
    (1,2) : {'left' : 0, 'right' : 0, 'up' : 0, 'down' : 0},
    (2,0) : {'left' : 0, 'right' : 0, 'up' : 0, 'down' : 0},
    (2,1) : {'left' : 0, 'right' : 0, 'up' : 0, 'down' : 0},
    (2,2) : {'left' : 0, 'right' : 0, 'up' : 0, 'down' : 0},
}

alpha = 0.1
gamma = 0.9
epsilon = 0.1


def choose_action(state, Q, epsilon = 0.1):

    if random.random() < epsilon:
        action = random.choice(actions)
        return action
    return max(Q[state], key=Q[state].get)

def step(state, action):

    if action == 'left':
        next_state = (state[0], state[1] - 1)
    elif action == 'right':
        next_state = (state[0], state[1] + 1)
    elif action == 'up':
        next_state = (state[0] + 1, state[1])
    else : #Action == down
        next_state = (state[0] - 1, state[1])

    if(next_state[0] < 0 or next_state[1] < 0):
        next_state = (0,0)

    if(next_state[0] > 2 or next_state[1] > 2):
        next_state = (0,0)

    if (next_state[0] == 2 and next_state[1] == 2): #Goal
        reward = 100
        done = True
    else:
        reward = -1
        done = False

    return next_state, reward, done

#Training
count = 0
for i in range(1000):

    done = False
    state = (0,0)

    while not done:

        action = choose_action(state, Q, epsilon)

        # print(action)
        

        next_state, reward, done = step(state, action)

        if done:
            best_next = 0
        else:
            best_next = max(Q[next_state].values())

        target = reward + (gamma * best_next)
        target = round(target, 3)

        old_val = Q[state][action]
        old_val = round(old_val, 3)

        Q[state][action] = round(old_val + (alpha * (target - old_val)), 3)

        state = next_state

        # print(next_state)

    count += 1
    print(count)


for key, value in Q.items():
    print(f"{key} : {value}")