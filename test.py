def compute_return (gamma, rewards):

    total_return = 0

    for steps, r in enumerate(rewards):

        total_return += r * (gamma ** steps)

    return total_return


actions = ['left', 'right']

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


gamma = 0.9
done = False
state = 0
rewards = []

while not done:

    state, reward, done = step(state, 'right')
    rewards.append(reward)

    print("State : ", state)
    print("Done : ", done)


print(*rewards, sep=", ")


total_return = compute_return(gamma, rewards)
print("Total Return : ", total_return)

