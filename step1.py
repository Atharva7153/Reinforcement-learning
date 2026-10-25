import random 

actions = ['left', 'right']
reward = 0

def environment(action):
      if(action == 'left'):
        return 1 if random.random() < 0.2 else 0
      else:
        return 1 if random.random() < 0.8 else 0

for i in range(10):
  action = random.choice(actions)
  reward += environment(action)
  print(f"Current Action : {action}")
  print(f"Current Reward : {reward}")
  print("---------------------------")

print("After Training : ")
print(f"Total Reward : {reward}")
