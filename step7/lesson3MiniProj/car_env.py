import random

class CarEnvironmen:
    def __init__(self):
        self.initial_velocity = random.randint(1, 15) # meter per second
        self.distance = 100 #100 meters
        self.buffer = 5 # 5 meters before the wall to stop
        self.time = 0

    def reset(self):
        self.initial_velocity = random.randint(1, 15) 
        self.distance = 100 
        self.buffer = 5
        self.time = 0

    def step(self, action):
        if action == 'brake':
            velocity = self.initial_velocity - 5
            self.initial_velocity = velocity

        if action == "acc":
            velocity = self.initial_velocity + 5
            self.initial_velocity = velocity

        if action == 'maintain':
            velocity = self.initial_velocity
            self.initial_velocity = velocity

        if self.initial_velocity < 0 :
            self.initial_velocity = 0
            velocity = 0

        self.time += 1

        self.distance = velocity * self.time

        #winning condition

        if(self.distance <= self.buffer and self.initial_velocity == 0):
            reward = 100
            done = True
        else:
            reward = -1
            done = False


