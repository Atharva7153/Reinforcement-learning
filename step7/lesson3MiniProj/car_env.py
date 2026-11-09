
import random

class CarEnvironment:

    def __init__(self):
        self.reset()

    def reset(self):
        self.initial_velocity = random.randint(1, 15)
        self.distance = 100
        self.buffer = 5
        self.time = 0
        self.distance_travelled = 0

        return (self.initial_velocity, self.distance)

    def step(self, action):

        if action == 'brake':
            self.initial_velocity -= 5

        elif action == 'acc':
            self.initial_velocity += 5

        elif action == 'maintain':
            pass

        else:
            raise ValueError("Invalid action")

        self.initial_velocity = max(0, self.initial_velocity)

        velocity = self.initial_velocity

        self.distance_travelled = velocity
        self.distance -= self.distance_travelled

        self.time += 1

        if self.distance <= 0:
            reward = -100
            done = True

        elif self.distance <= self.buffer and velocity == 0:
            reward = 100
            done = True

        elif velocity == 0 and self.distance > self.buffer:
            reward = -50
            done = True

        else:
            reward = -1
            done = False

        return reward, (velocity, self.distance), done
