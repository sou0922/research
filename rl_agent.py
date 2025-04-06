import random
import numpy as np

class RLAgent:
    def __init__(self, num_clients=3):
        self.q_table = np.zeros((num_clients, num_clients))
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.05
        self.alpha = 0.5
        self.gamma = 0.9

    def select_action(self, current_client):
        if random.random() < self.epsilon:
            action = random.randint(0, 2)
        else:
            action = np.argmax(self.q_table[current_client])

        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        return action

    def update(self, current_client, reward):
        next_best = np.max(self.q_table[current_client])
        self.q_table[current_client][current_client] += self.alpha * (reward + self.gamma * next_best - self.q_table[current_client][current_client])
