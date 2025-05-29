import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

MAX_COMM_TIME = 20

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 畳み込み層を増やし、BatchNormも追加
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        # 入力画像サイズが32x32の場合
        # 1回目pool後: 16x16, 2回目pool後: 8x8
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.reshape(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class SimpleAgent:
    def __init__(self, epsilon=1.0, epsilon_min=0.01, decay_type="linear", decay_param=0.001, max_hops=10):
        self.q_table = {("init", "init"): 0.0}
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_type = decay_type
        self.decay_param = decay_param
        self.step = 0
        self.max_hops = max_hops
        self.transition_counts = {}  # ノードごとの遷移回数を記録

    def get_action(self, state, actions):
        choice = ""
        # ペナルティ係数（大きいほど未遷移ノードを優先）
        penalty_coef = 0.05
        if np.random.rand() < self.epsilon:
            choice = "ランダム選択"
            action = random.choice(actions)
        else:
            q_values = []
            for a in actions:
                q = self.q_table.get((state, a), 0.0)
                count = self.transition_counts.get(a, 0)
                penalty = penalty_coef * count
                q_values.append(q - penalty)
            max_q = max(q_values)
            best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
            action = random.choice(best_actions)
            choice = "最適行動選択"
        # 遷移回数を記録
        self.transition_counts[action] = self.transition_counts.get(action, 0) + 1
        return action, choice

    def update(self, state, action, reward, next_state, alpha=0.1, gamma=0.9):
        key = (state, action)
        next_qs = [self.q_table.get((next_state, a), 0.0) for a in self.q_table if a[0] == next_state]
        max_next_q = max(next_qs) if next_qs else 0.0
        old_q = self.q_table.get(key, 0.0)
        self.q_table[key] = old_q + alpha * (reward + gamma * max_next_q - old_q)
        self.step += 1
    
    def calc_reward(self, acc_before, acc_after, comm_time):
        acc_w, comm_w = self.get_weights(self.step, self.max_hops)
        acc_w = comm_w = 1.0
        acc_diff = acc_after - acc_before
        acc_penalty = min(0, acc_diff * 10)  # 精度向上のペナルティ
        comm_norm = min(comm_time / MAX_COMM_TIME, 1.0) * 10
        # comm_timeが5秒以上のとき、超過分にペナルティ
        if comm_time >= 5.0:
            comm_penalty = comm_time - 5.0
        else:
            comm_penalty = 0.0
        return acc_w * (acc_after + acc_penalty) - comm_w * (comm_norm - comm_penalty)
    
    def get_weights(self, t, t_max):
        # 精度重視→通信重視の変化幅を拡大
        acc_w = 1.0 - 0.6 * (t / t_max)
        comm_w = 0.0 + 0.4 * (t / t_max)
        return acc_w, comm_w

    def decay_epsilon(self):
        if self.decay_type == "linear":
            T = self.max_hops
            k = self.step
            if k <= 3 * T // 4:
                decay = (self.epsilon - self.epsilon_min) / (3 * T // 4)
                self.epsilon = max(self.epsilon_min, self.epsilon - decay)
            else:
                self.epsilon = 0.0
            print(f"[INFO] ε減衰: {self.epsilon:.4f} (step: {self.step})", flush=True)
        elif self.decay_type == "exponential":
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_param)
        elif self.decay_type == "inverse":
            self.epsilon = max(self.epsilon_min, 1.0 / (1.0 + self.decay_param * self.step))
        return self.epsilon

# 報酬関数が精度のみを考慮する場合
class SimpleAccAgent:
    def __init__(self, epsilon=1.0, epsilon_min=0.01, decay_type="linear", decay_param=0.001, max_hops=10):
        self.q_table = {("init", "init"): 0.0}
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_type = decay_type
        self.decay_param = decay_param
        self.step = 0
        self.max_hops = max_hops
        self.transition_counts = {}  # ノードごとの遷移回数を記録

    def get_action(self, state, actions):
        choice = ""
        # ペナルティ係数（大きいほど未遷移ノードを優先）
        penalty_coef = 0.05
        if np.random.rand() < self.epsilon:
            choice = "ランダム選択"
            action = random.choice(actions)
        else:
            q_values = []
            for a in actions:
                q = self.q_table.get((state, a), 0.0)
                count = self.transition_counts.get(a, 0)
                penalty = penalty_coef * count
                q_values.append(q - penalty)
            max_q = max(q_values)
            best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
            action = random.choice(best_actions)
            choice = "最適行動選択"
        # 遷移回数を記録
        self.transition_counts[action] = self.transition_counts.get(action, 0) + 1
        return action, choice

    def update(self, state, action, reward, next_state, alpha=0.1, gamma=0.9):
        key = (state, action)
        next_qs = [self.q_table.get((next_state, a), 0.0) for a in self.q_table if a[0] == next_state]
        max_next_q = max(next_qs) if next_qs else 0.0
        old_q = self.q_table.get(key, 0.0)
        self.q_table[key] = old_q + alpha * (reward + gamma * max_next_q - old_q)
        self.step += 1

    def calc_reward(self, acc_before, acc_after):
        acc_diff = acc_after - acc_before
        return acc_after + min(0, acc_diff * 10)

    def decay_epsilon(self):
        if self.decay_type == "linear":
            T = self.max_hops
            k = self.step
            if k <= 3 * T // 4:
                decay = (self.epsilon - self.epsilon_min) / (3 * T // 4)
                self.epsilon = max(self.epsilon_min, self.epsilon - decay)
            else:
                self.epsilon = 0.0
            print(f"[INFO] ε減衰: {self.epsilon:.4f} (step: {self.step})", flush=True)
        elif self.decay_type == "exponential":
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_param)
        elif self.decay_type == "inverse":
            self.epsilon = max(self.epsilon_min, 1.0 / (1.0 + self.decay_param * self.step))
        
        return self.epsilon