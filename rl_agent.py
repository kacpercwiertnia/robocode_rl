import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

STATE_PATH = "io/state.csv"
DECISION_PATH = "io/decision.txt"
REWARD_PATH = "io/reward.txt"
MODEL_PATH = "model.pt"

STATE_SIZE = 9
ACTION_SPACE = [0.1, 1.0, 2.0, 3.0]
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, len(ACTION_SPACE))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class DQNAgent:
    def __init__(self):
        self.model = QNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = 1.0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(ACTION_SPACE) - 1)
        with torch.no_grad():
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def store(self, s, a, r, s_):
        self.memory.append((s, a, r, s_))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        q_values = self.model(states)
        next_q = self.model(next_states).max(1)[0].detach()
        q_target = rewards + GAMMA * next_q
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        loss = F.mse_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(0.1, self.epsilon * 0.995)

def run():
    agent = DQNAgent()
    last_state, last_action = None, None
    while True:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, 'r') as f:
                line = f.readline().strip()
                if not line or line.count(',') < 8:
                    continue
                state_values = list(map(float, line.split(',')))
                state = torch.tensor(state_values, dtype=torch.float32)


            if last_state is not None and os.path.exists(REWARD_PATH):
                with open(REWARD_PATH, 'r') as f:
                    reward = float(f.readline().strip())
                agent.store(last_state, last_action, reward, state)
                agent.train()
                os.remove(REWARD_PATH)

            action_index = agent.act(state)
            with open(DECISION_PATH, 'w') as f:
                f.write(str(ACTION_SPACE[action_index]))

            last_state, last_action = state, action_index
            os.remove(STATE_PATH)
        else:
            time.sleep(0.01)

if __name__ == "__main__":
    run()