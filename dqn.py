import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.prod(input_shape), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
    
    def forward(self, x):
        return self.net(x.flatten(1))

class DQNAgent:
    def __init__(self, input_shape, num_actions):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = DQN(input_shape, num_actions).to(self.device)
        self.target_net = DQN(input_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.net.num_actions)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.net(state_t)
        return q_values.argmax().item()
    
    def update(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(q_values.squeeze(), targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())