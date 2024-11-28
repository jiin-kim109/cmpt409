import torch
import numpy as np
from .base import BaseAgent

class VanillaTD(BaseAgent):
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=0.1):
        super().__init__(state_dim, action_dim)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        
        current_q = self.policy_net(state)[action]
        next_q = self.target_net(next_state).max(0)[0].detach()
        target = reward + (1 - done) * self.gamma * next_q
        
        loss = (current_q - target) ** 2
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
