import torch
import numpy as np
from .base import BaseAgent

class GTD2(BaseAgent):
    def __init__(self, state_dim, action_dim, lr=0.001, beta=0.5, gamma=0.99, epsilon=0.1):
        super().__init__(state_dim, action_dim)
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.h = torch.zeros(state_dim).to(self.device)

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
        
        td_error = reward + (1 - done) * self.gamma * next_q - current_q
        
        # GTD2 update
        delta_h = td_error * state - self.gamma * (1 - done) * next_state
        self.h = self.h + self.beta * delta_h
        
        grad_correction = -self.gamma * (1 - done) * next_state @ self.h
        
        self.optimizer.zero_grad()
        current_q.backward()
        
        with torch.no_grad():
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    param.grad += self.lr * grad_correction
        
        self.optimizer.step()
