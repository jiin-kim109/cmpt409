import torch
import numpy as np
from .base import BaseAgent

class TDRC(BaseAgent):
    def __init__(self, state_dim, action_dim, lr=0.001, beta=1.0, gamma=0.99, epsilon=0.1):
        super().__init__(state_dim, action_dim)
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Secondary weights for gradient correction
        self.h = torch.zeros(state_dim).to(self.device)
        
        # ADAM parameters for secondary weights
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.m = torch.zeros(state_dim).to(self.device)
        self.v = torch.zeros(state_dim).to(self.device)

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
        
        # TD error
        td_error = reward + (1 - done) * self.gamma * next_q - current_q
        
        # Compute expected TD error using secondary weights
        delta_hat = torch.dot(state, self.h)
        
        # Update secondary weights using ADAM
        grad_h = (td_error - delta_hat) * state - self.beta * self.h
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_h
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_h ** 2)
        
        # Update h weights
        self.h = self.h + self.lr * self.m / (torch.sqrt(self.v) + self.eps)
        
        # Primary weights update
        self.optimizer.zero_grad()
        loss = td_error * current_q
        loss.backward()
        self.optimizer.step()
