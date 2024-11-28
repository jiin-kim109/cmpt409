import torch
import numpy as np
from .base import BaseAgent

class ATD(BaseAgent):
    def __init__(self, state_dim, action_dim, lr=0.005, gamma=0.99, epsilon=0.1, momentum=0.95):
        super().__init__(state_dim, action_dim)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Initialize momentum buffer
        self.velocity = [torch.zeros_like(param) for param in self.policy_net.parameters()]

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
        
        # Compute loss
        loss = 0.5 * td_error ** 2
        
        # Compute gradients
        self.optimizer.zero_grad()
        loss.backward()
        
        # Apply momentum update
        with torch.no_grad():
            for i, param in enumerate(self.policy_net.parameters()):
                if param.grad is not None:
                    self.velocity[i] = self.momentum * self.velocity[i] + self.lr * param.grad
                    param.data.add_(-self.velocity[i])
