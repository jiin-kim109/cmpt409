import torch
import numpy as np
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, state_dim, action_dim, device=None, target_update_freq=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = self._build_network().to(self.device)
        self.target_net = self._build_network().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        
        self.target_update_freq = target_update_freq
        self.steps = 0
        
    def _build_network(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.action_dim)
        )
    
    @abstractmethod
    def select_action(self, state):
        pass
    
    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        pass
