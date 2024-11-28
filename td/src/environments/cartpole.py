import gymnasium as gym
import numpy as np

class CartPoleEnv:
    def __init__(self, max_steps=500):
        self.env = gym.make('CartPole-v1')
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.max_steps = max_steps
        self.steps = 0
        
    def reset(self):
        self.steps = 0
        state, _ = self.env.reset()
        return state
    
    def step(self, action):
        self.steps += 1
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        
        # End episode if max steps reached
        if self.steps >= self.max_steps:
            truncated = True
            
        done = terminated or truncated
        return next_state, reward, done
    
    def close(self):
        self.env.close()
