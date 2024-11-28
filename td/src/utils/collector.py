import numpy as np

class DataCollector:
    def __init__(self):
        self.data = {}
        
    def add(self, agent_name, run, episode, reward, steps):
        if agent_name not in self.data:
            self.data[agent_name] = {}
        if run not in self.data[agent_name]:
            self.data[agent_name][run] = []
        self.data[agent_name][run].append({
            'reward': reward,
            'steps': steps,
            'episode': episode
        })
    
    def get_statistics(self):
        stats = {}
        for agent_name in self.data:
            # Get max episodes across all runs
            max_episodes = max(len(run) for run in self.data[agent_name].values())
            
            # Initialize arrays for rewards and steps
            rewards = np.zeros((len(self.data[agent_name]), max_episodes))
            steps = np.zeros((len(self.data[agent_name]), max_episodes))
            
            # Fill arrays with data
            for run_idx, run_data in enumerate(self.data[agent_name].values()):
                for episode_idx, episode_data in enumerate(run_data):
                    rewards[run_idx, episode_idx] = episode_data['reward']
                    steps[run_idx, episode_idx] = episode_data['steps']
            
            # Calculate statistics
            stats[agent_name] = {
                'reward_mean': rewards.mean(axis=0),
                'reward_std': rewards.std(axis=0) / np.sqrt(len(self.data[agent_name])),
                'steps_mean': steps.mean(axis=0),
                'steps_std': steps.std(axis=0) / np.sqrt(len(self.data[agent_name]))
            }
            
        return stats
