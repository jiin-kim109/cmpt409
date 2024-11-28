import numpy as np
from tqdm import tqdm
from src.environments.cartpole import CartPoleEnv
from src.utils.collector import DataCollector

def run_experiment(agents, num_episodes=1000, num_runs=10):
    collector = DataCollector()
    total_iterations = len(agents) * num_runs
    
    with tqdm(total=total_iterations, desc="Training Progress") as pbar:
        for agent_name, AgentClass in agents.items():
            print(f"\n=== Starting {agent_name} Training ===")
            
            for run in range(num_runs):
                env = CartPoleEnv()
                agent = AgentClass(env.state_dim, env.action_dim)
                
                episode_pbar = tqdm(range(num_episodes), 
                                  desc=f"{agent_name} Run {run+1}/{num_runs}",
                                  leave=False)
                
                max_reward_so_far = float('-inf')
                moving_avg_reward = 0
                
                for episode in episode_pbar:
                    state = env.reset()
                    episode_reward = 0
                    steps = 0
                    done = False
                    
                    while not done:
                        action = agent.select_action(state)
                        next_state, reward, done = env.step(action)
                        agent.update(state, action, reward, next_state, done)
                        
                        state = next_state
                        episode_reward += reward
                        steps += 1
                    
                    collector.add(agent_name, run, episode, episode_reward, steps)
                    
                    # Update statistics
                    max_reward_so_far = max(max_reward_so_far, episode_reward)
                    moving_avg_reward = 0.95 * moving_avg_reward + 0.05 * episode_reward
                    
                    episode_pbar.set_postfix({
                        'reward': f'{episode_reward:.1f}',
                        'steps': steps,
                        'max': f'{max_reward_so_far:.1f}',
                        'avg': f'{moving_avg_reward:.1f}'
                    })
                    
                    if episode >= 100:
                        recent_rewards = [collector.data[agent_name][run][i]['reward'] 
                                        for i in range(episode-100, episode)]
                        if np.mean(recent_rewards) >= 475:
                            print(f"\n{agent_name} Run {run+1} solved environment at episode {episode}")
                            print(f"Final reward: {episode_reward:.1f}, Moving average: {moving_avg_reward:.1f}")
                            break
                
                env.close()
                episode_pbar.close()
                pbar.update(1)
    
    return collector
