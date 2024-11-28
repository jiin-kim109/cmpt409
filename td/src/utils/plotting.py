import matplotlib.pyplot as plt
import numpy as np
import os
from tabulate import tabulate

def plot_metrics(collector, save_path='src/output'):
    os.makedirs(save_path, exist_ok=True)
    stats = collector.get_statistics()
    
    print("\n=== Performance Summary ===")
    for agent_name in stats:
        print(f"\n{agent_name} Analysis:")
        
        # Episode Rewards Table
        rewards_table = []
        headers = ["Episode", "Mean Reward ± Std", "Mean Steps ± Std"]
        for i in range(0, len(stats[agent_name]['reward_mean']), 50):
            rewards_table.append([
                i,
                f"{stats[agent_name]['reward_mean'][i]:.2f} ± {stats[agent_name]['reward_std'][i]:.2f}",
                f"{stats[agent_name]['steps_mean'][i]:.2f} ± {stats[agent_name]['steps_std'][i]:.2f}"
            ])
        print("\nEpisode Statistics:")
        print(tabulate(rewards_table, headers=headers, tablefmt='grid'))
        
        # Solve Analysis
        solve_episodes = []
        for run in collector.data[agent_name].values():
            for episode_idx, episode_data in enumerate(run):
                if episode_data['reward'] >= 475:
                    solve_episodes.append(episode_idx)
                    break
        
        solve_table = []
        if solve_episodes:
            solve_table.append([
                f"{len(solve_episodes)}/{len(collector.data[agent_name])}",
                f"{np.mean(solve_episodes):.2f}",
                f"{np.std(solve_episodes):.2f}",
                min(solve_episodes),
                max(solve_episodes)
            ])
        else:
            solve_table.append(["0/10", "N/A", "N/A", "N/A", "N/A"])
            
        print("\nSolve Analysis:")
        print(tabulate(solve_table, 
                      headers=["Solve Rate", "Mean Episode", "Std Episode", "Min Episode", "Max Episode"],
                      tablefmt='grid'))
        
        # Per-run Analysis
        run_table = []
        for run_id, run_data in collector.data[agent_name].items():
            run_table.append([
                run_id,
                f"{run_data[-1]['reward']:.2f}",
                f"{max(data['reward'] for data in run_data):.2f}",
                f"{np.mean([data['reward'] for data in run_data]):.2f}",
                f"{np.mean([data['steps'] for data in run_data]):.2f}"
            ])
        
        print("\nPer-run Analysis:")
        print(tabulate(run_table, 
                      headers=["Run", "Final Reward", "Max Reward", "Avg Reward", "Avg Steps"],
                      tablefmt='grid'))
    
    # Set consistent colors for agents
    COLORS = {
        'VanillaTD': 'blue',
        'GTD2': 'grey',
        'TDRC': 'orange',
        'ATD': 'purple'
    }
    
    # 1. Learning Curves (Episode Rewards)
    plt.figure(figsize=(10, 6))
    for agent_name, data in stats.items():
        episodes = range(len(data['reward_mean']))
        plt.plot(episodes, data['reward_mean'], 
                label=agent_name, 
                color=COLORS[agent_name])
        plt.fill_between(episodes, 
                        data['reward_mean'] - data['reward_std'],
                        data['reward_mean'] + data['reward_std'],
                        alpha=0.2,
                        color=COLORS[agent_name])
    
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Learning Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, 'learning_curves.png'))
    plt.close()
    
    # 2. Steps to Solve Distribution
    plt.figure(figsize=(10, 6))
    data_to_plot = []
    labels = []
    
    for agent_name in stats:
        solve_steps = []
        for run in collector.data[agent_name].values():
            # Find first episode that reached 475+ reward
            for episode_data in run:
                if episode_data['reward'] >= 475:
                    solve_steps.append(episode_data['episode'])
                    break
        if solve_steps:
            data_to_plot.append(solve_steps)
            labels.append(agent_name)
    
    if data_to_plot:
        plt.boxplot(data_to_plot, labels=labels)
        plt.ylabel('Episodes to Solve')
        plt.title('Distribution of Episodes Required to Solve CartPole')
        for i, agent in enumerate(labels, 1):
            plt.scatter([i] * len(data_to_plot[i-1]), 
                       data_to_plot[i-1], 
                       color=COLORS[agent],
                       alpha=0.5)
    
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, 'solve_distribution.png'))
    plt.close()
    
    # 3. Average Steps per Episode
    plt.figure(figsize=(10, 6))
    for agent_name, data in stats.items():
        episodes = range(len(data['steps_mean']))
        plt.plot(episodes, data['steps_mean'],
                label=agent_name,
                color=COLORS[agent_name])
        plt.fill_between(episodes,
                        data['steps_mean'] - data['steps_std'],
                        data['steps_mean'] + data['steps_std'],
                        alpha=0.2,
                        color=COLORS[agent_name])
    
    plt.xlabel('Episodes')
    plt.ylabel('Average Steps per Episode')
    plt.title('Episode Length Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, 'episode_lengths.png'))
    plt.close()
