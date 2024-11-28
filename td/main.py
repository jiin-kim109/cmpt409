from src.experiments.cartpole_experiment import run_experiment
from src.agents.atd import ATD
from src.agents.vanilla_td import VanillaTD
from src.utils.plotting import plot_metrics
from tabulate import tabulate

def main():
    agents = {
        'VanillaTD': VanillaTD,
        'ATD': ATD,
    }
    
    collector = run_experiment(agents, num_episodes=500, num_runs=10)
    plot_metrics(collector)

if __name__ == "__main__":
    main()
