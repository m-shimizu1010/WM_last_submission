import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_learning_curves(tasks, seeds, modality='all', exp_name='default', output_file='learning_curves.pdf'):
    plt.figure(figsize=(10, 6))
    
    # Modern styling
    plt.style.use('ggplot')
    colors = plt.cm.tab10(np.linspace(0, 1, len(tasks)))
    
    for i, task in enumerate(tasks):
        all_rewards = []
        common_steps = None
        
        for seed in seeds:
            log_path = Path(f'logs/{task}/{modality}/{exp_name}/{seed}/eval.log')
            if log_path.exists():
                df = pd.read_csv(log_path)
                steps = df['env_step'].values
                rewards = df['episode_reward'].values
                
                if common_steps is None:
                    common_steps = steps
                
                # In case steps are slightly different, we could interpolate, 
                # but here we assume eval_freq is constant.
                # If they differ in length, we take the minimum length.
                if len(steps) != len(common_steps):
                    min_len = min(len(steps), len(common_steps))
                    common_steps = common_steps[:min_len]
                    rewards = rewards[:min_len]
                    # Adjust previous rewards in all_rewards if necessary
                    all_rewards = [r[:min_len] for r in all_rewards]
                
                all_rewards.append(rewards)
            else:
                print(f"Warning: Log file not found at {log_path}")
        
        if not all_rewards:
            print(f"No data found for task {task}")
            continue
            
        all_rewards = np.array(all_rewards)
        mean_reward = np.mean(all_rewards, axis=0)
        std_reward = np.std(all_rewards, axis=0)
        
        plt.plot(common_steps, mean_reward, label=task, color=colors[i], linewidth=2)
        plt.fill_between(common_steps, mean_reward - std_reward, mean_reward + std_reward, 
                         color=colors[i], alpha=0.2)
        
    plt.xlabel('Environment Steps')
    plt.ylabel('Episode Reward')
    plt.title('Learning Curves (Mean ± Std)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Use scientific notation for x-axis if steps are large
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close()

if __name__ == '__main__':
    TASKS = ['xarm_lift', 'xarm_push']
    SEEDS = [1, 2, 3, 4, 5]
    
    # Ensure logs directory exists
    if not os.path.exists('logs'):
        print("Warning: 'logs' directory not found. Please run experiments first.")
    
    plot_learning_curves(TASKS, SEEDS)
