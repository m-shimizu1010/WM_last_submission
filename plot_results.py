import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_method_comparison(task, domain, seeds, methods, modality='all'):
    plt.figure(figsize=(10, 6))
    
    # Modern styling
    plt.style.use('ggplot')
    # Fixed colors for methods for consistency across plots
    method_colors = {
        'Full-FT': '#E24A33',        # Red
        'Encoder-Frozen': '#348ABD', # Blue
        'Dynamics-LoRA': '#988ED5',  # Purple
    }
    
    for method in methods:
        exp_name = f"{domain}_{method}"
        all_rewards = []
        common_steps = None
        
        for seed in seeds:
            log_path = Path(f'logs/{task}/{modality}/{exp_name}/{seed}/eval.log')
            if log_path.exists():
                df = pd.read_csv(log_path)
                # Keep only steps where reward is present
                df = df.dropna(subset=['episode_reward'])
                steps = df['env_step'].values
                rewards = df['episode_reward'].values
                
                if common_steps is None:
                    common_steps = steps
                
                # Align data lengths
                if len(steps) != len(common_steps):
                    min_len = min(len(steps), len(common_steps))
                    common_steps = common_steps[:min_len]
                    rewards = rewards[:min_len]
                    all_rewards = [r[:min_len] for r in all_rewards]
                
                all_rewards.append(rewards)
            else:
                # Silently skip missing seeds to avoid cluttered output, 
                # but notify if no seeds found at all later.
                pass
        
        if not all_rewards:
            print(f"  Warning: No data found for {exp_name}")
            continue
            
        all_rewards = np.array(all_rewards)
        mean_reward = np.mean(all_rewards, axis=0)
        std_reward = np.std(all_rewards, axis=0)
        
        color = method_colors.get(method, plt.cm.tab10(np.random.rand()))
        plt.plot(common_steps, mean_reward, label=method, color=color, linewidth=2.5)
        plt.fill_between(common_steps, mean_reward - std_reward, mean_reward + std_reward, 
                         color=color, alpha=0.15)
        
    plt.xlabel('Environment Steps [1e5]', fontsize=25)
    plt.ylabel('Episode Reward', fontsize=25)
    # plt.title(f'Domain Adaptation: {domain.replace("_", " ").title()} ({task})', fontsize=20)
    plt.legend(frameon=True, fontsize=25, facecolor='white', edgecolor='black', framealpha=1)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Use scientific notation for x-axis if steps are large
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tick_params(axis='both', which='major', labelsize=20)
    
    output_file = f'results_{task}_{domain}.pdf'
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved plot: {output_file}")
    plt.close()

if __name__ == '__main__':
    TASKS = ['xarm_lift', 'xarm_push']
    DOMAINS = ['baseline', 'heavy_load', 'high_friction', 'mix_severe']
    METHODS = ['Full-FT', 'Encoder-Frozen', 'Dynamics-LoRA']
    SEEDS = [1, 2, 3, 4, 5]
    
    if not Path('logs').exists():
        print("Error: 'logs' directory not found.")
        exit(1)
        
    for task in TASKS:
        print(f"Plotting for task: {task}")
        for domain in DOMAINS:
            plot_method_comparison(task, domain, SEEDS, METHODS)
