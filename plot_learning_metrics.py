import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

def extract_metrics(csv_path):
    try:
        df = pd.read_csv(csv_path, comment='#')
        if len(df) == 0 or 'r' not in df.columns or 'l' not in df.columns:
            return None
            
        r = df['r'].values
        l = df['l'].values
        cum_timesteps = np.cumsum(l)
        
        r_smooth = pd.Series(r).rolling(window=100, min_periods=1).mean().values
        
        max_rew = np.max(r_smooth)
        min_rew = np.min(r_smooth)
        
        improvement = max_rew - min_rew
        if improvement == 0:
            threshold = max_rew
        else:
            threshold = min_rew + 0.9 * improvement
            
        burn_in_indices = np.where(r_smooth >= threshold)[0]
        if len(burn_in_indices) > 0:
            burn_in_idx = burn_in_indices[0]
            burn_in_point = cum_timesteps[burn_in_idx]
        else:
            burn_in_idx = len(r) - 1
            burn_in_point = cum_timesteps[-1]
            
        learning_speed = improvement / burn_in_point if burn_in_point > 0 else 0
        
        if burn_in_idx < len(r) - 10:
            volatility = np.std(r[burn_in_idx:])
        else:
            volatility = np.std(r[int(0.9 * len(r)):])
            
        return {
            'max_reward': max_rew,
            'learning_speed': learning_speed * 100000, # scaled for readability
            'volatility': volatility,
            'burn_in_point': burn_in_point / 1000.0
        }
    except Exception as e:
        return None

def process_generation(gen_dir):
    monitor_files = glob.glob(os.path.join(gen_dir, "individual_*", "monitor.csv"))
    metrics_list = [m for m in (extract_metrics(mf) for mf in monitor_files) if m is not None]
            
    if not metrics_list:
        return None
        
    return {
        'max_reward': np.mean([m['max_reward'] for m in metrics_list]),
        'learning_speed': np.mean([m['learning_speed'] for m in metrics_list]),
        'volatility': np.mean([m['volatility'] for m in metrics_list]),
        'burn_in_point': np.mean([m['burn_in_point'] for m in metrics_list])
    }

def process_run(run_dir):
    gen_dirs = sorted(glob.glob(os.path.join(run_dir, "rl_logs", "generation_*")))
    generations = []
    run_metrics = {'max_reward': [], 'learning_speed': [], 'volatility': [], 'burn_in_point': []}
    
    for idx, gen_dir in enumerate(gen_dirs):
        metrics = process_generation(gen_dir)
        if metrics is not None:
            generations.append(idx)
            for k in run_metrics.keys():
                run_metrics[k].append(metrics[k])
                
    return generations, run_metrics

def main():
    base_dir = "/Users/mikolajduchlinski/Desktop/results_final/results"
    runs = [f"exp_lamarckian_ppo_power_ea_rep{rep}" for rep in range(1, 6)]
    run_dirs = [os.path.join(base_dir, r) for r in runs if os.path.exists(os.path.join(base_dir, r))]
    
    print(f"Found {len(run_dirs)} run directories.")
    
    all_metrics = {'max_reward': {}, 'learning_speed': {}, 'volatility': {}, 'burn_in_point': {}}
    
    # Process runs using Pool for speed
    with Pool() as pool:
        results = pool.map(process_run, run_dirs)
        
    for i, (gens, metrics) in enumerate(results):
        for k in all_metrics.keys():
            all_metrics[k][i] = pd.Series(metrics[k], index=gens)
            
    metrics_info = {
        'max_reward': ('Maximum Reward Achieved', 'max_reward.png', 'Asymptotic performance'),
        'learning_speed': ('Learning Speed', 'learning_speed.png', 'Learning Speed'),
        'volatility': ('Stability of Learning', 'volatility.png', 'Stability of learning'),
        'burn_in_point': ('Point of Burn-in', 'burn_in_point.png', 'Time of burning phase')
    }
    
    plt.style.use('ggplot')
    color = '#4c72b0'
    
    for metric_key, (title, filename, ylabel) in metrics_info.items():
        df = pd.DataFrame(all_metrics[metric_key])
        
        plt.figure(figsize=(10, 6))
        for col in df.columns:
            plt.plot(df.index, df[col], alpha=0.3, color=color)
            
        mean_series = df.mean(axis=1)
        std_series = df.std(axis=1)
        
        plt.plot(df.index, mean_series, linewidth=2, color=color)
        plt.fill_between(df.index, 
                         mean_series - std_series,
                         mean_series + std_series, 
                         color=color, alpha=0.2)
                         
        plt.title(title, fontweight='bold')
        plt.xlabel('Generations')
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved {filename}")

if __name__ == '__main__':
    main()
