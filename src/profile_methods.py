import torch
import time
import numpy as np
from pathlib import Path
from copy import deepcopy
from omegaconf import OmegaConf

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent))

from algorithm.tdmpc import TDMPC
from cfg import parse_cfg
from env import make_env
from algorithm.helper import ReplayBuffer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def profile_method(name, method_cfg, base_cfg):
    print(f"\nProfiling {name}...")
    cfg = deepcopy(base_cfg)
    cfg.freeze_encoder = method_cfg['freeze_encoder']
    cfg.use_lora = method_cfg['use_lora']
    
    # Initialize Agent
    agent = TDMPC(cfg)
    agent.apply_adaptation()
    
    # 1. Parameter Count
    trainable_params = count_parameters(agent.model)
    total_params = sum(p.numel() for p in agent.model.parameters())
    
    # Setup for timing/memory
    obs_shape = cfg.obs_shape
    is_dict = hasattr(obs_shape, 'items')
    cfg.exp_name = 'profile_tmp' # Avoid overwriting real logs
    cfg.save_video = False
    cfg.save_model = False
    cfg.mpc = True # Ensure we profile the full MPC planning
    
    if is_dict:
        obs = {k: np.random.randn(*v).astype(np.float32) for k, v in obs_shape.items()}
    else:
        obs = np.random.randn(*obs_shape).astype(np.float32)
        
    # Dummy replay buffer for training profile
    num_samples = 256 # Match batch size for realistic profile
    if is_dict:
        dataset = {
            'observations': {k: np.random.randn(num_samples, *v).astype(np.float32) for k, v in obs_shape.items()},
            'next_observations': {k: np.random.randn(num_samples, *v).astype(np.float32) for k, v in obs_shape.items()},
            'actions': np.random.randn(num_samples, cfg.action_dim).astype(np.float32),
            'rewards': np.random.randn(num_samples).astype(np.float32),
            'masks': np.ones(num_samples, dtype=np.float32),
            'dones': np.zeros(num_samples, dtype=np.float32),
        }
    else:
        dataset = {
            'observations': np.random.randn(num_samples, *obs_shape).astype(np.float32),
            'next_observations': np.random.randn(num_samples, *obs_shape).astype(np.float32),
            'actions': np.random.randn(num_samples, cfg.action_dim).astype(np.float32),
            'rewards': np.random.randn(num_samples).astype(np.float32),
            'masks': np.ones(num_samples, dtype=np.float32),
            'dones': np.zeros(num_samples, dtype=np.float32),
        }
    buffer = ReplayBuffer(cfg, dataset=dataset)

    # 2. Inference (Act) Profiling
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup
    for _ in range(10):
        agent.act(obs, eval_mode=True, step=0)
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(50):
        agent.act(obs, eval_mode=True, step=0)
    torch.cuda.synchronize()
    inf_time = (time.time() - start_time) / 50.0
    inf_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB

    # 3. Training (Update) Profiling
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup
    for _ in range(5):
        agent.update(buffer, step=0)
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(20):
        agent.update(buffer, step=0)
    torch.cuda.synchronize()
    train_time = (time.time() - start_time) / 20.0
    train_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB

    return {
        'Method': name,
        'Trainable Params': trainable_params,
        'Total Params': total_params,
        'Inference Time (ms)': inf_time * 1000.0,
        'Training Time (ms)': train_time * 1000.0,
        'Peak Memory (MB)': max(inf_mem, train_mem)
    }

def run_profiling():
    # Load base configuration
    cfg = parse_cfg(Path(__file__).parent.parent / 'cfgs')
    
    # Initialize environment to populate cfg with shapes
    print("Initializing environment...")
    env = make_env(cfg)
    
    methods = {
        'Full-FT': {'freeze_encoder': False, 'use_lora': False},
        'Encoder-Frozen': {'freeze_encoder': True, 'use_lora': False},
        'Dynamics-LoRA': {'freeze_encoder': True, 'use_lora': True}
    }
    
    results = []
    for name, m_cfg in methods.items():
        res = profile_method(name, m_cfg, cfg)
        results.append(res)
        
    # Print results table
    print("\n" + "="*95)
    print(f"{'Method':<20} | {'Trainable Params':<18} | {'Inf Time (ms)':<15} | {'Train Time (ms)':<16} | {'Memory (MB)':<12}")
    print("-" * 95)
    for res in results:
        print(f"{res['Method']:<20} | {res['Trainable Params']:<18,} | {res['Inference Time (ms)']:<15.2f} | {res['Training Time (ms)']:<16.2f} | {res['Peak Memory (MB)']:<12.1f}")
    print("="*95)

if __name__ == '__main__':
    run_profiling()
