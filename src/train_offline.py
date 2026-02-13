import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import torch
import numpy as np
import time
from pathlib import Path
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import ReplayBuffer, get_dataset_dict
import logger
import gc

torch.backends.cudnn.benchmark = True

def train_offline(cfg):
    """Offline-only pretraining script for TD-MPC."""
    assert torch.cuda.is_available()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(seed=cfg.seed)
    
    # Save model directory
    model_dir = Path().cwd() / 'models'
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f'{cfg.task}_offline_seed{cfg.seed}.pt'
    
    if model_path.exists():
        print(f"Pretrained model already exists at {model_path}. Skipping.")
        return

    work_dir = Path().cwd() / 'logs' / cfg.task / cfg.modality / 'offline_pretrain' / str(cfg.seed)
    env = make_env(cfg)
    agent = TDMPC(cfg)
    
    print("Loading dataset...")
    dataset = get_dataset_dict(cfg, env)
    offline_buffer = ReplayBuffer(cfg, dataset=dataset)
    del dataset
    gc.collect()

    L = logger.Logger(work_dir, cfg)
    start_time = time.time()
    
    print(f"Offline pretraining starts for {cfg.offline_steps} steps...")
    for step in range(0, cfg.offline_steps, cfg.episode_length):
        train_metrics = {}
        for i in range(cfg.episode_length):
            train_metrics.update(agent.update(offline_buffer, step + i))
        
        metrics = {
            'step': step + cfg.episode_length,
            'env_step': (step + cfg.episode_length) * cfg.action_repeat,
            'total_time': time.time() - start_time,
            'episode': (step // cfg.episode_length) + 1,
            'phase': 'Off',
        }
        train_metrics.update(metrics)
        L.log(train_metrics, category='train')

    print(f"Saving offline model to {model_path}")
    agent.save(model_path)
    print("Offline pretraining completed.")

if __name__ == '__main__':
    cfg = parse_cfg(Path().cwd() / 'cfgs')
    train_offline(cfg)
