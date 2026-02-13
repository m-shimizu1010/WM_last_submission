import os
import time
import gc
import torch
from pathlib import Path
from copy import deepcopy
from cfg import parse_cfg
from train_off2on import train
from omegaconf import OmegaConf

# Fix Mujoco path for mujoco_py
paths_to_add = [
    '/home/takatoishii/.mujoco/mujoco210/bin',
    '/usr/lib/nvidia'
]
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
for p in paths_to_add:
    if p not in current_ld_path:
        current_ld_path += f':{p}'
os.environ['LD_LIBRARY_PATH'] = current_ld_path

def run_experiments():
    # Load base configuration
    cfg_base = parse_cfg(Path(__file__).parent.parent / 'cfgs')
    OmegaConf.set_struct(cfg_base, False)
    
    tasks = ['xarm_lift'] # Reduced for demo, can add xarm_push
    domains = ['baseline', 'heavy_load', 'high_friction', 'mix_severe']
    methods = {
        'Full-FT': {'freeze_encoder': False, 'use_lora': False},
        'Encoder-Frozen': {'freeze_encoder': True, 'use_lora': False},
        'Dynamics-LoRA': {'freeze_encoder': True, 'use_lora': True}
    }
    
    start_time_all = time.time()
    
    for task in tasks:
        # Step 1: Offline Pretraining (Fair baseline)
        print(f"\n{'='*60}")
        print(f"STAGE 1: Offline Pretraining for Task: {task}")
        print(f"{'='*60}")
        
        offline_cmd = [
            '/home/takatoishii/miniconda3/envs/fowm/bin/python',
            'src/train_offline.py',
            f'task={task}',
            f'offline_steps={cfg_base.offline_steps}',
            f'seed={cfg_base.seed}'
        ]
        # Pass through remaining args that are NOT related to experiments matrix
        import sys
        for arg in sys.argv[1:]:
             if not any(x in arg for x in ['task=', 'train_steps=', 'offline_steps=', 'exp_name=', 'domain_shift=', 'seed=']):
                 offline_cmd.append(arg)
        
        try:
            print(f"Running offline pretraining: {' '.join(offline_cmd)}")
            import subprocess
            subprocess.run(offline_cmd, check=True, cwd=str(Path(__file__).parent.parent))
        except Exception as e:
            print(f"ERROR: Offline pretraining failed for {task}!")
            print(e)
            continue

        # Path to the pretrained model we just created
        pretrained_model_path = Path().cwd() / 'models' / f'{task}_offline_seed{cfg_base.seed}.pt'

        # Step 2: Online Fine-tuning Experiments
        print(f"\n{'='*60}")
        print(f"STAGE 2: Online Fine-tuning Experiments for Task: {task}")
        print(f"{'='*60}")
        
        for domain in domains:
            for method_name, method_cfg in methods.items():
                exp_name = f"{domain}_{method_name}"
                print(f"\n--- Starting Experiment: {exp_name} ---")
                
                cmd = [
                    '/home/takatoishii/miniconda3/envs/fowm/bin/python',
                    'src/train_online.py',
                    f'task={task}',
                    f'exp_name={exp_name}',
                    f'domain_shift={domain}',
                    f'freeze_encoder={str(method_cfg["freeze_encoder"]).lower()}',
                    f'use_lora={str(method_cfg["use_lora"]).lower()}',
                    f'pretrained_model_path={pretrained_model_path}',
                    f'train_steps={cfg_base.train_steps}',
                    f'seed={cfg_base.seed}'
                ]
                # Pass through remaining args
                for arg in sys.argv[1:]:
                    if not any(x in arg for x in ['task=', 'train_steps=', 'offline_steps=', 'exp_name=', 'domain_shift=', 'seed=', 'pretrained_model_path=']):
                        cmd.append(arg)
                
                start_time_exp = time.time()
                try:
                    print(f"Running command: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True, cwd=str(Path(__file__).parent.parent))
                except Exception as e:
                    print(f"ERROR: Experiment {exp_name} failed!")
                    print(e)
                
                gc.collect()
                torch.cuda.empty_cache()
                duration_exp = time.time() - start_time_exp
                print(f"Done: {exp_name} in {duration_exp/60:.2f} min")
            
    total_duration = time.time() - start_time_all
    print(f"\n{'='*60}")
    print(f"Benchmark Run Completed Successfully!")
    print(f"Total Portfolio Duration: {total_duration/3600:.2f} hours")
    print(f"{'='*60}")
    
    # Run plotting script
    print("\nGenerating comparison plots...")
    plot_cmd = [
        '/home/takatoishii/miniconda3/envs/fowm/bin/python',
        'plot_results.py'
    ]
    try:
        import subprocess
        subprocess.run(plot_cmd, check=True, cwd=str(Path(__file__).parent.parent))
    except Exception as e:
        print("Warning: Plotting failed!")
        print(e)

if __name__ == '__main__':
    run_experiments()
