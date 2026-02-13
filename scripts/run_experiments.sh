#!/bin/bash

# Configuration
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/takatoishii/.mujoco/mujoco210/bin:/usr/lib/nvidia
TASKS=("xarm_lift" "xarm_push")
SEEDS=(1 2 3)

for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Running task=$task seed=$seed"
        python src/train_off2on.py task=$task seed=$seed
    done
done

# Generate plot
echo "Generating plots..."
python plot_results.py

