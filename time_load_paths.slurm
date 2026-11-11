#!/bin/bash
#SBATCH --job-name=int4_timing
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/timing_%j.log

# 3 generations of idx 130 per load path, each path in a fresh process.
#   sbatch time_load_paths.slurm
#   grep TIMING logs/timing_<jobid>.log

set -e
mkdir -p logs

module load cuda12.8/toolkit/12.8.1
source ~/envs/cube3d/bin/activate
cd ~/cube/cubepart

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1

echo "Job started: $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv
for src in bf16 pt st; do
    python time_load_paths.py --source $src --repeat 3
done
echo "Job finished: $(date)"
