#!/bin/bash
#SBATCH --job-name=cubepart_4config
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00
#SBATCH --output=logs/4config_%j.log

# Usage:
#   sbatch full_sweep_4config.slurm --idx-min 120 --idx-max 139   # furniture first
#   sbatch full_sweep_4config.slurm                                # full sweep, all 12 usable categories
#
# --time=23:00:00: gpu-l40s's MaxTime is 1-00:00:00 (24h) -- confirmed via
# `scontrol show partition gpu-l40s` on 2026-08-31 -- so this is the largest
# allocation the partition will grant, with a 1h safety margin. The furniture
# run (20 meshes) finishes in ~1-2h based on measured per-mesh timings, well
# inside this. The full 250-mesh sweep (report estimate: 25-40h) CANNOT fit
# in one job on this partition regardless of --time -- it relies on this
# script's per-(mesh,pair) resumability: if a full-sweep job hits the 23h
# wall before finishing, just `sbatch full_sweep_4config.slurm` again with
# the same (or no) idx filter and it picks up exactly where it left off via
# the <idx>_ab.json/<idx>_cd.json completion markers. Expect to resubmit
# 1-2 times for the full sweep.
#
# Resumable: rerunning (same or no idx filter) after a timeout or
# preemption skips any (mesh, pair) whose <idx>_ab.json / <idx>_cd.json
# already exists under results/full_sweep_4config/per_mesh/.

set -e
mkdir -p logs

module load cuda12.8/toolkit/12.8.1
source ~/envs/cube3d/bin/activate

cd ~/cube/cubepart

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

python benchmark_4config_sweep.py "$@"
python compute_4config_cd_stats.py

echo "Job finished: $(date)"
