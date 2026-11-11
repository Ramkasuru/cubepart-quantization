#!/bin/bash
#SBATCH --job-name=sh_render
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/render_%j.log

# 18 SH_BPS_AO views per mesh, for whole meshes and part meshes.
#   ONLY_PREFIX=020 sbatch render_sh_bps_ao.slurm   # one prompt, as a test
#   sbatch render_sh_bps_ao.slurm                   # all 36; finished meshes are skipped

set -e
mkdir -p logs

module load cuda12.8/toolkit/12.8.1
source ~/envs/cube3d/bin/activate
cd ~/cube/cubepart

echo "Job started: $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv
for sub in cube3d cubepart; do
    MESH_DIR=results/visual_assets/$sub SH_RENDER_OUT=results/renders/$sub python render_sh_bps_ao.py
done
echo "Job finished: $(date)"
