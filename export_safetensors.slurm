#!/bin/bash
#SBATCH --job-name=int4_safetensors
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/safetensors_%j.log

# converts the INT4 .pt checkpoint to .safetensors, then reloads it in a new process
# and compares a real generation bit for bit (idx 130)
#   sbatch export_safetensors.slurm

set -e
mkdir -p logs

module load cuda12.8/toolkit/12.8.1
source ~/envs/cube3d/bin/activate
cd ~/cube/cubepart

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1

echo "Job started: $(date) on $(hostname)"
ls -l results/checkpoint_export/multi_part_dit_int4_hqq_g128.pt results/checkpoint_export/reference_parts.npz
python -c "import safetensors, torch, torchao; print('safetensors', safetensors.__version__, 'torch', torch.__version__, 'torchao', torchao.__version__)"

python export_int4_safetensors.py --write
python export_int4_safetensors.py --verify   # separate process on purpose

ls -l results/checkpoint_export/
echo "Job finished: $(date)"
