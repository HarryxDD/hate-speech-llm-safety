#!/bin/bash
#SBATCH --account=project_2021905
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Runs one llm.py command on one GPU, e.g.
#   sbatch --job-name=cls-qwen3-4b run.sbatch classify --model qwen3-4b
# or another script when its name comes first, e.g.
#   sbatch --job-name=ft run.sbatch finetune.py --model qwen3-4b --method lora
# Quick test in the 15-minute test queue:
#   sbatch --partition=gputest --time=00:15:00 run.sbatch classify --model qwen3-4b --limit 200

module purge
module load python-pytorch/2.13
export HF_HUB_CACHE=/scratch/project_2021905/$USER/hf-cache   # downloaded on the login node
export HF_HUB_OFFLINE=1

if [[ "$1" == *.py ]]; then
  script=$1
  shift
else
  script=llm.py
fi
srun python "$script" "$@"
