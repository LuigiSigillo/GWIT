#!/bin/bash
#SBATCH -A IscrC_GenOpt
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00     # format: HH:MM:SS
#SBATCH --nodes=1              # 1 nodes
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gres=gpu:1       # 4 gpus per node out of 4
#SBATCH --cpus-per-task=4
#SBATCH --job-name=dreamdiff

echo "NODELIST="${SLURM_NODELIST}

export WANDB_MODE=offline
module load anaconda3
conda activate bendr
srun python src/DreamDiffusion/code/eeg_ldm.py