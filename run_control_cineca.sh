#!/bin/bash
#SBATCH -A IscrC_GenOpt
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00     # format: HH:MM:SS
#SBATCH --nodes=1              # 1 nodes
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gres=gpu:1       # 4 gpus per node out of 4
#SBATCH --cpus-per-task=4
#SBATCH --job-name=controlneteeg

echo "NODELIST="${SLURM_NODELIST}

export WANDB_MODE=offline
module load anaconda3
conda activate controlnet
srun accelerate launch /leonardo_scratch/fast/IscrC_GenOpt/luigi/Documents/DrEEam/src/diffusers/examples/controlnet/train_controlnet.py --caption_fixed --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-1-base --output_dir=/leonardo_scratch/fast/IscrC_GenOpt/luigi/Documents/DrEEam/src/diffusers/examples/controlnet/model_out_CVPR_MULTISUB_FIXED_CAPTION --dataset_name=luigi-s/EEG_Image_CVPR_ALL_subj --conditioning_image_column=conditioning_image --image_column=image --caption_column=caption --resolution=512 --learning_rate=1e-5 --train_batch_size=28 --num_train_epochs=500 --tracker_project_name=controlnet --enable_xformers_memory_efficient_attention --checkpointing_steps=1000 --validation_steps=500 --report_to wandb --validation_image ./using_VAL_DATASET_PLACEHOLDER.jpeg --validation_prompt "we are using val dataset hopefuly"