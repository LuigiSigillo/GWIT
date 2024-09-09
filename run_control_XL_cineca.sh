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
conda activate controlnetxl
srun accelerate launch src/diffusers/examples/controlnet/train_controlnet_sdxl.py  --caption_from_classifier  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"  --output_dir="/leonardo_scratch/fast/IscrC_GenOpt/luigi/Documents/DrEEam/src/diffusers/examples/controlnet/SDXL_model_out_CVPR_MULTISUB_CLASSIFIER_CAPTION"  --dataset_name=luigi-s/EEG_Image_CVPR_ALL_subj --conditioning_image_column=conditioning_image  --image_column=image  --caption_column=caption  --mixed_precision="fp16"  --resolution=1024  --learning_rate=1e-5  --max_train_steps=15000  --train_batch_size=4  --num_train_epochs=100  --gradient_accumulation_steps=4  --report_to="wandb"  --seed=42  --tracker_project_name=controlnet  --checkpointing_steps=1000  --validation_steps=500  --validation_image ./using_VAL_DATASET_PLACEHOLDER.jpeg  --validation_prompt "we are using val dataset hopefuly" 