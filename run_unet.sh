#!/bin/bash
#SBATCH --job-name="Unet BCELoss"
#SBATCH --mail-user=christian.piguet@students.unibe.ch
#SBATCH --mail-type=end,fail
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --error="unet_error.txt"
#SBATCH --output="unet_output.txt"

#### Your shell commands below this line ####
module load Workspace/home Anaconda3 CUDA
eval "$(conda shell.bash hook)"
conda activate unet


python run_train.py -e 40 -p "BCELoss" -v False -b 16