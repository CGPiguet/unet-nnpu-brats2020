#!/bin/bash
#SBATCH --job-name="Unet BCELoss numWorker=8"
#SBATCH --mail-user=christian.piguet@students.unibe.ch
#SBATCH --mail-type=end,fail
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=4 # 4 is the limit 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --error="unet_error_numworker8.txt"
#SBATCH --output="unet_output_numworker8.txt"

#### Your shell commands below this line ####
module load Workspace/home Anaconda3 CUDA
eval "$(conda shell.bash hook)"
conda activate unet


python run_train.py -e 100 -p "BCELoss" -v False -b 16 -n "BCELoss_NumWorker8"