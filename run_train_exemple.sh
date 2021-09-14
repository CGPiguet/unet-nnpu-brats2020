#!/bin/bash
#SBATCH --job-name="Exemple"
#SBATCH --mail-user=christian.piguet@students.unibe.ch
#SBATCH --mail-type=end,fail
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=4 # 4 is the limit 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --error="Exemple_error.txt"
#SBATCH --output="Exemple_output.txt"

#### Your shell commands below this line ####
module load Workspace/home Anaconda3 CUDA
eval "$(conda shell.bash hook)"
conda activate unet


srun python run_train.py  -n "Exemple" -pr 0.5 -s 0.001 -e 100 -opti "SGD" -img_mode "T1" -continue_training True

 
