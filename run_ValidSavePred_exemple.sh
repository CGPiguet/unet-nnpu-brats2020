#!/bin/bash
#SBATCH --job-name="Save Prediction"
#SBATCH --mail-user=christian.piguet@students.unibe.ch
#SBATCH --mail-type=end,fail
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=4 # 4 is the limit 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --error="ValidSavePred_error.txt"
#SBATCH --output="ValidSavePred_output.txt"

#### Your shell commands below this line ####
module load Workspace/home Anaconda3 CUDA
eval "$(conda shell.bash hook)"
conda activate unet


srun python run_ValidSavePred.py  -load_model "model_saved_Exemple/epoch_100_checkpoint.pth"
