#!/bin/bash
#SBATCH --job-name="Convert and downlaod Brats2020 to 2D"
#SBATCH --mail-user=christian.piguet@students.unibe.ch
#SBATCH --mail-type=end,fail
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=4 # 4 is the limit 



#### Your shell commands below this line ####
module load Workspace/home Anaconda3 CUDA
eval "$(conda shell.bash hook)"
conda activate unet


srun python preprocess2D.py -rtv 0.8 -rpu 0.95 --rootdir="MICCAI_BraTS2020_TrainingData"
