#!/bin/bash
#SBATCH --job-name=mytraining
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=60
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
module purge # Clean the environment
module load python/anaconda3 # Load the anaconda3 module
eval "$(conda shell.bash hook)" # Initialize the shell for Conda
conda activate myenv # Activate your Python environment
python train.py
