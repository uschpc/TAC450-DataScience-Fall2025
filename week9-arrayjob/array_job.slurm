#!/bin/bash

#SBATCH --job-name=mnist_array
#SBATCH --account=irahbari_1147
#SBATCH --partition=main
#SBATCH --array=1-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=mnist_%A_%a.out
#SBATCH --error=mnist_%A_%a.err

# Load required modules
module purge

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch-env

# Get the epoch value for this array task
# Array indices 1-11 correspond to epochs 3-13
EPOCH=$((SLURM_ARRAY_TASK_ID + 2))

# Run the MNIST training with the specific epoch count
python3 mnist_classify.py --epochs=$EPOCH 

echo "Completed training with $EPOCH epochs for array task $SLURM_ARRAY_TASK_ID"
