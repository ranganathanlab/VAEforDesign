#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --account=pi-andrewferguson
#SBATCH --job-name=sh3
#SBATCH --output=VAE3.out
#SBATCH --partition=gm4-pmext
#SBATCH --nodes=1            # SET NUM NODES 
#SBATCH --gres=gpu:1        # SET NUM GPUS
#SBATCH --ntasks-per-node=1  # SETS NUM MPI RANKS (1 PER GPU)
#SBATCH --cpus-per-task=10    # SET NUM THREADS 
#SBATCH --qos=gm4
# THIS EXAMPLE USES 1 GPU NODE - 1 MPI TASK - 4 THREADS PER TASK

module unload Anaconda3
module unload cuda
module load torch/7.0
module load Anaconda3/2019.03 cuda/10.0
source activate pytorch-gpu-1.2-cuda-10.0

python train_model.py -n SH3

wait
exit 0


