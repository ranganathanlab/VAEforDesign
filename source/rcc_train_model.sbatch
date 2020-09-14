#!/bin/zsh
#SBATCH --job-name=sh3VAE
#SBATCH --output=VAE3.out
#SBATCH --partition=fela
#SBATCH --account=pi-andrewferguson
#SBATCH --mail-type=END
#SBATCH --gres=gpu:1         # SET NUM GPUS
#SBATCH --ntasks-per-node=1  # SETS NUM MPI RANKS (1 PER GPU)
#SBATCH --cpus-per-task=10    # SET NUM THREADS 

module unload Anaconda3
module unload cuda
module load Anaconda3/2019.03 cuda/10.0
source activate pytorch-gpu-1.2-cuda-10.0

python train_model.py -n SH3

wait
exit 0

# EOF
