#!/bin/bash
#SBATCH --account=def-dividino          # your Compute Canada account
#SBATCH --gres=gpu:v100:1              # request 1 V100 GPU
#SBATCH --cpus-per-task=3              # ~3 CPU cores per GPU
#SBATCH --mem=12G                      # 12 GB RAM
#SBATCH --time=3-00:00                 # 1 day
#SBATCH --job-name=gpu_test
#SBATCH --output=gpu-%j.out            # stdout log
#SBATCH --error=gpu-%j.err             # stderr log

module load StdEnv/2023               # load standard environment
module load cuda                      # load CUDA (if needed)
nvidia-smi                            # debug GPU info
python src/main.py                # your training code
