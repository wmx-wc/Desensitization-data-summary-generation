#!/bin/bash
#SBATCH --job-name=varicos_p
#SBATCH --output=lr_pre_train.out
#SBATCH --error=vasp_pre_train.err
#SBATCH --nodelist=g-1-0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
source activate pytorch
python pre_train.py
