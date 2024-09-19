#!/bin/bash
#SBATCH -L gracehopper
#SBATCH -p highmem      # partition 
#SBATCH -q grp_scai_research_priority # queue
#SBATCH -t 7-00:00:00   # time in d-hh:mm:ss
#SBATCH -G 1            # number of GPU 
#SBATCH -N 1            # number of nodes
#SBATCH -c 1            # number of cores 
#SBATCH -o slurms/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurms/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%nngu2@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment

#Load CuDA
module load cuda-12.4.1-gcc-11.4.1
module load mamba/latest

mamba create -n nngu2 -c conda-forge
source activate nngu2

mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install anaconda::pandas
mamba install conda-forge::datasets
mamba install conda-forge::transformers[torch]
mamba install anaconda::pillow
conda install conda-forge::albumentations
conda install conda-forge::timm
conda install conda-forge::torchmetrics
conda install conda-forge::dataclasses
conda install anaconda::numpy
conda install conda-forge::pycocotools
conda install conda-forge::wandb
conda install anaconda::pandas

source train_airsim_models.sh