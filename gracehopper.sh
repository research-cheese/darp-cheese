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

mamba create -n venv -c conda-forge
source activate venv

mamba install -c conda-forge torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu125
mamba install -c conda-forge peft

# datasets
mamba install -c conda-forge datasets

# transformers
mamba install -c conda-forge transformers[torch]

mamba install -c conda-forge pillow
mamba install -c conda-forge albumentations
mamba install -c conda-forge timm
mamba install -c conda-forge torchmetrics
mamba install -c conda-forge dataclasses
mamba install -c conda-forge numpy
mamba install -c conda-forge pycocotools
mamba install -c conda-forge transformers
mamba install -c conda-forge wandb
source train_airsim_models.sh