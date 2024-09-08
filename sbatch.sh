#!/bin/bash
#SBATCH -L gracehopper
#SBATCH -p highmem      # partition 
#SBATCH -q grp_scai_research_priority # queue
#SBATCH -t 1-12:00:00   # time in d-hh:mm:ss
#SBATCH -G 1            # number of GPU 
#SBATCH -N 1            # number of nodes
#SBATCH -c 8            # number of cores 
#SBATCH -o slurms/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurms/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%nngu2@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment


#Load CuDA
module load cuda-12.6.1-gcc-12.1.0

source a.sh