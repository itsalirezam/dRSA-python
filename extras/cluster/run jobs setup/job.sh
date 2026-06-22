#!/bin/bash
#$ -N g1-syntacticTest                # Name of the job
#$ -cwd                       # Run from current directory
#$ -o _sh_outputs/job_output.txt             # Combined output file
#$ -j y                       # Join stderr with stdout
#$ -pe smp 1                 # Number of CPU cores to use
#$ -l h_rt=250:00:00          # Runtime limit
#$ -l mem_free=200G

# start the job with 150G free memory, kill it if it reaches 200

# -l hostname=compute-0-0 # you can choose a certain node

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate base

cd /mnt/storage/tier2/morwur/Projects/ALIREZA/speech_project_workspace/code

for i in {1..9}; do
    time python _main_v7_cluster.py $i
done

# time python _main_v7_cluster.py 1

# IFG:
# for i in 1 3 4 5 6 7 9; do
#     time python _main_v7_cluster.py $i
# done

# _main_v7_cluster