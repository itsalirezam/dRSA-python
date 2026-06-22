#!/bin/bash
#$ -N dask-scheduler
#$ -cwd
#$ -o _sh_outputs/scheduler_output.txt
#$ -j y
#$ -l h_rt=250:00:00
#$ -l mem_free=32G

# -l hostname=compute-0-0 # use the node with the least cpus

source ~/miniconda3/bin/activate
conda activate base

# Start the scheduler on this node and bind it to the node's IP
dask-scheduler --host $(hostname -f)

# run qsub scheduler.sh to get the ip - then give it to wrorker.sh
#  qsub scheduler.sh