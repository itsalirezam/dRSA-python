#!/bin/bash
#$ -N dask-worker
#$ -cwd
#$ -o _sh_outputs/worker_output.txt     # Combined output
#$ -j y                     # Join stderr into stdout
#$ -pe smp 15               # Number of cores to request - dont take all 64 to occupy the whole node!
#$ -l h_rt=250:00:00         # Runtime limit
#$ -l mem_free=20G         # only place this job on nodes that have at least nG free for EACH SLOT


# -l h_vmem=30G           # this is a hard limit, of memory usage exeeds this, SGE will kill the job (also per slot) (safety cap)

source ~/miniconda3/bin/activate
conda activate base

# Connect to the scheduler (replace IP)
dask-worker tcp://10.40.40.23:8786 --nthreads 1 --nworkers 15

# nworkers is how many python processes is done in parallel 

# submit like this: qsub -t 1-n_nodes worker.sh - e.g. " qsub -t 1-5 worker.sh " this uses 5 nodes with 64 cpu cores each: 5 * 64 = 320 cores


# change the ip to the one scheduler.sh outputs
# qsub -t 1-20 worker.sh