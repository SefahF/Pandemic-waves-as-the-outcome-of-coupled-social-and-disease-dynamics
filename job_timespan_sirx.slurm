#!/bin/bash
#SBATCH --job-name=SIRX     # create a short name for your job
#SBATCH --partition=hagrid_batch
#SBATCH --account=hagrid
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=40:59:00          # total run time limit (HH:MM:SS)
#SBATCH --array=1,7           # job array with index values 0, 1, 2, 3, 4


#module purge
#$ module --ignore_cache load "anaconda3/2020.10"

# python timespan_osi_copy.py


python timespan_osi.py
