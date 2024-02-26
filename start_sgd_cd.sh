#!/usr/bin/env bash
#SBATCH -A ####
#SBATCH -t 2-00:00:00
#SBATCH -o out_sgd_cd.txt
#SBATCH -e err_sgd_cd.txt
#SBATCH -n 1


ml SciPy-bundle/2022.05-foss-2022a
source ~/my_python/bin/activate
python sgd_cd.py seed=$SLURM_ARRAY_TASK_ID $*
