#!/bin/bash
#SBATCH --partition=short-serial                # partition
#SBATCH --time=8:00:00                          # walltime
#SBATCH --mem=64G                               # total memory (can also specify per-node, or per-core)
#SBATCH --ntasks=1                              # number of tasks (should force just 1 node and 1 CPU core)
#SBATCH --job-name="calc_DR_u-dc009"            # job name
#SBATCH --output=/home/users/dship/log/log_calc_DR_indicator_DS_u-dc009_%a.out      # output file
#SBATCH --error=/home/users/dship/log/log_calc_DR_indicator_DS_u-dc009_%a.err       # error file
#SBATCH

python3 $SCRIPTPATH/preprocess_u-dc009.py
