#!/bin/bash
#SBATCH --partition=short-serial                # partition
#SBATCH --array=27                            # job array (item identifier is %a)
#SBATCH --time=02:00:00                         # walltime
#SBATCH --mem=64G                               # total memory (can also specify per-node, or per-core)
#SBATCH --job-name="calc_DR_CTC5RAL"            # job name
#SBATCH --output=/home/users/dship/log/log_calc_DR_indicator_DS_CTC5RAL_%a.out      # output file
#SBATCH --error=/home/users/dship/log/log_calc_DR_indicator_DS_CTC5RAL_%a.err       # error file
#SBATCH

LOGPATH="/home/users/dship/log"
SCRIPTPATH="/home/users/dship/python/LoSSETT"

echo "Start Job $SLURM_ARRAY_TASK_ID on $HOSTNAME"  # Display job start information

python3 $SCRIPTPATH/CalcScaleIncrements.py "CTC5RAL" ${SLURM_ARRAY_TASK_ID}
