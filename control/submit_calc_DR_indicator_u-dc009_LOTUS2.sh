#!/bin/bash
#SBATCH --account=kscale                        # account (usually a GWS)
#SBATCH --partition=standard                    # partition
#SBATCH --qos=standard                          # quality of service
#SBATCH --array=1                               # batch array
#SBATCH --time=8:00:00                          # walltime
#SBATCH --mem=64G                               # total memory (can also specify per-node, or per-core)
#SBATCH --ntasks=1                              # number of tasks (should force just 1 node and 1 CPU core)
#SBATCH --job-name="calc_DR_u-dc009"            # job name
#SBATCH --output=/home/users/dship/log/log_calc_DR_indicator_DS_u-dc009_%a.out      # output file
#SBATCH --error=/home/users/dship/log/log_calc_DR_indicator_DS_u-dc009_%a.err       # error file
#SBATCH

SCRIPTPATH="/home/users/dship/python/LoSSETT"

python3 $SCRIPTPATH/preprocess_u-dc009.py
