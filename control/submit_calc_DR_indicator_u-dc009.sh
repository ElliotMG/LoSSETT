#!/bin/bash
#SBATCH --account=kscale                        # account (usually a GWS)
#SBATCH --partition=standard                    # partition
#SBATCH --qos=standard                          # quality of service
#SBATCH --array=1                               # batch array
#SBATCH --time=8:00:00                          # walltime
#SBATCH --mem=128G                              # total memory (can also specify per-node, or per-core)
#SBATCH --ntasks=1                              # number of tasks (should force just 1 node and 1 CPU core)
#SBATCH --job-name="calc_DR_u-dc009"            # job name
#SBATCH --output=/home/users/dship/log/log_calc_DR_indicator_u-dc009_1980_%a.out      # output file
#SBATCH --error=/home/users/dship/log/log_calc_DR_indicator_u-dc009_1980_%a.err       # error file
#SBATCH

month=$1

SCRIPTPATH="/home/users/dship/python"

cd $SCRIPTPATH

echo "Start Job $SLURM_ARRAY_TASK_ID on $HOSTNAME"  # Display job start information

echo "month = $1"

echo "day = ${SLURM_ARRAY_TASK_ID}"

python3 -m LoSSETT.control.run_lossett_u-dc009 $month ${SLURM_ARRAY_TASK_ID}
