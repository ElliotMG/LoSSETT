#!/bin/bash
#SBATCH --account=kscale                        # account (usually a GWS)
#SBATCH --partition=standard                    # partition
#SBATCH --qos=standard                          # quality of service
#SBATCH --array=[1-40]                          # job array (item identifier is %a)
#SBATCH --time=02:00:00                         # walltime
#SBATCH --mem=64G                               # total memory (can also specify per-node, or per-core)
#SBATCH --job-name="calc_DR_DS_CTC5RAL"         # job name
#SBATCH --output=/home/users/dship/log/log_calc_DR_indicator_DS_CTC5RAL_%a.out      # output file
#SBATCH --error=/home/users/dship/log/log_calc_DR_indicator_DS_CTC5RAL_%a.err       # error file
#SBATCH

SCRIPTPATH="/home/users/dship/python"

echo "Start Job $SLURM_ARRAY_TASK_ID on $HOSTNAME"  # Display job start information

cd $SCRIPTPATH

python3 -m LoSSETT.control.run_lossett_kscale_0p5deg "DS" "CTC5RAL" ${SLURM_ARRAY_TASK_ID}
