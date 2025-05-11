#!/bin/bash
#SBATCH --account=kscale                        # account (usually a GWS)
#SBATCH --partition=standard                    # partition
#SBATCH --qos=standard                          # quality of service
#SBATCH --time=00:30:00                         # walltime
#SBATCH --mem=32G                               # total memory (can also specify per-node, or per-core)
#SBATCH --job-name="plot_DR_vs_ell"          # job name
#SBATCH --output=/home/users/dship/log/log_plot_DR_vs_ell_ERA5.out      # output file
#SBATCH --error=/home/users/dship/log/log_plot_DR_vs_ell_ERA5.err       # error file
#SBATCH

LOGPATH="/home/users/dship/log"
SCRIPTPATH="/home/users/dship/python/LoSSETT/plotting"

#simid="CTC5RAL"
#simid="CTC5GAL"
simid="ERA5"

echo "Start Job $SLURM_ARRAY_TASK_ID on $HOSTNAME"  # Display job start information

python3 $SCRIPTPATH/plot_DR_vs_ell.py $simid
