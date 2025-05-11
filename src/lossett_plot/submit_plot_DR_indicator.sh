#!/bin/bash
#SBATCH --partition=short-serial                # partition
#SBATCH --array=0-39                            # job array (item identifier is %a)
#SBATCH --time=00:20:00                         # walltime
#SBATCH --mem=2G                               # total memory (can also specify per-node, or per-core)
#SBATCH --job-name="plot_DR_indicator"          # job name
#SBATCH --output=/home/users/dship/log/log_plot_DR_indicator_DS_CTC5RAL_%a.out      # output file
#SBATCH --error=/home/users/dship/log/log_plot_DR_indicator_DS_CTC5RAL_%a.err       # error file
#SBATCH

LOGPATH="/home/users/dship/log"
SCRIPTPATH="/home/users/dship/python/LoSSETT"

simid="CTC5RAL"
#simid="CTC5GAL"

# create list of dates
start_date="2016-08-01"
end_date="2016-09-10"
date=$start_date
dates=()
while [[ "$date" != "$end_date" ]]; do
    dates+=($date)
    date=$(date --date "$date + 1 day" +"%Y-%m-%d")
done

echo "Start Job $SLURM_ARRAY_TASK_ID on $HOSTNAME"  # Display job start information

python3 $SCRIPTPATH/plot_DR_indicator.py $simid ${dates[$SLURM_ARRAY_TASK_ID]}
