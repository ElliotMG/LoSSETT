#!/bin/bash
#SBATCH --account=kscale                        # account (usually a GWS)
#SBATCH --partition=standard                    # partition
#SBATCH --qos=standard                          # quality of service
#SBATCH --array=1 #[1-31]                               # batch array
#SBATCH --time=4:00:00 #8:00:00                          # walltime
#SBATCH --mem=128G #64G                               # total memory (can also specify per-node, or per-core)
#SBATCH --ntasks=1                              # number of tasks (should force just 1 node and 1 CPU core)
#SBATCH --job-name="calc_DR_era5"            # job name
#SBATCH --output=/home/users/dship/log/log_calc_DR_indicator_era5_%a.out      # output file
#SBATCH --error=/home/users/dship/log/log_calc_DR_indicator_era5_%a.err       # error file
#SBATCH

SCRIPTPATH="/home/users/dship/python"
year=2016
month=8

echo "Start Job $SLURM_ARRAY_TASK_ID on $HOSTNAME"  # Display job start information

cd $SCRIPTPATH

python3 -m LoSSETT.control.run_lossett_era5_0p5deg $year $month ${SLURM_ARRAY_TASK_ID}
