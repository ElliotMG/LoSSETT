#!/bin/bash
#SBATCH --account=kscale                        # account (usually a GWS)
#SBATCH --partition=debug                       # partition
#SBATCH --qos=debug                             # quality of service
#SBATCH --time=00:15:00                         # walltime
#SBATCH --mem=16G                               # total memory (can also specify per-node, or per-core)
#SBATCH --ntasks=1                              # number of tasks (should force just 1 node and 1 CPU core)
#SBATCH --job-name="calc_DR_chunk_test"         # job name
#SBATCH --output=/home/users/dship/log/log_calc_DR_indicator_chunk_test.out      # output file
#SBATCH --error=/home/users/dship/log/log_calc_DR_indicator_chunk_test.err       # error file
#SBATCH

SCRIPTPATH="/home/users/dship/python"
year=2005
month=1
day=5

echo "Start Job $SLURM_JOB_ID on $HOSTNAME"  # Display job start information

cd $SCRIPTPATH

python3 -m LoSSETT.control.run_lossett_era5_0p5deg $year $month $day
