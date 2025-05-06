#!/bin/bash
#SBATCH --account=kscale                        # account (usually a GWS)
#SBATCH --partition=standard                    # partition
#SBATCH --qos=high                              # quality of service
#SBATCH --array=[2]                             # job array (item identifier is %a)
#SBATCH --time=04:00:00                         # walltime
#SBATCH --ntasks=4                              # not quite sure if this is the right way to specify number of processes?
#SBATCH --mem=250G                              # total memory (can also specify per-node, or per-core)
#SBATCH --job-name="calc_DR_kscale_native"         # job name
#SBATCH --output=/home/users/dship/log/log_calc_DR_indicator_kscale_native_%a.out      # output file
#SBATCH --error=/home/users/dship/log/log_calc_DR_indicator_kscale_native_%a.err       # error file
#SBATCH

SCRIPTPATH="/home/users/dship/python"

max_r_deg=$1

echo "Start Job $SLURM_ARRAY_TASK_ID on $HOSTNAME"  # Display job start information

cd $SCRIPTPATH

# should really take options from a yaml options file! Then the run script could just load the opt file
python3 -m LoSSETT.control.run_lossett_kscale_native "DYAMOND3" "n2560RAL3" "none" "2020-01-21" "00" ${max_r_deg}
