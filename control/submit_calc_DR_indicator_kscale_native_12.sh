#!/bin/bash
# Account, partition, QOS
#SBATCH --account=kscale                        # account (usually a GWS)
#SBATCH --partition=standard                    # partition
#SBATCH --qos=high                              # quality of service

# Main job spec
#SBATCH --array=[1-10] #[1-27,30]                            # job array (item identifier is %a)
#SBATCH --time=06:00:00                        # walltime
#SBATCH --ntasks=4 #1                              # not quite sure if this is the right way to specify number of processes?
##SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
##SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50G #70G                       # memory per cpu (should end up as memory per task)

# Output & monitoring
#SBATCH --job-name="calc_DR_kscale_native_12Z"         # job name
#SBATCH --output=/home/users/dship/log/log_calc_DR_indicator_kscale_native_12Z_%a.out      # output file
#SBATCH --error=/home/users/dship/log/log_calc_DR_indicator_kscale_native_12Z_%a.err       # error file
#SBATCH --mail-user=daniel.shipley@reading.ac.uk
#SBATCH --mail-type=ALL
#SBATCH

SCRIPTPATH="/home/users/dship/python"

max_r_deg=$1
tstep=$2
plev=$3

echo "Start Job $SLURM_JOB_ID, array $SLURM_ARRAY_TASK_ID on $HOSTNAME"  # Display job start information

cd $SCRIPTPATH

# should really take options from a yaml options file! Then the run script could just load the opt file
python3 -m LoSSETT.control.run_lossett_kscale_native "DYAMOND3" "n2560RAL3" "none" 2020 9 ${SLURM_ARRAY_TASK_ID} "12" ${max_r_deg} ${tstep} ${plev}
