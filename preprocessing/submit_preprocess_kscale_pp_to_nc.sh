#!/bin/bash
#SBATCH --account=kscale                        # account (usually a GWS)
#SBATCH --partition=standard                    # partition
#SBATCH --qos=standard                          # quality of service
#SBATCH --array=[12] #[1-31]                             # job array (item identifier is %a)
#SBATCH --time=01:00:00                         # walltime
#SBATCH --ntasks=1                              # not quite sure if this is the right way to specify number of processes?
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1 
#SBATCH --mem-per-cpu=16G                       # memory per cpu (should end up as memory per task)
#SBATCH --job-name="kscale_pp_to_nc"            # job name
#SBATCH --output=/home/users/dship/log/log_kscale_pp_to_nc_%a.out      # output file
#SBATCH --error=/home/users/dship/log/log_kscale_pp_to_nc_%a.err       # error file
#SBATCH

SCRIPTPATH="/home/users/dship/python"

year=$1
month=$2

echo "Start Job $SLURM_JOB_ID, array $SLURM_ARRAY_TASK_ID on $HOSTNAME"  # Display job start information

cd $SCRIPTPATH

hours=("0" "12")

for hour in "${hours[@]}"
do
    echo $hour
    python3 -m LoSSETT.preprocessing.preprocess_kscale "DYAMOND3" "n2560RAL3" "none" ${year} ${month} ${SLURM_ARRAY_TASK_ID} ${hour}
done
