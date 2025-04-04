#!/bin/bash

# Example bash script for submission to SLURM-managed batch scripting system
# Queue names here are for JASMIN LOTUS 1

## LOTUS2
#account="parachute"
#account="kscale"
#qos="debug"
#partition="debug"
#n_proc=1
#mem=32000
#time=00:30:00
#qos="standard"
#partition="standard"
#n_proc=1
#mem=32000
#time=24:00:00

# queue to submit to, e.g. test or short-serial
queue="short-serial"
# number of nodes (can only be 1 if a serial queue)
n_proc=1
# memory in MB
mem=48000
# time HH:mm:ss
time=01:00:00

user=dship
LOGPATH="/home/users/${user}/log"
SCRIPTPATH="/home/users/dship/python/LoSSETT"

sbatch -p ${queue} -o $LOGPATH/log_calc_scale_increments.out -e $LOGPATH/log_calc_scale_increments.err --mem ${mem} --time ${time} -n ${n_proc} $SCRIPTPATH/CalcScaleIncrements.py
#sbatch --account=${account} --partition=${partition} --qos=${qos} -o $LOGPATH/log_calc_scale_increments.out -e $LOGPATH/log_calc_scale_increments.err --mem ${mem} --time ${time} -n ${n_proc} $SCRIPTPATH/CalcScaleIncrements.py
