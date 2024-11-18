#!/bin/bash

# Example bash script for submission to SLURM-managed batch scripting system
# Queue names here are for JASMIN LOTUS

# queue to submit to, e.g. test or short-serial
queue="test"
# number of nodes (can only be 1 if a serial queue)
n_proc=2
# memory in MB
mem=32000
# time HH:mm:ss
time=01:00:00

user=dship
LOGPATH="/home/users/${user}/log"
SCRIPTPATH="/home/users/dship/python/LoSSETT"

sbatch -p ${queue} -o $LOGPATH/log_lossett_dyn_TEST.out -e $LOGPATH/log_lossett_dyn_TEST.err --mem ${mem} --time ${time} -n ${n_proc} $SCRIPTPATH/LoSSETT_DYN.py

