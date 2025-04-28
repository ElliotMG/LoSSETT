#!/bin/bash

# Example bash script for submission to SLURM-managed batch scripting system
# Queue names here are for JASMIN LOTUS

# queue to submit to, e.g. test or short-serial
queue="short-serial"
# number of nodes (can only be 1 if a serial queue)
n_proc=1
# memory in MB
mem=48000
# time HH:mm:ss
time=24:00:00

# For migration to LOTUS 2:
# need to add --account ${ACCOUNT_NAME} --qos ${QOS_NAME}

user=emg97
LOGPATH="/home/users/${user}/log"
SCRIPTPATH="/home/users/${user}/emgScripts/LoSSETT"
# ACCOUNT_NAME=emg97
# QOS_NAME=

module load jaspy
which python3
python3 --version

sbatch -p ${queue} -o $LOGPATH/log_lossett_dyn_TEST.out -e $LOGPATH/log_lossett_dyn_TEST.err --mem ${mem} --time ${time} -n ${n_proc} $SCRIPTPATH/LoSSETT_DYN.py

