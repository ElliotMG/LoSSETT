#!/bin/bash

DIRI="/gws/nopw/j04/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/DYAMOND_SUMMER/n1280_regrid/embedded"
DIRO="/gws/nopw/j04/kscale/USERS/emg/data/DYAMOND_Summer/embedded/0p5deg"

# create output directory if it doesn't exist
mkdir -p "$DIRO"

# Load these modules: doesn't read the loaded modules in the terminal
module load jaspy jasmin-sci

for f in "$DIRI"/channel_n2560_GAL9*.nc; do
    fname=$(basename "$f")
    cdo remapcon,r720x360 "$f" "$DIRO/${fname/n1280/0p5deg}"
done