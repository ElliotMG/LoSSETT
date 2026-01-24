#!/usr/bin/bash

module load jaspy # for some reason jaspy contains cdo

DATA_DIR="/gws/nopw/j04/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/DYAMOND_SUMMER/"
#fname_root="CTC_n2560_RAL3p2.n1280_GAL9.uvw_201609"
fname_root="CTC_n2560_GAL9.n1280_GAL9.uvw_20160"

# interpolate w to u,v points; save as NetCDF
#sbatch /gws/nopw/j04/kscale/USERS/dship/lossett_run/submit_preprocess_kscale_pp_to_nc.sh "DYAMOND_SUMMER" "n1280GAL9" "CTC_n2560_GAL9" 2016 8

# 1st order conservative remapping to n1280
cd $DATA_DIR
mkdir -p n1280_regrid
for file in *${fname_root}*.nc; do
    cdo remapcon,/gws/nopw/j04/kscale/USERS/dship/grid_description_n1280.txt \
	"${file}" "n1280_regrid/${file/.nc/_n1280_untrimmed.nc}"
done

# subset CTC
cd $DATA_DIR/n1280_regrid
for file in *${fname_root}*_untrimmed.nc; do
    cdo sellonlatbox,180,-180,26,-40 "${file}" "${file/_untrimmed/}"
done

# clean up
# remove untrimmed n1280 files
cd $DATA_DIR/n1280_regrid
rm *${fname_root}*untrimmed*.nc

# rename CTC -> channel
cd $DATA_DIR/n1280_regrid
for file in *${fname_root}*.nc; do
    mv "${file}" "${file/CTC/channel}"
done

# remove n2560 files
#cd $DATA_DIR
#rm *${fname_root}*.nc
