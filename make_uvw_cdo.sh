#!/bin/bash
model="channel_n2560_GAL9"
mkdir -p output

output_file="/gws/nopw/j04/kscale/USERS/emg/data/DYAMOND_Summer/${model}_uvw_DS.nc"
temp_file="/tmp/${model}_uvw_temp.nc"

rm -f "$output_file"

# Generate dates from 20160801 to 20160909
python3 -c "
from datetime import date, timedelta
start = date(2016, 8, 1)
end = date(2016, 9, 9)
dates = [start + timedelta(days=x) for x in range((end-start).days + 1)]
print('\n'.join(d.strftime('%Y%m%d') for d in dates))
" | while read date; do
   for level in 1000 925 850 700 600 500 400 300 250 200 150 100; do
       dir="/gws/nopw/j04/kscale/DATA/outdir_20160801T0000Z/DMn1280GAL9/${model}/profile_${level}/"
       for file in "$dir"${date}_20160801T0000Z_channel_profile_3hourly_*_05deg.nc; do
           cdo -selvar,x_wind,y_wind,upward_air_velocity "$file" "$temp_file"
           
           if [ ! -f "$output_file" ]; then
               mv "$temp_file" "$output_file"
           else
               cdo merge "$output_file" "$temp_file" "$output_file.merged"
               mv "$output_file.merged" "$output_file"
           fi
       done
   done
done

rm -f "$temp_file"