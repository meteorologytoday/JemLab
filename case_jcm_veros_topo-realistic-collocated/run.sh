#!/bin/bash

grid_folder=../data/grid_ESM_collocated

runoff_to_ocean_mapping=$grid_folder/weight_river_routing_JCM_T31.nc
land_forcing_file=$grid_folder/forcing_era5_t31_gridlat.nc
ocean_init_file=$grid_folder/ocean_ic_t31.nc

python3 -c 'import jcm; import jem; import veros; print("jcm : ", jcm.__file__); print("jem: ", jem.__file__); print("veros : ", veros.__file__);'
time python3 main_forward.py \
    --total-simulation-days $(( 365 * 5 ))          \
    --simulation-name  spinup                       \
    --grid-folder $grid_folder                      \
    --ocean-initial-condition-file $ocean_init_file \
    --land-forcing-file $land_forcing_file          \
    --runoff-to-ocean-mapping $runoff_to_ocean_mapping \
    --land-free-temperature                         \
    --land-free-soil-water-availability
