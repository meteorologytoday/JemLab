#!/bin/bash

grid_folder=../data/grid_ESM_collocated
land_forcing_file=$grid_folder/forcing_era5_t31_gridlat.nc

python3 -c 'import jcm; import jem; import veros; print("jcm : ", jcm.__file__); print("jem: ", jem.__file__); print("veros : ", veros.__file__);'
time python3 main_forward.py \
    --total-simulation-days $(( 365 * 5 ))          \
    --simulation-interval-days 73                   \
    --checkpoint-interval 10                        \
    --truncation-number 31                          \
    --simulation-name  spinup                       \
    --jcm-timestep-min 30                           \
    --veros-timestep-min 60                         \
    --grid-folder $grid_folder                      \
    --ocean-initial-surface-temperature 10.0        \
    --land-forcing-file $land_forcing_file          \
    --ocean-surface-thickness 30                    \
    --no-seaice-transport                           \
    --jcm-advection semi_lagrangian                 #\
    #--do-not-average-time
