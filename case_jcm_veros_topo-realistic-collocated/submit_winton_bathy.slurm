#!/bin/bash
#
#SBATCH --job-name=JEM-winton-bathy
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --gres=gpu:1
#SBATCH --output=sbatch.64.%j.out
#SBATCH --mail-type=ALL,TIME_LIMIT
#SBATCH --time=24:00:00

nvidia-smi

echo "pwd: $(pwd)"
eval "$( /home/tienyiao/miniconda3/bin/conda shell.bash hook 2> /dev/null)"
conda activate jax

source /home/tienyiao/.bashrc_jem
echo "PYTHONPATH=$PYTHONPATH"

export PYTHONHASHSEED=0

#export XLA_FLAGS="--xla_gpu_deterministic_ops=true"

# Symlink to the grid/topography/weight files produced by EarthSystemGrids'
# preprocessing_workflow.
grid_folder=../grid_ESM_collocated

# Resubmitting this same script resumes the run from its checkpoint
# (output_T31/<simulation-name>/checkpoint).
# Checkpoints written before JEM split WintonForcing's precipitation (f759ae4)
# need one extra conversion; it leaves an already-converted one alone.
ckpt=output_T31/winton_bathy/checkpoint/carry.msgpack
[ -f $ckpt ] && python3 tools/upgrade_winton_checkpoint.py $ckpt

time python3 main_forward.py \
    --total-simulation-days $(( 365 * 500 ))         \
    --simulation-interval-days 73                   \
    --checkpoint-interval 10                        \
    --truncation-number 31                          \
    --simulation-name winton_bathy \
    --jcm-timestep-min 30                           \
    --veros-timestep-min 60                         \
    --grid-folder $grid_folder                      \
    --ocean-initial-surface-temperature 1.0         \
    --land-forcing-file forcing_era5_t31_gridlat.nc         \
    --ocean-surface-thickness 30                    \
    --no-seaice-transport                           \
    --jcm-advection semi_lagrangian                 #\
    #--do-not-average-time
