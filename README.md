# JemLab

This project is a testbed for [JAX-ESM (JEM)](https://github.com/climate-analytics-lab/jax-esm) runs. The latest version couples the atmosphere model [JAX-GCM (JCM)](https://github.com/climate-analytics-lab/jax-gcm), [jittable version ocean model Veros](https://github.com/meteorologytoday/veros-jittable), and Winton sea-ice model (within JEM), and a slab land model (within JEM) in a realistic-topography generated from [EarthSystemGrid.py](https://github.com/meteorologytoday/EarthSystemGrids.py) setup.

# Quickstart

Set up a new environment with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install) or [miniconda](https://continuumio-docs.readthedocs-hosted.com/miniconda/install/)
```
conda create -n jemlab python=3.13 -y
conda activate jemlab
```
Clone and set path
```
git clone -b fix-sea-ice-temperature-blending https://github.com/climate-analytics-lab/jax-gcm.git
git clone -b dev1.0 https://github.com/climate-analytics-lab/jax-esm.git
git clone https://github.com/meteorologytoday/veros-jittable.git

jcm_path=`pwd`/jax-gcm
jem_path=`pwd`/jax-esm
veros_path=`pwd`/veros-jittable

export PYTHONPATH=$jcm_path:$jem_path:$veros_path
```
OR install
```
git clone -b fix-sea-ice-temperature-blending https://github.com/climate-analytics-lab/jax-gcm.git
git clone -b dev1.0 https://github.com/climate-analytics-lab/jax-esm.git
git clone https://github.com/meteorologytoday/veros-jittable.git

for repo in jax-gcm jax-esm veros-jittable ; do
  pushd $repo
  pip install -e .
  popd
done
```

Run
```
git clone -b dev0.1 https://github.com/meteorologytoday/JemLab.git
cd JemLab/case_jcm_veros_topo-realistic-collocated
bash run.sh
```


