# JemLab (Temporary)

This project couples the atmosphere model JAX-GCM (JCM), ocean model Veros, and Winton sea-ice model, and a slab land model together through JAX-ESM (JEM), in a realistic-topography setup. 

# Quickstart

Install

```
git clone -b fix-sea-ice-temperature-blending https://github.com/climate-analytics-lab/jax-gcm.git
git clone -b dev0.1 https://github.com/climate-analytics-lab/jax-esm.git
git clone https://github.com/meteorologytoday/veros-jittable.git

jcm_path=`pwd`/jax-gcm
jem_path=`pwd`/jax-esm
veros_path=`pwd`/veros-jittable

export PYTHONPATH=$jcm_path:$jem_path:$veros_path
```

Run
```
git clone -b dev0.1 https://github.com/meteorologytoday/JemLab.git
cd JemLab/case_jcm_veros_topo-realistic-collocated
bash run.sh
```


