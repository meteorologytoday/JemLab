# MetESM

This project couples the atmosphere model JAX-GCM (JCM), the ocean model Veros, and the land model JAX-LAND through JAX-ESM (JEM), in a realistic-topography setup.

## Quickstart

Install

```
git clone -b fix-sea-ice-temperature-blending https://github.com/climate-analytics-lab/jax-gcm.git
git clone -b dev0.1 https://github.com/climate-analytics-lab/jax-esm.git
git clone https://github.com/meteorologytoday/veros-jittable.git

jcm\_path=`pwd`/jcm
jem\_path=`pwd`/jax-esm
veros\_path=`pwd`/veros-jittable

export PYTHONPATH=$jcm\_path:$jem\_path:$veros\_path
```

Run

```
git clone -b dev0.1 https://github.com/meteorologytoday/UnicornESM.git
cd UnicornESM/case\_jcm\_veros\_topo-realistic-collocated
bash run.sh
```


