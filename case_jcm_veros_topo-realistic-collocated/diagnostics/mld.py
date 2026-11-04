"""Density-threshold mixed layer depth, shared by the mixed-layer diagnostics.

The mixed layer is the depth where the potential density first exceeds its
value in the top layer by a threshold (0.03 kg/m3 by default, the criterion
of de Boyer Montegut et al. 2004), linearly interpolated between layer
centres. Density is the Veros linear equation of state the run integrates
with (eq_of_state_type = 1). A column that never exceeds the threshold is
mixed to its last wet layer, and its mixed layer depth is that layer's bottom.

The depth is computed per 73-day chunk, so one season's deep mixing is not
averaged away with the temperature and salinity; `winter_and_annual` then
takes, year by year, the deepest chunk (winter) and the mean of the chunks.
"""
import numpy as np
import xarray as xr

CHUNKS_PER_YEAR = 5
DEFAULT_THRESHOLD = 0.03
# Veros linear equation of state (eq_of_state_type = 1)
BETA_T, BETA_S, RHO_0, THETA_0, S_0 = 1.67e-4, 0.78e-3, 1024.0, 9.85, 35.0


class MixedLayer:
    """Vertical grid and wet mask of a run, read once from one of its ocn-*.nc files."""

    def __init__(self, ocn_file, threshold=DEFAULT_THRESHOLD):
        d0 = xr.open_dataset(ocn_file)
        self.lat, self.lon = d0.lat.values, d0.lon.values
        self.wet = d0.mask_T.transpose("lat", "lon", "depth").values[..., ::-1].astype(bool)   # surface first
        dz = d0.dzt.values[::-1]
        d0.close()
        self.zc = np.cumsum(dz) - 0.5 * dz                                              # layer centres
        zb = np.cumsum(dz)                                                              # layer bottoms
        nwet = self.wet.sum(axis=2)
        self.ocean = nwet > 0
        self.column_bottom = np.where(self.ocean, zb[np.maximum(nwet - 1, 0)], np.nan)
        self.threshold = threshold

    def depth(self, ocn_file):
        """Mixed layer depth (m, (lat, lon)) of one chunk, NaN over land."""
        ds = xr.open_dataset(ocn_file)
        T = ds.temp.mean("time").transpose("lat", "lon", "depth").values[..., ::-1]
        S = ds.salt.mean("time").transpose("lat", "lon", "depth").values[..., ::-1]
        ds.close()
        zc, ocean = self.zc, self.ocean
        rho = -(BETA_T * (T - THETA_0) - BETA_S * (S - S_0)) * RHO_0
        rho = np.where(self.wet, rho, np.inf)          # below the seafloor counts as "denser": never inside
        target = rho[..., :1] + self.threshold
        exceeded = rho > target
        k = np.argmax(exceeded, axis=2)                 # first layer denser than the target
        found = exceeded.any(axis=2) & ocean
        jj, ii = np.indices(k.shape)
        km = np.maximum(k - 1, 0)
        r0, r1 = rho[jj, ii, km], rho[jj, ii, k]
        r1 = np.where(np.isfinite(r1), r1, r0)          # the first "denser" layer is the seafloor
        frac = np.where(r1 > r0, (target[..., 0] - r0) / np.where(r1 > r0, r1 - r0, 1.0), 1.0)
        interior = zc[km] + np.clip(frac, 0.0, 1.0) * (zc[k] - zc[km])
        # the first layer below the reference already too dense -> the top layer centre;
        # the seafloor reached first, or never exceeded -> the column bottom
        hits_floor = found & ~np.isfinite(rho[jj, ii, k])
        mld = np.where(k == 0, zc[0], interior)
        mld = np.where(hits_floor | (~found & ocean), self.column_bottom, mld)
        return np.where(ocean, mld, np.nan)

    def winter_and_annual(self, ocn_files):
        """(winter, annual) mean mixed layer depth over consecutive years of five chunks.

        Winter is each year's deepest chunk, cell by cell; annual is the mean of
        its chunks. Both are averaged over the years. A trailing partial year is
        ignored.
        """
        nyear = len(ocn_files) // CHUNKS_PER_YEAR
        if nyear == 0:
            raise ValueError("need at least one complete year (five 73-day chunks)")
        winter, annual = [], []
        for y in range(nyear):
            chunks = np.array([self.depth(f) for f in ocn_files[CHUNKS_PER_YEAR * y:CHUNKS_PER_YEAR * (y + 1)]])
            winter.append(np.nanmax(chunks, axis=0))
            annual.append(np.nanmean(chunks, axis=0))
        return np.nanmean(winter, axis=0), np.nanmean(annual, axis=0)
