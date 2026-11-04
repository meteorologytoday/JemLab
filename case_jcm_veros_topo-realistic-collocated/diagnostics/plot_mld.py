"""Mixed layer depth of a JEM/Veros run from a density threshold.

The mixed layer is the depth where the potential density first exceeds its
value in the top layer by `--threshold` (default 0.03 kg/m3, the criterion of
de Boyer Montegut et al. 2004), linearly interpolated between layer centres.
Density is the Veros linear equation of state the run integrates with
(eq_of_state_type = 1). A column that never exceeds the threshold is mixed to
its last wet layer, and its mixed layer depth is that layer's bottom.

The depth is computed from every 73-day chunk separately, so the deep mixing
of one season is not averaged away with the temperature and salinity. The
figure shows, averaged over the chosen years:
  winter (maximum)  the deepest of each year's five chunks, cell by cell
  annual mean       the mean of the five chunks
Reads the run's ocn-*.nc output (temp, salt, mask_T, dzt).

Examples:
    python diagnostics/plot_mld.py output_T31/winton_ctl mld.png
    python diagnostics/plot_mld.py output_T31/winton_ctl mld.png --last-years 10 --threshold 0.125
"""
import argparse, glob, os
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm

CHUNKS_PER_YEAR = 5
# Veros linear equation of state (eq_of_state_type = 1)
BETA_T, BETA_S, RHO_0, THETA_0, S_0 = 1.67e-4, 0.78e-3, 1024.0, 9.85, 35.0

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run_dir", help="Run output directory (holds ocn-*.nc).")
parser.add_argument("out_png", help="Output figure.")
window = parser.add_mutually_exclusive_group()
window.add_argument("--last-years", type=int, default=5, help="Average over the last N complete years (default 5).")
window.add_argument("--years", type=int, nargs=2, metavar=("FIRST", "LAST"),
                    help="Average over model years FIRST..LAST (1-based, inclusive) instead.")
parser.add_argument("--threshold", type=float, default=0.03, help="Density threshold in kg/m3 (default 0.03).")
args = parser.parse_args()

files = sorted(glob.glob(f"{args.run_dir}/ocn-*.nc"))
nyear = len(files) // CHUNKS_PER_YEAR
if nyear == 0:
    parser.error(f"no complete year of ocn-*.nc output in {args.run_dir}")
first, last = args.years if args.years else (max(1, nyear - args.last_years + 1), nyear)
if not 1 <= first <= last <= nyear:
    parser.error(f"years must lie in 1..{nyear} (complete years so far)")

d0 = xr.open_dataset(files[0])
lat, lon = d0.lat.values, d0.lon.values
wet = d0.mask_T.transpose("lat", "lon", "depth").values[..., ::-1].astype(bool)     # surface first
dz = d0.dzt.values[::-1]
zc = np.cumsum(dz) - 0.5 * dz                                                       # layer centres
zb = np.cumsum(dz)                                                                  # layer bottoms
nwet = wet.sum(axis=2)
ocean = nwet > 0
column_bottom = np.where(ocean, zb[np.maximum(nwet - 1, 0)], np.nan)


def mixed_layer_depth(path):
    """Density-threshold mixed layer depth (m) of one chunk, NaN over land."""
    ds = xr.open_dataset(path)
    T = ds.temp.mean("time").transpose("lat", "lon", "depth").values[..., ::-1]
    S = ds.salt.mean("time").transpose("lat", "lon", "depth").values[..., ::-1]
    ds.close()
    rho = -(BETA_T * (T - THETA_0) - BETA_S * (S - S_0)) * RHO_0
    rho = np.where(wet, rho, np.inf)                    # below the seafloor counts as "denser": never inside
    target = rho[..., :1] + args.threshold
    exceeded = rho > target
    k = np.argmax(exceeded, axis=2)                      # first layer denser than the target
    found = exceeded.any(axis=2) & ocean
    jj, ii = np.indices(k.shape)
    km = np.maximum(k - 1, 0)
    r0, r1 = rho[jj, ii, km], rho[jj, ii, k]
    r1 = np.where(np.isfinite(r1), r1, r0)               # the first "denser" layer is the seafloor
    frac = np.where(r1 > r0, (target[..., 0] - r0) / np.where(r1 > r0, r1 - r0, 1.0), 1.0)
    interior = zc[km] + np.clip(frac, 0.0, 1.0) * (zc[k] - zc[km])
    # first wet layer already too dense below the reference -> the top layer centre;
    # the seafloor reached first -> the column bottom
    hits_floor = found & ~np.isfinite(rho[jj, ii, k])
    mld = np.where(k == 0, zc[0], interior)
    mld = np.where(hits_floor | (~found & ocean), column_bottom, mld)
    return np.where(ocean, mld, np.nan)


winter, annual = [], []
for y in range(first, last + 1):
    chunks = np.array([mixed_layer_depth(f) for f in files[CHUNKS_PER_YEAR * (y - 1):CHUNKS_PER_YEAR * y]])
    winter.append(np.nanmax(chunks, axis=0))
    annual.append(np.nanmean(chunks, axis=0))
winter, annual = np.nanmean(winter, axis=0), np.nanmean(annual, axis=0)

name = os.path.basename(os.path.realpath(args.run_dir).rstrip("/"))
levels = [0, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5500]
norm = BoundaryNorm(levels, plt.get_cmap("viridis_r").N)
proj, pc = ccrs.Robinson(central_longitude=180), ccrs.PlateCarree()
fig = plt.figure(figsize=(16, 5.5))
lon_c = np.append(lon, 360.0)
for k, (field, title) in enumerate([(winter, "winter (deepest 73-day chunk of each year)"),
                                    (annual, "annual mean")]):
    ax = fig.add_subplot(1, 2, k + 1, projection=proj)
    p = ax.pcolormesh(lon_c, lat, np.concatenate([field, field[:, :1]], axis=1), cmap="viridis_r", norm=norm,
                      shading="nearest", transform=pc)
    ax.coastlines(lw=0.5); ax.set_global()
    ax.set_title(f"{name} mixed layer depth, {title}\nyears {first}–{last}, Δρ = {args.threshold:g} kg/m³ "
                 f"from the top layer", fontsize=10)
    plt.colorbar(p, ax=ax, shrink=0.75, orientation="horizontal", pad=0.05, label="m", ticks=levels[1::2])
fig.tight_layout()
fig.savefig(args.out_png, dpi=110)

w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
LA, LO = np.meshgrid(lat, lon, indexing="ij")
regions = {"subpolar N Pacific (45-60N, 140-235E)": (LA >= 45) & (LA <= 60) & (LO >= 140) & (LO <= 235),
           "subpolar N Atlantic (50-65N, 300-350E)": (LA >= 50) & (LA <= 65) & (LO >= 300) & (LO <= 350),
           "Southern Ocean (65-50S)": (LA >= -65) & (LA <= -50),
           "tropics (20S-20N)": np.abs(LA) <= 20}
print(f"{name}, years {first}-{last}, threshold {args.threshold:g} kg/m3:")
for label, m in regions.items():
    m = m & ocean
    mean = lambda x: np.nansum((x * w)[m]) / np.sum(w[m])
    print(f"  {label}: winter mean {mean(winter):6.0f} m (deepest cell {np.nanmax(winter[m]):5.0f} m), "
          f"annual mean {mean(annual):5.0f} m")
print("saved", args.out_png)
