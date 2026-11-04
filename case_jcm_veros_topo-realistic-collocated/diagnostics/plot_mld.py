"""Mixed layer depth of a JEM/Veros run from a density threshold.

The definition and its computation are in diagnostics/mld.py: the depth
where the Veros linear-EOS density first exceeds the top layer's by
`--threshold` (default 0.03 kg/m3, de Boyer Montegut et al. 2004), computed
per 73-day chunk. The figure shows, averaged over the chosen years:
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

from mld import CHUNKS_PER_YEAR, DEFAULT_THRESHOLD, MixedLayer

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run_dir", help="Run output directory (holds ocn-*.nc).")
parser.add_argument("out_png", help="Output figure.")
window = parser.add_mutually_exclusive_group()
window.add_argument("--last-years", type=int, default=5, help="Average over the last N complete years (default 5).")
window.add_argument("--years", type=int, nargs=2, metavar=("FIRST", "LAST"),
                    help="Average over model years FIRST..LAST (1-based, inclusive) instead.")
parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="Density threshold in kg/m3 (default %(default)s).")
args = parser.parse_args()

files = sorted(glob.glob(f"{args.run_dir}/ocn-*.nc"))
nyear = len(files) // CHUNKS_PER_YEAR
if nyear == 0:
    parser.error(f"no complete year of ocn-*.nc output in {args.run_dir}")
first, last = args.years if args.years else (max(1, nyear - args.last_years + 1), nyear)
if not 1 <= first <= last <= nyear:
    parser.error(f"years must lie in 1..{nyear} (complete years so far)")

ml = MixedLayer(files[0], args.threshold)
lat, lon, ocean = ml.lat, ml.lon, ml.ocean
winter, annual = ml.winter_and_annual(files[CHUNKS_PER_YEAR * (first - 1):CHUNKS_PER_YEAR * last])

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
