"""Annual-mean sea-ice fraction maps of a JEM run with Winton sea ice.

One column per model year (mean of its five 73-day chunks), NH on top, SH
below. Prints, for every complete year: annual-mean ice area per hemisphere,
the largest and smallest 73-day-chunk area, and the annual-mean thickness over
cells whose annual-mean fraction exceeds 0.5.

Without --years, plots every complete year (at most 8, evenly spaced, always
including the last); the printout always covers every complete year. Reads the
run's seaice-*.nc output (and one ocn-*.nc file for the ocean mask).

Examples:
    python diagnostics/plot_seaice_annual.py output_T31/winton_bathy seaice.png
    python diagnostics/plot_seaice_annual.py output_T31/winton_bathy seaice.png --years 1 10 20
"""
import argparse, glob, os
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run_dir", help="Run output directory (holds seaice-*.nc and ocn-*.nc).")
parser.add_argument("out_png", help="Output figure.")
parser.add_argument("--years", type=int, nargs="+", default=[], help="Model years (1-based) to map.")
args = parser.parse_args()
out_png, rd, years = args.out_png, args.run_dir, args.years

R = 6.371e6
fs = sorted(glob.glob(f"{rd}/seaice-*.nc"))
nyear = len(fs) // 5
if nyear == 0:
    parser.error(f"no complete year of seaice-*.nc output in {rd}")
if any(not 1 <= y <= nyear for y in years):
    parser.error(f"--years must lie in 1..{nyear} (complete years so far)")

d0 = xr.open_dataset(fs[0])
lat, lon = d0.lat.values, d0.lon.values
ocean = xr.open_dataset(fs[0].replace("seaice-", "ocn-")).mask_surface_T.transpose("lat", "lon").values.astype(bool)
area = (R**2 * np.deg2rad(360 / len(lon)) * np.cos(np.deg2rad(lat)) * np.deg2rad(np.gradient(lat)))[:, None] \
    * np.ones((1, len(lon)))
LA = np.broadcast_to(lat[:, None], area.shape)
hemis = (("NH", LA > 0), ("SH", LA < 0))

annual = {}
for y in range(1, nyear + 1):
    fr_chunks, vol_chunks = [], []
    for f in fs[5 * (y - 1):5 * y]:
        si = xr.open_dataset(f).mean("time")
        fr_chunks.append(np.where(ocean, si.ice_fraction.transpose("lat", "lon").values, 0.0))
        vol_chunks.append(np.where(ocean, si.ice_volume.transpose("lat", "lon").values, 0.0))
    fr, vol = np.mean(fr_chunks, 0), np.mean(vol_chunks, 0)
    annual[y] = fr
    parts = []
    for name, hemi in hemis:
        chunk_area = [np.sum((c * area)[hemi]) / 1e12 for c in fr_chunks]
        icy = hemi & (fr > 0.5)
        h = np.sum((vol / np.maximum(fr, 1e-6) * area)[icy]) / np.sum(area[icy]) if icy.any() else np.nan
        parts.append(f"{name} mean {np.sum((fr * area)[hemi]) / 1e12:5.2f} (max {max(chunk_area):5.2f}, "
                     f"min {min(chunk_area):5.2f}) Mkm2, h {h:4.2f} m")
    print(f"year {y:3d}: " + " | ".join(parts))

if not years:
    years = sorted(set(np.linspace(1, nyear, min(8, nyear)).round().astype(int)))
fig = plt.figure(figsize=(3.6 * len(years) + 2, 7.5))
for c, y in enumerate(years):
    for r, (proj, ext) in enumerate([(ccrs.NorthPolarStereo(), [0, 359.9, 45, 90]),
                                     (ccrs.SouthPolarStereo(), [0, 359.9, -90, -50])]):
        ax = fig.add_subplot(2, len(years), r * len(years) + c + 1, projection=proj)
        ax.set_extent(ext, ccrs.PlateCarree()); ax.coastlines(lw=0.5)
        p = ax.pcolormesh(lon, lat, np.where(ocean, annual[y], np.nan), transform=ccrs.PlateCarree(),
                          shading="nearest", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"year {y}", fontsize=10)
fig.colorbar(p, ax=fig.axes, shrink=0.6, label="annual-mean ice fraction")
fig.suptitle(f"{os.path.basename(os.path.realpath(rd).rstrip('/'))}: annual-mean sea-ice fraction "
             "(top NH, bottom SH)", fontsize=12)
fig.savefig(out_png, dpi=90, bbox_inches="tight")
print("saved", out_png)
