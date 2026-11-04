"""Meridional overturning streamfunction (global, Atlantic, Pacific) of JEM/Veros runs.

One column per run; rows are the global, Atlantic and Pacific MOC from the
time-mean v over the run's last N years (default 5) or an explicit chunk
range. Each panel title carries the usual strength metrics:
  global    NH max (20-70N, 300-3000 m), bottom-water cell min (60S-20N,
            below 2000 m), Southern Ocean max (60-40S, above 3000 m)
  Atlantic  max at 26.5N (24-29N) and max 20-60N, both 300-3000 m
  Pacific   NH max (20-60N, below 300 m) and the full-column transport at 20S
            (Indonesian throughflow)
Basin masks and the streamfunction are in diagnostics/moc.py.

Examples:
    python diagnostics/plot_moc.py moc.png output_T31/winton_bathy
    python diagnostics/plot_moc.py moc.png output_T31/run_a output_T31/run_b --common --last-years 20
    python diagnostics/plot_moc.py moc.png output_T31/winton_bathy --chunks 226 250
"""
import argparse, glob, os
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moc import CHUNK_DAYS, basin_masks, bottom_row_at, interfaces, streamfunctions, window_stat

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("out_png", help="Output figure.")
parser.add_argument("run_dirs", nargs="+", help="Run output directories (hold ocn-*.nc).")
window = parser.add_mutually_exclusive_group()
window.add_argument("--last-years", type=int, default=5, help="Average over the last N years (default 5).")
window.add_argument("--chunks", type=int, nargs=2, metavar=("FIRST", "LAST"),
                    help="Average over chunks FIRST..LAST (1-based, inclusive) instead.")
parser.add_argument("--common", action="store_true",
                    help="With --last-years, end every run's window where the shortest run ends.")
args = parser.parse_args()

nchunks = int(round(args.last_years * 365 / CHUNK_DAYS))
n_end = min(len(glob.glob(f"{rd}/ocn-*.nc")) for rd in args.run_dirs) if args.common else None
runs = []
for rd in args.run_dirs:
    files = sorted(glob.glob(f"{rd}/ocn-*.nc"))
    if args.chunks:
        files = files[args.chunks[0] - 1:args.chunks[1]]
    else:
        files = files[:n_end][-nchunks:]
    if not files:
        parser.error(f"no ocn-*.nc files in the requested window of {rd}")
    ds = xr.open_mfdataset(files, combine="nested", concat_dim="time", data_vars=["v", "sea_surface_salinity"],
                           coords="minimal", compat="override")
    v = ds.v.mean("time").load().transpose("depth", "lat", "lon").values[::-1]  # surface first
    dz, z = interfaces(ds.dzt.values)
    lat, lon = ds.lat.values, ds.lon.values
    ocean = (ds.sea_surface_salinity.isel(time=-1) > 0).load().transpose("lat", "lon").values
    psi = streamfunctions(v, basin_masks(ocean, lat, lon), lat, lon, dz)
    d0, d1 = int(os.path.basename(files[0])[4:12]), int(os.path.basename(files[-1])[4:12]) + CHUNK_DAYS
    runs.append(dict(name=os.path.basename(os.path.realpath(rd).rstrip("/")), lat=lat, z=z,
                     period=f"yr {d0 / 365:.0f}–{d1 / 365:.0f}", **psi))

nc = len(runs)
fig, axs = plt.subplots(3, nc, figsize=(7 * nc, 15), squeeze=False)
fig.subplots_adjust(hspace=0.4)
lev = np.arange(-30, 32, 2)
for j, r in enumerate(runs):
    lat, z = r["lat"], r["z"]
    for i, (key, title, xlim) in enumerate([("global", "Global MOC", (-80, 80)),
                                            ("atlantic", "Atlantic MOC", (-34, 70)),
                                            ("pacific", "Pacific MOC", (-34, 65))]):
        ax = axs[i, j]
        p = r[key].copy()
        if key != "global":
            p[:, ~((lat >= xlim[0]) & (lat <= xlim[1]))] = np.nan
        cf = ax.contourf(lat, z, p, levels=lev, cmap="RdBu_r", extend="both")
        cs = ax.contour(lat, z, p, levels=[l for l in lev if l != 0][::2], colors="k", linewidths=0.4)
        ax.clabel(cs, fontsize=6, fmt="%d")
        ax.contour(lat, z, p, levels=[0], colors="k", linewidths=1)
        ax.set_ylim(z[-1], 0); ax.set_xlim(*xlim); ax.set_xlabel("latitude"); ax.set_ylabel("depth (m)")
        P = r[key]
        if key == "global":
            t = (f"NH max {window_stat(P, z, lat, (20, 70), (300, 3000), np.max):.1f}, "
                 f"deep cell min {window_stat(P, z, lat, (-60, 20), (2000, z[-1]), np.min):.1f}, "
                 f"SO (60–40S) max {window_stat(P, z, lat, (-60, -40), (0, 3000), np.max):.1f} Sv")
        elif key == "pacific":
            t = (f"NH max (20–60N, below 300 m) {window_stat(P, z, lat, (20, 60), (300, z[-1]), np.max):.1f}, "
                 f"net at 20S (Indonesian throughflow) {bottom_row_at(P, lat, (-21, -19)):.1f} Sv")
        else:
            t = (f"max at 26.5N {window_stat(P, z, lat, (24, 29), (300, 3000), np.max):.1f}, "
                 f"max 20–60N {window_stat(P, z, lat, (20, 60), (300, 3000), np.max):.1f} Sv")
        ax.set_title(f"{r['name']}, {r['period']}: {title}\n{t}", fontsize=9)
        print(f"{r['name']} {r['period']} {title}: {t}")
fig.colorbar(cf, ax=axs, label="Sv (positive = clockwise)", shrink=0.8)
fig.savefig(args.out_png, dpi=110)
print("saved", args.out_png)
