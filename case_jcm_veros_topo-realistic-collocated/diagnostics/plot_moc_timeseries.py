"""Time series of overturning strengths in JEM/Veros runs, one point per 73-day chunk.

Same streamfunction and basin masks as plot_moc.py (diagnostics/moc.py):
  AMOC at 26.5N        max of the Atlantic psi, 24-29N, 300-3000 m
  AMOC max 20-60N      max of the Atlantic psi, 20-60N, 300-3000 m
  PMOC NH max          max of the Pacific psi, 20-60N, below 300 m
  Global deep cell     min of the global psi, 60S-20N, below 2000 m (bottom-water cell)
  Indonesian throughflow  full-column Pacific transport at 20S
Thin lines: chunk values; thick: running annual mean (5 chunks).

Examples:
    python diagnostics/plot_moc_timeseries.py moc_ts.png output_T31/winton_bathy
    python diagnostics/plot_moc_timeseries.py moc_ts.png output_T31/run_a output_T31/run_b --xmax 100
"""
import argparse, glob, os
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moc import CHUNK_DAYS, basin_masks, bottom_row_at, interfaces, streamfunctions, window_stat

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("out_png", help="Output figure.")
parser.add_argument("run_dirs", nargs="+", help="Run output directories (hold ocn-*.nc).")
parser.add_argument("--xmax", type=float, help="Limit the time axis to model years 0..XMAX.")
args = parser.parse_args()

# (title, basin, statistic of psi, z and lat)
STATS = [
    ("AMOC at 26.5N", "atlantic", lambda p, z, lat: window_stat(p, z, lat, (24, 29), (300, 3000), np.max)),
    ("AMOC max 20–60N", "atlantic", lambda p, z, lat: window_stat(p, z, lat, (20, 60), (300, 3000), np.max)),
    ("PMOC NH max 20–60N (below 300 m)", "pacific", lambda p, z, lat: window_stat(p, z, lat, (20, 60), (300, z[-1]), np.max)),
    ("Global deep (bottom-water) cell, min", "global", lambda p, z, lat: window_stat(p, z, lat, (-60, 20), (2000, z[-1]), np.min)),
    ("Indonesian throughflow (full-column Pacific transport at 20S)", "pacific", lambda p, z, lat: bottom_row_at(p, lat, (-21, -19))),
]

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig, axes = plt.subplots(len(STATS), 1, figsize=(12, 2.6 * len(STATS)), sharex=True)
for k, rd in enumerate(args.run_dirs):
    files = sorted(glob.glob(f"{rd}/ocn-*.nc"))
    if not files:
        print(f"{rd}: no ocn-*.nc output"); continue
    first = xr.open_dataset(files[0])
    lat, lon = first.lat.values, first.lon.values
    ocean = first.sea_surface_salinity.isel(time=0).transpose("lat", "lon").values > 0
    masks = basin_masks(ocean, lat, lon)
    dz, z = interfaces(first.dzt.values)
    t, vals = [], {s[0]: [] for s in STATS}
    for f in files:
        ds = xr.open_dataset(f)
        v = ds.v.mean("time").transpose("depth", "lat", "lon").values[::-1]
        ds.close()
        psi = streamfunctions(v, masks, lat, lon, dz)
        for name, basin, fn in STATS:
            vals[name].append(fn(psi[basin], z, lat))
        t.append((int(os.path.basename(f)[4:12]) + CHUNK_DAYS / 2) / 365)
    t = np.array(t)
    label = os.path.basename(os.path.realpath(rd).rstrip("/"))
    for ax, (name, *_rest) in zip(axes, STATS):
        y = np.array(vals[name])
        ax.plot(t, y, color=colors[k], lw=0.5, alpha=0.5)
        if len(y) >= 5:
            ax.plot(t[2:-2], np.convolve(y, np.ones(5) / 5, mode="valid"), color=colors[k], lw=2, label=label)
    print(f"{label} (yr {t[-1]:.1f}), mean of the last 5 chunks: " +
          ", ".join(f"{name.split(' (')[0]} {np.mean(vals[name][-5:]):.1f}" for name, *_r in STATS) + " Sv")
for ax, (name, *_rest) in zip(axes, STATS):
    ax.set_title(name, fontsize=10); ax.set_ylabel("Sv"); ax.grid(alpha=0.3); ax.axhline(0, color="k", lw=0.5)
axes[0].legend(fontsize=8)
axes[-1].set_xlabel("model year")
if args.xmax is not None:
    for ax in axes:
        ax.set_xlim(0, args.xmax)
fig.tight_layout()
fig.savefig(args.out_png, dpi=110)
print("saved", args.out_png)
