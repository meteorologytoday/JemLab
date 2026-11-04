"""Time series of total sea-ice area and volume, per hemisphere, of a JEM run with Winton sea ice.

Area = sum of ice_fraction * cell area; volume = sum of ice_volume (the
cell-mean ice thickness) * cell area, over ocean cells. One point per 73-day
chunk (thin) and the running annual mean over five chunks (thick). Reads the
run's seaice-*.nc output (and one ocn-*.nc file for the ocean mask).

Several runs can be given; each gets its own colour, NH solid and SH dashed.

Examples:
    python diagnostics/plot_seaice_timeseries.py seaice_ts.png output_T31/winton_bathy
    python diagnostics/plot_seaice_timeseries.py seaice_ts.png output_T31/run_a output_T31/run_b
"""
import argparse, glob, os
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHUNK_DAYS = 73
R = 6.371e6

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("out_png", help="Output figure.")
parser.add_argument("run_dirs", nargs="+", help="Run output directories (hold seaice-*.nc and ocn-*.nc).")
args = parser.parse_args()

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for k, rd in enumerate(args.run_dirs):
    fs = sorted(glob.glob(f"{rd}/seaice-*.nc"))
    if not fs:
        print(f"{rd}: no seaice-*.nc output"); continue
    d0 = xr.open_dataset(fs[0])
    lat, lon = d0.lat.values, d0.lon.values
    ocean = xr.open_dataset(fs[0].replace("seaice-", "ocn-")).mask_surface_T.transpose("lat", "lon").values.astype(bool)
    area = (R**2 * np.deg2rad(360 / len(lon)) * np.cos(np.deg2rad(lat)) * np.deg2rad(np.gradient(lat)))[:, None] \
        * np.ones((1, len(lon)))
    LA = np.broadcast_to(lat[:, None], area.shape)
    hemis = {"NH": ocean & (LA > 0), "SH": ocean & (LA < 0)}
    t, series = [], {(h, q): [] for h in hemis for q in ("area", "volume")}
    for f in fs:
        si = xr.open_dataset(f).mean("time")
        fr = si.ice_fraction.transpose("lat", "lon").values
        vol = si.ice_volume.transpose("lat", "lon").values
        for h, m in hemis.items():
            series[h, "area"].append(np.sum((fr * area)[m]) / 1e12)        # 10^6 km^2
            series[h, "volume"].append(np.sum((vol * area)[m]) / 1e12)     # 10^3 km^3
        t.append((int(os.path.basename(f)[7:15]) + CHUNK_DAYS / 2) / 365)
    t = np.array(t)
    name = os.path.basename(os.path.realpath(rd).rstrip("/"))
    for ax, q in zip(axes, ("area", "volume")):
        for h, ls in (("NH", "-"), ("SH", "--")):
            y = np.array(series[h, q])
            ax.plot(t, y, color=colors[k], ls=ls, lw=0.6, alpha=0.6)
            if len(y) >= 5:
                ax.plot(t[2:-2], np.convolve(y, np.ones(5) / 5, mode="valid"), color=colors[k], ls=ls, lw=2,
                        label=f"{name} {h}")
    last = slice(-5, None)
    print(f"{name} (year {t[-1]:.1f}), mean of the last 5 chunks: " + ", ".join(
        f"{h} area {np.mean(series[h, 'area'][last]):.2f} Mkm2, volume {np.mean(series[h, 'volume'][last]):.2f} 10^3 km3"
        for h in hemis))

axes[0].set_ylabel("ice area (10$^6$ km$^2$)")
axes[0].set_title("Total sea-ice area (thin: 73-day means, thick: running annual mean; solid NH, dashed SH)", fontsize=10)
axes[1].set_ylabel("ice volume (10$^3$ km$^3$)")
axes[1].set_title("Total sea-ice volume", fontsize=10)
axes[1].set_xlabel("model year")
for ax in axes:
    ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(out_png := args.out_png, dpi=110)
print("saved", out_png)
