"""Time series of area-mean SST over a latitude band, for one or more JEM runs.

Area-weighted (cos latitude) mean of sea_surface_temperature over the ocean
cells between --lat-min and --lat-max (default the tropics, 20S-20N), one point
per 73-day chunk (thin) and the running annual mean over five chunks (thick).
The dashed line is the ERA5 annual-mean climatology over the same band, on the
model grid (ERA5's own land-sea mask).

Examples:
    python diagnostics/plot_sst_timeseries.py tropical_sst.png output_T31/winton_ctl output_T31/winton_bathy
    python diagnostics/plot_sst_timeseries.py sst_np.png output_T31/winton_ctl --lat-min 30 --lat-max 60
"""
import argparse, glob, os
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHUNK_DAYS = 73

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("out_png", help="Output figure.")
parser.add_argument("run_dirs", nargs="+", help="Run output directories (hold ocn-*.nc).")
parser.add_argument("--lat-min", type=float, default=-20.0, help="Southern edge of the band (default -20).")
parser.add_argument("--lat-max", type=float, default=20.0, help="Northern edge of the band (default 20).")
parser.add_argument("--era5-sst", default="forcing_era5_t31.nc",
                    help="ERA5 climatology on the model grid with sst and landsea_mask (default: %(default)s).")
args = parser.parse_args()


def band(lat):
    return (lat >= args.lat_min) & (lat <= args.lat_max)


def latstr(x):
    return f"{abs(x):g}{'S' if x < 0 else 'N'}"


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig, ax = plt.subplots(figsize=(12, 4.5))
for k, rd in enumerate(args.run_dirs):
    files = sorted(glob.glob(f"{rd}/ocn-*.nc"))
    if not files:
        print(f"{rd}: no ocn-*.nc output"); continue
    d0 = xr.open_dataset(files[0])
    lat = d0.lat.values
    ocean = d0.mask_surface_T.transpose("lat", "lon").values.astype(bool)
    w = np.cos(np.deg2rad(lat))[:, None] * (ocean & band(lat)[:, None])
    t, sst = [], []
    for f in files:
        ds = xr.open_dataset(f)
        x = ds.sea_surface_temperature.mean("time").transpose("lat", "lon").values - 273.15
        ds.close()
        sst.append(np.sum(np.where(w > 0, x, 0.0) * w) / w.sum())
        t.append((int(os.path.basename(f)[4:12]) + CHUNK_DAYS / 2) / 365)
    t, sst = np.array(t), np.array(sst)
    name = os.path.basename(os.path.realpath(rd).rstrip("/"))
    ax.plot(t, sst, color=colors[k], lw=0.6, alpha=0.5)
    if len(sst) >= 5:
        ax.plot(t[2:-2], np.convolve(sst, np.ones(5) / 5, mode="valid"), color=colors[k], lw=2, label=name)
    print(f"{name} (year {t[-1]:.1f}): last 5 chunks {sst[-5:].mean():.2f} C, "
          f"first year {sst[:5].mean():.2f} C")

era = xr.open_dataset(args.era5_sst)
era_sst = era.sst.mean("time").transpose("lat", "lon").values - 273.15
era_ocean = era.landsea_mask.transpose("lat", "lon").values == 0
elat = era.lat.values
ew = np.cos(np.deg2rad(elat))[:, None] * (era_ocean & band(elat)[:, None])
era_mean = np.nansum(np.where(ew > 0, era_sst, 0.0) * ew) / ew.sum()
ax.axhline(era_mean, color="k", ls="--", lw=1.2, label=f"ERA5 climatology ({era_mean:.2f} °C)")
print(f"ERA5 annual-mean climatology: {era_mean:.2f} C")

ax.set_xlabel("model year"); ax.set_ylabel("SST (°C)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax.set_title(f"Mean SST over the ocean, {latstr(args.lat_min)}–{latstr(args.lat_max)} "
             "(thin: 73-day means, thick: running annual mean)", fontsize=10)
fig.tight_layout()
fig.savefig(args.out_png, dpi=110)
print("saved", args.out_png)
