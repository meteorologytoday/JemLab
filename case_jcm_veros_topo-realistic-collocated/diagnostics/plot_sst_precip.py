"""SST and precipitation of one JEM run vs ERA5, side by side.

Left column: surface temperature (SST over ocean, land surface temperature over
land; ERA5 sst/stl), ERA5, model minus ERA5, and equatorial (5S-5N) SST vs
longitude. The magenta line is the 0 degC SST contour (ocean only).
Right column: total JCM precipitation (precnv + precls), ERA5 (annual mean of
monthly `tp`, interpolated to the model grid), model minus ERA5, and the zonal
mean of both.

The mean is over the run's last N years (default 1), or over an explicit
range of 73-day chunks. Reads the run's ocn-/lnd-/atm-*.nc output.

Examples:
    python diagnostics/plot_sst_precip.py output_T31/winton_bathy sst_precip.png
    python diagnostics/plot_sst_precip.py output_T31/winton_bathy sst_precip.png --last-years 5
    python diagnostics/plot_sst_precip.py output_T31/winton_bathy sst_precip.png --chunks 16 20
"""
import argparse, glob, os
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

CHUNK_DAYS = 73
FREEZE_LINE = 0.0   # highlighted SST contour, degC

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run_dir", help="Run output directory (holds ocn-/lnd-/atm-*.nc).")
parser.add_argument("out_png", help="Output figure.")
window = parser.add_mutually_exclusive_group()
window.add_argument("--last-years", type=int, default=1, help="Average over the last N years (default 1).")
window.add_argument("--chunks", type=int, nargs=2, metavar=("FIRST", "LAST"),
                    help="Average over chunks FIRST..LAST (1-based, inclusive) instead.")
parser.add_argument("--era5-sst", default="forcing_era5_t31.nc",
                    help="ERA5 climatology on the model grid with sst, stl and landsea_mask (default: %(default)s).")
parser.add_argument("--era5-tp", default="/p/tmp/tienyiao/era5_tp/era5_tp_monthly_1971-2000_1deg.nc",
                    help="ERA5 monthly total precipitation tp in m/day (default: %(default)s).")
args = parser.parse_args()
era_file, tp_file, out_png, rd = args.era5_sst, args.era5_tp, args.out_png, args.run_dir

all_files = sorted(glob.glob(f"{rd}/ocn-*.nc"))
if args.chunks:
    files = all_files[args.chunks[0] - 1:args.chunks[1]]
else:
    files = all_files[-int(round(args.last_years * 365 / CHUNK_DAYS)):]
if not files:
    parser.error(f"no ocn-*.nc files in the requested window of {rd}")


def open_series(paths, names):
    return xr.open_mfdataset(paths, combine="nested", concat_dim="time", data_vars=names,
                             coords="minimal", compat="override")


ds = open_series(files, ["sea_surface_temperature", "sea_surface_salinity"])
ocean = (ds.sea_surface_salinity.isel(time=-1) > 0).load().transpose("lat", "lon")
sst = (ds.sea_surface_temperature.mean("time").load() - 273.15).transpose("lat", "lon").where(ocean)
lnd = open_series([f.replace("/ocn-", "/lnd-") for f in files], ["land_surface_temperature"])
land = (lnd.land_fraction.isel(time=0, missing_dims="ignore") == 1).load().transpose("lat", "lon").values
lst = (lnd.land_surface_temperature.mean("time").load() - 273.15).transpose("lat", "lon").values
lst = xr.DataArray(np.where(land, lst, np.nan), coords=sst.coords, dims=sst.dims)
atm = open_series([f.replace("/ocn-", "/atm-") for f in files], ["convection.precnv", "condensation.precls"])
pr = ((atm["convection.precnv"] + atm["condensation.precls"]).mean("time").load() * 86.4).transpose("lat", "lon")  # g m-2 s-1 -> mm/day
pr = xr.DataArray(pr.values, coords=sst.coords, dims=sst.dims)
tsfc = sst.fillna(lst)
t = ds.time.values
d_first, d_last = int(os.path.basename(files[0])[4:12]), int(os.path.basename(files[-1])[4:12]) + CHUNK_DAYS
name = os.path.basename(os.path.dirname(os.path.realpath(files[0])))

lat, lon = sst.lat.values, sst.lon.values
era = xr.open_dataset(era_file).assign_coords(lat=lat, lon=lon)
era_sst = (era.sst.mean("time") - 273.15).where(era.landsea_mask == 0)
era_lst = (era.stl.mean("time") - 273.15).where(era.landsea_mask == 1)
era_tsfc = era_sst.fillna(era_lst)
e = xr.open_dataset(tp_file)
tdim = "valid_time" if "valid_time" in e.dims else "time"
tp = (e.tp.mean(tdim) * 1000.0).rename(latitude="lat", longitude="lon").sortby("lat")
tp = tp.assign_coords(lon=tp.lon % 360).sortby("lon")
tp = xr.concat([tp, tp.isel(lon=0).assign_coords(lon=tp.lon[0] + 360)], "lon")  # wrap for interpolation
era_pr = xr.DataArray(tp.interp(lat=lat, lon=lon).transpose("lat", "lon").values, coords=sst.coords, dims=sst.dims)

w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))


def gmean(x):
    x = np.asarray(x); ok = np.isfinite(x)
    return (x[ok] * w[ok]).sum() / w[ok].sum()


def cyc(da):
    return np.append(lon, 360.0), np.concatenate([np.asarray(da), np.asarray(da)[:, :1]], axis=1)


eq = (lat >= -5) & (lat <= 5)
def eq_profile(da):
    x = np.asarray(da)[eq]; ww = w[eq]
    with np.errstate(invalid="ignore", divide="ignore"):  # all-land longitudes -> NaN
        return np.nansum(x * ww, 0) / np.where(np.isfinite(x), ww, 0).sum(0)


proj, pc = ccrs.Robinson(central_longitude=180), ccrs.PlateCarree()
fig = plt.figure(figsize=(14, 14.5))
lev, dlev = np.arange(-30, 36, 2), np.arange(-10, 11, 1)
plev = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15]
pdlev = [-8, -6, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 6, 8]


def map_panel(k, field, levels, cmap, extend, title, unit, sst_field=None, zero_line=False):
    ax = fig.add_subplot(4, 2, k, projection=proj)
    lo, v = cyc(field)
    cf = ax.contourf(lo, lat, v, levels=levels, cmap=cmap, extend=extend, transform=pc)
    if zero_line:
        ax.contour(lo, lat, v, levels=[0], colors="k", linewidths=0.5, transform=pc)
    if sst_field is not None:
        ax.contour(lo, lat, v, levels=[0, 20, 28], colors="k", linewidths=0.5, transform=pc)
        lo_s, v_s = cyc(sst_field)
        ax.contour(lo_s, lat, v_s, levels=[FREEZE_LINE], colors="magenta", linewidths=1.2, transform=pc)
    ax.coastlines(lw=0.5); ax.set_global()
    ax.set_title(title, fontsize=9)
    fig.colorbar(cf, ax=ax, shrink=0.75, label=unit)


period = f"{str(t[0])[:7]} – {str(t[-1])[:7]}, model years {d_first / 365:.0f}–{d_last / 365:.0f}"
dsst, dlst, dpr = sst - era_sst, lst - era_lst, pr - era_pr
map_panel(1, tsfc, lev, "RdYlBu_r", "both",
          f"{name}: SST + land T ({period})\n[ocean {gmean(sst):.1f} °C, land {gmean(lst):.1f} °C; magenta: SST = 0 °C]",
          "°C", sst_field=sst)
map_panel(2, pr, plev, "YlGnBu", "max", f"{name}: total precipitation\n[global {gmean(pr):.2f} mm/day]", "mm/day")
map_panel(3, era_tsfc, lev, "RdYlBu_r", "both",
          f"ERA5 annual-mean sst + stl\n[ocean {gmean(era_sst):.1f} °C, land {gmean(era_lst):.1f} °C]", "°C", sst_field=era_sst)
map_panel(4, era_pr, plev, "YlGnBu", "max", f"ERA5 1971–2000 annual-mean precipitation\n[global {gmean(era_pr):.2f} mm/day]", "mm/day")
map_panel(5, tsfc - era_tsfc, dlev, "RdBu_r", "both",
          f"minus ERA5  RMSE [ocean {np.sqrt(gmean(dsst ** 2)):.2f} °C, land {np.sqrt(gmean(dlst ** 2)):.2f} °C]", "°C", zero_line=True)
map_panel(6, dpr, pdlev, "BrBG", "both",
          f"minus ERA5  [{gmean(dpr):+.2f} mm/day, RMS {np.sqrt(gmean(dpr ** 2)):.2f}]", "mm/day")

ax = fig.add_subplot(4, 2, 7)
ax.plot(lon, eq_profile(sst), color="C0", label="model")
ax.plot(lon, eq_profile(era_sst), color="k", ls="--", label="ERA5")
ax.axvspan(120, 280, color="0.92", zorder=0)
ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 60)); ax.grid(alpha=0.3)
ax.set_xlabel("longitude (°E)"); ax.set_ylabel("SST, 5°S–5°N (°C)"); ax.legend(fontsize=8)
ax.set_title("Equatorial SST (shaded: Pacific 120–280°E)", fontsize=9)

ax = fig.add_subplot(4, 2, 8)
ax.plot(lat, np.nanmean(np.asarray(pr), 1), color="C0", label="model")
ax.plot(lat, np.nanmean(np.asarray(era_pr), 1), color="k", ls="--", label="ERA5")
ax.set_xlim(-90, 90); ax.set_xticks(range(-90, 91, 30)); ax.grid(alpha=0.3)
ax.set_xlabel("latitude"); ax.set_ylabel("mm/day"); ax.legend(fontsize=8)
ax.set_title("Zonal-mean precipitation", fontsize=9)

fig.tight_layout()
fig.savefig(out_png, dpi=100)
p = eq_profile(sst); pe = eq_profile(era_sst); wp = (lon >= 130) & (lon <= 160); ep = (lon >= 240) & (lon <= 270)
print(f"{name} (model years {d_first / 365:.1f}-{d_last / 365:.1f}): ocean SST {gmean(sst):.2f} C (ERA5 {gmean(era_sst):.2f}, RMSE {np.sqrt(gmean(dsst ** 2)):.2f}); "
      f"eq Pacific W-E {np.nanmean(p[wp]) - np.nanmean(p[ep]):+.2f} K (ERA5 {np.nanmean(pe[wp]) - np.nanmean(pe[ep]):+.2f}); "
      f"precip global {gmean(pr):.2f} mm/day (ERA5 {gmean(era_pr):.2f}), "
      f"zonal-mean max {np.nanmean(np.asarray(pr), 1).max():.2f} at {lat[np.nanargmax(np.nanmean(np.asarray(pr), 1))]:.1f} "
      f"(ERA5 {np.nanmean(np.asarray(era_pr), 1).max():.2f} at {lat[np.nanargmax(np.nanmean(np.asarray(era_pr), 1))]:.1f})")
print("saved", out_png)
