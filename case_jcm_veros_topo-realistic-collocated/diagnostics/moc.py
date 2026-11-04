"""Basin masks and meridional overturning streamfunctions shared by the MOC diagnostics.

psi(z, y) is the northward volume transport above depth z, in Sv (positive =
clockwise in a latitude-depth section: northward near the surface, southward
below), from the meridional velocity v. v is taken at T-row latitudes and the
zonal cell width uses cos(lat). Veros output is bottom-first along `depth`;
everything here is surface-first. Cells below the seafloor hold NaN or 0 and
contribute nothing, so the bottom row of psi is the net transport of the whole
column (for the Pacific mask at 20S, the Indonesian throughflow).
"""
import sys

import numpy as np
from scipy import ndimage

R_EARTH = 6.371e6
CHUNK_DAYS = 73


def basin_masks(ocean, lat, lon):
    """Return {"global", "atlantic", "pacific"} boolean (lat, lon) masks.

    Atlantic: surface ocean cells between 34S and 70N that are 4-connected to a
    seed at (0N, 330E); includes the Gulf of Mexico, Caribbean and
    Mediterranean. Pacific: ocean cells 34S-65N, 105E-290E that are not
    Atlantic, excluding south of 8S west of 147E (Timor/Arafura seas,
    Australian Bight), 4-connected to a seed at (0N, 200E). Exits if the
    Atlantic fill leaks into the Pacific or the Pacific seed is land.
    """
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    band = ocean & (LA >= -34) & (LA <= 70)
    shift = len(lon) // 2  # put 180E at the array edge so the Atlantic does not straddle it
    lab, _ = ndimage.label(np.roll(band, shift, axis=1))
    j0, i0 = np.argmin(np.abs(lat)), (np.argmin(np.abs(lon - 330)) + shift) % len(lon)
    atl = np.roll(lab == lab[j0, i0], -shift, axis=1)
    jp, ip = np.argmin(np.abs(lat)), np.argmin(np.abs(lon - 200))
    if atl[jp, ip]:
        sys.exit("Atlantic flood fill leaked into the Pacific")
    pband = ocean & ~atl & (LA >= -34) & (LA <= 65) & (LO >= 105) & (LO <= 290) & ~((LA < -8) & (LO < 147))
    plab, _ = ndimage.label(pband)
    pac = plab == plab[jp, ip]
    if not pac[jp, ip]:
        sys.exit("Pacific seed is not ocean")
    return {"global": np.ones_like(atl), "atlantic": atl, "pacific": pac}


def interfaces(dzt_bottom_first):
    """Layer thicknesses (surface first) and interface depths z = 0 .. bottom."""
    dz = np.asarray(dzt_bottom_first)[::-1]
    return dz, np.concatenate([[0.0], np.cumsum(dz)])


def streamfunctions(v, masks, lat, lon, dz):
    """Overturning streamfunctions (Sv), one (nz + 1, nlat) array per mask.

    v is (depth, lat, lon), surface first; row k of psi is the transport above
    interface k (row 0 is the surface, zero).
    """
    v = np.nan_to_num(v)
    dx = R_EARTH * np.cos(np.deg2rad(lat)) * np.deg2rad(lon[1] - lon[0])
    out = {}
    for key, m in masks.items():
        V = np.sum(v * m[None] * dx[None, :, None], axis=2) * dz[:, None]   # m3/s per layer
        out[key] = np.vstack([np.zeros((1, len(lat))), np.cumsum(V, axis=0) / 1e6])
    return out


def window_stat(psi, z, lat, lats, depths, fn):
    """fn of psi over latitudes lats=(a, b) and interface depths depths=(a, b)."""
    sel = np.ix_((z >= depths[0]) & (z <= depths[1]), (lat >= lats[0]) & (lat <= lats[1]))
    return fn(psi[sel])


def bottom_row_at(psi, lat, lats):
    """Mean full-column (bottom-row) transport over latitudes lats=(a, b)."""
    return np.mean(psi[-1, (lat >= lats[0]) & (lat <= lats[1])])
