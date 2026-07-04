#!/usr/bin/env python
"""Build committed CI fixtures for the geometry-provenance contract tests.

Crops the real T144-308011-IW2 CSLC + STATIC granules (validation/real_data/, see
fetch_real.py) down to their metadata groups plus a tiny data window, and prints
independently-derived oracle values (heading, azimuth spacing, mean incidence) to
hardcode in the Rust contract tests.

    oracle/.venv/bin/python validation/make_geomprov_fixture.py
"""

from __future__ import annotations

import glob
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "oracle" / "fixtures"

CSLC = sorted(glob.glob(str(ROOT / "validation/real_data/OPERA_L2_CSLC-S1_T144*.h5")))[0]
STATIC = glob.glob(str(ROOT / "validation/real_data/OPERA_L2_CSLC-S1-STATIC_T144*.h5"))[0]

IDENT_KEYS = [
    "orbit_pass_direction",
    "look_direction",
    "mission_id",
    "burst_id",
    "track_number",
    "zero_doppler_start_time",
    "zero_doppler_end_time",
    "product_version",
]
ORBIT_KEYS = [
    "time",
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "reference_epoch",
    "orbit_direction",
    "orbit_type",
]
BURST_KEYS = [
    "range_pixel_spacing",
    "azimuth_time_interval",
    "center",
    "wavelength",
    "platform_id",
]


def copy_keys(src: h5py.Group, dst: h5py.File, group: str, keys: list[str]) -> None:
    g = dst.require_group(group)
    for k in keys:
        if k in src[group]:
            src.copy(f"{group}/{k}", g, name=k)


def make_cslc_fixture() -> None:
    dst_path = OUT / "geomprov_ci_cslc.h5"
    with h5py.File(CSLC, "r") as f, h5py.File(dst_path, "w") as g:
        copy_keys(f, g, "identification", IDENT_KEYS)
        copy_keys(f, g, "metadata/orbit", ORBIT_KEYS)
        copy_keys(f, g, "metadata/processing_information/input_burst_metadata", BURST_KEYS)
        d = g.create_group("data")
        d.create_dataset("VV", data=f["/data/VV"][:8, :8])
        d.create_dataset("x_coordinates", data=f["/data/x_coordinates"][:8])
        d.create_dataset("y_coordinates", data=f["/data/y_coordinates"][:8])
        d.create_dataset("projection", data=f["/data/projection"][()])
    print(f"wrote {dst_path} ({dst_path.stat().st_size / 1e3:.0f} kB)")


def make_data_only_fixture() -> None:
    """A /data-only CSLC (like the cropped validation granules): all provenance absent."""
    dst_path = OUT / "geomprov_ci_data_only.h5"
    with h5py.File(CSLC, "r") as f, h5py.File(dst_path, "w") as g:
        d = g.create_group("data")
        d.create_dataset("VV", data=f["/data/VV"][:8, :8])
        d.create_dataset("x_coordinates", data=f["/data/x_coordinates"][:8])
        d.create_dataset("y_coordinates", data=f["/data/y_coordinates"][:8])
        d.create_dataset("projection", data=f["/data/projection"][()])
    print(f"wrote {dst_path} ({dst_path.stat().st_size / 1e3:.0f} kB)")


def make_static_fixture() -> None:
    dst_path = OUT / "geomprov_ci_static.h5"
    r0, c0, n = 2200, 9700, 64  # same window family as crop_real.py
    with h5py.File(STATIC, "r") as f, h5py.File(dst_path, "w") as g:
        copy_keys(f, g, "identification", IDENT_KEYS)
        d = g.create_group("data")
        for layer in ("los_east", "los_north"):
            d.create_dataset(layer, data=f[f"/data/{layer}"][r0 : r0 + n, c0 : c0 + n])
        d.create_dataset("x_coordinates", data=f["/data/x_coordinates"][c0 : c0 + n])
        d.create_dataset("y_coordinates", data=f["/data/y_coordinates"][r0 : r0 + n])
        d.create_dataset("projection", data=np.int32(f["/data/projection"][()]))
    print(f"wrote {dst_path} ({dst_path.stat().st_size / 1e3:.0f} kB)")


def parse_dt(raw: bytes) -> datetime:
    return datetime.strptime(raw.decode(), "%Y-%m-%d %H:%M:%S.%f")


def oracle_values() -> None:
    with h5py.File(CSLC, "r") as f:
        ref = parse_dt(f["/metadata/orbit/reference_epoch"][()])
        t0 = parse_dt(f["/identification/zero_doppler_start_time"][()])
        t1 = parse_dt(f["/identification/zero_doppler_end_time"][()])
        t_mid = ((t0 - ref).total_seconds() + (t1 - ref).total_seconds()) / 2.0
        t = f["/metadata/orbit/time"][:]
        v = np.stack([f[f"/metadata/orbit/velocity_{a}"][:] for a in "xyz"], axis=1)
        p = np.stack([f[f"/metadata/orbit/position_{a}"][:] for a in "xyz"], axis=1)
        v_mid = np.array([np.interp(t_mid, t, v[:, i]) for i in range(3)])
        p_mid = np.array([np.interp(t_mid, t, p[:, i]) for i in range(3)])
        lon, lat = np.deg2rad(f["/metadata/processing_information/input_burst_metadata/center"][:])
        az_dt = float(f["/metadata/processing_information/input_burst_metadata/azimuth_time_interval"][()])
        rg = float(f["/metadata/processing_information/input_burst_metadata/range_pixel_spacing"][()])

    # ECEF -> ENU at (lat, lon)
    e_hat = np.array([-np.sin(lon), np.cos(lon), 0.0])
    n_hat = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    heading = np.rad2deg(np.arctan2(v_mid @ e_hat, v_mid @ n_hat)) % 360.0

    # ground-projected azimuth spacing: dt * |v| * R_earth(lat) / |r_platform|
    a, b = 6378137.0, 6356752.314245  # WGS84
    r_earth = np.sqrt(
        ((a**2 * np.cos(lat)) ** 2 + (b**2 * np.sin(lat)) ** 2)
        / ((a * np.cos(lat)) ** 2 + (b * np.sin(lat)) ** 2)
    )
    az_spacing = az_dt * np.linalg.norm(v_mid) * r_earth / np.linalg.norm(p_mid)

    with h5py.File(STATIC, "r") as f:
        e = f["/data/los_east"][::10, ::10].astype(np.float64)
        n = f["/data/los_north"][::10, ::10].astype(np.float64)
        valid = np.isfinite(e) & np.isfinite(n) & (np.abs(e) <= 1) & (np.abs(n) <= 1)
        up = np.sqrt(np.clip(1.0 - e[valid] ** 2 - n[valid] ** 2, 0.0, 1.0))
        inc_full = np.rad2deg(np.arccos(up))
        # fixture-window incidence (what the Rust test over the 64x64 crop sees)
        ef = f["/data/los_east"][2200:2264, 9700:9764].astype(np.float64)
        nf = f["/data/los_north"][2200:2264, 9700:9764].astype(np.float64)
        vf = np.isfinite(ef) & np.isfinite(nf) & (np.abs(ef) <= 1) & (np.abs(nf) <= 1)
        upf = np.sqrt(np.clip(1.0 - ef[vf] ** 2 - nf[vf] ** 2, 0.0, 1.0))
        inc_fix = np.rad2deg(np.arccos(upf))
        # LOS-implied heading (right-looking: heading = target->sensor azimuth + 90)
        los_az = np.rad2deg(np.arctan2(np.nanmean(e[valid]), np.nanmean(n[valid])))
        los_heading = (los_az + 90.0) % 360.0

    print(f"heading_deg (orbit ENU)      = {heading:.6f}")
    print(f"heading_deg (LOS-implied)    = {los_heading:.4f}   (cross-check, coarser)")
    print(f"native_range_spacing_m       = {rg!r}")
    print(f"native_azimuth_spacing_m     = {az_spacing:.6f}")
    print(f"incidence mean full burst    = {inc_full.mean():.4f}  (spread {inc_full.min():.2f}..{inc_full.max():.2f})")
    print(f"incidence mean fixture crop  = {inc_fix.mean():.6f}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    make_cslc_fixture()
    make_data_only_fixture()
    make_static_fixture()
    oracle_values()
