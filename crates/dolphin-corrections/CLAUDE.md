# dolphin-corrections — atmospheric phase corrections

Ionospheric + tropospheric range-delay modelling and the apply stage that subtracts
per-date delay from the displacement time series. Scientific reference: dolphin
`atmosphere/` (Yunjun et al. 2022; Chen & Zebker 2012). CPU only — this is delay
modelling + raster subtraction, no solver.

## Domain

- **Range delay → displacement units.** Every correction produces a per-acquisition
  **range delay in meters** on the frame grid. The apply stage subtracts the delay
  *relative to acquisition 0* (the series' own reference) from the LOS-phase series via
  `φ = d · (-4π/λ)` — the inverse of the pipeline's `phase → displacement` factor
  `-λ/4π` — so corrected displacement = `measured − relative_delay` exactly. Needs
  `input_options.wavelength`; corrections error without it.

- **Ionosphere (dispersive, `1/f²`).** From GNSS IONEX vertical-TEC maps:
  `delay = TEC_LOS · K / f²` with `K = 40.31`, TEC in el/m² (1 TECU = 1e16), `f` the
  radar carrier. Zenith→LOS uses the thin-shell refraction angle (Yunjun eq. 8). At
  vertical incidence this is the exact analytic anchor `delay = vtec·1e16·K/f²`. **This
  is the dominant L-band term**: `delay ∝ 1/f²`, so NISAR L-band (`f≈1.257 GHz`) is
  `(f_C/f_L)² ≈ 18×` the Sentinel-1 C-band (`f≈5.405 GHz`) effect for the same TEC —
  always scale to the *configured* λ, never a C-band constant. IONEX is coarse
  (2.5°×5°), so VTEC is sampled once at the frame centre per date (`grid_centroid_lonlat`
  → lon/lat; acquisition time-of-day from the granule name) and projected to a uniform
  delay grid.

- **Troposphere (non-dispersive).** Same delay in meters for L- and C-band. Primary
  source: the public OPERA L4 tropospheric netCDF (DISP-S1-aligned), read via GDAL's
  `NETCDF:` driver and resampled onto the frame grid — `resample_bilinear` when the
  product shares the frame CRS, `warp_to_frame` (GDAL bilinear `reproject`) when it
  differs (the global EPSG:4326 L4 product → a UTM frame). The L4 grid carries no CRS
  through GDAL's NETCDF driver, so a geographic-degree-range geotransform is assigned
  EPSG:4326 (the plate-carrée product). Fallback: RAiDER
  (`raider.py` subprocess) — **gated behind an availability check like SNAPHU, never
  stubbed**; absent ⇒ `RaiderUnavailable` and the path is skipped (deferred), not faked.

## Config (dolphin parity + forward divergence)

`CorrectionOptions` mirrors dolphin's `ionosphere_files` / `geometry_files` / `dem_file`
so a dolphin YAML round-trips. `troposphere_files` (direct OPERA-L4 ingest),
`incidence_angle_deg`, `troposphere_variable` are dolphinRust-only — dolphin instead
derives troposphere from `dem_file` via RAiDER. **Both corrections are opt-in (off by
default)**: empty file lists ⇒ no-op ⇒ output bit-identical to the uncorrected run.

## Conventions

- Correction math is proven on analytic/synthesized fixtures (closed-form TEC→delay,
  IONEX parse, synthesized L4 netCDF) regardless of network. **Real-source fetch (IONEX
  from CDDIS, OPERA L4 from ASF/PODAAC, RAiDER) is best-effort and deferred-with-receipts
  if unreachable** — see VALIDATION.md. Never stub a source.
- Keep delay functions pure and small; the apply stage is the only mutator.
