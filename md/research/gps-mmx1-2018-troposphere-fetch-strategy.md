# MMX1/ICMX 2018 troposphere fetch strategy

**Intake:** DR-TROPO-2018 / T06
**Scope:** decide the transfer budget before fetching the 52-epoch OPERA L4 cohort.

## Decision

Server-side spatial subsetting is unavailable for CMR collection
`C3717139408-ASF` (`OPERA_L4_TROPO-ZENITH_V1`): its CMR metadata advertises
`has-spatial-subsetting: false` and distribution is by complete HTTPS/S3 objects.

On-ingest reduction is plausible but unmeasured. GDAL can open netCDF/HDF5 over
`/vsis3/` or `/vsicurl/`, and dolphinRust already requests only the source window and
height bands needed by the frame after a granule is local. Remote HDF5 chunk and metadata
reads can amplify that logical window to a large fraction of the object, so cell count is
not a transfer estimate.

**Superseded 2026-09-03 by the probe result below.** The fallback gate this section set,
`projected_total_transfer_bytes=111638814943`, was never a measurement — it was the
whole-object penalty standing in until someone ran the probe. The probe has now run and
replaced it with a measured integer.

## Known quantities

| Quantity | Value |
|---|---:|
| Matching HRES epochs | 52 |
| Exact bulk transfer | 111,638,814,943 bytes (111.639 GB; 103.972 GiB) |
| Mean object size | 2,146,900,287 bytes |
| Object-size range | 2,059,653,340-2,223,500,438 bytes |
| Source raster | 5,120 columns x 2,560 rows = 13,107,200 cells per height plane |
| 2018 frame bounds | longitude -99.17691 to -98.97694; latitude 19.40265 to 19.48732 |
| Required horizontal source window | 6 columns x 5 rows = 30 cells per height plane |
| Logical cell fraction | 30 / 13,107,200 = 0.0000022888 |

The 6x5 result includes the bounded native window needed for the burst footprint. It says
nothing about transferred bytes because HDF5 chunk shape, coordinate reads, metadata, and
range-request behavior control the actual transfer. The required height-band count is also
unknown until a DEM covering the frame is staged and its finite elevation range is mapped to
the product's bracketing `height` coordinates.

The current implementation already performs the desired local access pattern:

- `crates/dolphin-corrections/src/troposphere.rs` transforms the destination bounds to the
  source CRS and reads a native pixel window.
- `crates/dolphin-workflows/src/corrections.rs` selects the inclusive height-band range
  bracketing the frame DEM, reads those planes, and interpolates each pixel to terrain.
- `total` is the sum of `hydrostatic_delay` and `wet_delay`; neither variable may be omitted.

## One-granule metered probe

Use this exact HRES granule because its object size is known and it is the first cohort epoch:

```text
CMR concept: G3752930776-ASF
Granule: OPERA_L4_TROPO-ZENITH_20180106T000000Z_20250922T192334Z_HRES_v1.0
Epoch: 2018-01-06T00:00:00Z
Object bytes: 2147849917
```

The later probe must:

1. Stage only the burst DEM first. Record the minimum and maximum finite terrain elevations,
   then select the inclusive L4 `height` indices that bracket that range.
2. Start with empty GDAL/VSI caches and an isolated byte counter. Obtain temporary Earthdata
   S3 credentials or use authenticated HTTPS, open the named object remotely, and require
   byte-range responses.
3. Read the coordinate metadata needed to resolve `time`, the selected `height` values, and
   the 6x5 `latitude`/`longitude` window. Extract every selected height plane for both
   `hydrostatic_delay` and `wet_delay`.
4. Meter `probe_transfer_bytes` from the first remote object read through completion. Count
   all object response-body bytes, including repeated chunks, HDF5 metadata/index reads,
   retries, and both variables. Do not infer this value from the staged subset size.
5. Write the staged subset without changing dimension order, coordinate values, band order,
   `_FillValue`/mask behavior, units, scale/offset metadata, or the EPSG:4326 geotransform.
   Preserve `time`, `height`, `latitude`, `longitude`, `hydrostatic_delay`, and `wet_delay`.
6. Compare remote-window and staged-window values and masks exactly, then run the four Rust
   contracts below against the staged result before recording the gate value.

If range requests are rejected or ignored, extraction cannot preserve that contract, or
`probe_transfer_bytes` is at least half of the 2,147,849,917-byte probe object, record the
bulk fallback instead of extrapolating.

## Sole transfer gate

For a successful probe, compute with integer ceiling:

```text
projected_total_transfer_bytes =
    ceil(probe_transfer_bytes * 52 * 2223500438 / 2147849917)
```

The maximum observed object size makes the extrapolation conservative. The probe receipt must
contain exactly one unsigned-decimal decision field:

```text
projected_total_transfer_bytes=<integer>
```

Missing/non-integer output or the fallback value `111638814943` is a no-go. A lower measured
integer is the one number presented for the cohort transfer decision; logical cells, staged
disk size, and compressed subset size are supporting diagnostics only.

## Contracts before any cohort fetch

```sh
cargo test -p dolphin-corrections bounded_l4_read_is_native_windowed_and_resamples_exactly
cargo test -p dolphin-workflows delay_interpolates_to_terrain_elevation
cargo test -p dolphin-workflows bracketing_levels_cover_the_terrain_range
cargo test -p dolphin-workflows build_troposphere_warps_4326_onto_utm_frame
```

The remote/staged exact comparison and all four contracts must pass before replacing the
fallback. This strategy does not authorize a fetch, run the 2018 cohort, or claim a
troposphere residual result.

## Sources

- [CMR collection metadata](https://cmr.earthdata.nasa.gov/search/collections.umm_json?concept_id=C3717139408-ASF&page_size=1)
- [CMR probe-granule metadata](https://cmr.earthdata.nasa.gov/search/granules.umm_json?concept_id=G3752930776-ASF&page_size=1)
- [OPERA L4 TROPO product specification](https://d2pn8kiwq2w21t.cloudfront.net/documents/OPERA_TROPO_CalVal_Product_Spec.pdf)
- [GDAL netCDF driver](https://gdal.org/en/stable/drivers/raster/netcdf.html)
- [GDAL virtual file systems](https://gdal.org/en/stable/user/virtual_file_systems.html)

## Probe result — 2026-09-03

Run against the exact granule this strategy names. Range requests are honored (HTTP 206,
CloudFront-signed URL after one authenticated redirect); auth is the `GP_EARTHDATA_TOKEN`
bearer, not `~/.netrc`, which is stale.

```text
projected_total_transfer_bytes=68794097
```

**0.069 GB against the 111.639 GB fallback — 1,623x smaller.** The cohort fetch is a go.

| Quantity | Value |
|---|---:|
| `probe_transfer_bytes` (metered, all response bodies) | 1,277,952 |
| Range requests | 78 |
| Read block size (GDAL `/vsicurl` default) | 16,384 |
| Half-object no-go threshold | 1,073,924,958 |
| Frame terrain range (Copernicus 30 m DEM, windowed read) | 2217.65 - 2298.48 m |
| Bracketing `height` indices | 35, 36, 37 (2081.09 / 2261.80 / 2452.99 m) |
| Staged window | 3 lat x 5 lon x 3 heights, both variables |
| Staged file size | 14,752 bytes |

Extrapolation per the gate formula, using the cohort's largest object size:
`ceil(1277952 * 52 * 2223500438 / 2147849917) = 68794097`.

**Why it collapsed.** The delay variables are `(time, height, lat, lon)` = `(1, 145, 2560,
5120)` float32, gzip, **chunked `(1, 64, 64, 64)`**. The frame window and its three
bracketing height bands fall inside a handful of chunks, so the transfer is chunk-bound,
not object-bound. That also means the number is insensitive to the exact window: this probe
staged 3x5 cells where the strategy anticipated 6x5, and both land in the same chunks.

**Verification.** Staged-vs-remote comparison is bit-identical for `time`, `height`,
`latitude`, `longitude`, `hydrostatic_delay` and `wet_delay` — values, `_FillValue` masks
and units — with `spatial_ref` (EPSG:4326) preserved. The window strictly covers the frame
bounds and the height bands strictly bracket the terrain, both asserted rather than assumed.
Recovered delays are physically sensible for Mexico City in January: hydrostatic
1.7109-1.7888 m, wet 0.0667-0.0959 m.

All four required contracts pass:

```text
troposphere::tests::bounded_l4_read_is_native_windowed_and_resamples_exactly ... ok
corrections::tests::delay_interpolates_to_terrain_elevation ... ok
corrections::tests::bracketing_levels_cover_the_terrain_range ... ok
corrections::tests::build_troposphere_warps_4326_onto_utm_frame ... ok
```

**Caveats on the number.** It is measured under a 16 KiB block cache; a client using a
larger read block would transfer more. Terrain here needs only 3 of 145 height levels and
they fall inside one 64-level chunk — a frame whose terrain straddles a height-chunk
boundary would roughly double the per-epoch cost, which the 52x extrapolation does not
model. Both leave the result orders of magnitude inside the fallback.

Reproduce:

```sh
source validation/creds.sh
oracle/.venv/bin/python validation/probe_l4_tropo_transfer.py
oracle/.venv/bin/python validation/probe_l4_tropo_stage_verify.py
```

This result authorizes the cohort transfer. It does not run the 2018 cohort or claim a
troposphere residual result.
