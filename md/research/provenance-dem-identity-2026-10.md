# Full DEM identity in geometry provenance (#131)

**Date:** 2026-10-31 · **Base commit:** `25ab07a` · **Issue:** #131 (from eo#483)
**Question:** where should DEM dataset, DSM/DTM class, vertical accuracy and validity epoch
come from, and does dolphinRust record any of them?

## Result

**Adopt, narrowed to dataset identity.** #131 assumes the dataset name is not recoverable
from product metadata. It is. Every real OPERA CSLC-S1 and CSLC-S1-STATIC granule on this disk
names its DEM in `/metadata/processing_information/inputs/dem_source`, and dolphinRust can
record that string verbatim from the granules it already opens. No config field is needed. The
other three items (DSM/DTM, vertical accuracy, validity epoch) are not in the product. They are
facts about the Copernicus DEM, documented in its handbook, and belong in eo's report layer,
not in dolphinRust's provenance.

The review also found a gap the issue missed. The four `dem_*` fields shipped in #118
describe `correction_options.dem_file`, the DEM used by the troposphere correction, not the DEM
that shaped the CSLC phase. GroundPulse never sets `dem_file` for Sentinel-1
(`eo/crates/gp-dolphin/src/lib.rs:944-949` sets only `geometry_files` and, for NISAR,
`nisar_ellipsoidal_dem_file`). So every GroundPulse S1 run records all four as absent ("no DEM
configured"), and the report prints "not recorded" for grid spacing and datum as well
(`eo/crates/gp-reports/src/sections/method_and_limitations.rs:149-161`).

## What the products actually carry

**Inspected locally** (h5dump, HDF5 from `/opt/homebrew/opt/hdf5`, read-only):

- 7 distinct CSLC-S1-STATIC granules under `../dolphinRust/validation/real_data/` (T005-008703,
  -008704, -008705, T035-073270, T100-213486, T137-292316, T137-292329; all
  `product_specification_version = 0.1.7`, COMPASS 0.5.4, processed 2023-12-02), plus dolphin's
  test fixture T087-185683. All eight carry the same string:
  `dem_source = "Digital Elevation Model (DEM) for the NASA OPERA project version 1.1 (v1.1)
  based on the Copernicus DEM 30-m and Copernicus 90-m referenced to the WGS84 ellipsoid"`
  (attribute `description = "Description of the DEM used for processing"`).
- In the same granules: `algorithms/dem_interpolation = "biquintic"`,
  `algorithms/topography_algorithm = "isce3.geometry.topo"`, and the embedded `runconfig`
  repeats the description as `dem_description` with `dem_file: /home/compass_user/input_dir/dem.vrt`.
- 262 CSLC-S1 v1.1 granules: **200 carry the identical string**. The other 62 sit in the repo's
  `cropped*` fixture directories and have no `/metadata` group at all (stripped by the crop),
  so a missing key is a real case the code must handle.
- No granule has a field for DSM/DTM class, vertical accuracy, acquisition or validity epoch,
  or a machine-readable DEM version beyond the free text.

**Product spec.** *OPERA CSLC-S1 Static Layers Product Specification*, JPL D-108762 v1.0.1
(2023-11-07), d2pn8kiwq2w21t.cloudfront.net/documents/ProductSpec_CSLC-S1-STATIC.pdf:
- Table 2-1 lists the input as "GLO-30 Copernicus Digital Elevation Model".
- §3.6: STATIC is regenerated "a limited number of times to account for changes in the DEM".
- The only DEM field is `/metadata/processing_information/inputs/dem_source`, "string, scalar,
  Description of the DEM used for processing".
- Appendix A: the SDS ingests "any DEM whose vertical datum represents height above the WGS84
  Ellipsoid".

*OPERA DISP-S1 Static Layers Product Specification* v1.0.0 (2025-03-20) describes its DEM layer
as "the OPERA DEM v1.1, derived from the Copernicus DEM" and carries only `INPUT_DEM_SOURCE`,
"Description of the input DEM" (§4.1.2, Table 4-2).

**Copernicus DEM.** *Copernicus DEM Product Handbook* v5.0 (Airbus, 2022-11-29),
dataspace.copernicus.eu/…/geo1988-copernicusdem-spe-002_producthandbook_i5.0.pdf:
- "The Copernicus DEM is a Digital Surface Model (DSM) which represents the surface of the
  Earth including buildings, infrastructure and vegetation."
- Vertical reference is EGM2008 (EPSG 3855).
- GLO-30/GLO-90 cover "the time frame of data acquisition (2011-2015)", and TanDEM-X was
  "acquired between December 2010 and January 2015".
- Absolute vertical accuracy, DLR LE90ABS against ICESat (Table 12): global mean 2.57 m;
  1.92 m excluding Greenland and Antarctica. The specification statement is "LE90ABS < 4m for
  the global product".

**What that does and does not license.** The STATIC string says *OPERA DEM v1.1, built from
Copernicus 30 m and 90 m, ellipsoid-referenced*. The handbook's DSM class, epoch and accuracy
describe the *source* Copernicus DEM. Nothing fetched states the vertical accuracy of the OPERA
v1.1 derivative (geoid-to-ellipsoid conversion, GLO-90 fill), or which Copernicus release it
was built from. Writing "DSM, LE90 < 4 m, 2011–2015" into dolphinRust's provenance would be a
lookup from a free-text string to documentation values. That is the "platform nominal" this
module refuses by design (`crates/dolphin-workflows/src/provenance.rs:3-4`: "never from config
knobs, filename guesses, or platform nominals").

## The boundary question in #131

#131 offered (a) a caller-asserted `CorrectionOptions.dem_product` block or (b) eo keying DEM
identity on a file path or hash in `DolphinBuildIdentityV1`. Neither is right:

- (a) records what the caller says, not what the product says, which inverts the module's
  rule, and it would sit beside a sourced value that already exists.
- (b) keys on a DEM file GroundPulse does not use for S1. The DEM that matters is
  upstream, inside COMPASS, and only the granules record it.

**(c) instead:** dolphinRust records `dem_source` verbatim from the granules, gated on all
granules agreeing, as it already does for `orbit_type`. eo owns any mapping from that exact
string to documented properties (DSM, epoch, LE90), cites the handbook in the report, and marks
the values as properties of the source dataset rather than measurements of this run. eo#483 is
closed (2026-10-19) and eo#661 added the report line that prints "dataset, DSM/DTM class, geoid
model, vertical accuracy: not recorded", so #131's re-entry gate ("the report generator has a
concrete field to populate") is met.

## Conclusion: ADOPT (dataset identity only)

**Spec.**

- **dolphin-io** `crates/dolphin-io/src/cslc_metadata.rs`: `pub fn read_cslc_dem_source(path:
  &Path) -> Result<String>`, reading `/metadata/processing_information/inputs/dem_source`
  through the existing `read_string` (l.142). Same error style as `read_cslc_orbit_type`
  (l.60). Unit tests beside the existing `orbit_type` fixture tests (~l.172, 247, 260):
  present → value; missing → `Err`.
- **dolphin-workflows** `crates/dolphin-workflows/src/provenance.rs`:
  - New field `cslc_dem_source: Option<String>` with
    `#[serde(default, skip_serializing_if = "Option::is_none")]`.
  - Assembled by `fn cslc_dem_source(cfg, &mut fields)`, modelled on `orbit_ephemeris_class`
    (l.429). Read every `cfg.cslc_file_list` granule plus every
    `correction_options.geometry_files` STATIC granule. Trim NUL and ASCII whitespace only.
    Byte-equal across all granules → `Sourced { source_files: [granule names],
    source_keys: ["/metadata/processing_information/inputs/dem_source"], method: "verbatim
    CSLC/CSLC-S1-STATIC dem_source; all granules agree", raw_value: Some(first untrimmed) }`.
  - Otherwise `Absent`: on a read failure, name the granule and key; with disagreement,
    "mixed dem_source across granules (k distinct values)"; for NISAR,
    "NISAR GSLC DEM source mapping not implemented".
  - No new fields for DSM/DTM, accuracy or epoch.
  - Doc-comment the existing four `dem_*` fields (l.89-102) and `dem_identity` (l.501-513)
    as describing `correction_options.dem_file`, the correction DEM, so no reader takes them
    for the CSLC DEM.
  - Schema `dolphinrust-geometry-provenance/5` → `/6`, `METHOD_VERSION` `5.0.0` → `6.0.0`
    (l.30-31).
- **Docs:** add a `cslc_dem_source` row to the per-field source table in
  `md/design/geometry-provenance.md`; add a CHANGELOG entry.
- **Contract tests** in `crates/dolphin-workflows/tests/geometry_provenance_contract.rs`
  (synthetic HDF5 fixtures, serialized via the local HDF5 lock):
  1. Two CSLC + one STATIC fixture with identical `dem_source` → field equals the string
     exactly, status `sourced`, `source_files.len() == 3`.
  2. One CSLC differs by one character → `None`, `absent`, reason contains `mixed`.
  3. One CSLC missing the key → `None`, `absent`, reason names that granule.
  4. NISAR `input_type` → `None`, `absent`.
  5. A trailing-NUL padded value (the STATIC string is `H5T_STR_NULLPAD`, STRSIZE 162) →
     trimmed value, untrimmed `raw_value`.
  6. Serialized `schema == "dolphinrust-geometry-provenance/6"`; the key is omitted when
     `None`.
  7. Opt-in (`#[ignore]`) real-data check on
     `validation/real_data/t035_073270_iw1_p169_p170` STATIC plus a v1.1 CSLC → exactly the
     OPERA DEM v1.1 string quoted above.

  Acceptance: all seven pass, the existing provenance contract tests pass unchanged apart from
  the schema string, and no value is normalised beyond the trim.
- **Follow-up in eo (not dolphinRust):** print `cslc_dem_source` as the report's DEM dataset
  line. If eo wants DSM/epoch/LE90, it maps from that exact string, cites the Copernicus
  handbook v5.0, and labels the values as source-dataset properties. That is eo's issue to file.

**Out of scope for dolphinRust, with reasons:**
- DSM/DTM class, vertical accuracy, validity epoch: not carried by any OPERA product field;
  recording them would be a documentation lookup.
- A config-asserted `dem_product` block: inverts the module's sourcing rule and duplicates a
  sourced value.

**Evidence that would change this:** an OPERA product revision (STATIC spec > 0.1.7, or a
CSLC release) that adds structured DEM fields. dolphinRust would then read those the same way.
