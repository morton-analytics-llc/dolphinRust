# dolphin-ingest — S3 read-staging (feature `s3`)

## Domain
dolphinRust *consumes* CSLC data already in S3; it never writes raw data back. Sliding-
window phase linking re-reads every pixel many times and OPERA CSLC HDF5 is not
cloud-optimized, so **stage, don't stream**: download each granule once to local scratch,
process locally, delete. Concurrent download (object_store + bounded tokio) is the only
async stage, hidden behind a synchronous `stage(uris, scratch) -> Vec<PathBuf>` facade.

## Conventions
- Off by default; local-path callers pull zero async deps.
- On the GroundPulse path this crate is bypassed — GroundPulse's `gp-storage` already
  stages S3→local. Primary use is the standalone `dolphin` CLI.
