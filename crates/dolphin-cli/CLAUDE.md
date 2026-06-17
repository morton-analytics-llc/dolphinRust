# dolphin-cli — `dolphin` binary

## Domain
CLI mirroring Python `dolphin run`. `dolphin run --config <yaml>` drives the displacement
workflow. Optional S3 inputs via dolphin-ingest (feature `s3`).

## Conventions
- Thin shell over dolphin-workflows: parse, configure tracing, dispatch. No domain logic.
- Until a stage lands, fail honestly (no stubbed success).
