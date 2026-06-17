# dolphin-workflows — pipeline orchestration (reference: `dolphin/workflows/`)

## Domain
Displacement pipeline in execution order: group inputs by burst → per-burst wrapped_phase
(mask → PS → SHP → covariance → phase-link → compress → ifg network) → stitch bursts →
unwrap → timeseries → velocity. Owns the YAML config models and the burst-parallel
executor (rayon).

## Conventions
- Public entry points are **synchronous** (`fn run(cfg) -> Result<…>`) — the host app
  bridges to its runtime. No tokio here.
- Don't start orchestration until the per-stage crates each carry green validation tests.
