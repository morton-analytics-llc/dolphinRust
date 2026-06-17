# dolphin-unwrap — unwrapping dispatch (reference: `dolphin/unwrap/`)

## Domain
dolphin contains NO unwrapping math — it wraps external solvers, and so do we. This crate
shells out to the **SNAPHU** binary (subprocess): tiling, cost model / init method, NPROC
parallelism, nodata propagation, connected-component regrow. tophu / spurt / whirlwind are
documented gaps unless required. Not a rebuild target.

## Conventions
- Treat SNAPHU as a hard dependency; fail fast if the binary is absent.
