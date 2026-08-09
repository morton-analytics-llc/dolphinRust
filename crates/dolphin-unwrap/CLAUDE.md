# dolphin-unwrap — unwrapping dispatch (reference: `dolphin/unwrap/`)

## Domain
dolphin contains NO unwrapping math — it wraps external solvers. dolphinRust **diverges**: it
ships its own unwrapper and only falls back to a subprocess on request.

- **`native` (the default)** — clean-room in-process Costantini MCF via network simplex, no
  SNAPHU and no CS2. Owns cost model, network simplex, auto-tiling (64-pixel core floor),
  seam reconciliation, and connected-component regrow. Matches SNAPHU per-component to
  ≤0.5% on the MMX1 live common frame while running faster. Modules: `native/{cost, simplex,
  mcf, tile, conncomp}.rs`.
- **`snaphu`** — selectable fallback; shells out to the SNAPHU binary (tiling, cost model /
  init method, NPROC parallelism, nodata propagation).
- **`tophu`** — multi-scale driver over the SNAPHU per-tile solver (coarse init → overlapping
  tiled SNAPHU → 2π-reconciled merge). dolphin reserves `multiscale_unwrap` for ICU/PHASS;
  we expose tophu as first-class over the solver we ship.
- `icu` / `phass` / `spurt` / `whirlwind` are documented gaps unless required.

## Conventions
- SNAPHU is **not** a hard dependency — it is required only when `unwrap_method` selects
  `snaphu` or `tophu`. Fail fast, with the method named, if a selected backend's binary is
  absent; never silently substitute another backend.
- Seam tie-breaks must be deterministic (no `HashMap` iteration order) — non-determinism here
  produced a one-fringe slip that failed the GNSS gate on 2026-07-14.
