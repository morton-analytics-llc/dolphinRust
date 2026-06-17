# dolphin-ps — persistent scatterers (reference: `dolphin/ps.py`)

## Domain
- Amplitude dispersion `D_A = std(|z|) / mean(|z|)` over the temporal stack; threshold
  (default 0.25) → uint8 mask (1=PS, 255=nodata). Tiled (512×512).
- **PS-fill rule:** PS pixels bypass covariance — they take phase from the brightest PS in
  the look window and get `temporal_coherence = 1.0`.

## Contracts
- Validate `D_A` and the threshold decision against a fixture with known per-pixel stats.
