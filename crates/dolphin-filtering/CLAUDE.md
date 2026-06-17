# dolphin-filtering — phase filters (reference: `dolphin/filtering.py`, `goldstein.py`)

## Domain
- Long-wavelength FFT Gaussian high-pass filter; Goldstein adaptive filter. Both via
  rustfft. Optional pre-unwrap stages.

## Contracts
- Validate against synthetic ramps / known spectra; dolphin as a reference oracle
  (atol ~1e-4).
