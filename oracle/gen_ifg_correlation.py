#!/usr/bin/env python
"""Generate phase-correlation contracts with installed dolphin 0.35.0.

Run: oracle/.venv/bin/python oracle/gen_ifg_correlation.py
"""

from pathlib import Path
import hashlib
import importlib.metadata
import json

import numpy as np
from scipy.ndimage import gaussian_filter

from dolphin.interferogram import estimate_correlation_from_phase


OUT = Path(__file__).resolve().parents[1] / "crates/dolphin-workflows/tests/fixtures/ifg_correlation.json"
SHAPE = (33, 35)
SIGMA = 11 / 3
TRUNCATE = 4.0


def main():
    assert importlib.metadata.version("dolphin") == "0.35.0"
    yy, xx = np.indices(SHAPE, dtype=np.float64)
    amplitude = 0.25 + (1 + (3 * xx + 7 * yy) % 23) / 4
    ramp = 0.13 * xx - 0.09 * yy + 0.27
    varying = ramp + 1.7 * np.sin(xx * 0.53) * np.cos(yy * 0.37) + 0.003 * xx * yy
    cases = []
    for name, phase in [("smooth_ramp", ramp), ("varying_phase", varying), ("missing_support", varying)]:
        # Freeze source values at complex64, then compute the oracle in float64.
        ifg = (amplitude * np.exp(1j * phase)).astype(np.complex64).astype(np.complex128)
        support = np.ones(SHAPE, dtype=bool)
        if name == "missing_support":
            ifg[0:4, 0:6] = complex(np.nan, 0)
            ifg[12:17, 16:21] = complex(0, np.nan)
            ifg[27:33, 30:35] = 0
            support[6:11, 24:31] = False
            support[20:24, 0:3] = False
        valid = support & np.isfinite(ifg.real) & np.isfinite(ifg.imag) & (abs(ifg) > 0)
        unit = np.zeros(SHAPE, dtype=np.complex128)
        unit[valid] = ifg[valid] / abs(ifg[valid])
        numerator = gaussian_filter(unit, SIGMA, mode="nearest", truncate=TRUNCATE)
        denominator = gaussian_filter(valid.astype(np.float64), SIGMA, mode="nearest", truncate=TRUNCATE)
        normalized = np.zeros(SHAPE, dtype=np.float64)
        np.divide(abs(numerator), denominator, out=normalized, where=denominator > 0)
        normalized = np.clip(normalized, 0, 1)
        normalized[~valid] = 0
        parity = None
        if name != "missing_support":
            expected = estimate_correlation_from_phase(ifg, window_size=11)
            parity = float(np.max(abs(expected - normalized)))
            assert parity < 2e-15
        else:
            expected = normalized
        case = {
            "name": name,
            "shape": list(SHAPE),
            "ifg_re": [float(v) if np.isfinite(v) else None for v in ifg.real.ravel()],
            "ifg_im": [float(v) if np.isfinite(v) else None for v in ifg.imag.ravel()],
            "support": support.ravel().tolist(),
            "expected": expected.astype(np.float32).ravel().tolist(),
            "upstream_vs_normalized_max_abs": parity,
            "invalid_input_centers": int((~valid).sum()),
        }
        cases.append(case)
    document = {
        "metadata": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {p: importlib.metadata.version(p) for p in ("dolphin", "numpy", "scipy")},
            "window_size": 11,
            "sigma": SIGMA,
            "truncate": TRUNCATE,
            "radius": int(TRUNCATE * SIGMA + 0.5),
            "mode": "nearest",
            "input_precision": "complex64 values promoted exactly to complex128",
            "expected_precision": "float32, before IO mantissa rounding",
            "mantissa_policy": "Dolphin estimate_correlation_from_phase returns unrounded correlation. Mantissa rounding belongs to raster IO compression, not the estimator; preserve all float32 bits here.",
            "null_input": "null real or imaginary component represents NaN",
            "missing_formula": "valid = support & finite(real) & finite(imag) & abs(z)>0; u=valid*z/abs(z) with zero elsewhere; correlation=clip(abs(G(u))/G(valid),0,1); zero where denominator is zero or center invalid. G is sigma11/3, nearest, truncate4.",
            "missing_reference": "Independent SciPy normalized convolution. Excludes invalid samples before unit normalization; does not copy upstream nan_to_num phase-zero contributions.",
            "support_semantics": "Input support remains a separate structural predicate; zero correlation at a valid center does not make that center structurally invalid.",
        },
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({"path": str(OUT), "bytes": OUT.stat().st_size, "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(), "cases": [{"name": c["name"], "finite_parity": c["upstream_vs_normalized_max_abs"], "invalid_centers": c["invalid_input_centers"]} for c in cases]}))


if __name__ == "__main__":
    main()
