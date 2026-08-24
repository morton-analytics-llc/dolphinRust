# Spatial reference covariance kernel

`reference_specific_influence_v1` is the local phase-linking contract for one
target/reference pair. It evaluates both looked pixels against the same native
source-key set and contracts the returned factors as

\[
  C_{p-r}=F_pF_p^T+F_rF_r^T-F_pF_r^T-F_rF_p^T.
\]

The implementation is CPU/f64, rectangular support only, and uses the
production covariance replay, normalization, and fixed EVD/EMI phase JVPs. For
each native source key it binds the complete real/imaginary JVP basis through
the canonical `ProperComplexFactor::real_embedding`, retaining source model
identity and numeric receipt digest. Effective-look scaling is explicit under
`source_factor_declared_v1`. A source key absent from one looked support
contributes a zero block for that pixel; no marginal subtraction or
independent-pixel fallback is allowed.

The module does not implement sequential compression ancestry, L2 inversion,
artifact persistence, or calibration. Those layers consume the bounded factor
and retain the `conditional_only`/uncalibrated boundary until the #54
approximation/resource/review receipt and #53 gates pass.

Unsupported estimator branches, amplitude-floor transitions, tied modes,
non-finite state, and invalid output references fail closed with stable status
names. The source-key factor is bounded by the union of the two local Rect
supports, including border-clamped windows, strides, and fixed native masks.
