# dolphin-phaselink — phase linking (reference: `dolphin/phase_link/`)

The numerical core and highest-value crate — the main reason for the Rust rebuild. dolphin
defines the algorithms; the data layout, solver choice, and parallelism are ours to
optimize. All math in `Cf64`.

## Domain
- **Covariance:** sliding-window sample coherence matrix
  `C_ij = Σ(z_i z_j*) / sqrt(Σ|z_i|² · Σ|z_j|²)`, optionally masked by the SHP neighbor
  array. The #1 hot path. Two kernels: the **direct** per-pixel path (`_direct`,
  parallel over output pixels — the SHP-masked implementation and the tolerance oracle)
  and the **row-separable box-sum** (default for the unmasked rectangular window,
  `neighbors: None` = the production path): parallel over output *rows*, reusing each
  row's per-column vertical sums across its output columns and summing each window in
  fixed left-to-right order. The sliding kernel matches direct to coherence ~1e-4 (order
  differs), **not** bit-exactly; but `fused==staged` and `tiled==whole` stay bit-identical
  because both share the one `sliding_row_numerators` and each window depends only on its
  own samples. Do **not** materialize an `nslc²·area` cube — per-row buffers only.
- **EVD:** largest eigenvector of `C ⊙ |C|` (power iteration).
- **EMI (default):** smallest eigenvector of `Γ⁻¹ ⊙ C`, where `Γ = |C|`. Regularize
  `Γ ← (1−β)Γ + βI`, threshold near-zero entries, Cholesky-invert with 1e-6 jitter,
  inverse iteration (shift μ=0.99). **On singular `Γ⁻¹` (NaN), fall back to EVD** — this
  fallback is part of the algorithm, keep it.
- Eigensolvers: power / inverse iteration are dolphin's approach; we are free to use faer's
  direct dense eigensolvers if faster, as long as the result converges to the correct
  eigenvector. The N×N systems are small (N = ministack size).
- **Phase referencing:** `θ ← θ · exp(−j·∠θ[ref_idx])`.
- **Quality:** temporal coherence `|Σ_{i>j} C_ij e^{−j(θ_i−θ_j)} W_ij| / Σ W_ij`; average
  coherence magnitude per real SLC date `mean_j |C_ij|` (optionally reduced to the distinct
  workflow `phase_linking_coherence` raster); CRLB from the Fisher matrix; closure phase on
  nearest triples. Dolphin v0.35's public `avg_coh` is an argmax date index, so do not expose
  or name that integer as a coherence value.
- **Compressed SLC:** `Σ_k z_k e^{−jθ_k} / N` projection (carried forward by dolphin-stack).
- **Phase-bias correction** (`phasebias`, Michaelides 2022; forward divergence, **not in dolphin**):
  nearest-neighbour closure `Ξ_k = β_k + β_{k+1}` ⇒ first-order constant bias velocity
  `β̄ = mean_k(Ξ_k)/2`, subtract cumulative `B_n = n·β̄` from the linked phase. Opt-in
  (`correct_phase_bias`, off by default). Validate analytically (exact for constant bias) + by
  measured non-closure reduction — no oracle (it leads dolphin).

## Contracts
- Validate against analytic covariance fixtures (known dominant eigenvector) plus dolphin
  as a reference oracle, to physically-meaningful tolerances — not bit-exactness.
  Eigenvector compared as `|⟨v_rust, v_oracle⟩|` (sign / global-phase ambiguity); referenced
  phase ~1e-3 rad; coherence ~1e-4. Write the contract test first.
