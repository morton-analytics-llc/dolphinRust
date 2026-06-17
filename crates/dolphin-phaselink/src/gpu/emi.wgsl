// GPU EMI phase linking (port of the EMI branch of estimator.rs, f32).
//
// Per output pixel: Γ = |C| (β=0 path), Cholesky-invert it, form the Hermitian
// M = Γ⁻¹ ⊙ C, and take its *least* eigenvector. M is PSD (Schur product of the
// PSD Γ⁻¹ and PSD C), so we first power-iterate M for its dominant eigenvalue
// λ_max, then power-iterate B = shift·I − M (shift = λ_max·SHIFT_MARGIN, strictly
// above λ_max so B stays PSD and its dominant mode is M's *least* — a tight ‖Mv‖
// shift underestimates λ_max and can lock onto the wrong mode). No per-iteration
// linear solve. On a non-PD Γ (Cholesky failure) we fall back to EVD (dominant
// eigenvector of C ⊙ |C|), matching the CPU NaN-triggered fallback.
//
// First-class hybrid: the least eigenvector of M is numerically ill-defined when
// M's two smallest eigenvalues are nearly degenerate — there an f32 iterative
// solver and faer's f64 direct decomposition pick different vectors in the
// near-degenerate subspace (the spike's π-rad tail). We recover M's second-least
// eigenvalue with one Hotelling-deflated power pass and emit a per-pixel
// `reliable` flag: a pixel is unreliable when the bottom eigengap (λ_2nd − λ_min)
// is small, the Rayleigh quotient of the "least" vector is too large (the
// iteration locked onto a high mode), or the mean coherence is low. The host
// recomputes the flagged minority on the f64 CPU path — so EMI matches the CPU
// reference on *every* pixel, sub-mm, with no π-rad tail.
//
// Scratch (Γ → Cholesky L → Γ⁻¹) is **threadgroup** memory, not per-thread
// private: at nslc up to 32 the nslc² scratch (~8 KB/thread for Γ + Γ⁻¹) spilled
// out of registers and produced run-to-run nondeterministic output at 384².
// Threadgroup memory never spills, so EMI is deterministic at every size. Each
// thread owns a private slice `[goff, goff + nslc²)` of the shared arrays (no
// barrier — there is no cross-thread sharing); the host sets the workgroup size
// WG and array length GAM_LEN = WG·nslc² so the two arrays fit the 32 KB
// threadgroup budget (WG ≈ 24 at nslc 13, 4 at nslc 32). The iterate vectors stay
// private (nslc-length, tiny). nslc ≤ MAX_NSLC. Complex = vec2<f32>.

const MAX_NSLC: u32 = 32u;

// Reliability thresholds (tuned on the real Mexico stack so the host fallback
// removes the π-rad tail; generous — false positives only cost a CPU recompute).
const GAP_TOL: f32 = 0.07;    // bottom relative eigengap floor (λ_2nd−λ_min)/λ_max
const COH_FLOOR: f32 = 0.10;  // mean off-diagonal |C| floor
const RHO_FRAC: f32 = 0.50;   // Rayleigh(v_least)/λ_max ceiling (wrong-mode capture)
const SHIFT_MARGIN: f32 = 1.02; // shift = λ_max·margin (‖Mv‖ underestimates λ_max)

// Host-set pipeline overrides: workgroup size and threadgroup-scratch length.
override WG: u32 = 4u;          // pixels (threads) per workgroup
override GAM_LEN: u32 = 4096u;  // WG · nslc² (≤ 4096 so 2 arrays fit 32 KB)

struct Params {
    nslc: u32,
    n_pix: u32,
    ref_idx: u32,
    iters: u32,
};

@group(0) @binding(0) var<storage, read> cmat: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> p: Params;
@group(0) @binding(2) var<storage, read_write> phase_out: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> eig_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> estimator_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> reliable_out: array<u32>;

var<workgroup> gam_wg: array<f32, GAM_LEN>; // Γ, then its Cholesky factor L (lower)
var<workgroup> inv_wg: array<f32, GAM_LEN>; // Γ⁻¹
var<private> goff: u32;                      // this thread's scratch slice base
var<private> v: array<vec2<f32>, MAX_NSLC>;
var<private> w: array<vec2<f32>, MAX_NSLC>;
var<private> v1: array<vec2<f32>, MAX_NSLC>; // saved least eigenvector

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Cholesky factor of Γ (lower, in place over `gam_wg` slice). Returns false if not PD.
fn cholesky(n: u32) -> bool {
    for (var j = 0u; j < n; j = j + 1u) {
        var sum = gam_wg[goff + j * n + j];
        for (var k = 0u; k < j; k = k + 1u) {
            sum = sum - gam_wg[goff + j * n + k] * gam_wg[goff + j * n + k];
        }
        if (sum <= 1e-12) { return false; }
        let ljj = sqrt(sum);
        gam_wg[goff + j * n + j] = ljj;
        for (var i = j + 1u; i < n; i = i + 1u) {
            var s = gam_wg[goff + i * n + j];
            for (var k = 0u; k < j; k = k + 1u) {
                s = s - gam_wg[goff + i * n + k] * gam_wg[goff + j * n + k];
            }
            gam_wg[goff + i * n + j] = s / ljj;
        }
    }
    return true;
}

// Given the Cholesky factor L in `gam_wg`, write Γ⁻¹ into `inv_wg` (solve L Lᵀ X = I).
fn invert_from_cholesky(n: u32) {
    for (var c = 0u; c < n; c = c + 1u) {
        for (var i = 0u; i < n; i = i + 1u) {
            var b = select(0.0, 1.0, i == c);
            for (var k = 0u; k < i; k = k + 1u) {
                b = b - gam_wg[goff + i * n + k] * inv_wg[goff + k * n + c];
            }
            inv_wg[goff + i * n + c] = b / gam_wg[goff + i * n + i];
        }
        for (var ii = 0u; ii < n; ii = ii + 1u) {
            let i = n - 1u - ii;
            var b = inv_wg[goff + i * n + c];
            for (var k = i + 1u; k < n; k = k + 1u) {
                b = b - gam_wg[goff + k * n + i] * inv_wg[goff + k * n + c];
            }
            inv_wg[goff + i * n + c] = b / gam_wg[goff + i * n + i];
        }
    }
}

// w ← M v, where M_ij = (use_emi ? Γ⁻¹_ij : |C_ij|) · C_ij. Returns ‖w‖₂.
fn matvec(n: u32, base: u32, use_emi: bool) -> f32 {
    var nrm = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        var acc = vec2<f32>(0.0, 0.0);
        for (var j = 0u; j < n; j = j + 1u) {
            let c = cmat[base + i * n + j];
            let wgt = select(length(c), inv_wg[goff + i * n + j], use_emi);
            acc = acc + wgt * cmul(c, v[j]);
        }
        w[i] = acc;
        nrm = nrm + dot(acc, acc);
    }
    return sqrt(nrm);
}

fn set_v_ones(n: u32) {
    for (var i = 0u; i < n; i = i + 1u) {
        v[i] = vec2<f32>(1.0, 0.0);
    }
}

// Normalize `w` into `v` by ‖w‖; returns ‖w‖ (the dominant eigenvalue estimate).
fn normalize_w_into_v(n: u32, nrm: f32) {
    let s = select(0.0, 1.0 / nrm, nrm > 0.0);
    for (var i = 0u; i < n; i = i + 1u) {
        v[i] = w[i] * s;
    }
}

// Least eigenvector of M into `v` via shifted power iteration on B = shift·I − M
// (from the current `v` start). Returns B's dominant eigenvalue (= shift − λ_min).
fn least_pass(n: u32, base: u32, shift: f32, iters: u32) -> f32 {
    var beta = 0.0;
    for (var it = 0u; it < iters; it = it + 1u) {
        _ = matvec(n, base, true);
        var nrm = 0.0;
        for (var i = 0u; i < n; i = i + 1u) {
            w[i] = shift * v[i] - w[i];
            nrm = nrm + dot(w[i], w[i]);
        }
        beta = sqrt(nrm);
        normalize_w_into_v(n, beta);
    }
    return beta;
}

// Second-least eigenvalue of M via Hotelling deflation of v₁ out of B = shift·I − M.
// Iterates the current `v` and returns B's deflated dominant (= shift − λ_2nd).
fn deflated_pass(n: u32, base: u32, shift: f32, beta1: f32, iters: u32) -> f32 {
    var beta = 0.0;
    for (var it = 0u; it < iters; it = it + 1u) {
        _ = matvec(n, base, true);
        var proj = vec2<f32>(0.0, 0.0);
        for (var i = 0u; i < n; i = i + 1u) {
            proj = proj + cmul(vec2<f32>(v1[i].x, -v1[i].y), v[i]);
        }
        var nrm = 0.0;
        for (var i = 0u; i < n; i = i + 1u) {
            let bv = shift * v[i] - w[i];
            w[i] = bv - beta1 * cmul(proj, v1[i]);
            nrm = nrm + dot(w[i], w[i]);
        }
        beta = sqrt(nrm);
        normalize_w_into_v(n, beta);
    }
    return beta;
}

// Power-iterate the dominant mode of M into `v` (iters steps).
fn power_dominant(n: u32, base: u32, use_emi: bool, iters: u32) {
    set_v_ones(n);
    for (var it = 0u; it < iters; it = it + 1u) {
        let nrm = matvec(n, base, use_emi);
        normalize_w_into_v(n, nrm);
    }
}

@compute @workgroup_size(WG)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let pix = gid.x;
    let n = p.nslc;
    goff = lid * n * n; // this thread's scratch slice (set before any guard return)
    if (pix >= p.n_pix) { return; }
    let base = pix * n * n;

    // Γ = |C|, mean off-diagonal coherence, Cholesky-invert (EVD fallback if non-PD).
    var coh_sum = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        for (var j = 0u; j < n; j = j + 1u) {
            let mag = length(cmat[base + i * n + j]);
            gam_wg[goff + i * n + j] = mag;
            if (i != j) { coh_sum = coh_sum + mag; }
        }
    }
    let off = max(1.0, f32(n * (n - 1u)));
    let coh_mean = coh_sum / off;
    let use_emi = cholesky(n);
    if (use_emi) {
        invert_from_cholesky(n);
    }

    // Dominant pass → eigenvector + λ_max (Rayleigh v^H M v).
    power_dominant(n, base, use_emi, p.iters);
    let lambda_max = matvec(n, base, use_emi);

    var reliable = 1u;
    if (use_emi) {
        // Shift strictly above λ_max so B = shift·I − M stays PSD and its dominant
        // mode is M's *least* (a tight ‖Mv‖ shift can undershoot → wrong mode).
        let shift = lambda_max * SHIFT_MARGIN;
        // Least eigenvector (β₁ = shift − λ_min); save it in v1.
        set_v_ones(n);
        let beta1 = least_pass(n, base, shift, p.iters);
        for (var i = 0u; i < n; i = i + 1u) { v1[i] = v[i]; }
        // Second-least eigenvalue via deflation (β₂ = shift − λ_2nd).
        set_v_ones(n);
        let beta2 = deflated_pass(n, base, shift, beta1, p.iters);
        let gap = beta1 - beta2; // = λ_2nd − λ_min

        // Rayleigh ρ = v₁ᴴ M v₁: the true least mode has the smallest ρ; a ρ near
        // λ_max means the iteration locked onto a high mode (non-convergence).
        for (var i = 0u; i < n; i = i + 1u) { v[i] = v1[i]; }
        _ = matvec(n, base, true);
        var rho = 0.0;
        for (var i = 0u; i < n; i = i + 1u) { rho = rho + dot(v1[i], w[i]); }

        // Flag ill-defined / non-converged / decorrelated pixels for CPU recompute.
        if (gap < GAP_TOL * lambda_max) { reliable = 0u; }
        if (rho > RHO_FRAC * lambda_max) { reliable = 0u; }
        if (coh_mean <= COH_FLOOR) { reliable = 0u; }
    }

    // Reference to ref_idx: multiply every entry by exp(-j·∠v[ref]).
    let r = v[p.ref_idx];
    let ang = atan2(r.y, r.x);
    let s = vec2<f32>(cos(-ang), sin(-ang));
    let pbase = pix * n;
    for (var i = 0u; i < n; i = i + 1u) {
        phase_out[pbase + i] = cmul(v[i], s);
    }
    eig_out[pix] = lambda_max;
    estimator_out[pix] = select(0u, 1u, use_emi);
    reliable_out[pix] = reliable;
}
