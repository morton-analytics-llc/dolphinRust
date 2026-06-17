// GPU EMI phase linking (port of the EMI branch of estimator.rs, f32).
//
// Per output pixel: Γ = |C| (β=0 path), Cholesky-invert it, form the Hermitian
// M = Γ⁻¹ ⊙ C, and take its *least* eigenvector. M is PSD (Schur product of the
// PSD Γ⁻¹ and PSD C), so we first power-iterate M for its dominant eigenvalue
// λ_max, then power-iterate B = λ_max·I − M whose dominant mode is M's least —
// no per-iteration linear solve, and a tight shift (so B's spectrum is well
// separated, unlike a loose Gershgorin bound). On a non-PD Γ (Cholesky failure)
// we fall back to EVD (dominant eigenvector of C ⊙ |C|), matching the CPU
// NaN-triggered fallback.
//
// Γ, its Cholesky factor, and the inverse share private nslc² scratch; the
// iterate vectors are private nslc-length. nslc ≤ MAX_NSLC. Complex = vec2<f32>.

const MAX_NSLC: u32 = 32u;
const MAX_NN: u32 = 1024u; // MAX_NSLC * MAX_NSLC

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

var<private> gam: array<f32, MAX_NN>; // Γ, then its Cholesky factor L (lower)
var<private> inv: array<f32, MAX_NN>; // Γ⁻¹
var<private> v: array<vec2<f32>, MAX_NSLC>;
var<private> w: array<vec2<f32>, MAX_NSLC>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Cholesky factor of Γ (lower, in place over `gam`). Returns false if not PD.
fn cholesky(n: u32) -> bool {
    for (var j = 0u; j < n; j = j + 1u) {
        var sum = gam[j * n + j];
        for (var k = 0u; k < j; k = k + 1u) {
            sum = sum - gam[j * n + k] * gam[j * n + k];
        }
        if (sum <= 1e-12) { return false; }
        let ljj = sqrt(sum);
        gam[j * n + j] = ljj;
        for (var i = j + 1u; i < n; i = i + 1u) {
            var s = gam[i * n + j];
            for (var k = 0u; k < j; k = k + 1u) {
                s = s - gam[i * n + k] * gam[j * n + k];
            }
            gam[i * n + j] = s / ljj;
        }
    }
    return true;
}

// Given the Cholesky factor L in `gam`, write Γ⁻¹ into `inv` (solve L Lᵀ X = I).
fn invert_from_cholesky(n: u32) {
    for (var c = 0u; c < n; c = c + 1u) {
        for (var i = 0u; i < n; i = i + 1u) {
            var b = select(0.0, 1.0, i == c);
            for (var k = 0u; k < i; k = k + 1u) {
                b = b - gam[i * n + k] * inv[k * n + c];
            }
            inv[i * n + c] = b / gam[i * n + i];
        }
        for (var ii = 0u; ii < n; ii = ii + 1u) {
            let i = n - 1u - ii;
            var b = inv[i * n + c];
            for (var k = i + 1u; k < n; k = k + 1u) {
                b = b - gam[k * n + i] * inv[k * n + c];
            }
            inv[i * n + c] = b / gam[i * n + i];
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
            let wgt = select(length(c), inv[i * n + j], use_emi);
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

// Power-iterate the dominant mode of M into `v` (iters steps).
fn power_dominant(n: u32, base: u32, use_emi: bool, iters: u32) {
    set_v_ones(n);
    for (var it = 0u; it < iters; it = it + 1u) {
        let nrm = matvec(n, base, use_emi);
        let s = select(0.0, 1.0 / nrm, nrm > 0.0);
        for (var i = 0u; i < n; i = i + 1u) {
            v[i] = w[i] * s;
        }
    }
}

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pix = gid.x;
    if (pix >= p.n_pix) { return; }
    let n = p.nslc;
    let base = pix * n * n;

    // Γ = |C|, Cholesky-invert (EVD fallback if non-PD).
    for (var i = 0u; i < n; i = i + 1u) {
        for (var j = 0u; j < n; j = j + 1u) {
            gam[i * n + j] = length(cmat[base + i * n + j]);
        }
    }
    let use_emi = cholesky(n);
    if (use_emi) {
        invert_from_cholesky(n);
    }

    // Dominant pass → eigenvector + λ_max (Rayleigh v^H M v).
    power_dominant(n, base, use_emi, p.iters);
    let lambda_max = matvec(n, base, use_emi);

    // EMI: shifted pass for the least mode of M = dominant of λ_max·I − M.
    if (use_emi) {
        set_v_ones(n);
        for (var it = 0u; it < p.iters; it = it + 1u) {
            _ = matvec(n, base, true);
            var nrm = 0.0;
            for (var i = 0u; i < n; i = i + 1u) {
                w[i] = lambda_max * v[i] - w[i];
                nrm = nrm + dot(w[i], w[i]);
            }
            nrm = sqrt(nrm);
            let s = select(0.0, 1.0 / nrm, nrm > 0.0);
            for (var i = 0u; i < n; i = i + 1u) {
                v[i] = w[i] * s;
            }
        }
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
}
