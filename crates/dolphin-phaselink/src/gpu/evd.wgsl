// GPU EVD phase linking (port of the EVD branch of estimator.rs, f32).
//
// One invocation per output pixel: recover the dominant eigenvector of
// M = C ⊙ |C| (Hermitian) by power iteration, then reference its phase to
// `ref_idx`: θ ← θ · exp(-j·∠θ[ref]). Complex = vec2<f32>. The matrix is
// rebuilt entry-by-entry inside the matvec (M_ij = C_ij·|C_ij|), so only the
// length-nslc iterate vectors live in registers.

const MAX_NSLC: u32 = 16u;

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

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pix = gid.x;
    if (pix >= p.n_pix) { return; }
    let n = p.nslc;
    let base = pix * n * n;

    var v: array<vec2<f32>, MAX_NSLC>;
    for (var i = 0u; i < n; i = i + 1u) {
        v[i] = vec2<f32>(1.0, 0.0);
    }

    var lambda = 0.0;
    for (var it = 0u; it < p.iters; it = it + 1u) {
        var w: array<vec2<f32>, MAX_NSLC>;
        var nrm = 0.0;
        for (var i = 0u; i < n; i = i + 1u) {
            var acc = vec2<f32>(0.0, 0.0);
            for (var j = 0u; j < n; j = j + 1u) {
                let c = cmat[base + i * n + j];
                acc = acc + length(c) * cmul(c, v[j]);
            }
            w[i] = acc;
            nrm = nrm + dot(acc, acc);
        }
        nrm = sqrt(nrm);
        lambda = nrm;
        let inv = select(0.0, 1.0 / nrm, nrm > 0.0);
        for (var i = 0u; i < n; i = i + 1u) {
            v[i] = w[i] * inv;
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
    eig_out[pix] = lambda;
}
