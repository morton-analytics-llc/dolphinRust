// GPU sliding-window coherence (port of covariance.rs, f32).
//
// One invocation per output pixel: build the nslc x nslc normalized coherence
// matrix C_ij = sum(z_i z_j*) / sqrt(sum|z_i|^2 * sum|z_j|^2) over the window,
// clamped inward at borders to match the CPU `window_origin`. Complex values are
// vec2<f32> = (re, im). When `has_mask`, the SHP neighbor array gates which window
// samples contribute (per output pixel, shape win_h x win_w), matching the CPU
// `neighbors` path. Non-finite samples are treated as zero.

const MAX_NSLC: u32 = 32u;
const AMP_FLOOR: f32 = 1e-6;

struct Params {
    nslc: u32,
    rows: u32,
    cols: u32,
    half_y: u32,
    half_x: u32,
    stride_y: u32,
    stride_x: u32,
    out_rows: u32,
    out_cols: u32,
    win_h: u32,
    win_w: u32,
    has_mask: u32,
};

@group(0) @binding(0) var<storage, read> stack: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> p: Params;
@group(0) @binding(2) var<storage, read_write> out: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> neighbors: array<u32>; // SHP mask (1=keep), or dummy

var<private> amp: array<f32, MAX_NSLC>;

// saturating_sub(a, b) for u32, then clamp so the window stays full-size.
fn window_origin(out_r: u32, out_c: u32) -> vec2<u32> {
    let in_r = p.stride_y / 2u + out_r * p.stride_y;
    let in_c = p.stride_x / 2u + out_c * p.stride_x;
    let sub_r = select(0u, in_r - p.half_y, in_r >= p.half_y);
    let sub_c = select(0u, in_c - p.half_x, in_c >= p.half_x);
    let r0 = min(sub_r, p.rows - p.win_h);
    let c0 = min(sub_c, p.cols - p.win_w);
    return vec2<u32>(r0, c0);
}

// SHP keep-factor for window position (wr, wc) of output pixel `pix`: 1.0 unless a
// mask is bound and this sample is not a statistical homogeneous neighbor.
fn keep(pix: u32, wr: u32, wc: u32) -> f32 {
    if (p.has_mask == 0u) { return 1.0; }
    let idx = (pix * p.win_h + wr) * p.win_w + wc;
    return f32(neighbors[idx]);
}

// Load stack[slc, r, c], mapping any non-finite component to zero.
fn load_sample(slc: u32, r: u32, c: u32) -> vec2<f32> {
    let z = stack[(slc * p.rows + r) * p.cols + c];
    let finite = z.x == z.x && z.y == z.y && abs(z.x) < 3.0e38 && abs(z.y) < 3.0e38;
    return select(vec2<f32>(0.0, 0.0), z, finite);
}

// sqrt(sum |z_slc|^2) over the (masked) window.
fn amp_of(pix: u32, slc: u32, origin: vec2<u32>) -> f32 {
    var acc = 0.0;
    for (var wr = 0u; wr < p.win_h; wr = wr + 1u) {
        for (var wc = 0u; wc < p.win_w; wc = wc + 1u) {
            let z = load_sample(slc, origin.x + wr, origin.y + wc);
            acc = acc + keep(pix, wr, wc) * (z.x * z.x + z.y * z.y);
        }
    }
    return sqrt(acc);
}

// sum(z_i z_j*) over the (masked) window.
fn numer_of(pix: u32, i: u32, j: u32, origin: vec2<u32>) -> vec2<f32> {
    var num = vec2<f32>(0.0, 0.0);
    for (var wr = 0u; wr < p.win_h; wr = wr + 1u) {
        for (var wc = 0u; wc < p.win_w; wc = wc + 1u) {
            let m = keep(pix, wr, wc);
            let zi = load_sample(i, origin.x + wr, origin.y + wc);
            let zj = load_sample(j, origin.x + wr, origin.y + wc);
            num.x = num.x + m * (zi.x * zj.x + zi.y * zj.y);
            num.y = num.y + m * (zi.y * zj.x - zi.x * zj.y);
        }
    }
    return num;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pix = gid.x;
    let n_pix = p.out_rows * p.out_cols;
    if (pix >= n_pix) { return; }

    let origin = window_origin(pix / p.out_cols, pix % p.out_cols);
    let n = p.nslc;
    for (var i = 0u; i < n; i = i + 1u) {
        amp[i] = amp_of(pix, i, origin);
    }

    let base = pix * n * n;
    for (var i = 0u; i < n; i = i + 1u) {
        for (var j = 0u; j < n; j = j + 1u) {
            let num = numer_of(pix, i, j, origin);
            let denom = amp[i] * amp[j];
            let coh = select(vec2<f32>(0.0, 0.0), num / denom, denom > AMP_FLOOR);
            out[base + i * n + j] = coh;
        }
    }
}
