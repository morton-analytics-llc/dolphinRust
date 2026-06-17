// GPU sliding-window coherence (port of covariance.rs, f32, maskless).
//
// One invocation per output pixel: build the nslc x nslc normalized coherence
// matrix C_ij = sum(z_i z_j*) / sqrt(sum|z_i|^2 * sum|z_j|^2) over the window,
// clamped inward at borders to match the CPU `window_origin`. Complex values are
// vec2<f32> = (re, im). SHP masking is not modeled here (the spike compares the
// rectangular-window path); non-finite samples are treated as zero.

const MAX_NSLC: u32 = 16u;
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
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> stack: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> p: Params;
@group(0) @binding(2) var<storage, read_write> out: array<vec2<f32>>;

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

// Load stack[slc, r, c], mapping any non-finite component to zero.
fn load_sample(slc: u32, r: u32, c: u32) -> vec2<f32> {
    let z = stack[(slc * p.rows + r) * p.cols + c];
    let finite = z.x == z.x && z.y == z.y && abs(z.x) < 3.0e38 && abs(z.y) < 3.0e38;
    return select(vec2<f32>(0.0, 0.0), z, finite);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pix = gid.x;
    let n_pix = p.out_rows * p.out_cols;
    if (pix >= n_pix) { return; }

    let origin = window_origin(pix / p.out_cols, pix % p.out_cols);
    let n = p.nslc;

    // Per-SLC amplitude sqrt(sum |z|^2) over the window.
    var amp: array<f32, MAX_NSLC>;
    for (var i = 0u; i < n; i = i + 1u) {
        var acc = 0.0;
        for (var wr = 0u; wr < p.win_h; wr = wr + 1u) {
            for (var wc = 0u; wc < p.win_w; wc = wc + 1u) {
                let z = load_sample(i, origin.x + wr, origin.y + wc);
                acc = acc + z.x * z.x + z.y * z.y;
            }
        }
        amp[i] = sqrt(acc);
    }

    // Numerator sum(z_i z_j*) per pair, then normalize.
    let base = pix * n * n;
    for (var i = 0u; i < n; i = i + 1u) {
        for (var j = 0u; j < n; j = j + 1u) {
            var num = vec2<f32>(0.0, 0.0);
            for (var wr = 0u; wr < p.win_h; wr = wr + 1u) {
                for (var wc = 0u; wc < p.win_w; wc = wc + 1u) {
                    let zi = load_sample(i, origin.x + wr, origin.y + wc);
                    let zj = load_sample(j, origin.x + wr, origin.y + wc);
                    num.x = num.x + zi.x * zj.x + zi.y * zj.y;
                    num.y = num.y + zi.y * zj.x - zi.x * zj.y;
                }
            }
            let denom = amp[i] * amp[j];
            let coh = select(vec2<f32>(0.0, 0.0), num / denom, denom > AMP_FLOOR);
            out[base + i * n + j] = coh;
        }
    }
}
