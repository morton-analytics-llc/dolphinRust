//! GPU phase-linking contract tests (feature `gpu`).
//!
//! Step 1 — the adapter gate: a GPU adapter must initialize on this machine, and
//! on macOS it must be the Metal backend. Later steps add covariance/EVD parity
//! tests against the CPU (f64) reference to f32 tolerance.
#![cfg(feature = "gpu")]

use dolphin_phaselink::gpu::{enumerate_adapters, GpuContext};

#[test]
fn metal_adapter_initializes() {
    let adapters = enumerate_adapters();
    assert!(
        !adapters.is_empty(),
        "no GPU adapter initialized — GPU path cannot run on this machine"
    );
    for a in &adapters {
        eprintln!("adapter: {a}");
    }

    let ctx = GpuContext::new().expect("acquire a GPU device");
    eprintln!("bound: {}", ctx.adapter);

    if cfg!(target_os = "macos") {
        assert_eq!(
            ctx.adapter.backend, "Metal",
            "expected the Metal backend on macOS, got {}",
            ctx.adapter.backend
        );
    }
}
