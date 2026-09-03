//! Contract: `estimator.rs`'s `PhaseAngleLinearization` prepare/apply path carries no
//! `unwrap`/`expect`/`panic!` on the happy path (CLAUDE.md, "Result-based errors in
//! library code"). Issue #98: `apply` reached for `gamma_inverse` with an `expect` that
//! held only by convention — nothing in the type system stopped an `Emi` branch from
//! being prepared without an inverse. The fix folds the inverse into the branch variant
//! that owns it so that state is unrepresentable, which is exactly what this source-scan
//! proves: it fails red while the `expect` exists and passes green once it is gone.

const ESTIMATOR_SOURCE: &str = include_str!("../src/estimator.rs");

#[test]
fn estimator_source_has_no_unwrap_expect_or_panic() {
    let forbidden = [".expect(", ".unwrap()", "panic!("];
    for marker in forbidden {
        assert!(
            !ESTIMATOR_SOURCE.contains(marker),
            "estimator.rs contains {marker:?} — library code must propagate with `?` and \
             crate error enums instead of panicking on the happy path"
        );
    }
}
