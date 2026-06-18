//! Phase-5 planner contract tests.
//!
//! Primary (analytic): compressed-SLC carry-forward grows by one per ministack,
//! capped at `max_num_compressed`; the trailing batch is partial. Secondary
//! (oracle): the full plan (num_compressed, num_real, output/compressed
//! reference indices per ministack) matches dolphin v0.35.0 for several
//! (N, size, max_compressed, plan) combos. Oracle tests skip without fixtures.

use std::path::{Path, PathBuf};

use dolphin_core::config::CompressedSlcPlan;
use dolphin_stack::{MiniStack, MiniStackPlanner};
use ndarray::Array2;

fn planner(num_slc: usize, max_num_compressed: usize, plan: CompressedSlcPlan) -> MiniStackPlanner {
    MiniStackPlanner {
        num_slc,
        max_num_compressed,
        output_reference_idx: 0,
        compressed_slc_plan: plan,
    }
}

// ------------------------------- analytic (primary) ---------------------------

#[test]
fn carry_forward_grows_and_caps() {
    let p = planner(12, 2, CompressedSlcPlan::AlwaysFirst);
    let stacks = p.plan(5).unwrap();
    let compressed: Vec<usize> = stacks.iter().map(|m| m.num_compressed).collect();
    assert_eq!(
        compressed,
        vec![0, 1, 2],
        "carry-forward grows then caps at 2"
    );
    let reals: Vec<usize> = stacks.iter().map(|m| m.num_real).collect();
    assert_eq!(reals, vec![5, 5, 2], "trailing ministack is partial");
    assert_eq!(stacks[1].size(), 1 + 5, "size = compressed + real");
}

/// `plan_with_offset` resumes the carry-forward: planning the tail of a stack
/// with `batch_offset` already-sealed ministacks must reproduce exactly the
/// matching suffix of the full plan (the NRT incrementality invariant). Tail
/// `real_start` is relative; `num_compressed`/reference indices continue the
/// prior sequence.
#[test]
fn plan_with_offset_resumes_full_plan() {
    // Full 13-SLC plan, size 5 → ministacks at real starts [0,5,10].
    let full = planner(13, 10, CompressedSlcPlan::AlwaysFirst)
        .plan(5)
        .unwrap();
    // Two ministacks (10 real SLCs) already sealed; the tail is the last 3 SLCs.
    let tail = planner(3, 10, CompressedSlcPlan::AlwaysFirst)
        .plan_with_offset(5, 2)
        .unwrap();
    assert_eq!(tail.len(), 1, "tail has one (open) ministack");
    let t = tail[0];
    let f = full[2];
    assert_eq!(t.num_compressed, f.num_compressed, "carry continues to 2");
    assert_eq!(t.num_real, f.num_real, "3 real SLCs");
    assert_eq!(t.real_start, 0, "tail real_start is tail-relative");
    assert_eq!(
        (t.output_reference_idx, t.compressed_reference_idx),
        (f.output_reference_idx, f.compressed_reference_idx),
        "reference indices continue the prior sequence",
    );
}

#[test]
fn rejects_degenerate_size() {
    assert!(planner(10, 10, CompressedSlcPlan::AlwaysFirst)
        .plan(1)
        .is_err());
}

// ------------------------------- oracle (secondary) ---------------------------

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/fixtures")
}

/// (N, size, max_compressed, plan) — mirrors `oracle/gen_stack.py` COMBOS.
fn combos() -> Vec<(usize, usize, usize, &'static str, CompressedSlcPlan)> {
    vec![
        (10, 5, 10, "always_first", CompressedSlcPlan::AlwaysFirst),
        (12, 5, 2, "always_first", CompressedSlcPlan::AlwaysFirst),
        (7, 3, 10, "always_first", CompressedSlcPlan::AlwaysFirst),
        (
            10,
            4,
            10,
            "first_per_ministack",
            CompressedSlcPlan::FirstPerMinistack,
        ),
        (
            10,
            4,
            10,
            "last_per_ministack",
            CompressedSlcPlan::LastPerMinistack,
        ),
    ]
}

fn row(m: &MiniStack) -> [i64; 4] {
    [
        m.num_compressed as i64,
        m.num_real as i64,
        m.output_reference_idx as i64,
        m.compressed_reference_idx as i64,
    ]
}

#[test]
fn plans_match_oracle() {
    let dir = fixtures();
    for (n, size, maxc, name, plan) in combos() {
        let path = dir.join(format!("plan_{n}_{size}_{maxc}_{name}.npy"));
        if !path.exists() {
            eprintln!("skipping plan oracle ({name}): no fixtures");
            continue;
        }
        let oracle: Array2<i64> = ndarray_npy::read_npy(&path).unwrap();
        let stacks = planner(n, maxc, plan).plan(size).unwrap();
        assert_eq!(stacks.len(), oracle.nrows(), "ministack count for {name}");
        for (i, m) in stacks.iter().enumerate() {
            let want = [
                oracle[(i, 0)],
                oracle[(i, 1)],
                oracle[(i, 2)],
                oracle[(i, 3)],
            ];
            assert_eq!(row(m), want, "ministack {i} of {name}_{n}_{size}_{maxc}");
        }
    }
}
