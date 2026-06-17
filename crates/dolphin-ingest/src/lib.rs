//! S3 read-staging for CSLC stacks (feature `s3`).
//!
//! dolphinRust *consumes* raw InSAR data already present in S3 (the host app
//! puts it there); it never writes raw data back. Because phase linking is a
//! sliding-window algorithm — each pixel is read many times across overlapping
//! covariance windows — and OPERA CSLC HDF5 is not cloud-optimized, blocks are
//! never streamed over `/vsis3/`. Instead each granule is downloaded **once**
//! to local scratch, processed locally, and removed.
//!
//! The download is the only async stage. It runs `object_store` on a bounded
//! `tokio` runtime behind a synchronous `stage(uris, scratch) -> Vec<PathBuf>`
//! facade, so the rest of the pipeline — and any host app calling it — stays
//! runtime-agnostic. The host app bridges the (sync, CPU-bound) compute run via
//! `spawn_blocking` / a dedicated thread; see PLAYBOOK.md §Architecture #6–7.
