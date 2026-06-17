//! Phase-8 ingest contract test (feature `s3`).
//!
//! `stage_from_store` downloads objects concurrently to local scratch, in input
//! order, with byte-exact contents. Validated against an in-memory object store
//! (no real S3 needed). The `s3://` `stage` wrapper shares the same download
//! path (parse_url → store).

#![cfg(feature = "s3")]

use std::sync::Arc;

use dolphin_ingest::stage_from_store;
use object_store::memory::InMemory;
use object_store::{path::Path as ObjPath, ObjectStore, PutPayload};

#[test]
fn stages_objects_in_order_byte_exact() {
    let store = Arc::new(InMemory::new());
    let keys = ["a/g1.h5", "a/g2.h5", "a/g3.h5"];
    let payloads: Vec<Vec<u8>> = (0..keys.len()).map(|i| vec![i as u8; 16 + i]).collect();

    // Seed the in-memory store.
    let rt = tokio::runtime::Runtime::new().unwrap();
    let seeds: Vec<(String, Vec<u8>)> = keys
        .iter()
        .map(|k| k.to_string())
        .zip(payloads.clone())
        .collect();
    rt.block_on(seed(store.clone(), seeds));

    let scratch = std::env::temp_dir().join("dolphinrust_ingest_test");
    std::fs::create_dir_all(&scratch).unwrap();
    let key_vec: Vec<String> = keys.iter().map(|s| s.to_string()).collect();

    let local = stage_from_store(store.clone(), &key_vec, &scratch).unwrap();

    assert_eq!(local.len(), keys.len(), "one local path per key, in order");
    for (path, expected) in local.iter().zip(&payloads) {
        assert_eq!(
            &std::fs::read(path).unwrap(),
            expected,
            "staged bytes match source"
        );
    }
    // filenames preserve the object basename, in order.
    assert!(local[0]
        .file_name()
        .unwrap()
        .to_string_lossy()
        .starts_with("g1"));
    assert!(local[2]
        .file_name()
        .unwrap()
        .to_string_lossy()
        .starts_with("g3"));

    for p in &local {
        let _ = std::fs::remove_file(p);
    }
}

/// Put each `(key, bytes)` into the store.
async fn seed(store: Arc<InMemory>, items: Vec<(String, Vec<u8>)>) {
    for (key, bytes) in items {
        store
            .put(&ObjPath::from(key.as_str()), PutPayload::from(bytes))
            .await
            .unwrap();
    }
}
