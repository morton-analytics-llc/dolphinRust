//! Concurrent S3→local staging behind a synchronous facade.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::stream::{self, StreamExt, TryStreamExt};
use object_store::{path::Path as ObjPath, ObjectStore};
use tokio::runtime::Builder;

/// Max concurrent downloads / runtime worker threads.
const MAX_CONCURRENT: usize = 8;

/// Errors raised while staging granules.
#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    #[error("object_store: {0}")]
    Store(#[from] object_store::Error),
    #[error("url parse: {0}")]
    Url(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience alias for fallible staging operations.
pub type Result<T> = std::result::Result<T, IngestError>;

/// Download `uris` (e.g. `s3://bucket/key`) to `scratch`, returning the local
/// paths in input order. Synchronous facade over a bounded `tokio` runtime.
///
/// # Errors
/// Returns `Err` on URL parse, object-store, or filesystem failure.
pub fn stage(uris: &[String], scratch: &Path) -> Result<Vec<PathBuf>> {
    runtime()?.block_on(download_all(uris, scratch, |uri: &String, dir: &Path| {
        fetch_uri(uri, dir)
    }))
}

/// Download `keys` from a single `store` to `scratch`, in input order. Used for
/// the standalone-CLI path and for testing against an in-memory store.
///
/// # Errors
/// Returns `Err` on object-store or filesystem failure.
pub fn stage_from_store(
    store: Arc<dyn ObjectStore>,
    keys: &[String],
    scratch: &Path,
) -> Result<Vec<PathBuf>> {
    let fetch = move |key: &String, scratch: &Path| {
        fetch_key(store.clone(), key.clone(), scratch.to_owned())
    };
    runtime()?.block_on(download_all(keys, scratch, fetch))
}

/// A bounded multi-thread runtime for the (only) async stage.
fn runtime() -> Result<tokio::runtime::Runtime> {
    Ok(Builder::new_multi_thread()
        .worker_threads(MAX_CONCURRENT)
        .enable_all()
        .build()?)
}

/// Run `fetch` over every item concurrently, preserving input order.
async fn download_all<I, F, Fut>(items: &[I], scratch: &Path, fetch: F) -> Result<Vec<PathBuf>>
where
    F: Fn(&I, &Path) -> Fut,
    Fut: std::future::Future<Output = Result<PathBuf>>,
{
    stream::iter(items.iter().map(|item| fetch(item, scratch)))
        .buffered(MAX_CONCURRENT)
        .try_collect()
        .await
}

/// Resolve an `s3://`-style URI to a store and fetch it.
fn fetch_uri(uri: &str, scratch: &Path) -> impl std::future::Future<Output = Result<PathBuf>> {
    let (uri, scratch) = (uri.to_owned(), scratch.to_owned());
    async move {
        let url = url::Url::parse(&uri).map_err(|e| IngestError::Url(e.to_string()))?;
        let (store, path) = object_store::parse_url(&url).map_err(IngestError::from)?;
        write_local(&store, &path, &scratch).await
    }
}

/// Fetch a single key from `store`.
async fn fetch_key(store: Arc<dyn ObjectStore>, key: String, scratch: PathBuf) -> Result<PathBuf> {
    let path = ObjPath::from(key.as_str());
    write_local(store.as_ref(), &path, &scratch).await
}

/// Download one object to `scratch/<filename>` and return the local path.
async fn write_local(store: &dyn ObjectStore, path: &ObjPath, scratch: &Path) -> Result<PathBuf> {
    let bytes = store.get(path).await?.bytes().await?;
    let dest = scratch.join(path.filename().unwrap_or("granule"));
    tokio::fs::write(&dest, &bytes).await?;
    Ok(dest)
}
