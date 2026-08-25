#!/usr/bin/env python
"""Range-read a bounded OPERA HDF5 crop without downloading the product."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

import fsspec
import h5py
import requests

if __package__:
    from .crop_real import Window, validate_window
    from .fetch_real import authenticated_session, sha256_file
else:
    from crop_real import Window, validate_window
    from fetch_real import authenticated_session, sha256_file


CONTENT_RANGE = re.compile(r"bytes 0-0/(\d+)")
HASH = re.compile(r"[0-9a-f]{64}")
DATASETS = {
    "cslc": ("/data/VV",),
    "static": ("/data/los_east", "/data/los_north"),
}
GRID_DATASETS = ("/data/x_coordinates", "/data/y_coordinates", "/data/projection")


class RemoteCropError(RuntimeError):
    """A remote product could not be read within the frozen transport contract."""


@dataclass(frozen=True)
class HttpIdentity:
    content_length: int
    etag: str | None
    last_modified: str | None


class ByteBudget:
    def __init__(self, maximum: int) -> None:
        if maximum <= 0:
            raise RemoteCropError("transfer cap must be positive")
        self.maximum = maximum
        self.bytes_read = 0

    @property
    def remaining(self) -> int:
        return self.maximum - self.bytes_read

    def authorize(self, size: int) -> None:
        if size < 0 or size > self.remaining:
            raise RemoteCropError("transfer cap would be exceeded")

    def consume(self, size: int) -> None:
        self.authorize(size)
        self.bytes_read += size


class BudgetedReader:
    def __init__(self, source: Any, budget: ByteBudget) -> None:
        self.source = source
        self.budget = budget

    def read(self, size: int = -1) -> bytes:
        self.budget.authorize(size)
        value = self.source.read(size)
        self.budget.consume(len(value))
        return value

    def readinto(self, buffer: Any) -> int:
        size = len(buffer)
        self.budget.authorize(size)
        count = self.source.readinto(buffer)
        self.budget.consume(count)
        return count

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return self.source.seek(offset, whence)

    def tell(self) -> int:
        return self.source.tell()

    def flush(self) -> None:
        self.source.flush()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True


def conditional_headers(identity: HttpIdentity | None) -> dict[str, str]:
    headers = {"Accept-Encoding": "identity"}
    if identity is None:
        return headers
    if identity.etag and not identity.etag.startswith("W/"):
        headers["If-Match"] = identity.etag
    elif identity.last_modified:
        headers["If-Unmodified-Since"] = identity.last_modified
    return headers


def probe_identity(
    session: requests.Session,
    url: str,
    budget: ByteBudget,
    expected: HttpIdentity | None = None,
) -> HttpIdentity:
    headers = {**conditional_headers(expected), "Range": "bytes=0-0"}
    try:
        response = session.get(url, headers=headers, stream=True, timeout=60)
    except requests.RequestException as error:
        raise RemoteCropError("authenticated range probe failed") from error
    with response:
        if response.status_code == 412 and expected is not None:
            raise RemoteCropError("remote product identity changed during crop")
        if response.status_code != 206:
            raise RemoteCropError("server ignored the required HTTP range request")
        match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
        if match is None or response.headers.get("Content-Length") != "1":
            raise RemoteCropError("server returned an invalid HTTP range identity")
        etag = response.headers.get("ETag")
        last_modified = response.headers.get("Last-Modified")
        if etag is None and last_modified is None:
            raise RemoteCropError("remote product has no HTTP identity validator")
        budget.authorize(1)
        byte = response.raw.read(1)
        if len(byte) != 1:
            raise RemoteCropError("server returned an incomplete HTTP range")
        budget.consume(1)
    identity = HttpIdentity(int(match.group(1)), etag, last_modified)
    if expected is not None and identity != expected:
        raise RemoteCropError("remote product identity changed during crop")
    return identity


def safe_url_identity(url: str) -> tuple[str, str, str]:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise RemoteCropError("source URL must be HTTP(S)")
    netloc = parts.hostname
    if parts.port is not None:
        netloc += f":{parts.port}"
    public_url = urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    file_name = Path(unquote(parts.path)).name
    return public_url, hashlib.sha256(url.encode()).hexdigest(), file_name


def request_headers(session: requests.Session, identity: HttpIdentity) -> dict[str, str]:
    headers = {
        key: value
        for key, value in session.headers.items()
        if key.lower() not in {"range", "accept-encoding"}
    }
    headers.update(conditional_headers(identity))
    return headers


def raw_window_bytes(source: h5py.File, datasets: tuple[str, ...], window: Window) -> int:
    total = 0
    for path in datasets:
        dataset = source[path]
        if dataset.ndim != 2:
            raise RemoteCropError(f"declared raster dataset is not two-dimensional: {path}")
        total += window.height * window.width * dataset.dtype.itemsize
    x = source[GRID_DATASETS[0]]
    y = source[GRID_DATASETS[1]]
    projection = source[GRID_DATASETS[2]]
    if x.ndim != 1 or y.ndim != 1 or projection.shape != ():
        raise RemoteCropError("declared grid metadata has unsupported dimensions")
    return total + window.width * x.dtype.itemsize + window.height * y.dtype.itemsize + projection.dtype.itemsize


def copy_dataset_attributes(source: h5py.Dataset, destination: h5py.Dataset) -> None:
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def write_crop(source: h5py.File, destination: Path, product_type: str, window: Window) -> list[dict[str, Any]]:
    datasets = DATASETS.get(product_type)
    if datasets is None:
        raise RemoteCropError(f"unsupported product type: {product_type}")
    for path in (*datasets, *GRID_DATASETS):
        if path not in source:
            raise RemoteCropError(f"remote product is missing declared dataset: {path}")
    x = source[GRID_DATASETS[0]]
    y = source[GRID_DATASETS[1]]
    validate_window(window, (len(y), len(x)))
    for path in datasets:
        if source[path].shape != (len(y), len(x)):
            raise RemoteCropError(f"raster and grid shapes disagree: {path}")
    entries: list[dict[str, Any]] = []
    with h5py.File(destination, "w") as output:
        data = output.create_group("data")
        for path in datasets:
            source_dataset = source[path]
            values = source_dataset[window.row0 : window.row1, window.col0 : window.col1]
            output_dataset = data.create_dataset(Path(path).name, data=values)
            copy_dataset_attributes(source_dataset, output_dataset)
            entries.append(
                {
                    "path": path,
                    "source_shape": list(source_dataset.shape),
                    "output_shape": list(values.shape),
                    "dtype": str(source_dataset.dtype),
                }
            )
        for path, values in (
            (GRID_DATASETS[0], x[window.col0 : window.col1]),
            (GRID_DATASETS[1], y[window.row0 : window.row1]),
        ):
            source_dataset = source[path]
            output_dataset = data.create_dataset(Path(path).name, data=values)
            copy_dataset_attributes(source_dataset, output_dataset)
        projection = source[GRID_DATASETS[2]]
        output_projection = data.create_dataset("projection", data=projection[()])
        copy_dataset_attributes(projection, output_projection)
    return entries


def temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, value = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(value)


def crop_remote_hdf5(
    *,
    url: str,
    expected_file_name: str,
    destination: Path,
    receipt_path: Path,
    product_type: str,
    window: Window,
    source_catalog_sha256: str,
    session: requests.Session,
    max_transfer_bytes: int,
) -> dict[str, Any]:
    if destination.exists() or receipt_path.exists():
        raise RemoteCropError("destination or receipt already exists")
    if HASH.fullmatch(source_catalog_sha256) is None:
        raise RemoteCropError("source catalog hash is not a SHA-256")
    public_url, url_sha256, file_name = safe_url_identity(url)
    if file_name != expected_file_name:
        raise RemoteCropError("source URL file identity does not match the catalog")
    budget = ByteBudget(max_transfer_bytes)
    identity = probe_identity(session, url, budget)
    output_temporary = temporary_path(destination)
    receipt_temporary = temporary_path(receipt_path)
    filesystem = None
    try:
        headers = request_headers(session, identity)
        filesystem = fsspec.filesystem(
            "http",
            headers=headers,
            client_kwargs={"cookies": session.cookies.get_dict()},
        )
        with filesystem.open(
            url,
            "rb",
            block_size=64 * 1024,
            cache_type="none",
            size=identity.content_length,
        ) as remote:
            budgeted = BudgetedReader(remote, budget)
            try:
                with h5py.File(budgeted, "r") as source:
                    datasets = DATASETS.get(product_type)
                    if datasets is None:
                        raise RemoteCropError(f"unsupported product type: {product_type}")
                    for path in (*datasets, *GRID_DATASETS):
                        if path not in source:
                            raise RemoteCropError(f"remote product is missing declared dataset: {path}")
                    validate_window(
                        window,
                        (len(source[GRID_DATASETS[1]]), len(source[GRID_DATASETS[0]])),
                    )
                    declared_bytes = raw_window_bytes(source, datasets, window)
                    if declared_bytes > budget.remaining:
                        raise RemoteCropError("declared crop exceeds the transfer cap before data reads")
                    dataset_receipts = write_crop(source, output_temporary, product_type, window)
            except RemoteCropError:
                raise
            except (OSError, ValueError, KeyError) as error:
                raise RemoteCropError("remote HDF5 read failed") from error
        probe_identity(session, url, budget, identity)
        output_hash = sha256_file(output_temporary)
        receipt = {
            "schema": "dolphinrust.remote_hdf5_crop",
            "schema_version": 1,
            "source": {
                "url": public_url,
                "url_sha256": url_sha256,
                "file_name": expected_file_name,
                "catalog_sha256": source_catalog_sha256,
                "content_length": identity.content_length,
                "etag": identity.etag,
                "last_modified": identity.last_modified,
            },
            "product_type": product_type,
            "window": asdict(window),
            "datasets": dataset_receipts,
            "grid_datasets": list(GRID_DATASETS),
            "transfer": {
                "maximum_bytes": budget.maximum,
                "bytes_read": budget.bytes_read,
                "range_required": True,
                "cache": "none",
            },
            "output": {
                "path": str(destination),
                "bytes": output_temporary.stat().st_size,
                "sha256": output_hash,
            },
        }
        receipt_temporary.write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(output_temporary, destination)
        try:
            os.replace(receipt_temporary, receipt_path)
        except OSError:
            destination.unlink(missing_ok=True)
            raise
        return receipt
    except RemoteCropError:
        raise
    except Exception as error:
        raise RemoteCropError("remote crop failed without publishing output") from error
    finally:
        output_temporary.unlink(missing_ok=True)
        receipt_temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--file-name", required=True)
    parser.add_argument("--product-type", choices=sorted(DATASETS), required=True)
    parser.add_argument("--row0", type=int, required=True)
    parser.add_argument("--col0", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--source-catalog-sha256", required=True)
    parser.add_argument("--max-transfer-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = crop_remote_hdf5(
        url=args.url,
        expected_file_name=args.file_name,
        destination=args.output,
        receipt_path=args.receipt,
        product_type=args.product_type,
        window=Window(args.row0, args.col0, args.height, args.width),
        source_catalog_sha256=args.source_catalog_sha256,
        session=authenticated_session(),
        max_transfer_bytes=args.max_transfer_bytes,
    )
    print(
        json.dumps(
            {
                "output": receipt["output"]["path"],
                "output_sha256": receipt["output"]["sha256"],
                "bytes_read": receipt["transfer"]["bytes_read"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
