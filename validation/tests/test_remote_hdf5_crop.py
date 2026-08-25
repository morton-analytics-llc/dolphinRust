from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import h5py
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crop_real import Window
from remote_hdf5_crop import RemoteCropError, crop_remote_hdf5


TOKEN = "local-secret-token"
CATALOG_SHA256 = "a" * 64


def source_product(path: Path) -> None:
    with h5py.File(path, "w") as product:
        data = product.create_group("data")
        data.create_dataset(
            "VV",
            data=(np.arange(80).reshape(8, 10) + 1j).astype(np.complex64),
            chunks=(4, 5),
        )
        data.create_dataset(
            "los_east",
            data=np.arange(80, dtype=np.float32).reshape(8, 10),
            chunks=(4, 5),
        )
        data.create_dataset(
            "los_north",
            data=np.arange(80, dtype=np.float32).reshape(8, 10) * -1,
            chunks=(4, 5),
        )
        data.create_dataset("x_coordinates", data=np.arange(10, dtype=np.float64))
        data.create_dataset("y_coordinates", data=np.arange(8, dtype=np.float64))
        data.create_dataset("projection", data=np.int32(32611))


@contextmanager
def range_server(path: Path, *, ignore_range: bool = False, mutate_identity: bool = False):
    content = path.read_bytes()
    state = {"probe_count": 0, "transferred": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("ETag", '"source-v1"')
            self.send_header("Last-Modified", "Tue, 25 Aug 2026 00:00:00 GMT")
            self.end_headers()

        def do_GET(self):
            if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                self.send_response(401)
                self.end_headers()
                return
            range_header = self.headers.get("Range")
            if ignore_range or range_header is None:
                self.send_response(200)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("ETag", '"source-v1"')
                self.end_headers()
                return
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", range_header)
            if match is None:
                self.send_response(416)
                self.end_headers()
                return
            start, end = map(int, match.groups())
            end = min(end, len(content) - 1)
            if start > end:
                self.send_response(416)
                self.end_headers()
                return
            exact_probe = start == 0 and end == 0
            if exact_probe:
                state["probe_count"] += 1
            etag = '"source-v2"' if mutate_identity and state["probe_count"] > 1 else '"source-v1"'
            if_match = self.headers.get("If-Match")
            if if_match is not None and if_match != etag:
                self.send_response(412)
                self.end_headers()
                return
            body = content[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
            self.send_header("ETag", etag)
            self.send_header("Last-Modified", "Tue, 25 Aug 2026 00:00:00 GMT")
            self.end_headers()
            self.wfile.write(body)
            state["transferred"] += len(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/{path.name}", state
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class RemoteHdf5CropContract(unittest.TestCase):
    def session(self) -> requests.Session:
        session = requests.Session()
        session.headers["Authorization"] = f"Bearer {TOKEN}"
        return session

    def test_range_crop_writes_exact_window_and_redacted_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.h5"
            output = root / "crop.h5"
            receipt = root / "crop.receipt.json"
            source_product(source)
            with range_server(source) as (url, state):
                result = crop_remote_hdf5(
                    url=url,
                    expected_file_name=source.name,
                    destination=output,
                    receipt_path=receipt,
                    product_type="cslc",
                    window=Window(2, 3, 3, 4),
                    source_catalog_sha256=CATALOG_SHA256,
                    session=self.session(),
                    max_transfer_bytes=1_000_000,
                )
            with h5py.File(output, "r") as product:
                np.testing.assert_array_equal(
                    product["/data/VV"][:],
                    (np.arange(80).reshape(8, 10) + 1j).astype(np.complex64)[2:5, 3:7],
                )
                np.testing.assert_array_equal(product["/data/x_coordinates"][:], np.arange(10)[3:7])
                np.testing.assert_array_equal(product["/data/y_coordinates"][:], np.arange(8)[2:5])
                self.assertEqual(int(product["/data/projection"][()]), 32611)
            payload = json.loads(receipt.read_text())
            self.assertEqual(payload, result)
            self.assertEqual(payload["source"]["catalog_sha256"], CATALOG_SHA256)
            self.assertEqual(payload["source"]["content_length"], source.stat().st_size)
            self.assertEqual(payload["window"], {"row0": 2, "col0": 3, "height": 3, "width": 4})
            self.assertEqual(payload["output"]["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertLessEqual(state["transferred"], 1_000_000)
            self.assertEqual(state["transferred"], payload["transfer"]["bytes_read"])
            self.assertLessEqual(payload["transfer"]["bytes_read"], 1_000_000)
            self.assertNotIn(TOKEN, receipt.read_text())

    def test_ignored_range_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.h5"
            source_product(source)
            output = root / "crop.h5"
            receipt = root / "crop.receipt.json"
            with range_server(source, ignore_range=True) as (url, _):
                with self.assertRaisesRegex(RemoteCropError, "range"):
                    crop_remote_hdf5(
                        url=url,
                        expected_file_name=source.name,
                        destination=output,
                        receipt_path=receipt,
                        product_type="cslc",
                        window=Window(0, 0, 1, 1),
                        source_catalog_sha256=CATALOG_SHA256,
                        session=self.session(),
                        max_transfer_bytes=1_000_000,
                    )
            self.assertFalse(output.exists())
            self.assertFalse(receipt.exists())

    def test_static_crop_reads_only_declared_los_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "static.h5"
            output = root / "static-crop.h5"
            receipt = root / "static-crop.receipt.json"
            source_product(source)
            with range_server(source) as (url, _):
                payload = crop_remote_hdf5(
                    url=url,
                    expected_file_name=source.name,
                    destination=output,
                    receipt_path=receipt,
                    product_type="static",
                    window=Window(1, 2, 2, 3),
                    source_catalog_sha256=CATALOG_SHA256,
                    session=self.session(),
                    max_transfer_bytes=1_000_000,
                )
            with h5py.File(output, "r") as product:
                np.testing.assert_array_equal(
                    product["/data/los_east"][:],
                    np.arange(80, dtype=np.float32).reshape(8, 10)[1:3, 2:5],
                )
                np.testing.assert_array_equal(
                    product["/data/los_north"][:],
                    (np.arange(80, dtype=np.float32).reshape(8, 10) * -1)[1:3, 2:5],
                )
                self.assertNotIn("VV", product["/data"])
            self.assertEqual(
                [entry["path"] for entry in payload["datasets"]],
                ["/data/los_east", "/data/los_north"],
            )

    def test_transfer_cap_and_changed_identity_leave_no_partial_output(self) -> None:
        for kwargs, message in [({"max_transfer_bytes": 32}, "cap"), ({"mutate_identity": True}, "identity")]:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.h5"
                source_product(source)
                output = root / "crop.h5"
                receipt = root / "crop.receipt.json"
                server_kwargs = {key: value for key, value in kwargs.items() if key == "mutate_identity"}
                cap = kwargs.get("max_transfer_bytes", 1_000_000)
                with range_server(source, **server_kwargs) as (url, _):
                    with self.assertRaisesRegex(RemoteCropError, message):
                        crop_remote_hdf5(
                            url=url,
                            expected_file_name=source.name,
                            destination=output,
                            receipt_path=receipt,
                            product_type="static",
                            window=Window(0, 0, 2, 2),
                            source_catalog_sha256=CATALOG_SHA256,
                            session=self.session(),
                            max_transfer_bytes=cap,
                        )
                self.assertFalse(output.exists())
                self.assertFalse(receipt.exists())
                self.assertEqual(list(root.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
