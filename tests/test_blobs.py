"""CAS de blobs: dedupe, verificação, hardlink, pins e GC."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import pytest

from vramd import protocol as P
from vramd.blobs import BlobConfig, BlobError, BlobStore


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _store(tmp_path: Path, **kwargs) -> BlobStore:
    return BlobStore(BlobConfig(root=tmp_path / "blobs", **kwargs))


def _put(store: BlobStore, data: bytes) -> str:
    sha = _sha(data)
    writer = store.open_writer(sha, len(data))
    writer.write(data)
    writer.finish()
    return sha


class TestBlobStore:
    def test_put_has_get(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        data = b"glTF" + bytes(range(256)) * 1000
        sha = _put(store, data)
        assert store.has(sha)["exists"] is True
        assert store.has(sha, len(data))["exists"] is True
        assert store.has(sha, len(data) + 1)["exists"] is False
        with store.open_reader(sha) as handle:
            assert handle.read() == data

    def test_dedupe_skips_existing(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        data = b"x" * 1024
        sha = _put(store, data)
        before = store.path(sha).stat().st_mtime_ns
        writer = store.open_writer(sha, len(data))
        writer.write(data)
        result = writer.finish()
        assert result["deduped"] is True
        assert store.path(sha).stat().st_mtime_ns == before

    def test_hash_mismatch_is_rejected(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        writer = store.open_writer("0" * 64, 3)
        writer.write(b"abc")
        with pytest.raises(BlobError) as exc:
            writer.finish()
        assert exc.value.code == P.ERR_BLOB_HASH
        assert not store.path("0" * 64).exists()

    def test_size_mismatch_is_rejected(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        data = b"abc"
        writer = store.open_writer(_sha(data), 4)
        writer.write(data)
        with pytest.raises(BlobError) as exc:
            writer.finish()
        assert exc.value.code == P.ERR_BLOB_SIZE

    def test_upload_overflow_is_rejected(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        writer = store.open_writer(_sha(b"ab"), 2)
        writer.write(b"a")
        with pytest.raises(BlobError):
            writer.write(b"bc")
        writer.abort()

    def test_abort_removes_temp(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        writer = store.open_writer(_sha(b"abc"), 3)
        writer.write(b"ab")
        writer.abort()
        assert list((store.root / ".tmp").iterdir()) == []

    def test_materialize_hardlinks(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        data = b"mesh" * 5000
        sha = _put(store, data)
        target = tmp_path / "job" / "in" / "mesh.glb"
        size = store.materialize(sha, target)
        assert size == len(data)
        assert target.read_bytes() == data
        assert target.stat().st_ino == store.path(sha).stat().st_ino
        assert store.pinned(sha) is True

    def test_missing_blob(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(BlobError) as exc:
            store.stat(_sha(b"nope"))
        assert exc.value.code == P.ERR_BLOB_UNKNOWN

    def test_invalid_sha(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        for bad in ("", "../../etc/passwd", "abc"):
            with pytest.raises(BlobError):
                store.has(bad)

    def test_quota(self, tmp_path: Path) -> None:
        store = _store(tmp_path, max_blob_bytes=1024)
        with pytest.raises(BlobError) as exc:
            store.open_writer(_sha(b"x" * 2048), 2048)
        assert exc.value.code == P.ERR_BLOB_QUOTA

    def test_pin_blocks_delete(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        sha = _put(store, b"pinned")
        store.pin(sha)
        with pytest.raises(BlobError):
            store.delete(sha)
        assert store.path(sha).exists()
        store.unpin(sha)
        assert store.delete(sha)["removed"] is True

    def test_gc_keeps_pinned_and_drops_oldest(self, tmp_path: Path) -> None:
        store = _store(tmp_path, cache_bytes=1000)
        old = _put(store, b"a" * 100)
        new = _put(store, b"b" * 100)
        pinned = _put(store, b"c" * 100)
        store.pin(pinned)
        past = time.time() - 3600
        os.utime(store.path(old), (past, past))
        os.utime(store.path(new), (past + 10, past + 10))
        result = store.gc(target_bytes=250)
        assert result["removed"] >= 1
        assert not store.path(old).exists()
        assert store.path(pinned).exists()

    def test_stats(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        sha = _put(store, b"abc")
        store.pin(sha)
        stats = store.stats()
        assert stats["count"] == 1
        assert stats["bytes"] == 3
        assert stats["pinned"] == 1

    def test_sweep_ttl(self, tmp_path: Path) -> None:
        store = _store(tmp_path, ttl_sec=60)
        sha = _put(store, b"old")
        past = time.time() - 3600
        os.utime(store.path(sha), (past, past))
        assert store.sweep() == [sha]
        assert not store.path(sha).exists()
