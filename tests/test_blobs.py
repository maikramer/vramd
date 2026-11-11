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
        with pytest.raises(BlobError) as exc:
            writer.write(b"bc")
        assert exc.value.code == P.ERR_BLOB_SIZE
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

    def test_materialize_survives_gc_between_check_and_link(self, tmp_path: Path, monkeypatch) -> None:
        """O pin tem de existir antes do link: o GC não pode apagar a fonte pelo meio."""
        store = _store(tmp_path)
        sha = _put(store, b"race")
        real_link = os.link

        def link_after_gc(src, dst, **kwargs) -> None:
            store.gc(target_bytes=0)  # corrida: GC entre a validação e o hardlink
            real_link(src, dst, **kwargs)

        monkeypatch.setattr(os, "link", link_after_gc)
        target = tmp_path / "job" / "in" / "f.bin"
        assert store.materialize(sha, target) == 4
        assert target.read_bytes() == b"race"

    def test_upload_temp_is_private(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        writer = store.open_writer(_sha(b"segredo"), 7)
        try:
            if os.name != "nt":
                assert (writer._tmp.stat().st_mode & 0o777) == 0o600
        finally:
            writer.abort()

    @pytest.mark.skipif(os.name == "nt", reason="modos POSIX")
    def test_dedupe_restores_private_mode(self, tmp_path: Path) -> None:
        """Um blob antigo 0644 não fica legível só porque o upload foi deduplicado."""
        store = _store(tmp_path)
        data = b"x" * 10
        sha = _put(store, data)
        os.chmod(store.path(sha), 0o644)
        writer = store.open_writer(sha, len(data))
        writer.write(data)
        result = writer.finish()
        assert result["deduped"] is True
        assert (store.path(sha).stat().st_mode & 0o777) == 0o600

    def test_same_size_corruption_is_replaced(self, tmp_path: Path) -> None:
        """Dedupe só pelo tamanho aceitava um CAS corrompido com a mesma size."""
        store = _store(tmp_path)
        data = b"bom"
        sha = _put(store, data)
        store.path(sha).write_bytes(b"mal")  # mesmo tamanho, conteúdo errado
        writer = store.open_writer(sha, len(data))
        writer.write(data)
        result = writer.finish()
        assert result["deduped"] is False
        assert store.path(sha).read_bytes() == data

    @pytest.mark.skipif(os.name == "nt", reason="symlinks POSIX")
    def test_symlink_is_not_a_blob(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        data = b"conteudo"
        sha = _sha(data)
        outside = tmp_path / "outside.bin"
        outside.write_bytes(data)
        blob = store.path(sha)
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.symlink_to(outside)

        assert store.has(sha)["exists"] is False
        assert store.stats()["count"] == 0
        with pytest.raises(BlobError) as exc:
            store.stat(sha)
        assert exc.value.code == P.ERR_BLOB_UNKNOWN
        with pytest.raises(BlobError):
            store.materialize(sha, tmp_path / "job" / "f")
        store.gc(target_bytes=0)
        assert outside.read_bytes() == data  # o GC não segue o link

    @pytest.mark.skipif(os.name == "nt", reason="symlinks POSIX")
    def test_upload_replaces_symlink_instead_of_deduping(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        data = b"conteudo"
        sha = _put(store, data)
        outside = tmp_path / "outside.bin"
        outside.write_bytes(data)
        blob = store.path(sha)
        blob.unlink()
        blob.symlink_to(outside)

        writer = store.open_writer(sha, len(data))
        writer.write(data)
        result = writer.finish()
        assert result["deduped"] is False
        assert not blob.is_symlink()
        assert blob.read_bytes() == data
        assert outside.read_bytes() == data

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
        assert result["removed"] == 1
        assert not store.path(old).exists()
        assert store.path(new).exists()
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

    def test_sweep_removes_aborted_uploads(self, tmp_path: Path) -> None:
        """`.part` de um cliente/supervisor morto não fica para sempre no disco."""
        store = _store(tmp_path, ttl_sec=60)
        writer = store.open_writer("a" * 64, 10)
        writer.write(b"x" * 5)  # sem finish/abort
        part = writer._tmp
        writer._handle.close()
        past = time.time() - 3600
        os.utime(part, (past, past))
        assert store.sweep() == [part.name]
        assert not part.exists()

    def test_stray_files_are_not_blobs(self, tmp_path: Path) -> None:
        """Só `<2 hex>/<sha256>` conta; o GC não apaga ficheiros alheios."""
        store = _store(tmp_path)
        sha = _put(store, b"ok")
        stray = store.root / "lost+found"
        stray.mkdir()
        (stray / "precious.txt").write_text("não apagar", encoding="utf-8")
        assert store.stats()["count"] == 1
        store.gc(target_bytes=0)
        assert (stray / "precious.txt").is_file()
        assert not store.path(sha).exists()
