"""Blobs content-addressed para transferências remotas.

O sandbox histórico escreve cada ficheiro no diretório do job. Isso reenvia o
mesmo conteúdo em cada job (a mesma imagem de referência em várias vistas, o
mesmo ``_to_paint`` num retry, etc.). Um CAS por SHA-256 resolve os dois
problemas: o cliente pergunta ``blob-has`` antes de enviar e o servidor
materializa o blob no sandbox por hardlink (sem copiar bytes).

Layout::

    <root>/<2 primeiros hex>/<sha256>

A escrita é sempre para ``<root>/.tmp/<sha>.<uuid>.part`` e só passa ao nome
final depois de verificar tamanho e SHA-256, com ``os.replace`` atómico.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from . import protocol as P

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_TMP_DIR = ".tmp"
_GIB = 1024**3
_MIB = 1024**2


class BlobError(Exception):
    """Erro de blob com código estável para a resposta JSON."""

    def __init__(self, message: str, *, code: str = P.ERR_BLOB_INVALID, hint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint


@dataclass(frozen=True)
class BlobConfig:
    root: Path
    max_blob_bytes: int = 4 * _GIB
    cache_bytes: int = 200 * _GIB
    ttl_sec: float = 30 * 24 * 3600.0


class BlobWriter:
    """Upload sequencial de um blob; verifica SHA antes de publicar."""

    def __init__(self, store: BlobStore, sha256: str, size: int) -> None:
        self.store = store
        self.sha256 = sha256
        self.size = size
        self.written = 0
        self._hash = hashlib.sha256()
        self._finished = False
        self._tmp = store.root / _TMP_DIR / f"{sha256}.{uuid.uuid4().hex}.part"
        self._handle = self._tmp.open("wb")

    def write(self, data: bytes) -> None:
        if self._finished:
            raise BlobError("escrita depois do finish")
        if self.written + len(data) > self.size:
            raise BlobError(
                f"upload excede o tamanho declarado ({self.size} bytes)",
                code=P.ERR_BLOB_SIZE,
            )
        self._handle.write(data)
        self._hash.update(data)
        self.written += len(data)

    def finish(self) -> dict[str, Any]:
        if self._finished:
            raise BlobError("finish repetido")
        self._finished = True
        try:
            self._handle.close()
            if self.written != self.size:
                raise BlobError(
                    f"upload incompleto: {self.written}/{self.size} bytes",
                    code=P.ERR_BLOB_SIZE,
                )
            actual = self._hash.hexdigest()
            if actual != self.sha256:
                raise BlobError(
                    "sha256 do blob não bate",
                    code=P.ERR_BLOB_HASH,
                    hint=f"calculado={actual}; declarado={self.sha256}",
                )
            final = self.store.path(self.sha256)
            final.parent.mkdir(parents=True, exist_ok=True)
            existing = final.is_file() and final.stat().st_size == self.size
            if existing:
                # Outro cliente publicou o mesmo conteúdo entretanto.
                self._tmp.unlink(missing_ok=True)
            else:
                os.replace(self._tmp, final)
                with contextlib.suppress(OSError):
                    final.chmod(0o600)
            self.store._touch(self.sha256)
            self.store.gc_if_needed()
            return {
                "status": P.STATUS_OK,
                "sha256": self.sha256,
                "size": self.size,
                "deduped": existing,
            }
        except BaseException:
            self._tmp.unlink(missing_ok=True)
            raise

    def abort(self) -> None:
        if not self._finished:
            self._finished = True
            with contextlib.suppress(OSError):
                self._handle.close()
            self._tmp.unlink(missing_ok=True)


class BlobStore:
    """CAS local com quotas, pins e GC LRU."""

    def __init__(self, config: BlobConfig) -> None:
        self.config = config
        self.root = config.root
        (self.root / _TMP_DIR).mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._pins: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Paths / validação
    # ------------------------------------------------------------------

    @staticmethod
    def validate_sha(sha256: str) -> str:
        text = str(sha256 or "").strip().lower()
        if not _SHA_RE.match(text):
            raise BlobError("sha256 inválido (esperado 64 hex)", code=P.ERR_BLOB_INVALID)
        return text

    def path(self, sha256: str) -> Path:
        sha = self.validate_sha(sha256)
        return self.root / sha[:2] / sha

    def _touch(self, sha256: str) -> None:
        path = self.path(sha256)
        with contextlib.suppress(OSError):
            stat = path.stat()
            now = time.time()
            os.utime(path, (now, stat.st_mtime))

    # ------------------------------------------------------------------
    # Pins (blobs referenciados por jobs/sessões activos)
    # ------------------------------------------------------------------

    def pin(self, sha256: str) -> None:
        sha = self.validate_sha(sha256)
        with self._lock:
            self._pins[sha] = self._pins.get(sha, 0) + 1

    def unpin(self, sha256: str) -> None:
        sha = self.validate_sha(sha256)
        with self._lock:
            current = self._pins.get(sha, 0)
            if current <= 1:
                self._pins.pop(sha, None)
            else:
                self._pins[sha] = current - 1

    def pinned(self, sha256: str) -> bool:
        with self._lock:
            return self._pins.get(sha256, 0) > 0

    # ------------------------------------------------------------------
    # Leitura / escrita
    # ------------------------------------------------------------------

    def has(self, sha256: str, size: int | None = None) -> dict[str, Any]:
        sha = self.validate_sha(sha256)
        path = self.path(sha)
        exists = path.is_file()
        actual = path.stat().st_size if exists else None
        if exists and size is not None and int(size) != actual:
            exists = False
        return {
            "status": P.STATUS_OK,
            "sha256": sha,
            "exists": exists,
            "size": actual,
        }

    def stat(self, sha256: str) -> dict[str, Any]:
        sha = self.validate_sha(sha256)
        path = self.path(sha)
        try:
            info = path.stat()
        except OSError as exc:
            raise BlobError(
                f"blob não existe: {sha[:12]}…",
                code=P.ERR_BLOB_UNKNOWN,
                hint="Faz upload com `blob-put` antes de o referenciar.",
            ) from exc
        if not path.is_file():
            raise BlobError(f"blob não é ficheiro: {sha[:12]}…", code=P.ERR_BLOB_UNKNOWN)
        return {"status": P.STATUS_OK, "sha256": sha, "size": info.st_size, "path": str(path)}

    def open_writer(self, sha256: str, size: int) -> BlobWriter:
        sha = self.validate_sha(sha256)
        size = int(size)
        if size < 0:
            raise BlobError("tamanho do blob negativo", code=P.ERR_BLOB_SIZE)
        if size > self.config.max_blob_bytes:
            raise BlobError(
                f"blob excede {self.config.max_blob_bytes // _MIB} MiB",
                code=P.ERR_BLOB_QUOTA,
            )
        self.assert_capacity(size)
        return BlobWriter(self, sha, size)

    def open_reader(self, sha256: str) -> BinaryIO:
        sha = self.validate_sha(sha256)
        path = self.path(sha)
        try:
            handle = path.open("rb")
        except OSError as exc:
            raise BlobError(
                f"blob não existe: {sha[:12]}…",
                code=P.ERR_BLOB_UNKNOWN,
            ) from exc
        self._touch(sha)
        return handle

    def materialize(self, sha256: str, target: Path, *, pin: bool = True) -> int:
        """Liga (hardlink) ou copia o blob para ``target``; devolve o tamanho."""
        sha = self.validate_sha(sha256)
        source = self.path(sha)
        if not source.is_file():
            raise BlobError(
                f"blob não existe: {sha[:12]}…",
                code=P.ERR_BLOB_UNKNOWN,
                hint="Faz upload com `blob-put` antes de o referenciar.",
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            target.unlink()
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
        if pin:
            self.pin(sha)
        self._touch(sha)
        return target.stat().st_size

    def delete(self, sha256: str, *, force: bool = False) -> dict[str, Any]:
        sha = self.validate_sha(sha256)
        if not force and self.pinned(sha):
            raise BlobError(
                "blob está em uso por um job/sessão activo",
                code=P.ERR_BLOB_INVALID,
                hint="Espera o job terminar ou usa force=true.",
            )
        path = self.path(sha)
        if not path.is_file():
            return {"status": P.STATUS_OK, "sha256": sha, "removed": False}
        size = path.stat().st_size
        path.unlink(missing_ok=True)
        self._remove_empty_shard(path.parent)
        return {"status": P.STATUS_OK, "sha256": sha, "removed": True, "freed_bytes": size}

    # ------------------------------------------------------------------
    # Quotas / GC
    # ------------------------------------------------------------------

    def assert_capacity(self, incoming: int = 0) -> None:
        """Recusa cedo se o cache está cheio e nada é evictável."""
        total = self.total_bytes()
        if total + incoming <= self.config.cache_bytes:
            return
        self.gc(target_bytes=max(0, self.config.cache_bytes - incoming))
        if self.total_bytes() + incoming > self.config.cache_bytes:
            raise BlobError(
                f"cache de blobs cheio ({self.config.cache_bytes // _GIB} GiB); "
                "aumenta VRAMD_BLOB_CACHE_GIB ou corre `vramd blobs gc`",
                code=P.ERR_BLOB_QUOTA,
            )

    def total_bytes(self) -> int:
        return sum(path.stat().st_size for path in self._iter_blobs())

    def stats(self) -> dict[str, Any]:
        files = list(self._iter_blobs())
        total = 0
        for path in files:
            with contextlib.suppress(OSError):
                total += path.stat().st_size
        with self._lock:
            pinned = sum(1 for sha, count in self._pins.items() if count > 0)
        return {
            "status": P.STATUS_OK,
            "root": str(self.root),
            "count": len(files),
            "bytes": total,
            "pinned": pinned,
            "max_blob_bytes": self.config.max_blob_bytes,
            "cache_bytes": self.config.cache_bytes,
        }

    def gc(self, *, target_bytes: int | None = None) -> dict[str, Any]:
        """Apaga blobs não pinados por ordem de acesso até caber no alvo."""
        target = self.config.cache_bytes if target_bytes is None else max(0, target_bytes)
        removed = 0
        freed = 0
        with self._lock:
            files: list[tuple[float, Path]] = []
            for path in self._iter_blobs():
                try:
                    info = path.stat()
                except OSError:
                    continue
                stamp = info.st_atime or info.st_mtime
                files.append((stamp, path))
            total = sum(path.stat().st_size for _stamp, path in files)
            if total <= target:
                return {"status": P.STATUS_OK, "removed": 0, "freed_bytes": 0, "bytes": total}
            for _stamp, path in sorted(files, key=lambda item: item[0]):
                if total <= target:
                    break
                if self._pins.get(path.name, 0) > 0:
                    continue
                try:
                    size = path.stat().st_size
                    path.unlink()
                except OSError:
                    continue
                total -= size
                freed += size
                removed += 1
                self._remove_empty_shard(path.parent)
        return {"status": P.STATUS_OK, "removed": removed, "freed_bytes": freed, "bytes": total}

    def gc_if_needed(self) -> None:
        with contextlib.suppress(BlobError, OSError):
            if self.total_bytes() > self.config.cache_bytes:
                self.gc()

    def sweep(self, *, now: float | None = None) -> list[str]:
        """Apaga blobs não pinados mais velhos que o TTL."""
        moment = time.time() if now is None else now
        removed: list[str] = []
        with self._lock:
            for path in self._iter_blobs():
                if self._pins.get(path.name, 0) > 0:
                    continue
                try:
                    info = path.stat()
                except OSError:
                    continue
                if moment - max(info.st_atime, info.st_mtime) > self.config.ttl_sec:
                    with contextlib.suppress(OSError):
                        path.unlink()
                        removed.append(path.name)
                        self._remove_empty_shard(path.parent)
        return removed

    def _iter_blobs(self):
        if not self.root.is_dir():
            return
        for shard in self.root.iterdir():
            if not shard.is_dir() or shard.name == _TMP_DIR:
                continue
            for path in shard.iterdir():
                if path.is_file() and not path.name.endswith(".part"):
                    yield path

    @staticmethod
    def _remove_empty_shard(path: Path) -> None:
        with contextlib.suppress(OSError):
            path.rmdir()
