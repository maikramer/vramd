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
from stat import S_ISREG
from typing import Any, BinaryIO

from . import protocol as P

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_SHARD_RE = re.compile(r"^[0-9a-f]{2}$")
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
        # 0600 desde a criação: o conteúdo de um upload não pode ficar legível
        # por outros utilizadores locais enquanto está em ``.tmp``.
        fd = os.open(
            self._tmp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        self._handle = os.fdopen(fd, "wb")

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
            # ``lstat``: um symlink plantado no CAS com o tamanho certo não é
            # o blob (nem pode ser aceite como dedupe). Nesse caso o
            # ``os.replace`` troca-o pelo ficheiro verificado.
            existing = False
            try:
                info = os.lstat(final)
                existing = S_ISREG(info.st_mode) and info.st_size == self.size
            except OSError:
                existing = False
            if existing:
                # Tamanho igual não é prova de conteúdo igual: confirma o hash
                # do blob existente antes de o aceitar como dedupe. Um CAS
                # corrompido/injectado com o mesmo tamanho é substituído pelo
                # ficheiro que acabámos de verificar.
                try:
                    existing = _file_sha256(final) == self.sha256
                except OSError:
                    existing = False
            if existing:
                # Outro cliente publicou o mesmo conteúdo entretanto. O blob
                # pode ter sido criado por uma versão antiga com 0644; o
                # conteúdo é o mesmo, mas a privacidade tem de ser reposta.
                self._tmp.unlink(missing_ok=True)
                with contextlib.suppress(OSError):
                    final.chmod(0o600)
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
            info = os.lstat(path)
            if not S_ISREG(info.st_mode):
                return
            now = time.time()
            os.utime(path, (now, info.st_mtime))

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

    def _blob_stat(self, sha256: str) -> tuple[Path, os.stat_result] | None:
        """Path + ``lstat`` de um blob regular; ``None`` se faltar/não for ficheiro.

        ``lstat`` e não ``stat``: um symlink plantado em ``<root>/<shard>/<sha>``
        passava por ``is_file()`` e o CAS lia/escrevia no destino do link. Só
        ficheiros regulares são blobs.
        """
        path = self.path(sha256)
        try:
            info = os.lstat(path)
        except OSError:
            return None
        if not S_ISREG(info.st_mode):
            return None
        return path, info

    def has(self, sha256: str, size: int | None = None) -> dict[str, Any]:
        sha = self.validate_sha(sha256)
        validated = self._blob_stat(sha)
        actual = validated[1].st_size if validated is not None else None
        exists = validated is not None
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
        validated = self._blob_stat(sha)
        if validated is None:
            raise BlobError(
                f"blob não existe: {sha[:12]}…",
                code=P.ERR_BLOB_UNKNOWN,
                hint="Faz upload com `blob-put` antes de o referenciar.",
            )
        path, info = validated
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
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise BlobError(
                f"blob não existe: {sha[:12]}…",
                code=P.ERR_BLOB_UNKNOWN,
            ) from exc
        try:
            info = os.fstat(fd)
        except OSError as exc:
            os.close(fd)
            raise BlobError(f"blob não é ficheiro: {sha[:12]}…", code=P.ERR_BLOB_UNKNOWN) from exc
        if not S_ISREG(info.st_mode):
            os.close(fd)
            raise BlobError(f"blob não é ficheiro: {sha[:12]}…", code=P.ERR_BLOB_UNKNOWN)
        self._touch(sha)
        return os.fdopen(fd, "rb")

    def materialize(self, sha256: str, target: Path, *, pin: bool = True) -> int:
        """Liga (hardlink) ou copia o blob para ``target``; devolve o tamanho."""
        sha = self.validate_sha(sha256)
        # O pin vem ANTES de validar/ligar: o GC só apaga blobs não pinados,
        # e sem isto podia removê-lo entre o ``lstat`` e o ``os.link`` (link e
        # fallback de cópia falhavam com BLOB_UNKNOWN/ENOENT numa corrida).
        if pin:
            self.pin(sha)
        try:
            validated = self._blob_stat(sha)
            if validated is None:
                raise BlobError(
                    f"blob não existe: {sha[:12]}…",
                    code=P.ERR_BLOB_UNKNOWN,
                    hint="Faz upload com `blob-put` antes de o referenciar.",
                )
            source = validated[0]
            target.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(FileNotFoundError):
                target.unlink()
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
            self._touch(sha)
            return target.stat().st_size
        except BaseException:
            if pin:
                self.unpin(sha)
            raise

    def delete(self, sha256: str, *, force: bool = False) -> dict[str, Any]:
        sha = self.validate_sha(sha256)
        # Pin/delete sob o mesmo lock do GC: verificar ``pinned`` e só depois
        # apagar deixava uma janela em que outro job pinava o blob já condenado.
        with self._lock:
            if not force and self._pins.get(sha, 0) > 0:
                raise BlobError(
                    "blob está em uso por um job/sessão activo",
                    code=P.ERR_BLOB_INVALID,
                    hint="Espera o job terminar ou usa force=true.",
                )
            validated = self._blob_stat(sha)
            if validated is None:
                return {"status": P.STATUS_OK, "sha256": sha, "removed": False}
            path, info = validated
            try:
                path.unlink()
            except FileNotFoundError:
                return {"status": P.STATUS_OK, "sha256": sha, "removed": False}
            self._remove_empty_shard(path.parent)
            return {"status": P.STATUS_OK, "sha256": sha, "removed": True, "freed_bytes": info.st_size}

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
        total = 0
        for path in self._iter_blobs():
            with contextlib.suppress(OSError):
                total += path.stat().st_size
        return total

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
            total = 0
            for _stamp, path in files:
                with contextlib.suppress(OSError):
                    total += path.stat().st_size
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
        """Apaga blobs não pinados e uploads abortados mais velhos que o TTL."""
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
            # Uploads cujo cliente/supervisor morreu deixam `.part` para trás;
            # não são blobs e nunca seriam recolhidos pelo GC.
            tmp_dir = self.root / _TMP_DIR
            if tmp_dir.is_dir():
                for path in tmp_dir.iterdir():
                    try:
                        info = path.stat()
                    except OSError:
                        continue
                    if not path.is_file() or moment - info.st_mtime <= self.config.ttl_sec:
                        continue
                    with contextlib.suppress(OSError):
                        path.unlink()
                        removed.append(path.name)
        return removed

    def _iter_blobs(self):
        """Só pares ``<2 hex>/<sha256>`` regulares; o resto (``lost+found``) não é nosso."""
        if not self.root.is_dir():
            return
        for shard in self.root.iterdir():
            if not shard.is_dir() or shard.name == _TMP_DIR or not _SHARD_RE.match(shard.name):
                continue
            for path in shard.iterdir():
                if not _SHA_RE.match(path.name) or not path.name.startswith(shard.name):
                    continue
                try:
                    if not S_ISREG(os.lstat(path).st_mode):
                        continue
                except OSError:
                    continue
                yield path

    @staticmethod
    def _remove_empty_shard(path: Path) -> None:
        with contextlib.suppress(OSError):
            path.rmdir()


def _file_sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    """SHA-256 de um ficheiro (verificação de dedupe no CAS)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()
