"""Sandbox de transferência do modo remoto (``stage``/``put``/``get``/``ls``/``clean``).

Um job remoto tem um diretório próprio (``<root>/<job>/{in,out}``) que é o
**único** sítio onde o cliente pode escrever e ler. O supervisor não interpreta
o payload do job (isso é do cliente, que conhece os campos de cada backend):
aqui só se guardam e devolvem bytes, com validação canónica de paths.
"""

from __future__ import annotations

import base64
import binascii
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
from typing import Any

from . import host
from . import protocol as P

_JOB_RE = re.compile(r"^[0-9a-f]{12,32}$")
_JANITOR_INTERVAL_SEC = 600.0


class RemoteError(Exception):
    """Erro de sandbox com código estável para a resposta JSON."""

    def __init__(self, message: str, *, code: str = P.ERR_INVALID_REQUEST, hint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint


@dataclass(frozen=True)
class RemoteConfig:
    root: Path
    max_file_bytes: int = 512 * 1024 * 1024
    max_job_bytes: int = 4 * 1024 * 1024 * 1024
    ttl_sec: float = 24 * 3600

    @classmethod
    def from_env(cls, cache_dir: Path | None = None) -> RemoteConfig:
        base = cache_dir if cache_dir is not None else host.cache_dir()

        def _int(name: str, default: int) -> int:
            raw = os.environ.get(name, "").strip()
            try:
                return int(raw) if raw else default
            except ValueError:
                return default

        root_env = os.environ.get("VRAMD_REMOTE_ROOT", "").strip()
        root = Path(os.path.expanduser(root_env)) if root_env else base / "remote"
        return cls(
            root=root,
            max_file_bytes=max(1, _int("VRAMD_REMOTE_MAX_FILE_MIB", 512)) * 1024 * 1024,
            max_job_bytes=max(1, _int("VRAMD_REMOTE_MAX_JOB_MIB", 4096)) * 1024 * 1024,
            ttl_sec=max(60, _int("VRAMD_REMOTE_TTL_HOURS", 24)) * 3600.0,
        )


class RemoteSandbox:
    """Operações de ficheiros confinadas a ``<root>/<job>``."""

    def __init__(self, config: RemoteConfig) -> None:
        self.config = config
        self.root = config.root
        self.root.mkdir(parents=True, exist_ok=True)
        self._janitor: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # Comandos
    # ------------------------------------------------------------------

    def stage(self) -> dict[str, Any]:
        job = uuid.uuid4().hex
        job_dir = self.root / job
        (job_dir / "in").mkdir(parents=True, exist_ok=False)
        (job_dir / "out").mkdir(parents=True, exist_ok=False)
        return {"status": P.STATUS_OK, "job": job, "root": str(job_dir)}

    def put(
        self,
        job: str,
        path: str,
        *,
        offset: int,
        data_b64: str,
        eof: bool = False,
        sha256: str | None = None,
    ) -> dict[str, Any]:
        job_dir = self._job_dir(job)
        target = self._target(job_dir, path)
        try:
            payload = base64.b64decode(data_b64.encode("ascii"), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RemoteError(f"chunk base64 inválido: {exc}", code=P.ERR_INVALID_REQUEST) from exc
        offset = int(offset)
        if offset < 0:
            raise RemoteError("offset negativo", code=P.ERR_REMOTE_PATH)
        current = target.stat().st_size if target.is_file() else 0
        if offset != current:
            raise RemoteError(
                f"offset {offset} != tamanho remoto {current}",
                code=P.ERR_REMOTE_PATH,
                hint="Retoma a transferência a partir de `size`.",
            )
        new_size = current + len(payload)
        if new_size > self.config.max_file_bytes:
            raise RemoteError(
                f"ficheiro excede {self.config.max_file_bytes // (1024 * 1024)} MiB",
                code=P.ERR_REMOTE_QUOTA,
            )
        job_bytes = self._job_bytes(job_dir)
        if job_bytes + len(payload) > self.config.max_job_bytes:
            raise RemoteError(
                f"sandbox excede {self.config.max_job_bytes // (1024 * 1024)} MiB",
                code=P.ERR_REMOTE_QUOTA,
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("r+b" if target.exists() else "wb") as handle:
            handle.seek(offset)
            handle.write(payload)
        result: dict[str, Any] = {"status": P.STATUS_OK, "size": new_size}
        if eof:
            if sha256:
                actual = _sha256(target)
                if actual.lower() != sha256.strip().lower():
                    raise RemoteError(
                        "sha256 do upload não bate com o ficheiro",
                        code=P.ERR_REMOTE_PATH,
                        hint=f"Remoto={actual}; local={sha256}",
                    )
                result["sha256"] = actual
            result["eof"] = True
        return result

    def get(
        self,
        job: str,
        path: str,
        *,
        offset: int = 0,
        length: int = 2 * 1024 * 1024,
    ) -> dict[str, Any]:
        job_dir = self._job_dir(job)
        target = self._target(job_dir, path)
        if not target.is_file():
            raise RemoteError(f"ficheiro não existe no sandbox: {path}", code=P.ERR_REMOTE_PATH)
        offset = max(0, int(offset))
        length = max(1, min(int(length), 8 * 1024 * 1024))
        with target.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read(length)
        size = target.stat().st_size
        return {
            "status": P.STATUS_OK,
            "data": base64.b64encode(chunk).decode("ascii"),
            "size": size,
            "offset": offset,
            "eof": offset + len(chunk) >= size,
        }

    def ls(self, job: str, path: str = ".") -> dict[str, Any]:
        job_dir = self._job_dir(job)
        target = self._target(job_dir, path)
        if not target.is_dir():
            raise RemoteError(f"diretório não existe no sandbox: {path}", code=P.ERR_REMOTE_PATH)
        entries: list[dict[str, Any]] = []
        for entry in sorted(target.iterdir(), key=lambda item: item.name):
            try:
                stat = entry.stat()
            except OSError:
                continue
            entries.append(
                {
                    "name": entry.name,
                    "size": stat.st_size if entry.is_file() else 0,
                    "is_dir": entry.is_dir(),
                }
            )
        return {"status": P.STATUS_OK, "path": path, "entries": entries}

    def clean(self, job: str) -> dict[str, Any]:
        job_dir = self._job_dir(job)
        freed = self._job_bytes(job_dir)
        shutil.rmtree(job_dir, ignore_errors=True)
        return {"status": P.STATUS_OK, "removed": True, "freed_bytes": freed}

    # ------------------------------------------------------------------
    # Validação
    # ------------------------------------------------------------------

    def _job_dir(self, job: str) -> Path:
        text = str(job or "").strip().lower()
        if not _JOB_RE.match(text):
            raise RemoteError("job id inválido (esperado hex de 12–32)", code=P.ERR_REMOTE_JOB)
        job_dir = (self.root / text).resolve()
        if self.root.resolve() not in job_dir.parents:
            raise RemoteError("job fora da raiz remota", code=P.ERR_REMOTE_JOB)
        if not job_dir.is_dir():
            raise RemoteError(
                f"sandbox desconhecido: {text}",
                code=P.ERR_REMOTE_JOB,
                hint="Faz `stage` antes de transferir.",
            )
        return job_dir

    def _target(self, job_dir: Path, path: str) -> Path:
        rel = str(path or "").strip()
        if not rel or Path(rel).is_absolute():
            raise RemoteError("path tem de ser relativo ao sandbox", code=P.ERR_REMOTE_PATH)
        if any(part == ".." for part in Path(rel).parts):
            raise RemoteError("path com '..' recusado", code=P.ERR_REMOTE_PATH)
        candidate = (job_dir / rel).resolve()
        if candidate != job_dir and job_dir not in candidate.parents:
            raise RemoteError("path escapa ao sandbox", code=P.ERR_REMOTE_PATH)
        return candidate

    def _job_bytes(self, job_dir: Path) -> int:
        total = 0
        for current, _dirs, files in os.walk(job_dir, followlinks=False):
            for name in files:
                file_path = Path(current) / name
                try:
                    if file_path.is_symlink():
                        continue
                    total += file_path.stat().st_size
                except OSError:
                    continue
        return total

    # ------------------------------------------------------------------
    # Janitor
    # ------------------------------------------------------------------

    def start_janitor(self) -> None:
        if self._janitor is not None and self._janitor.is_alive():
            return
        self._stop.clear()
        self._janitor = threading.Thread(
            target=self._janitor_loop, name="vramd-remote-janitor", daemon=True
        )
        self._janitor.start()

    def stop_janitor(self) -> None:
        self._stop.set()
        with contextlib.suppress(Exception):
            if self._janitor is not None:
                self._janitor.join(timeout=2.0)

    def sweep(self, *, now: float | None = None) -> list[str]:
        """Apaga sandboxes mais velhos que o TTL; devolve os ids removidos."""
        moment = time.time() if now is None else now
        removed: list[str] = []
        for entry in list(self.root.iterdir()):
            if not entry.is_dir():
                continue
            try:
                age = moment - entry.stat().st_mtime
            except OSError:
                continue
            if age > self.config.ttl_sec:
                shutil.rmtree(entry, ignore_errors=True)
                removed.append(entry.name)
        return removed

    def _janitor_loop(self) -> None:
        while not self._stop.wait(_JANITOR_INTERVAL_SEC):
            with contextlib.suppress(Exception):
                self.sweep()


def _sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()
