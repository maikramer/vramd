"""Ambientes Python content-addressed para sessões de código do cliente.

O servidor não mantém um checkout nem venvs do pipeline. O cliente envia um
freeze exacto (``pip freeze --exclude-editable``) dentro do bundle; o vramd
constrói uma vez ``<cache>/envs/<hash>`` com ``uv`` e reutiliza-o em todas as
sessões com o mesmo lock. O hash cobre a versão de Python, a plataforma e o
conteúdo do lock — mudar uma dependência cria um ambiente novo sem tocar no
antigo.

Para testes e para um ambiente base já instalado, uma env pode declarar
``path`` (interpretador existente) ou ``system`` (o Python do supervisor); não
há build nem download.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import host
from . import protocol as P

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]
try:  # Windows
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]

_GIB = 1024**3
DEFAULT_BUILD_TIMEOUT_SEC = 3600.0


class EnvError(Exception):
    """Falha de ambiente com código estável para a resposta JSON."""

    def __init__(self, message: str, *, code: str = P.ERR_SESSION_ENV, hint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint


@dataclass(frozen=True)
class EnvSpec:
    """Entrada ``envs.<nome>`` do manifesto de sessão."""

    name: str
    python: str | None = None
    path: str | None = None
    lock: str | None = None
    index_url: str | None = None
    extra_index_url: str | None = None
    system: bool = False
    hash: str | None = None

    @classmethod
    def from_dict(cls, name: str, raw: Any) -> EnvSpec:
        if not isinstance(raw, dict):
            raise EnvError(f"env {name!r} deve ser um objecto", code=P.ERR_SESSION_INVALID)
        return cls(
            name=name,
            python=str(raw["python"]) if raw.get("python") else None,
            path=str(raw["path"]) if raw.get("path") else None,
            lock=str(raw["lock"]) if raw.get("lock") else None,
            index_url=str(raw["index_url"]) if raw.get("index_url") else None,
            extra_index_url=str(raw["extra_index_url"]) if raw.get("extra_index_url") else None,
            system=bool(raw.get("system")),
            hash=str(raw["hash"]).lower() if raw.get("hash") else None,
        )


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _venv_python(env_dir: Path) -> Path:
    return env_dir / ("Scripts/python.exe" if host.IS_WINDOWS else "bin/python")


class EnvManager:
    """Constrói e reutiliza ambientes por hash de lock."""

    def __init__(self, root: Path | None = None, *, uv: str | None = None) -> None:
        self.root = Path(root) if root is not None else host.cache_dir() / "envs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.uv = uv or os.environ.get("VRAMD_UV", "").strip() or shutil.which("uv")
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ------------------------------------------------------------------
    # Resolução / cache
    # ------------------------------------------------------------------

    def describe(self, spec: EnvSpec, bundle_root: Path) -> dict[str, Any]:
        """Resolve a env sem construir: devolve hash, caminho do Python e kind."""
        if spec.path:
            path = Path(spec.path).expanduser().resolve()
            if not path.is_file():
                raise EnvError(
                    f"env {spec.name!r}: interpretador não existe: {path}",
                    code=P.ERR_SESSION_ENV,
                )
            env_hash = spec.hash or _sha_text(f"path:{path}")
            return {
                "name": spec.name,
                "kind": "path",
                "hash": env_hash,
                "python_path": str(path),
                "exists": True,
            }
        if spec.system:
            env_hash = spec.hash or _sha_text(f"system:{sys.executable}")
            return {
                "name": spec.name,
                "kind": "system",
                "hash": env_hash,
                "python_path": sys.executable,
                "exists": True,
            }
        if not spec.lock:
            raise EnvError(
                f"env {spec.name!r}: declara `lock` (freeze), `path` ou `system`",
                code=P.ERR_SESSION_INVALID,
            )
        lock_path = self._resolve_lock(spec.lock, bundle_root)
        content = lock_path.read_bytes()
        python = spec.python or "3.13"
        actual = hashlib.sha256(b"lock\0" + python.encode() + b"\0" + content).hexdigest()
        env_hash = spec.hash or actual
        if spec.hash and spec.hash != actual:
            # O cliente assinou o lock; um hash que não bate é corrupção.
            raise EnvError(
                f"env {spec.name!r}: hash {spec.hash[:12]}… não bate com o lock ({actual[:12]}…)",
                code=P.ERR_SESSION_INVALID,
            )
        env_dir = self.root / env_hash
        return {
            "name": spec.name,
            "kind": "lock",
            "hash": env_hash,
            "python_path": str(_venv_python(env_dir)),
            "env_dir": str(env_dir),
            "lock_path": str(lock_path),
            "python": python,
            "exists": (env_dir / ".ready").is_file(),
        }

    def ensure(self, spec: EnvSpec, bundle_root: Path, *, allow_build: bool = True) -> dict[str, Any]:
        info = self.describe(spec, bundle_root)
        if info["kind"] != "lock" or info["exists"]:
            return info
        if not allow_build:
            raise EnvError(
                f"env {spec.name!r} ainda não foi construída e o servidor está sem `--allow-client-env`",
                code=P.ERR_SESSION_ENV_DISABLED,
                hint="Constrói a env no servidor (`vramd env build`) ou liga `--allow-client-env`.",
            )
        env_dir = Path(info["env_dir"])
        lock_path = Path(info["lock_path"])
        with self._build_lock(info["hash"]):
            if (env_dir / ".ready").is_file():
                info["exists"] = True
                return info
            self._build(
                env_hash=info["hash"],
                env_dir=env_dir,
                lock_path=lock_path,
                python=info["python"],
                index_url=spec.index_url,
                extra_index_url=spec.extra_index_url,
            )
        info["exists"] = True
        return info

    def build_lock(
        self,
        lock_path: str | Path,
        *,
        python: str = "3.13",
        index_url: str | None = None,
        extra_index_url: str | None = None,
    ) -> dict[str, Any]:
        """Constrói/reutiliza uma env a partir de um lock absoluto (CLI)."""
        lock = Path(lock_path).expanduser().resolve()
        if not lock.is_file():
            raise EnvError(f"lock não existe: {lock}", code=P.ERR_SESSION_INVALID)
        content = lock.read_bytes()
        env_hash = hashlib.sha256(b"lock\0" + python.encode() + b"\0" + content).hexdigest()
        env_dir = self.root / env_hash
        with self._build_lock(env_hash):
            if not (env_dir / ".ready").is_file():
                self._build(
                    env_hash=env_hash,
                    env_dir=env_dir,
                    lock_path=lock,
                    python=python,
                    index_url=index_url,
                    extra_index_url=extra_index_url,
                )
        return {
            "hash": env_hash,
            "python": python,
            "python_path": str(_venv_python(env_dir)),
            "env_dir": str(env_dir),
            "lock": str(lock),
        }

    def has(self, env_hash: str) -> bool:
        return (self.root / env_hash / ".ready").is_file()

    def list(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            ready = entry / ".ready"
            if not ready.is_file():
                continue
            with contextlib.suppress(OSError, json.JSONDecodeError):
                meta = json.loads(ready.read_text(encoding="utf-8"))
                out.append({"hash": entry.name, "path": str(entry), **meta})
        return out

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(
        self,
        *,
        env_hash: str,
        env_dir: Path,
        lock_path: Path,
        python: str,
        index_url: str | None,
        extra_index_url: str | None,
    ) -> None:
        uv = self.uv
        if not uv:
            raise EnvError(
                "`uv` não encontrado no servidor",
                code=P.ERR_SESSION_ENV,
                hint="Instala `uv` (https://docs.astral.sh/uv/) ou define VRAMD_UV.",
            )
        tmp = self.root / f".{env_hash}.{uuid.uuid4().hex}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        log_path = self.root / f"{env_hash}.build.log"
        env = {
            **os.environ,
            "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str(host.cache_dir() / "uv-cache")),
            "UV_LINK_MODE": os.environ.get("UV_LINK_MODE", "copy"),
            "UV_NO_PROGRESS": "1",
            "PYTHONNOUSERSITE": "1",
        }
        timeout = float(os.environ.get("VRAMD_ENV_BUILD_TIMEOUT_SEC", DEFAULT_BUILD_TIMEOUT_SEC))
        log: list[str] = []
        try:
            self._run(
                [uv, "venv", "--python", python, str(tmp)],
                env=env,
                timeout=timeout,
                log=log,
            )
            python_path = _venv_python(tmp)
            if not python_path.is_file():
                raise EnvError(
                    f"`uv venv` não criou o interpretador em {python_path}",
                    code=P.ERR_SESSION_ENV,
                )
            install = [
                uv,
                "pip",
                "install",
                "--python",
                str(python_path),
                "-r",
                str(lock_path),
                # PyPI + índice PyTorch: sem isto o uv fixa-se no primeiro
                # índice onde vê o pacote e recusa a versão do outro
                # (certifi/torch). Os dois índices são de confiança.
                "--index-strategy",
                "unsafe-best-match",
            ]
            if index_url:
                install += ["--index-url", index_url]
            if extra_index_url:
                install += ["--extra-index-url", extra_index_url]
            self._run(install, env=env, timeout=timeout, log=log)
            meta = {
                "hash": env_hash,
                "python": python,
                "lock": lock_path.name,
                "built_at": time.time(),
                "uv": uv,
            }
            (tmp / ".ready").write_text(json.dumps(meta), encoding="utf-8")
            if env_dir.exists():
                shutil.rmtree(env_dir, ignore_errors=True)
            os.replace(tmp, env_dir)
            with contextlib.suppress(OSError):
                log_path.unlink(missing_ok=True)
        except BaseException as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            with contextlib.suppress(OSError):
                log_path.write_text("\n".join(log[-200:]), encoding="utf-8")
            if isinstance(exc, EnvError):
                raise
            if isinstance(exc, subprocess.TimeoutExpired):
                raise EnvError(
                    f"build da env {env_hash[:12]}… excedeu {timeout:.0f}s",
                    code=P.ERR_SESSION_ENV,
                    hint=f"Log: {log_path}",
                ) from exc
            raise EnvError(
                f"build da env {env_hash[:12]}… falhou: {exc}",
                code=P.ERR_SESSION_ENV,
                hint=f"Log: {log_path}",
            ) from exc

    @staticmethod
    def _run(cmd: list[str], *, env: dict[str, str], timeout: float, log: list[str]) -> None:
        log.append("$ " + " ".join(cmd))
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.stdout:
            log.append(proc.stdout[-8000:])
        if proc.stderr:
            log.append(proc.stderr[-8000:])
        if proc.returncode != 0:
            raise EnvError(
                f"comando falhou (exit {proc.returncode}): {' '.join(cmd[:3])}…",
                code=P.ERR_SESSION_ENV,
            )

    def _resolve_lock(self, rel: str, bundle_root: Path) -> Path:
        raw = str(rel or "").strip()
        if not raw or Path(raw).is_absolute() or ".." in Path(raw).parts:
            raise EnvError(f"lock inválido: {rel!r}", code=P.ERR_SESSION_INVALID)
        path = (bundle_root / raw).resolve()
        if path != bundle_root and bundle_root not in path.parents:
            raise EnvError(f"lock escapa ao bundle: {rel!r}", code=P.ERR_SESSION_INVALID)
        if not path.is_file():
            raise EnvError(f"lock não existe no bundle: {rel!r}", code=P.ERR_SESSION_INVALID)
        return path

    def _build_lock(self, env_hash: str) -> _BuildLock:
        with self._locks_guard:
            lock = self._locks.setdefault(env_hash, threading.Lock())
        return _BuildLock(lock, self.root / f"{env_hash}.lock")


class _BuildLock:
    """Lock de processo + ficheiro (builds concorrentes da mesma env)."""

    def __init__(self, thread_lock: threading.Lock, path: Path) -> None:
        self._thread_lock = thread_lock
        self._path = path
        self._handle: Any = None

    def __enter__(self) -> _BuildLock:
        self._thread_lock.acquire()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a+")
        if fcntl is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:  # pragma: no cover - Windows
            self._handle.seek(0)
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_LOCK, 1)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        try:
            if fcntl is not None and self._handle is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None and self._handle is not None:  # pragma: no cover
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            self._thread_lock.release()
