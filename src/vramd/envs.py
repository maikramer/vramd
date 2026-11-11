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
import re
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


def _reject_cu12(lock_path: Path, content: bytes) -> None:
    """O pipeline é CUDA 13; um lock com cu12 é recusado no servidor."""
    text = content.decode("utf-8", "replace")
    if "-cu12" in text or "+cu12" in text:
        raise EnvError(
            f"{lock_path.name}: lock com CUDA 12; o pipeline é CUDA 13 (cu130)",
            code=P.ERR_SESSION_ENV,
            hint=(
                "Remove os pacotes `nvidia-*-cu12`/`torch+cu12` do venv, volta a "
                "gerar o freeze e reabre a sessão. O Hi3D (cu124) não entra em sessões."
            ),
        )


def _cu13_specs(lock_path: Path) -> list[str]:
    """Pins ``nvidia-*-cu13`` do lock.

    Os pacotes ``nvidia-*-cu12`` e ``nvidia-*-cu13`` escrevem no **mesmo**
    ``site-packages/nvidia/<lib>/lib`` (nccl, cudnn, ...). Se a cu12 ficar por
    cima, o torch cu13 falha com símbolos em falta (``ncclCommResume``) ou
    ``CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED``. Reinstalar as cu13 no fim
    replica a ordem que funciona no venv do cliente.
    """
    specs: list[str] = []
    try:
        lines = lock_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return specs
    for line in lines:
        text = line.strip()
        if text.startswith("nvidia-") and "-cu13==" in text:
            specs.append(text)
    return specs


# Pacotes cujo `setup.py` importa `torch` em tempo de build. O uv instala-os
# com isolamento por omissão (venv temporária sem torch) e o build falha com
# "You have PyTorch installed / --no-build-isolation". Instalam-se à parte,
# depois do resto do lock, sem isolamento — com o torch e o setuptools do
# ambiente final já presentes.
_NO_BUILD_ISOLATION = {"nvdiffrast"}

_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
# Credenciais em URLs de índice (`https://user:token@host/simple`) e pares
# `token=…` não podem ficar no log de build (legível por outros utilizadores).
_REDACT_URL = re.compile(r"(://)[^/@\s]+@")
_REDACT_SECRET = re.compile(r"((?:token|api_key|apikey|password|passwd)=)[^&\s]+", re.IGNORECASE)


def _redact_command(cmd: list[str]) -> str:
    """Argv para o log com credenciais mascaradas."""
    text = " ".join(str(part) for part in cmd)
    text = _REDACT_URL.sub(r"\1***@", text)
    return _REDACT_SECRET.sub(r"\1***", text)


def _write_private_log(path: Path, text: str) -> None:
    """Log de build 0600 desde o ``open`` (pode conter URLs/envs)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).strip().lower()


def _split_no_build_isolation(content: str) -> tuple[list[str], list[str]]:
    """Separa do lock as linhas que não podem ser construídas isoladas.

    Devolve ``(linhas_restantes, requisitos_especiais)``. Linhas de
    continuação (``--hash=...`` indentadas) acompanham o requisito anterior.
    """
    kept: list[str] = []
    special: list[str] = []
    current: str | None = None
    for line in content.splitlines():
        if line[:1].isspace() and current is not None:
            # Continuação de um requisito (hashes, marcadores partidos).
            if current == "special":
                special[-1] += "\n" + line
            else:
                kept.append(line)
            continue
        match = _REQ_NAME.match(line)
        name = _canonical_name(match.group(1)) if match else ""
        if name in _NO_BUILD_ISOLATION:
            current = "special"
            special.append(line.strip())
        else:
            current = "main"
            kept.append(line)
    return kept, special


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
            if not host.IS_WINDOWS and not os.access(path, os.X_OK):
                raise EnvError(
                    f"env {spec.name!r}: interpretador sem permissão de execução: {path}",
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
        _reject_cu12(lock_path, content)
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
        _reject_cu12(lock, content)
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
        try:
            entries = sorted(self.root.iterdir())
        except OSError:
            # ``vramd env list`` é read-only: cache apagado/ilegível dá lista
            # vazia, não um traceback.
            return out
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            ready = entry / ".ready"
            if not ready.is_file():
                continue
            try:
                meta = json.loads(ready.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(meta, dict):
                continue
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
        # `nvdiffrast` (paint3d) compila com nvcc; o serviço systemd não herda
        # o PATH do /usr/local/cuda. Exporta-o no build, não no worker.
        cuda_home = os.environ.get("CUDA_HOME", "").strip()
        if not cuda_home:
            candidate = Path("/usr/local/cuda")
            if (candidate / "bin" / "nvcc").is_file():
                cuda_home = str(candidate)
        if cuda_home:
            env.setdefault("CUDA_HOME", cuda_home)
            env["PATH"] = f"{cuda_home}/bin:" + env.get(
                "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            )
        timeout = float(os.environ.get("VRAMD_ENV_BUILD_TIMEOUT_SEC", DEFAULT_BUILD_TIMEOUT_SEC))
        log: list[str] = []
        split_lock: Path | None = None
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
            content = lock_path.read_text(encoding="utf-8")
            kept, special = _split_no_build_isolation(content)
            if special:
                split_lock = self.root / f".{env_hash}.{uuid.uuid4().hex}.lock"
                split_lock.write_text("\n".join(kept) + "\n", encoding="utf-8")
            install_lock = split_lock or lock_path
            install = [
                uv,
                "pip",
                "install",
                "--python",
                str(python_path),
                "-r",
                str(install_lock),
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
            if special:
                # `nvdiffrast` importa torch no setup.py: sem isolamento, com
                # o torch já instalado acima. setuptools/wheel também têm de
                # existir no venv (a venv do uv não os traz).
                lowered = {_canonical_name(m.group(1)) for m in map(_REQ_NAME.match, kept) if m}
                bootstrap = [name for name in ("setuptools", "wheel") if name not in lowered]
                isolated = [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(python_path),
                    "--no-build-isolation",
                    "--index-strategy",
                    "unsafe-best-match",
                ]
                if index_url:
                    isolated += ["--index-url", index_url]
                if extra_index_url:
                    isolated += ["--extra-index-url", extra_index_url]
                self._run(isolated + bootstrap + special, env=env, timeout=timeout, log=log)
            # As libs cu13 têm de ficar por cima das cu12 (mesmo path).
            cu13 = _cu13_specs(lock_path)
            if cu13:
                repair = [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(python_path),
                    "--no-deps",
                    "--reinstall",
                    "--index-strategy",
                    "unsafe-best-match",
                    *cu13,
                ]
                self._run(repair, env=env, timeout=timeout, log=log)
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
            if split_lock is not None:
                with contextlib.suppress(OSError):
                    split_lock.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                _write_private_log(log_path, "\n".join(log[-200:]))
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
        finally:
            if split_lock is not None:
                with contextlib.suppress(OSError):
                    split_lock.unlink(missing_ok=True)

    @staticmethod
    def _run(cmd: list[str], *, env: dict[str, str], timeout: float, log: list[str]) -> None:
        log.append("$ " + _redact_command(cmd))
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
        root = Path(bundle_root).resolve()
        path = (root / raw).resolve()
        if path != root and root not in path.parents:
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
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("a+")
            if fcntl is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
            elif msvcrt is not None:  # pragma: no cover - Windows
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_LOCK, 1)
        except BaseException:
            # Sem isto, um open/flock falhado (lockfile num path inválido)
            # deixava a threading.Lock presa para sempre: o build seguinte
            # da mesma env bloqueava o servidor inteiro.
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            self._thread_lock.release()
            raise
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
