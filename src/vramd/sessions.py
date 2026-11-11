"""Sessões de código do cliente (bundle content-addressed + envs + backends).

Uma sessão é a unidade que faz o servidor executar **o código do cliente** sem
ter o pipeline instalado: o cliente envia um tar.gz com ``Shared/src``,
``<Tool>/src`` e os locks de dependências (via CAS de blobs), o vramd extrai-o
para ``<cache>/sessions/bundles/<sha>``, constrói/reutiliza as envs por hash e
regista backends efémeros ``<backend>@<session>`` no ``Registry``. O worker é
arrancado com ``PYTHONPATH=<bundle>/...`` e ``cwd=<bundle>``; o código do
servidor nunca é fonte de verdade.

Segurança: ``session-open``/env-build só funcionam com ``--allow-client-code``
e ``--allow-client-env``. A extracção valida cada membro contra o manifesto
(paths relativos, sem symlinks/hardlinks/.., tamanho e sha256) antes de
publicar o bundle.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import tarfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import protocol as P
from .blobs import BlobStore
from .envs import EnvError, EnvManager, EnvSpec
from .models import ModelError, ModelManager, ModelSpec
from .registry import BackendDescriptor, Registry, RuntimeSpec

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_TOOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_JANITOR_INTERVAL_SEC = 600.0

MAX_BUNDLE_FILES = 200_000
MAX_BUNDLE_BYTES = 2 * 1024**3
MAX_FILE_BYTES = 256 * 1024**2
MAX_MANIFEST_FILES = 200_000
# O tar pode trazer directórios vazios além dos ficheiros declarados; um
# limite global evita encher o filesystem com inodes a partir de um bundle
# comprimido minúsculo.
MAX_BUNDLE_MEMBERS = 250_000


def _int_field(value: Any, field: str) -> int:
    """Converte campos numéricos do manifesto com erro de protocolo claro."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SessionError(f"{field} inválido: {value!r}") from exc


class SessionError(Exception):
    """Erro de sessão com código estável para a resposta JSON."""

    def __init__(self, message: str, *, code: str = P.ERR_SESSION_INVALID, hint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint


@dataclass(frozen=True)
class BackendSpec:
    """Backend declarado no manifesto da sessão."""

    name: str
    tool: str
    env: str
    code_roots: tuple[str, ...]
    vram_mib: int
    priority: int
    load_keys: frozenset[str] | None
    shape_keys: frozenset[str] | None
    remote: dict[str, Any] = field(default_factory=dict)
    models: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, raw: Any) -> BackendSpec:
        if not isinstance(raw, dict):
            raise SessionError("backend da sessão deve ser um objecto")
        name = str(raw.get("name") or "").strip()
        if not _NAME_RE.match(name) or "@" in name:
            raise SessionError(f"backend name inválido: {name!r}")
        tool = str(raw.get("tool") or "").strip()
        if not _TOOL_RE.match(tool):
            raise SessionError(f"backend {name!r}: tool/módulo inválido: {tool!r}")
        env = str(raw.get("env") or "").strip()
        if not env:
            raise SessionError(f"backend {name!r}: falta `env`")
        roots_raw = raw.get("code_roots") or []
        if not isinstance(roots_raw, list) or not roots_raw:
            raise SessionError(f"backend {name!r}: `code_roots` tem de ser uma lista não vazia")
        roots = tuple(_safe_rel(str(item)) for item in roots_raw)
        vram_mib = _int_field(raw.get("vram_mib") or 4096, f"backend {name!r}.vram_mib")
        if vram_mib < 1:
            raise SessionError(f"backend {name!r}: vram_mib inválido")
        priority = _int_field(raw.get("priority") or 0, f"backend {name!r}.priority")
        load_keys = _key_set(raw.get("load_keys"))
        shape_keys = _key_set(raw.get("shape_keys"))
        remote = raw.get("remote") if isinstance(raw.get("remote"), dict) else {}
        models_raw = raw.get("models") or []
        models: list[dict[str, Any]] = []
        if isinstance(models_raw, list):
            models = [dict(item) for item in models_raw if isinstance(item, dict)]
        return cls(
            name=name,
            tool=tool,
            env=env,
            code_roots=roots,
            vram_mib=vram_mib,
            priority=priority,
            load_keys=load_keys,
            shape_keys=shape_keys,
            remote=dict(remote),
            models=tuple(models),
        )


@dataclass
class Session:
    id: str
    key: str
    bundle_sha: str
    bundle_root: Path
    envs: dict[str, dict[str, Any]]
    backends: list[str]
    backend_specs: list[BackendSpec]
    created: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.id,
            "bundle_sha": self.bundle_sha,
            "envs": {
                name: {"hash": info.get("hash"), "python": info.get("python_path")} for name, info in self.envs.items()
            },
            "backends": [spec.name for spec in self.backend_specs],
            "age_sec": round(time.monotonic() - self.created, 1),
        }


def _safe_rel(raw: str) -> str:
    text = str(raw or "").strip().replace("\\", "/")
    if not text or text.startswith("/") or Path(text).is_absolute():
        raise SessionError(f"path inválido no bundle: {raw!r}")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise SessionError(f"path inválido no bundle: {raw!r}")
    return "/".join(parts)


def _key_set(raw: Any) -> frozenset[str] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise SessionError("load_keys/shape_keys deve ser uma lista")
    return frozenset(str(item) for item in raw)


class SessionManager:
    """Abre/fecha sessões e mantém os backends efémeros registados."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        registry: Registry,
        manager: Any,
        job_queue: Any,
        blobs: BlobStore,
        allow_client_code: bool = False,
        allow_client_env: bool = False,
        env_manager: EnvManager | None = None,
        models: ModelManager | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.registry = registry
        self.manager = manager
        self.job_queue = job_queue
        self.blobs = blobs
        self.allow_client_code = allow_client_code
        self.allow_client_env = allow_client_env
        self.envs = env_manager or EnvManager(self.cache_dir / "envs")
        self.models = models or ModelManager()
        self.root = self.cache_dir / "sessions"
        self.bundles_root = self.root / "bundles"
        self.bundles_root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()
        self._bundle_locks: dict[str, threading.Lock] = {}
        self._janitor: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # Comandos
    # ------------------------------------------------------------------

    def open(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self.allow_client_code:
            raise SessionError(
                "sessões de código do cliente estão desativadas",
                code=P.ERR_SESSION_DISABLED,
                hint="Arranca o vramd com `--allow-client-code` (só em rede de confiança).",
            )
        manifest = request.get("manifest")
        bundle, files, envs, backends = self._parse_manifest(manifest)
        bundle_sha = bundle["sha256"]
        key_raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        session_key = hashlib.sha256(bundle_sha.encode() + b"\0" + key_raw).hexdigest()

        with self._lock:
            for current in self._sessions.values():
                if current.key == session_key:
                    current.last_used = time.monotonic()
                    return {"status": P.STATUS_OK, "session": current.id, **current.to_dict()}

        bundle_root = self._prepare_bundle(bundle, files)
        envs_info: dict[str, dict[str, Any]] = {}
        try:
            for name, spec in envs.items():
                envs_info[name] = self.envs.ensure(
                    spec,
                    bundle_root,
                    allow_build=self.allow_client_env,
                )
            for spec in backends:
                if spec.env not in envs_info:
                    raise SessionError(
                        f"backend {spec.name!r}: env {spec.env!r} não declarada",
                    )
        except EnvError as exc:
            raise SessionError(str(exc), code=exc.code, hint=exc.hint) from exc

        model_specs: list[ModelSpec] = []
        try:
            for spec in backends:
                for raw in spec.models:
                    model_specs.append(ModelSpec.from_dict(raw))
            if model_specs:
                self.models.ensure(model_specs)
        except ModelError as exc:
            for model_spec in model_specs:
                self.models.unpin(model_spec)
            raise SessionError(str(exc), code=exc.code, hint=exc.hint) from exc

        session_id = self._new_session_id()
        registered: list[str] = []
        try:
            for spec in backends:
                registered.append(self._register_backend(session_id, spec, bundle_root, envs_info))
        except BaseException:
            for name in registered:
                self.registry.unregister(name)
            for model_spec in model_specs:
                self.models.unpin(model_spec)
            raise

        session = Session(
            id=session_id,
            key=session_key,
            bundle_sha=bundle_sha,
            bundle_root=bundle_root,
            envs=envs_info,
            backends=registered,
            backend_specs=backends,
        )
        with self._lock:
            self._sessions[session_id] = session
        return {
            "status": P.STATUS_OK,
            "session": session_id,
            "bundle_sha": bundle_sha,
            "bundle_root": str(bundle_root),
            **{k: v for k, v in session.to_dict().items() if k != "session"},
        }

    def close(self, session_id: str, *, force: bool = False) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return {"status": P.STATUS_OK, "closed": False, "session": session_id}
        if not force and self._busy(session_id):
            raise SessionError(
                f"sessão {session_id} tem jobs na fila/em curso",
                code=P.ERR_SESSION_BUSY,
                hint="Espera o batch terminar ou usa `force=true` (mata workers).",
            )
        stopped = 0
        for name in session.backends:
            with contextlib.suppress(Exception):
                if self.manager.shutdown_worker(name):
                    stopped += 1
            with contextlib.suppress(Exception):
                self.manager.forget(name)
            self.registry.unregister(name)
        for backend_spec in session.backend_specs:
            for raw in backend_spec.models:
                with contextlib.suppress(ModelError):
                    self.models.unpin(ModelSpec.from_dict(raw))
        with self._lock:
            self._sessions.pop(session_id, None)
        return {
            "status": P.STATUS_OK,
            "closed": True,
            "session": session_id,
            "workers_stopped": stopped,
        }

    def list(self) -> dict[str, Any]:
        with self._lock:
            sessions = [session.to_dict() for session in self._sessions.values()]
        return {
            "status": P.STATUS_OK,
            "count": len(sessions),
            "sessions": sessions,
            "envs": self.envs.list(),
        }

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": len(self._sessions),
                "allow_client_code": self.allow_client_code,
                "allow_client_env": self.allow_client_env,
                "sessions": [
                    {"session": s.id, "bundle_sha": s.bundle_sha, "backends": [b.name for b in s.backend_specs]}
                    for s in self._sessions.values()
                ],
            }

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def touch(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_used = time.monotonic()

    def bundle_sha(self, session_id: str) -> str | None:
        session = self.get(session_id)
        return session.bundle_sha if session is not None else None

    # ------------------------------------------------------------------
    # Manifesto / bundle
    # ------------------------------------------------------------------

    def _parse_manifest(
        self, raw: Any
    ) -> tuple[dict[str, Any], dict[str, tuple[int, str]], dict[str, EnvSpec], list[BackendSpec]]:
        if not isinstance(raw, dict):
            raise SessionError("manifesto da sessão ausente ou não é objecto")
        version = _int_field(raw.get("version") or 1, "manifest.version")
        if version != 1:
            raise SessionError(f"versão de manifesto não suportada: {version}")
        bundle = raw.get("bundle")
        if not isinstance(bundle, dict):
            raise SessionError("manifesto sem `bundle`")
        sha = str(bundle.get("sha256") or "").strip().lower()
        if not _SHA_RE.match(sha):
            raise SessionError("bundle.sha256 inválido")
        fmt = str(bundle.get("format") or "tar.gz").strip().lower()
        if fmt not in ("tar", "tar.gz", "tgz", "tar.zst"):
            raise SessionError(f"formato de bundle não suportado: {fmt}")
        size = bundle.get("size")
        if size is not None:
            size = _int_field(size, "bundle.size")
            if size < 0:
                raise SessionError("bundle.size inválido")

        files_raw = raw.get("files")
        if not isinstance(files_raw, list) or not files_raw:
            raise SessionError("manifesto sem `files` (lista de ficheiros do bundle)")
        if len(files_raw) > MAX_MANIFEST_FILES:
            raise SessionError(f"manifesto com demasiados ficheiros (> {MAX_MANIFEST_FILES})")
        files: dict[str, tuple[int, str]] = {}
        for item in files_raw:
            if not isinstance(item, dict):
                raise SessionError("entrada de `files` inválida")
            path = _safe_rel(str(item.get("path") or ""))
            file_size = _int_field(item.get("size") or 0, f"files[{path!r}].size")
            file_sha = str(item.get("sha256") or "").strip().lower()
            if file_size < 0 or file_size > MAX_FILE_BYTES:
                raise SessionError(f"ficheiro {path!r}: tamanho inválido")
            if not _SHA_RE.match(file_sha):
                raise SessionError(f"ficheiro {path!r}: sha256 inválido")
            if path in files:
                raise SessionError(f"ficheiro repetido no manifesto: {path!r}")
            files[path] = (file_size, file_sha)
        if len(files) > MAX_BUNDLE_FILES:
            raise SessionError(f"bundle com demasiados ficheiros (> {MAX_BUNDLE_FILES})")
        if sum(size for size, _sha in files.values()) > MAX_BUNDLE_BYTES:
            raise SessionError(f"bundle excede {MAX_BUNDLE_BYTES // (1024**3)} GiB descomprimido")

        envs_raw = raw.get("envs")
        if not isinstance(envs_raw, dict) or not envs_raw:
            raise SessionError("manifesto sem `envs`")
        envs = {str(name): EnvSpec.from_dict(str(name), spec) for name, spec in envs_raw.items()}

        backends_raw = raw.get("backends")
        if not isinstance(backends_raw, list) or not backends_raw:
            raise SessionError("manifesto sem `backends`")
        backends = [BackendSpec.from_dict(item) for item in backends_raw]
        names = [spec.name for spec in backends]
        if len(set(names)) != len(names):
            raise SessionError("nomes de backend repetidos no manifesto")
        for spec in backends:
            if spec.env not in envs:
                raise SessionError(f"backend {spec.name!r}: env {spec.env!r} não declarada")
        return (
            {"sha256": sha, "format": fmt, "size": size},
            files,
            envs,
            backends,
        )

    def _prepare_bundle(self, bundle: dict[str, Any], files: dict[str, tuple[int, str]]) -> Path:
        sha = bundle["sha256"]
        dest = self.bundles_root / sha
        if (dest / ".ready").is_file():
            return dest
        blob_path = self.blobs.path(sha)
        if not blob_path.is_file():
            raise SessionError(
                f"bundle {sha[:12]}… não está no servidor",
                code=P.ERR_SESSION_BUNDLE,
                hint="Faz `blob-put` do bundle antes de `session-open`.",
            )
        expected_size = bundle.get("size")
        if expected_size is not None and blob_path.stat().st_size != int(expected_size):
            raise SessionError(
                f"bundle {sha[:12]}…: tamanho {blob_path.stat().st_size} != {expected_size}",
                code=P.ERR_SESSION_BUNDLE,
            )
        # O GC do CAS é LRU por atime; um `blobs gc` concorrente entre o
        # upload e a extracção podia apagar o bundle. O pin dura só a
        # extracção: depois o código vive em `sessions/bundles/<sha>`.
        self.blobs.pin(sha)
        try:
            with self._bundle_lock(sha):
                if (dest / ".ready").is_file():
                    return dest
                if dest.exists():
                    shutil.rmtree(dest, ignore_errors=True)
                tmp = self.bundles_root / f".{sha}.{uuid.uuid4().hex}.tmp"
                tmp.mkdir(parents=True, exist_ok=False)
                try:
                    try:
                        self._extract_bundle(blob_path, bundle["format"], tmp, files)
                    except (tarfile.TarError, OSError) as exc:
                        # Um tar que colide com o filesystem (directório e
                        # ficheiro com o mesmo path, permissões) é bundle
                        # inválido, não um erro genérico de geração.
                        raise SessionError(f"bundle inválido: {exc}", code=P.ERR_SESSION_BUNDLE) from exc
                    (tmp / ".ready").write_text(
                        json.dumps({"sha256": sha, "files": len(files), "format": bundle["format"]}),
                        encoding="utf-8",
                    )
                    os.replace(tmp, dest)
                except BaseException:
                    shutil.rmtree(tmp, ignore_errors=True)
                    raise
        finally:
            self.blobs.unpin(sha)
        return dest

    def _extract_bundle(
        self,
        blob_path: Path,
        fmt: str,
        dest: Path,
        files: dict[str, tuple[int, str]],
    ) -> None:
        expected = dict(files)
        seen: set[str] = set()
        total = 0
        raw: Any = None
        try:
            if fmt in ("tar.gz", "tgz"):
                tf = tarfile.open(blob_path, mode="r:gz")  # noqa: SIM115 (fechado no with abaixo)
            elif fmt == "tar":
                tf = tarfile.open(blob_path, mode="r:")  # noqa: SIM115
            else:
                try:
                    import zstandard
                except ImportError as exc:  # pragma: no cover
                    raise SessionError(
                        "bundle tar.zst requer o extra `zstandard` no servidor",
                        code=P.ERR_SESSION_BUNDLE,
                    ) from exc
                raw = blob_path.open("rb")
                stream = zstandard.ZstdDecompressor().stream_reader(raw)
                tf = tarfile.open(fileobj=stream, mode="r|")  # noqa: SIM115
        except (tarfile.TarError, OSError) as exc:
            if raw is not None:
                raw.close()
            raise SessionError(f"bundle inválido: {exc}", code=P.ERR_SESSION_BUNDLE) from exc

        members = 0
        try:
            with contextlib.closing(tf):
                for member in tf:
                    members += 1
                    if members > MAX_BUNDLE_MEMBERS:
                        raise SessionError(
                            f"bundle tem mais de {MAX_BUNDLE_MEMBERS} entradas",
                            code=P.ERR_SESSION_BUNDLE,
                        )
                    name = _safe_rel(member.name)
                    if member.isdir():
                        (dest / name).mkdir(parents=True, exist_ok=True)
                        continue
                    if member.issym() or member.islnk() or not member.isfile():
                        raise SessionError(
                            f"bundle contém entrada não regular: {name!r}",
                            code=P.ERR_SESSION_BUNDLE,
                        )
                    if name not in expected:
                        raise SessionError(
                            f"bundle contém ficheiro não declarado: {name!r}",
                            code=P.ERR_SESSION_BUNDLE,
                        )
                    if name in seen:
                        # Um tar com o mesmo path duas vezes tem duas versões
                        # possíveis do ficheiro; o manifesto só declara uma.
                        raise SessionError(
                            f"bundle repete o ficheiro: {name!r}",
                            code=P.ERR_SESSION_BUNDLE,
                        )
                    size, sha = expected[name]
                    if member.size != size:
                        raise SessionError(
                            f"bundle {name!r}: tamanho {member.size} != {size}",
                            code=P.ERR_SESSION_BUNDLE,
                        )
                    if total + size > MAX_BUNDLE_BYTES:
                        raise SessionError(
                            f"bundle excede {MAX_BUNDLE_BYTES // (1024**3)} GiB descomprimido",
                            code=P.ERR_SESSION_BUNDLE,
                        )
                    src = tf.extractfile(member)
                    if src is None:
                        raise SessionError(f"bundle não legível: {name!r}", code=P.ERR_SESSION_BUNDLE)
                    target = dest / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    written = 0
                    with src, target.open("wb") as out:
                        while chunk := src.read(1 << 20):
                            out.write(chunk)
                            digest.update(chunk)
                            written += len(chunk)
                            if written > MAX_FILE_BYTES:
                                raise SessionError(
                                    f"bundle {name!r} excede {MAX_FILE_BYTES // (1024**2)} MiB",
                                    code=P.ERR_SESSION_BUNDLE,
                                )
                    if written != size or digest.hexdigest() != sha:
                        raise SessionError(
                            f"bundle {name!r}: sha256/tamanho não batem com o manifesto",
                            code=P.ERR_SESSION_BUNDLE,
                        )
                    mode = member.mode & 0o777
                    with contextlib.suppress(OSError):
                        os.chmod(target, 0o755 if mode & 0o111 else 0o644)
                    total += written
                    seen.add(name)
        finally:
            if raw is not None:
                with contextlib.suppress(OSError):
                    raw.close()

        missing = set(expected) - seen
        if missing:
            sample = ", ".join(sorted(missing)[:5])
            raise SessionError(
                f"bundle sem {len(missing)} ficheiro(s) declarados (ex.: {sample})",
                code=P.ERR_SESSION_BUNDLE,
            )

    def _bundle_lock(self, sha: str) -> threading.Lock:
        with self._lock:
            return self._bundle_locks.setdefault(sha, threading.Lock())

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _register_backend(
        self,
        session_id: str,
        spec: BackendSpec,
        bundle_root: Path,
        envs: dict[str, dict[str, Any]],
    ) -> str:
        code_paths: list[Path] = []
        for root in spec.code_roots:
            path = (bundle_root / root).resolve()
            if path != bundle_root and bundle_root not in path.parents:
                raise SessionError(f"code_root escapa ao bundle: {root!r}")
            if not path.is_dir():
                raise SessionError(f"code_root não existe no bundle: {root!r}")
            code_paths.append(path)
        env_info = envs[spec.env]
        python_path = str(env_info["python_path"])
        pythonpath = os.pathsep.join([*(str(path) for path in code_paths), str(bundle_root)])
        runtime = RuntimeSpec(
            command=(python_path, "-m", spec.tool, "serve", "--ums-worker"),
            cwd=str(bundle_root),
            env={
                "PYTHONPATH": pythonpath,
                "PYTHONNOUSERSITE": "1",
                "VRAMD_BUNDLE_SHA": bundle_root.name,
                "VRAMD_SESSION": session_id,
            },
        )
        base: BackendDescriptor | None = None
        if self.registry.has(spec.name):
            with contextlib.suppress(KeyError):
                base = self.registry.descriptor(spec.name)
        synthetic = f"{spec.name}@{session_id}"
        descriptor = BackendDescriptor(
            name=synthetic,
            adapter="vramd.session",
            vram_mib=spec.vram_mib or (base.vram_mib if base else 4096),
            priority=spec.priority or (base.priority if base else 0),
            footprint_key=base.footprint_key if base else None,
            tool=None,
            runtime=runtime,
            load_keys=spec.load_keys if spec.load_keys is not None else (base.load_keys if base else None),
            shape_keys=spec.shape_keys if spec.shape_keys is not None else (base.shape_keys if base else None),
            vram=dict(base.vram) if base else {},
            peak_profile=dict(base.peak_profile) if base else {},
            calibrate_request=dict(base.calibrate_request) if base else {},
            calibrate_load_kwargs=dict(base.calibrate_load_kwargs) if base else {},
            placement=dict(base.placement) if base else {},
        )
        self.registry.register(descriptor)
        return synthetic

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def _new_session_id(self) -> str:
        with self._lock:
            while True:
                candidate = uuid.uuid4().hex[:16]
                if candidate not in self._sessions:
                    return candidate

    def _busy(self, session_id: str) -> bool:
        for job in (*self.job_queue.queued_jobs(), *self.job_queue.running_jobs()):
            request = getattr(job, "request", {})
            if isinstance(request, dict) and request.get("session") == session_id:
                return True
        return False

    def start_janitor(self) -> None:
        if self._janitor is not None and self._janitor.is_alive():
            return
        self._stop.clear()
        self._janitor = threading.Thread(target=self._janitor_loop, name="vramd-sessions-janitor", daemon=True)
        self._janitor.start()

    def stop_janitor(self) -> None:
        self._stop.set()
        with contextlib.suppress(Exception):
            if self._janitor is not None:
                self._janitor.join(timeout=2.0)

    def _janitor_loop(self) -> None:
        ttl = float(P._env_int("VRAMD_SESSION_TTL_HOURS", 24)) * 3600.0
        while not self._stop.wait(_JANITOR_INTERVAL_SEC):
            now = time.monotonic()
            with self._lock:
                stale = [s.id for s in self._sessions.values() if now - s.last_used > ttl]
            for session_id in stale:
                if self._busy(session_id):
                    continue
                with contextlib.suppress(Exception):
                    self.close(session_id)
            with contextlib.suppress(Exception):
                self.prune_bundles()

    def prune_bundles(self, *, now: float | None = None) -> list[str]:
        """Remove bundles extraídos que nenhuma sessão activa referencia.

        O código é re-extraído do CAS se voltar a ser pedido (ou reenviado pelo
        cliente se o blob já tiver sido recolhido). Sem isto, cada mudança de
        código deixava dezenas de MB em ``sessions/bundles`` para sempre.
        """
        ttl = float(P._env_int("VRAMD_BUNDLE_TTL_DAYS", 14)) * 86400.0
        moment = time.time() if now is None else now
        with self._lock:
            active = {session.bundle_root.name for session in self._sessions.values()}
        removed: list[str] = []
        if not self.bundles_root.is_dir():
            return removed
        for entry in list(self.bundles_root.iterdir()):
            if not entry.is_dir() or entry.name in active:
                continue
            try:
                age = moment - entry.stat().st_mtime
            except OSError:
                continue
            if age <= ttl:
                continue
            shutil.rmtree(entry, ignore_errors=True)
            if not entry.name.startswith("."):
                removed.append(entry.name)
            with self._lock:
                self._bundle_locks.pop(entry.name, None)
        return removed
