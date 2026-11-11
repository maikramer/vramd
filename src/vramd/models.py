"""Gestão do cache Hugging Face (layout normal) usada pelas sessões.

O vramd não inventa um layout: usa ``HF_HOME`` (default
``~/.cache/huggingface``) e as APIs do ``huggingface_hub``. Uma sessão declara
os modelos de cada backend; antes do primeiro load o vramd verifica o cache e,
conforme a política, descarrega ou falha com uma mensagem accionável. Os
modelos ficam pinados enquanto a sessão estiver aberta para o GC não os levar.

``move_hf_cache`` migra um cache existente (ex.: o volume Docker
``/srv/vramd-cache/models/hf``) para ``HF_HOME`` sem re-download: rename no
mesmo filesystem, cópia com symlinks noutro, verificação e só depois remoção.
"""

from __future__ import annotations

import contextlib
import math
import os
import shutil
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import protocol as P

_GIB = 1024**3


class ModelError(Exception):
    """Falha de cache de modelos com código estável."""

    def __init__(self, message: str, *, code: str = P.ERR_MODEL, hint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    revision: str = "main"
    allow: tuple[str, ...] = ()
    ignore: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: Any) -> ModelSpec:
        if not isinstance(raw, dict):
            raise ModelError("modelo deve ser um objecto")
        repo = str(raw.get("repo") or raw.get("repo_id") or "").strip()
        # O Hub exige `namespace/nome`; um path local/relativo aqui só dava
        # um erro opaco do huggingface_hub no primeiro download.
        parts = repo.split("/")
        bad_repo = (
            not repo
            or repo.startswith(("/", "\\"))
            # O Hub aceita ``modelo`` (legado sem namespace) ou ``org/modelo``.
            or len(parts) > 2
            or ".." in repo
            or "\\" in repo
            or len(repo) > 96
            or any(ch.isspace() or ord(ch) < 32 for ch in repo)
            or any(not part or part.startswith(("-", ".")) or part.endswith(("-", ".")) for part in parts)
        )
        if bad_repo:
            raise ModelError(f"repo HF inválido: {repo!r} (esperado 'org/modelo')")
        revision = str(raw.get("revision") or "main").strip() or "main"
        if ".." in revision or any(ord(ch) < 32 for ch in revision):
            raise ModelError(f"revision HF inválida: {revision!r}")
        allow = tuple(str(item) for item in raw.get("allow") or raw.get("allow_patterns") or ())
        ignore = tuple(str(item) for item in raw.get("ignore") or raw.get("ignore_patterns") or ())
        return cls(repo=repo, revision=revision, allow=allow, ignore=ignore)

    @property
    def key(self) -> str:
        return f"{self.repo}@{self.revision}"


def hf_home() -> Path:
    """``HF_HOME`` normal; sem a var, o default do huggingface_hub."""
    raw = os.environ.get("HF_HOME", "").strip()
    return Path(os.path.expanduser(raw)) if raw else Path.home() / ".cache" / "huggingface"


def _revision_stamp(revision: Any) -> float:
    """Timestamp de uma revision; a API 2.4 trocou ``last_accessed`` por
    ``last_modified`` no ``CachedRevisionInfo`` (e o valor é ``datetime``)."""
    value = getattr(revision, "last_accessed", None)
    if value is None:
        value = getattr(revision, "last_modified", None)
    if value is None:
        return 0.0
    if hasattr(value, "timestamp"):
        try:
            return float(value.timestamp())
        except (TypeError, ValueError, OSError):
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_token(token_file: str | None) -> str | None:
    if token_file:
        try:
            return Path(os.path.expanduser(token_file)).read_text(encoding="utf-8").strip() or None
        except OSError:
            return None
    return None


class ModelManager:
    """Verifica/descarrega modelos e protege-os do GC enquanto há sessões."""

    def __init__(
        self,
        *,
        home: Path | None = None,
        policy: str | None = None,
        max_gib: int | None = None,
        token: str | None = None,
        token_file: str | None = None,
    ) -> None:
        self.home = Path(home) if home is not None else hf_home()
        self.policy = (policy or os.environ.get("VRAMD_MODELS_POLICY", "auto")).strip().lower()
        if self.policy not in ("auto", "hub", "off"):
            self.policy = "auto"
        if max_gib is None:
            raw = os.environ.get("VRAMD_MODELS_MAX_GIB", "").strip()
            try:
                max_gib = int(raw) if raw else 150
            except ValueError:
                # Env malformada (`10G`) derrubava o arranque do supervisor
                # com um traceback; o default é o comportamento documentado.
                max_gib = 150
        if int(max_gib) < 0:
            raise ModelError(
                f"tecto do cache HF inválido: {max_gib} GiB (deve ser ≥ 0)",
                code=P.ERR_MODEL,
            )
        self.max_bytes = int(max_gib) * _GIB
        self.token = token or os.environ.get("HF_TOKEN", "").strip() or _read_token(token_file)
        self._pins: dict[str, int] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Cache / download
    # ------------------------------------------------------------------

    def ensure(self, specs: Iterable[ModelSpec], *, allow_download: bool = True) -> list[dict[str, Any]]:
        specs = list(specs)
        if not specs:
            return []
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - dependência core
            raise ModelError(
                "huggingface-hub não instalado",
                code=P.ERR_MODEL,
                hint="pip install -U vramd (a dependência é core).",
            ) from exc
        os.environ.setdefault("HF_HOME", str(self.home))
        result: list[dict[str, Any]] = []
        for spec in specs:
            path = self._local_path(snapshot_download, spec)
            if path is None:
                if self.policy == "off" or not allow_download:
                    raise ModelError(
                        f"modelo não está no cache: {spec.key}",
                        code=P.ERR_MODEL_MISSING,
                        hint=(
                            f"Move o cache para {self.home} ou corre `vramd models ensure "
                            f"{spec.repo} --revision {spec.revision}`."
                        ),
                    )
                try:
                    path = snapshot_download(
                        repo_id=spec.repo,
                        revision=spec.revision,
                        cache_dir=str(self._hub_cache_dir()),
                        allow_patterns=list(spec.allow) or None,
                        ignore_patterns=list(spec.ignore) or None,
                        token=self.token,
                    )
                except Exception as exc:
                    raise ModelError(
                        f"não foi possível descarregar {spec.key}: {exc}",
                        code=P.ERR_MODEL,
                        hint=(
                            "Verifica a rede/`HF_TOKEN`; se o servidor estiver offline, "
                            f"copia o cache (hub/ + blobs/) para {self.home} antes de abrir a sessão."
                        ),
                    ) from exc
            self.pin(spec)
            result.append({"repo": spec.repo, "revision": spec.revision, "path": str(path)})
        return result

    def _hub_cache_dir(self) -> Path:
        """``HF_HOME/hub`` — o layout normal do huggingface_hub.

        Passar ``cache_dir`` explicitamente evita depender de ``HF_HOME`` no
        ambiente: as constantes do hub são lidas no import e um ``HF_HOME``
        definido depois (ou um ``home`` custom no ``config.yml``) não chegava
        ao ``snapshot_download``.
        """
        return self.home / "hub"

    def _local_path(self, snapshot_download: Any, spec: ModelSpec) -> str | None:
        try:
            return snapshot_download(
                repo_id=spec.repo,
                revision=spec.revision,
                cache_dir=str(self._hub_cache_dir()),
                allow_patterns=list(spec.allow) or None,
                ignore_patterns=list(spec.ignore) or None,
                local_files_only=True,
            )
        except Exception:
            return None

    def has(self, spec: ModelSpec) -> bool:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:  # pragma: no cover
            return False
        return self._local_path(snapshot_download, spec) is not None

    # ------------------------------------------------------------------
    # Pins
    # ------------------------------------------------------------------

    def pin(self, spec: ModelSpec) -> None:
        with self._lock:
            self._pins[spec.key] = self._pins.get(spec.key, 0) + 1

    def unpin(self, spec: ModelSpec) -> None:
        with self._lock:
            current = self._pins.get(spec.key, 0)
            if current <= 1:
                self._pins.pop(spec.key, None)
            else:
                self._pins[spec.key] = current - 1

    def pinned(self) -> set[str]:
        with self._lock:
            return {key for key, count in self._pins.items() if count > 0}

    # ------------------------------------------------------------------
    # Inventário / GC
    # ------------------------------------------------------------------

    def list(self) -> dict[str, Any]:
        try:
            from huggingface_hub import scan_cache_dir
        except ImportError as exc:  # pragma: no cover
            raise ModelError("huggingface-hub não instalado", code=P.ERR_MODEL) from exc
        try:
            info = scan_cache_dir(cache_dir=str(self._hub_cache_dir()))
        except Exception as exc:
            # STATUS_OK com bytes=0 fazia `vramd models list` anunciar um cache
            # vazio quando na verdade o scan rebentou (permissões/corrupção).
            raise ModelError(f"não foi possível ler o cache HF: {exc}", code=P.ERR_MODEL) from exc
        repos: list[dict[str, Any]] = []
        for repo in info.repos:
            revisions = [
                {
                    "commit": revision.commit_hash,
                    "refs": list(revision.refs),
                    "size": revision.size_on_disk,
                    "last_accessed": _revision_stamp(revision),
                }
                for revision in repo.revisions
            ]
            repos.append(
                {
                    "repo": repo.repo_id,
                    "type": repo.repo_type,
                    "size": repo.size_on_disk,
                    "revisions": revisions,
                }
            )
        return {
            "status": P.STATUS_OK,
            "home": str(info.cache_dir),
            "bytes": info.size_on_disk,
            "repos": sorted(repos, key=lambda item: item["repo"]),
            "pinned": sorted(self.pinned()),
            "max_bytes": self.max_bytes,
        }

    def gc(self) -> dict[str, Any]:
        try:
            from huggingface_hub import scan_cache_dir
        except ImportError as exc:  # pragma: no cover
            raise ModelError("huggingface-hub não instalado", code=P.ERR_MODEL) from exc
        try:
            info = scan_cache_dir(cache_dir=str(self._hub_cache_dir()))
        except Exception as exc:
            raise ModelError(f"não foi possível ler o cache HF: {exc}", code=P.ERR_MODEL) from exc
        total = info.size_on_disk
        if total <= self.max_bytes:
            return {
                "status": P.STATUS_OK,
                "removed": 0,
                "freed_bytes": 0,
                "bytes": total,
            }
        pins = self.pinned()
        # Não apagar um modelo acabado de descarregar só porque o cache está
        # acima do tecto; o GC é para revisions antigas, não para o download
        # que acabou de alimentar a próxima sessão.
        raw_min_age = os.environ.get("VRAMD_MODELS_MIN_AGE_DAYS", "").strip()
        try:
            min_age_days = float(raw_min_age) if raw_min_age else 7.0
        except ValueError:
            min_age_days = 7.0
        if not math.isfinite(min_age_days) or min_age_days < 0:
            min_age_days = 7.0
        min_age = min_age_days * 86400.0
        now = time.time()
        candidates: list[tuple[int, Any, str]] = []
        for repo in info.repos:
            for revision in repo.revisions:
                keys = {f"{repo.repo_id}@{ref}" for ref in revision.refs}
                keys.add(f"{repo.repo_id}@{revision.commit_hash}")
                if keys & pins:
                    continue
                stamp = _revision_stamp(revision)
                if stamp and now - stamp < min_age:
                    continue
                candidates.append((stamp, revision, repo.repo_id))
        candidates.sort(key=lambda item: item[0])
        remove: list[str] = []
        freed = 0
        for _accessed, revision, _repo in candidates:
            if total <= self.max_bytes:
                break
            remove.append(revision.commit_hash)
            total -= revision.size_on_disk
            freed += revision.size_on_disk
        if remove:
            try:
                info.delete_revisions(*remove).execute()
            except Exception as exc:
                raise ModelError(f"GC do cache HF falhou: {exc}", code=P.ERR_MODEL) from exc
        return {
            "status": P.STATUS_OK,
            "removed": len(remove),
            "freed_bytes": freed,
            "bytes": total,
        }


# ---------------------------------------------------------------------------
# Migração do cache existente (ex.: volume Docker)
# ---------------------------------------------------------------------------


def _tree_stats(root: Path) -> tuple[int, int, int]:
    files = 0
    links = 0
    total = 0
    for current, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(current) / name).is_symlink()]
        for name in names:
            path = Path(current) / name
            try:
                if path.is_symlink():
                    links += 1
                elif path.is_file():
                    files += 1
                    total += path.stat().st_size
            except OSError:
                continue
    return files, links, total


def _verify_against(source: Path, dest: Path) -> None:
    """Confirma que cada entrada de ``source`` existe igual em ``dest``."""
    for current, dirs, names in os.walk(source, followlinks=False):
        rel_dir = Path(current).relative_to(source)
        dirs[:] = [name for name in dirs if not (Path(current) / name).is_symlink()]
        for name in names:
            src = Path(current) / name
            dst = dest / rel_dir / name
            if src.is_symlink():
                if not dst.is_symlink() or os.readlink(src) != os.readlink(dst):
                    raise ModelError(f"verificação falhou (symlink): {src}")
                continue
            if not src.is_file():
                continue
            if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
                raise ModelError(f"verificação falhou (ficheiro): {src}")


def move_hf_cache(
    source: Path | str,
    dest: Path | str,
    *,
    delete_source: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Move/merge um cache HF para ``dest`` sem re-download.

    Mesmo filesystem: rename por entrada (instantâneo). Filesystems diferentes:
    cópia preservando symlinks, verificação de tamanho/target e só depois
    remoção da origem. ``dry_run`` devolve o plano sem tocar em nada.
    """
    src = Path(source).expanduser().resolve()
    dst = Path(dest).expanduser().resolve()
    if src == dst:
        return {"status": "ok", "moved": False, "reason": "same path", "files": 0, "bytes": 0}
    if not src.is_dir():
        raise ModelError(f"cache de origem não existe: {src}")
    files, links, total = _tree_stats(src)
    probe = dst if dst.exists() else dst.parent
    same_device = probe.exists() and src.stat().st_dev == probe.stat().st_dev
    if dry_run:
        return {
            "status": "ok",
            "dry_run": True,
            "source": str(src),
            "dest": str(dst),
            "files": files,
            "symlinks": links,
            "bytes": total,
            "same_filesystem": same_device,
        }
    dst.mkdir(parents=True, exist_ok=True)
    # `delete_source=False` é uma cópia: nunca renomear (a origem tem de ficar).
    same_fs = bool(delete_source and same_device)
    if same_fs:
        for entry in list(src.iterdir()):
            target = dst / entry.name
            if not target.exists():
                os.rename(entry, target)
            elif entry.is_dir() and target.is_dir():
                _merge_dir(entry, target)
                shutil.rmtree(entry, ignore_errors=True)
            elif entry.is_file() and target.is_file() and entry.stat().st_size == target.stat().st_size:
                entry.unlink()
            else:
                _copy_entry(entry, target)
                if delete_source:
                    if entry.is_dir():
                        shutil.rmtree(entry, ignore_errors=True)
                    else:
                        entry.unlink(missing_ok=True)
    else:
        for entry in list(src.iterdir()):
            target = dst / entry.name
            _copy_entry(entry, target)
        _verify_against(src, dst)
        if delete_source:
            shutil.rmtree(src, ignore_errors=True)
    if delete_source and same_fs and src.is_dir():
        with contextlib.suppress(OSError):
            src.rmdir()
    return {
        "status": "ok",
        "moved": True,
        "source": str(src),
        "dest": str(dst),
        "files": files,
        "symlinks": links,
        "bytes": total,
        "same_filesystem": same_fs,
        "source_removed": delete_source and not src.exists(),
    }


def _copy_entry(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        link = os.readlink(source)
        with contextlib.suppress(FileNotFoundError):
            target.unlink()
        os.symlink(link, target)
        return
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True, symlinks=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _merge_dir(source: Path, target: Path) -> None:
    for entry in list(source.iterdir()):
        child = target / entry.name
        if not child.exists():
            os.rename(entry, child)
        elif entry.is_dir() and child.is_dir() and not entry.is_symlink():
            _merge_dir(entry, child)
            with contextlib.suppress(OSError):
                entry.rmdir()
        elif entry.is_file() and child.is_file() and entry.stat().st_size == child.stat().st_size:
            entry.unlink(missing_ok=True)
        else:
            _copy_entry(entry, child)
