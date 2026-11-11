"""Configuração do vramd: ``config.yml``, wizard e unidade systemd.

Precedência: flags do CLI > variáveis de ambiente > ``config.yml`` > defaults.
O ficheiro é YAML simples; o wizard só preenche os valores e chama
:func:`write_config`, :func:`ensure_token` e :func:`install_service`.

O objectivo é o servidor ficar com **vramd + modelos** e nada mais: o
``config.yml`` descreve o listener, o cache, o HF_HOME, as permissões de
sessões e o serviço systemd; o código do pipeline viaja por sessão.
"""

from __future__ import annotations

import contextlib
import getpass
import os
import secrets
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .models import move_hf_cache

try:
    import pwd
except ImportError:  # pragma: no cover - Windows
    pwd = None  # type: ignore[assignment]

DEFAULT_LISTEN = "0.0.0.0:8790"
DEFAULT_CACHE_DIR = "/var/cache/vramd"
DEFAULT_TOKEN_FILE = "/etc/vramd/token"
DEFAULT_HF_HOME = "~/.cache/huggingface"
DEFAULT_HF_TOKEN_FILE = "/etc/vramd/hf-token"
OLD_CACHE_CANDIDATES = ("/srv/vramd-cache/models/hf",)


def _is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


@dataclass
class VramdConfig:
    """Configuração efectiva do servidor vramd."""

    listen: str = DEFAULT_LISTEN
    token_file: str = DEFAULT_TOKEN_FILE
    cache_dir: str = DEFAULT_CACHE_DIR
    allow_client_code: bool = True
    allow_client_env: bool = True
    gpus: str = "auto"
    session_ttl_hours: int = 24
    models_home: str = DEFAULT_HF_HOME
    models_policy: str = "auto"
    models_max_gib: int = 150
    hf_token_file: str = DEFAULT_HF_TOKEN_FILE
    service_user: str = field(default_factory=lambda: getpass.getuser())
    install_service: bool = False
    old_cache: str = OLD_CACHE_CANDIDATES[0]
    move_models: bool = True
    keep_source: bool = False

    @classmethod
    def for_current_user(cls) -> VramdConfig:
        """Defaults seguros para correr sem root."""
        if _is_root():
            return cls()
        home = Path.home()
        return cls(
            listen="127.0.0.1:8790",
            token_file=str(home / ".config" / "vramd" / "token"),
            cache_dir=str(home / ".cache" / "vramd"),
            hf_token_file=str(home / ".config" / "vramd" / "hf-token"),
        )

    @classmethod
    def from_dict(cls, raw: Any) -> VramdConfig:
        cfg = cls.for_current_user()
        if not isinstance(raw, dict):
            return cfg
        models = raw.get("models") if isinstance(raw.get("models"), dict) else {}
        sessions = raw.get("sessions") if isinstance(raw.get("sessions"), dict) else {}
        policy_raw = models.get("policy", raw.get("models_policy"))
        # YAML 1.1 lê `off`/`on` como booleanos; o policy é string.
        if policy_raw is False:
            policy_raw = "off"
        elif policy_raw is True:
            policy_raw = "auto"
        elif policy_raw is not None:
            policy_raw = str(policy_raw)
        mapping = {
            "listen": raw.get("listen"),
            "token_file": raw.get("token_file"),
            "cache_dir": raw.get("cache_dir"),
            "allow_client_code": raw.get("allow_client_code"),
            "allow_client_env": raw.get("allow_client_env"),
            "gpus": raw.get("gpus"),
            "session_ttl_hours": sessions.get("ttl_hours", raw.get("session_ttl_hours")),
            "models_home": models.get("home", raw.get("models_home")),
            "models_policy": policy_raw,
            "models_max_gib": models.get("max_gib", raw.get("models_max_gib")),
            "hf_token_file": models.get("token_file", raw.get("hf_token_file")),
            "service_user": raw.get("service_user"),
            "install_service": raw.get("install_service"),
            "old_cache": raw.get("old_cache"),
            "move_models": raw.get("move_models"),
            "keep_source": raw.get("keep_source"),
        }
        for key, value in mapping.items():
            if value is not None:
                setattr(cfg, key, value)
        cfg.session_ttl_hours = int(cfg.session_ttl_hours)
        cfg.models_max_gib = int(cfg.models_max_gib)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {
            "listen": data["listen"],
            "token_file": data["token_file"],
            "cache_dir": data["cache_dir"],
            "allow_client_code": data["allow_client_code"],
            "allow_client_env": data["allow_client_env"],
            "gpus": data["gpus"],
            "service_user": data["service_user"],
            "install_service": data["install_service"],
            "sessions": {"ttl_hours": data["session_ttl_hours"]},
            "models": {
                "home": data["models_home"],
                "policy": data["models_policy"],
                "max_gib": data["models_max_gib"],
                "token_file": data["hf_token_file"],
            },
            "old_cache": data["old_cache"],
            "move_models": data["move_models"],
            "keep_source": data["keep_source"],
        }


def config_paths(explicit: str | Path | None = None) -> list[Path]:
    paths: list[Path] = []
    if explicit:
        paths.append(Path(explicit).expanduser())
    env = os.environ.get("VRAMD_CONFIG_FILE", "").strip()
    if env:
        paths.append(Path(env).expanduser())
    paths.append(Path("/etc/vramd/config.yml"))
    paths.append(Path.home() / ".config" / "vramd" / "config.yml")
    return paths


def default_config_path(*, root: bool | None = None) -> Path:
    as_root = _is_root() if root is None else root
    return Path("/etc/vramd/config.yml") if as_root else Path.home() / ".config" / "vramd" / "config.yml"


def load_config(path: str | Path | None = None, *, strict: bool = False) -> VramdConfig:
    """Lê o primeiro ``config.yml`` existente; sem ficheiro devolve defaults."""
    for candidate in config_paths(path):
        if candidate.is_file():
            try:
                raw = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError) as exc:
                raise ValueError(f"config inválida em {candidate}: {exc}") from exc
            return VramdConfig.from_dict(raw)
    if strict and path is not None:
        raise FileNotFoundError(str(path))
    return VramdConfig.for_current_user()


def write_config(cfg: VramdConfig, path: str | Path | None = None) -> Path:
    target = Path(path).expanduser() if path else default_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    header = "# vramd — runtime GPU remoto\n# Gerado por `vramd config`; variáveis VRAMD_*/HF_* no ambiente vencem.\n"
    body = yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True)
    target.write_text(header + body, encoding="utf-8")
    with contextlib.suppress(OSError):
        target.chmod(0o644)
    return target


def ensure_token(path: str | Path) -> Path:
    """Garante um token estável 0600 (32 bytes URL-safe)."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        target.write_text(secrets.token_urlsafe(32), encoding="utf-8")
    with contextlib.suppress(OSError):
        target.chmod(0o600)
    return target


def write_secret(path: str | Path, value: str) -> Path:
    """Escreve um segredo 0600 (ex.: HF_TOKEN)."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(value.strip() + "\n", encoding="utf-8")
    with contextlib.suppress(OSError):
        target.chmod(0o600)
    return target


def validate_user(user: str) -> tuple[int, int] | None:
    """Valida o utilizador do serviço; devolve ``(uid, gid)`` ou ``None``."""
    if not user or pwd is None:
        return None
    try:
        entry = pwd.getpwnam(user)
    except KeyError as exc:
        raise ValueError(f"utilizador do serviço não existe: {user!r}") from exc
    return entry.pw_uid, entry.pw_gid


def resolve_path_for_user(path: str | Path, user: str) -> str:
    """Expande ``~`` contra a home do utilizador do serviço (não a de root)."""
    text = str(path)
    if not text.startswith("~"):
        return str(Path(text).expanduser())
    home: Path
    if _is_root() and pwd is not None and user:
        try:
            home = Path(pwd.getpwnam(user).pw_dir)
        except KeyError:
            home = Path.home()
    else:
        home = Path.home()
    rest = text[2:].lstrip("/")
    return str(home / rest) if rest else str(home)


def chown_for_user(path: str | Path, user: str) -> None:
    """Dá ao utilizador do serviço acesso ao token/cache/HF (só como root)."""
    if not _is_root() or not user or not hasattr(os, "chown"):
        return
    ids = validate_user(user)
    if ids is None or ids[0] == 0:
        return
    target = Path(path).expanduser()
    if not target.exists():
        return
    with contextlib.suppress(OSError):
        os.chown(target, ids[0], ids[1])
        for current, dirs, names in os.walk(target):
            for name in (*dirs, *names):
                child = Path(current) / name
                with contextlib.suppress(OSError):
                    os.chown(child, ids[0], ids[1], follow_symlinks=False)


def apply_env(cfg: VramdConfig) -> None:
    """Exporta a config para o ambiente (só se a env não tiver valor)."""
    os.environ.setdefault("VRAMD_CACHE_DIR", str(Path(cfg.cache_dir).expanduser()))
    os.environ.setdefault("VRAMD_GPUS", cfg.gpus)
    os.environ.setdefault("VRAMD_SESSION_TTL_HOURS", str(cfg.session_ttl_hours))
    os.environ.setdefault("VRAMD_MODELS_POLICY", cfg.models_policy)
    os.environ.setdefault("VRAMD_MODELS_MAX_GIB", str(cfg.models_max_gib))
    os.environ.setdefault("HF_HOME", str(Path(cfg.models_home).expanduser()))
    token_path = Path(cfg.hf_token_file).expanduser()
    if "HF_TOKEN" not in os.environ and token_path.is_file():
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            os.environ["HF_TOKEN"] = token


def detect_old_cache(cfg: VramdConfig) -> Path | None:
    for raw in (cfg.old_cache, *OLD_CACHE_CANDIDATES):
        candidate = Path(raw).expanduser()
        if candidate.is_dir():
            return candidate
    return None


def migrate_models(
    cfg: VramdConfig,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any] | None:
    """Move o cache HF antigo para ``cfg.models_home`` sem re-download."""
    source = detect_old_cache(cfg)
    if source is None:
        return None
    dest = Path(cfg.models_home).expanduser()
    if source.resolve() == dest.resolve():
        return None
    if not force and not cfg.move_models:
        return None
    return move_hf_cache(source, dest, delete_source=not cfg.keep_source, dry_run=dry_run)


def systemd_unit(
    cfg: VramdConfig,
    *,
    config_file: str | Path,
    exec_python: str | None = None,
    exec_module: str = "vramd",
) -> str:
    """Unidade systemd do servidor (ExecStart usa ``python -m vramd``)."""
    python = exec_python or sys.executable
    user = cfg.service_user or getpass.getuser()
    cache = Path(cfg.cache_dir).expanduser()
    token = Path(cfg.token_file).expanduser()
    env_lines = [f"Environment=VRAMD_CONFIG_FILE={Path(config_file).expanduser()}"]
    return (
        "[Unit]\n"
        "Description=vramd GPU runtime (queue, VRAM admission, sessions)\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n"
        "\n"
        "[Service]\n"
        f"User={user}\n"
        f"WorkingDirectory={cache}\n"
        f"Environment=VRAMD_CACHE_DIR={cache}\n"
        f"Environment=VRAMD_TOKEN_FILE={token}\n"
        f"Environment=HF_HOME={Path(cfg.models_home).expanduser()}\n" + "\n".join(env_lines) + "\n"
        f"ExecStart={python} -m {exec_module} start --config {Path(config_file).expanduser()}\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
        "KillMode=mixed\n"
        "TimeoutStopSec=30\n"
        "LimitNOFILE=65536\n"
        "LimitMEMLOCK=infinity\n"
        "SupplementaryGroups=video render\n"
        "\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )


def install_service(
    cfg: VramdConfig,
    *,
    config_file: str | Path,
    dry_run: bool = False,
    exec_python: str | None = None,
) -> dict[str, Any]:
    """Escreve a unidade e (se root) activa via systemctl."""
    unit = systemd_unit(cfg, config_file=config_file, exec_python=exec_python)
    root = _is_root()
    target = Path("/etc/systemd/system/vramd.service") if root else Path.home() / ".config" / "vramd" / "vramd.service"
    commands: list[str] = []
    if root:
        commands = ["systemctl daemon-reload", "systemctl enable --now vramd.service"]
    else:
        commands = [
            f"sudo cp {target} /etc/systemd/system/vramd.service",
            "sudo systemctl daemon-reload",
            "sudo systemctl enable --now vramd.service",
        ]
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(unit, encoding="utf-8")
        with contextlib.suppress(OSError):
            target.chmod(0o644)
        if root:
            for command in commands:
                subprocess.run(command.split(), check=False)
    return {"unit_path": str(target), "commands": commands, "installed": bool(root and not dry_run)}
