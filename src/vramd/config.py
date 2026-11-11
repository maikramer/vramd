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
import time
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


def _config_int(value: Any, key: str, *, minimum: int | None = None) -> int:
    """Converte um inteiro do YAML com mensagem accionável (nunca traceback)."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{key} deve ser um inteiro, recebi {value!r}")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{key} deve ser um inteiro, recebi {value!r}")
        number = int(value)
    else:
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} deve ser um inteiro, recebi {value!r}") from None
    if minimum is not None and number < minimum:
        raise ValueError(f"{key} deve ser ≥ {minimum}, recebi {number}")
    return number


def _config_bool(value: Any, key: str) -> bool:
    """Um booleano a sério: ``"false"`` (string) avalia como verdadeiro."""
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} deve ser true/false (YAML), recebi {value!r}")


def _config_path(value: Any, key: str) -> str:
    if isinstance(value, (str, os.PathLike)):
        return str(value)
    raise ValueError(f"{key} deve ser um caminho (string), recebi {value!r}")


def _config_listen(value: Any) -> str:
    """``HOST:PORT``/``[::1]:PORT`` — validado antes de chegar ao ``start``."""
    if not isinstance(value, str):
        raise ValueError(f"listen deve ser 'HOST:PORT', recebi {value!r}")
    text = value.strip()
    if text.startswith("["):
        host, sep, rest = text[1:].partition("]")
        if not sep or not rest.startswith(":"):
            raise ValueError(f"listen inválido: {value!r} (esperado [IPv6]:PORT)")
        port_text = rest[1:]
    else:
        if text.count(":") != 1:
            raise ValueError(f"listen inválido: {value!r} (esperado HOST:PORT)")
        host, _, port_text = text.partition(":")
    if not host:
        raise ValueError(f"listen sem host: {value!r}")
    try:
        port = int(port_text)
    except ValueError:
        raise ValueError(f"listen com porta inválida: {port_text!r}") from None
    if not 1 <= port <= 65535:
        raise ValueError(f"listen com porta fora de 1..65535: {port}")
    return text


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
        if raw is None:
            return cfg
        if not isinstance(raw, dict):
            # Um ficheiro com uma lista no topo (ex.: `- listen: ...`) era
            # ignorado em silêncio e o servidor arrancava com defaults errados.
            raise ValueError(f"config.yml deve ser um mapa no topo, recebi {type(raw).__name__}")
        models = raw.get("models")
        if models is None:
            models = {}
        elif not isinstance(models, dict):
            raise ValueError(f"config.yml: 'models' deve ser um mapa, recebi {type(models).__name__}")
        sessions = raw.get("sessions")
        if sessions is None:
            sessions = {}
        elif not isinstance(sessions, dict):
            raise ValueError(f"config.yml: 'sessions' deve ser um mapa, recebi {type(sessions).__name__}")
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

        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Normaliza e valida a config efectiva (ficheiro **e** flags do CLI).

        Raises:
            ValueError: campo com tipo/valor inválido, com o nome do campo na
                mensagem (o CLI mostra-a em vez de um traceback no arranque).
        """
        self.listen = _config_listen(self.listen)
        for key in ("token_file", "cache_dir", "models_home", "hf_token_file", "old_cache"):
            setattr(self, key, _config_path(getattr(self, key), key))
        if self.models_policy not in ("auto", "hub", "off"):
            raise ValueError(f"models.policy deve ser auto, hub ou off, recebi {self.models_policy!r}")
        gpus = self.gpus
        # `gpus: off` sem aspas chega como bool no YAML 1.1; `gpus: 0` chega
        # como int. Ambos são intenções válidas do operador.
        if gpus is False:
            gpus = "off"
        elif gpus is True:
            gpus = "auto"
        elif isinstance(gpus, int):
            gpus = str(gpus)
        elif isinstance(gpus, list):
            gpus = ",".join(str(item) for item in gpus)
        if not isinstance(gpus, str) or not gpus.strip():
            raise ValueError(f"gpus deve ser 'auto', 'off' ou uma lista de índices, recebi {self.gpus!r}")
        self.gpus = gpus
        if not isinstance(self.service_user, str) or not self.service_user.strip():
            raise ValueError(f"service_user deve ser um nome de utilizador, recebi {self.service_user!r}")
        for key in ("allow_client_code", "allow_client_env", "install_service", "move_models", "keep_source"):
            setattr(self, key, _config_bool(getattr(self, key), key))
        self.session_ttl_hours = _config_int(self.session_ttl_hours, "sessions.ttl_hours", minimum=1)
        self.models_max_gib = _config_int(self.models_max_gib, "models.max_gib", minimum=0)

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


def _write_private(target: Path, text: str, *, exclusive: bool = False) -> None:
    """Escreve 0600 desde o ``open`` (sem janela 0644 do umask)."""
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    handle = os.open(target, flags, 0o600)
    with os.fdopen(handle, "w", encoding="utf-8") as fh:
        fh.write(text)
    with contextlib.suppress(OSError):
        os.chmod(target, 0o600)


def ensure_token(path: str | Path) -> Path:
    """Garante um token estável 0600 (32 bytes URL-safe).

    A criação usa ``O_EXCL``: dois ``vramd config`` concorrentes não podem
    ficar com tokens diferentes (um deles lê o que o outro escreveu).
    """
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = ""
    if target.is_file():
        with contextlib.suppress(OSError, UnicodeDecodeError):
            existing = target.read_text(encoding="utf-8")
    if existing.strip():
        with contextlib.suppress(OSError):
            target.chmod(0o600)
        return target
    try:
        _write_private(target, secrets.token_urlsafe(32), exclusive=True)
    except FileExistsError:
        # Outro processo ganhou a corrida: espera que ele escreva e usa o
        # token dele (nunca um segundo token válido em paralelo).
        for _ in range(50):
            existing = target.read_text(encoding="utf-8").strip()
            if existing:
                with contextlib.suppress(OSError):
                    target.chmod(0o600)
                return target
            time.sleep(0.01)
        # Ficheiro criado mas nunca preenchido (processo morreu a meio):
        # garantir um token em vez de devolver um path vazio.
        _write_private(target, secrets.token_urlsafe(32))
    return target


def write_secret(path: str | Path, value: str) -> Path:
    """Escreve um segredo 0600 (ex.: HF_TOKEN)."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_private(target, value.strip() + "\n")
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


def _sd_quote(value: Any) -> str:
    """Aspas systemd para caminhos/argv com espaços (``Environment=``/``ExecStart=``)."""
    text = str(value).replace("\n", " ")
    if not text or any(ch in text for ch in ' \t"\\'):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


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
    config_path = Path(config_file).expanduser()
    env_lines = [f"Environment=VRAMD_CONFIG_FILE={_sd_quote(config_path)}"]
    exec_start = " ".join(
        _sd_quote(part) for part in (python, "-m", exec_module, "start", "--config", str(config_path))
    )
    return (
        "[Unit]\n"
        "Description=vramd GPU runtime (queue, VRAM admission, sessions)\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n"
        "\n"
        "[Service]\n"
        f"User={user}\n"
        f"WorkingDirectory={_sd_quote(cache)}\n"
        f"Environment=VRAMD_CACHE_DIR={_sd_quote(cache)}\n"
        f"Environment=VRAMD_TOKEN_FILE={_sd_quote(token)}\n"
        f"Environment=HF_HOME={_sd_quote(Path(cfg.models_home).expanduser())}\n" + "\n".join(env_lines) + "\n"
        f"ExecStart={exec_start}\n"
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
