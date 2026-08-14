"""Registry de backends — carrega ``data/backends.yaml`` e resolve adapters por lazy import.

O registry é declarativo (YAML) e resolution de adapter é lazy: só importamos
o módulo da tool quando o backend é efetivamente pedido. Isto mantém o vramd
importável sem todas as deps GPU (torch/diffusers/stable-audio) instaladas.

**Camadas de configuração (v2).** O YAML empacotado é a base; por cima dele
sobrepõem-se ficheiros do utilizador, por ordem::

    data/backends.yaml  →  $VRAMD_BACKENDS_FILE  →  ~/.config/vramd/backends.d/*.yaml

A sobreposição é **por chave**, não por entrada: um ficheiro do utilizador com
``{name: motion3d, vram_mib: 5632}`` corrige só esse campo e herda o resto. É o
que permite instalar um descriptor calibrado (`vramd calibrate --out`) sem tocar
no package — e é a razão de o merge existir.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import import_module, resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterator

# Env vars da sobreposição de configuração.
ENV_BACKENDS_FILE = "VRAMD_BACKENDS_FILE"
ENV_BACKENDS_DIR = "VRAMD_BACKENDS_DIR"

# ~/.config/vramd/backends.d — alinhado com a docstring do módulo e o README
# (era ~/.config/ums/backends.d, legado do AiGameKit: quem seguisse a doc
# punha overlays num diretório que nunca era lido).
DEFAULT_BACKENDS_DIR = Path.home() / ".config" / "vramd" / "backends.d"


@dataclass(frozen=True)
class RuntimeSpec:
    """Como arrancar o worker deste backend.

    Um backend do monorepo diz apenas ``monorepo_tool: text3d`` e o comando sai
    do checkout (``Text3D/.venv/bin/python -m text3d serve --ums-worker``). Um
    backend externo declara o ``command`` completo — é isto que torna o vramd
    utilizável fora deste repositório.

    Attributes:
        command: Argv do worker. Suporta ``${env:VAR}`` e ``${monorepo:tool}``.
        monorepo_tool: Açúcar para backends do checkout (deriva ``command``).
        cwd: Diretório de trabalho do worker.
        env: Variáveis de ambiente extra (juntam-se às herdadas).
        load_timeout_sec: Timeout do ``load`` (``None`` = default do pool).
        event_timeout_sec: Timeout entre eventos de progresso.
    """

    command: tuple[str, ...] | None = None
    monorepo_tool: str | None = None
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    load_timeout_sec: float | None = None
    event_timeout_sec: float | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> RuntimeSpec | None:
        """Constrói a partir do bloco ``runtime:`` do YAML (``None`` se vazio)."""
        if not raw:
            return None
        command = raw.get("command")
        if isinstance(command, str):
            command = [command]
        env = raw.get("env") or {}
        if not isinstance(env, Mapping):
            raise ValueError(f"runtime.env deve ser um mapa, recebido {type(env).__name__}")
        return cls(
            command=tuple(str(c) for c in command) if command else None,
            monorepo_tool=raw.get("monorepo_tool"),
            cwd=str(raw["cwd"]) if raw.get("cwd") else None,
            env={str(k): str(v) for k, v in env.items()},
            load_timeout_sec=_opt_float(raw.get("load_timeout_sec")),
            event_timeout_sec=_opt_float(raw.get("event_timeout_sec")),
        )

    def resolve_command(self, *, tool: str | None = None) -> list[str] | None:
        """Argv final, com ``${env:…}``/``${monorepo:…}`` expandidos.

        Args:
            tool: Tool a usar quando nem ``command`` nem ``monorepo_tool`` estão
                definidos (o ``tool:`` do descriptor).

        Returns:
            Argv pronto para ``Popen``, ou ``None`` se não houver como resolver
            (o caller cai no comportamento legado).
        """
        target = self.monorepo_tool or tool
        if not self.command:
            return _monorepo_worker_cmd(target) if target else None
        resolved = [_expand_token(part, tool=target) for part in self.command]
        return None if any(part is None for part in resolved) else [str(p) for p in resolved]

    def to_dict(self) -> dict[str, Any]:
        """Bloco ``runtime:`` serializável (round-trip do ``from_dict``).

        Usado pelo ``vramd calibrate --out`` para preservar command/cwd/env/
        timeouts ao reescrever o descriptor — sem isto o YAML emitido
        regenerava ``monorepo_tool`` e perdia a configuração de arranque.
        """
        out: dict[str, Any] = {}
        if self.command:
            out["command"] = list(self.command)
        if self.monorepo_tool:
            out["monorepo_tool"] = self.monorepo_tool
        if self.cwd:
            out["cwd"] = self.cwd
        if self.env:
            out["env"] = dict(self.env)
        if self.load_timeout_sec is not None:
            out["load_timeout_sec"] = self.load_timeout_sec
        if self.event_timeout_sec is not None:
            out["event_timeout_sec"] = self.event_timeout_sec
        return out

    def resolve_env(self) -> dict[str, str]:
        """Ambiente extra com ``${env:…}`` e ``~`` expandidos."""
        out: dict[str, str] = {}
        for key, value in self.env.items():
            expanded = _expand_token(value, tool=self.monorepo_tool)
            if expanded is not None:
                out[key] = os.path.expanduser(str(expanded))
        return out

    def resolve_cwd(self) -> str | None:
        """``cwd`` com ``~`` expandido (``None`` se não definido)."""
        return os.path.expanduser(self.cwd) if self.cwd else None


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _expand_token(token: str, *, tool: str | None) -> str | None:
    """Expande ``${env:VAR}`` / ``${monorepo:tool}`` num token do argv.

    Devolve ``None`` quando a referência não resolve — o caller trata isso como
    "não sei arrancar este worker", que é melhor que arrancar com um caminho
    literal ``${env:FOO}`` e falhar com um erro incompreensível.
    """
    text = str(token)
    # ${monorepo:python} / ${monorepo} ANTES do branch genérico ${monorepo:*}
    # — senão o primeiro caía em _monorepo_tool_python("python") (venv chamado
    # "python", quase sempre inexistente) e o caso especial nunca corria.
    if text == "${monorepo:python}" or (text == "${monorepo}" and tool):
        return _monorepo_tool_python(tool or "")
    if text.startswith("${env:") and text.endswith("}"):
        return os.environ.get(text[6:-1])
    if text.startswith("${monorepo:") and text.endswith("}"):
        return _monorepo_tool_python(text[11:-1])
    return os.path.expanduser(text) if text.startswith("~") else text


def _monorepo_tool_python(tool: str) -> str | None:
    """Interpretador do venv da tool no checkout (``None`` se não existir)."""
    if not tool:
        return None
    from .toolchain import resolve_tool_python

    return resolve_tool_python(tool)


def _monorepo_worker_cmd(tool: str) -> list[str] | None:
    """Comando canónico do worker de uma tool do monorepo."""
    python = _monorepo_tool_python(tool)
    return None if python is None else [python, "-m", tool, "serve", "--ums-worker"]


@dataclass(frozen=True)
class BackendDescriptor:
    """Descriptor declarativo de um backend (uma ferramenta GPU).

    Attributes:
        name: Identificador canónico (ex: ``text2icon``).
        adapter: Dotted path do módulo que exporta a classe ``BackendAdapter``.
        vram_mib: Estimativa do footprint em VRAM (MiB) quando carregado.
        priority: Prioridade de evicção — valores MAIORES = menos provável de ser
            evicted (backends "pesados" que compensa manter quentes). Tie-break: LRU.
        footprint_key: Chave do registry ``vramd.footprints.FOOTPRINTS`` (ex:
            ``"flux-klein-9b"``). Se definida, o ``vram_mib`` é derivado do footprint
            (mais preciso que o valor estático). Opcional.
        tool: Nome da tool monorepo (ex: ``text3d``, ``paint3d``) para o modo
            subprocess-per-backend. Se definido, o BackendManager despacha jobs
            para um worker persistente no venv da tool em vez de importar o
            adapter in-process. ``None`` = backend in-process (durante migração).
        runtime: Bloco ``runtime:`` v2 (comando/cwd/env/timeouts do worker).
        load_keys: Kwargs do request que influenciam a carga **deste** backend.
            ``None`` = usar a allowlist global do BackendManager.
        shape_keys: Subconjunto que força reload quando muda. ``None`` = global.
        vram: Bloco ``vram:`` medido (`vramd calibrate`), quando existe.
        peak_profile: Bloco ``peak_profile:`` (quant, staged, unload_frees_vram…).
        calibrate_request: Request de geração default para ``vramd calibrate``
            (ex.: prompt + output com a extensão certa). Sem isto, o calibrador
            usa ``{}`` e backends que exigem inputs (mesh_path/output) falham
            com mensagens pouco acionáveis — o default vive no descriptor para
            que ``vramd calibrate <backend>`` funcione sempre.
        calibrate_load_kwargs: Kwargs de load default para ``vramd calibrate``
            (ex.: ``sdnq_preset`` quando o perfil da tool não o resolve).
            Fundidos com o hw-auto (este ganha) e com os explícitos do CLI
            (``--load-kwargs``/``--quant`` ganham a este).
    """

    name: str
    adapter: str
    vram_mib: int
    priority: int
    footprint_key: str | None = None
    tool: str | None = None
    runtime: RuntimeSpec | None = None
    load_keys: frozenset[str] | None = None
    shape_keys: frozenset[str] | None = None
    vram: Mapping[str, Any] = field(default_factory=dict)
    peak_profile: Mapping[str, Any] = field(default_factory=dict)
    calibrate_request: Mapping[str, Any] = field(default_factory=dict)
    calibrate_load_kwargs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def unload_frees_vram(self) -> bool:
        """``False`` quando a calibração provou que evictar não liberta VRAM."""
        return bool(self.peak_profile.get("unload_frees_vram", True))

    def worker_command(self) -> list[str] | None:
        """Argv do worker subprocesso (``None`` = sem forma de arrancar)."""
        if self.runtime is not None:
            return self.runtime.resolve_command(tool=self.tool)
        return _monorepo_worker_cmd(self.tool) if self.tool else None


def _default_yaml_path() -> str:
    """Path canónico do ``backends.yaml`` empacotado (package-data)."""
    return str(resources.files("vramd").joinpath("data", "backends.yaml"))


def descriptor_sources(yaml_path: str | None = None) -> list[str]:
    """Ficheiros a fundir, do mais genérico para o mais específico.

    Args:
        yaml_path: Se dado, é a **única** fonte (usado em testes e por
            ``Registry(yaml_path=…)``).

    Returns:
        Lista de caminhos existentes, por ordem de precedência crescente.
    """
    if yaml_path:
        return [yaml_path]

    sources = [_default_yaml_path()]
    env_file = os.environ.get(ENV_BACKENDS_FILE, "").strip()
    if env_file:
        env_paths = [p.strip() for p in env_file.split(os.pathsep) if p.strip()]
        # Path declarado pelo operador que não existe: quase sempre typo — a
        # calibração escreve ali e seria silenciosamente ignorada (admit/evict
        # com números errados sem ninguém perceber).
        for p in env_paths:
            if not Path(os.path.expanduser(p)).is_file():
                print(
                    f"[vramd] WARNING: {ENV_BACKENDS_FILE}={p} não existe — a ser ignorado.",
                    file=sys.stderr,
                )
        sources.extend(env_paths)

    raw_dir = os.environ.get(ENV_BACKENDS_DIR, "").strip()
    conf_dir = Path(os.path.expanduser(raw_dir)) if raw_dir else DEFAULT_BACKENDS_DIR
    if conf_dir.is_dir():
        sources.extend(sorted(str(p) for p in conf_dir.iterdir() if p.suffix in (".yaml", ".yml")))

    return [s for s in sources if Path(os.path.expanduser(s)).is_file()]


def _read_entries(path: str) -> list[dict[str, Any]]:
    """Lê a lista ``backends:`` de um ficheiro.

    Raises:
        ValueError: YAML malformado (incl. ``yaml.YAMLError`` normalizado —
            antes escapava cru de ``safe_load``) ou sem a chave ``backends``.
    """
    try:
        with open(os.path.expanduser(path), encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"YAML malformado em {path}: {e}") from e

    if not isinstance(data, dict) or "backends" not in data:
        raise ValueError(f"backends.yaml malformado: falta a chave 'backends' ({path})")
    entries = data["backends"]
    if not isinstance(entries, list):
        raise ValueError(f"backends.yaml: 'backends' deve ser uma lista ({path})")
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ValueError(f"backends.yaml: entrada sem 'name' ({path})")
    return entries


def _deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge recursivo: mappings fundem campo a campo; resto é substituído.

    Puro: devolve um dict NOVO (cópia de ``base`` fundida com ``override``) —
    mutar ``base`` inplace escrevia nos dicts aninhados PARTILHADOS com a
    camada de origem (``dict(entry)`` é shallow), corrompendo entradas que o
    caller pudesse reutilizar. O ``dict.update`` shallow original, esse,
    destruía blocos aninhados — um overlay com ``runtime: {cwd: X}`` apagava o
    ``runtime.command`` da base.
    """
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def merge_entries(sources: Iterable[list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Funde listas de entradas por ``name``, com sobreposição profunda.

    A última fonte ganha campo a campo — um override parcial (só ``vram_mib``)
    herda tudo o resto da camada de baixo, incluindo blocos aninhados
    (``runtime``, ``vram``, ``peak_profile``) que agora fundem recursivamente.
    """
    merged: dict[str, dict[str, Any]] = {}
    for entries in sources:
        for entry in entries:
            name = str(entry["name"])
            if name in merged:
                merged[name] = _deep_update(merged[name], entry)
            else:
                merged[name] = dict(entry)
    return merged


def _to_descriptor(entry: Mapping[str, Any], *, source: str) -> BackendDescriptor:
    """Converte uma entrada fundida em :class:`BackendDescriptor`."""
    name = str(entry["name"])
    missing = [k for k in ("adapter", "vram_mib") if entry.get(k) is None]
    if missing:
        raise ValueError(f"backend {name!r}: falta {', '.join(missing)} (fontes: {source})")
    try:
        vram_mib = int(entry["vram_mib"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"backend {name!r}: vram_mib={entry['vram_mib']!r} não é inteiro") from e
    if vram_mib < 0:
        raise ValueError(f"backend {name!r}: vram_mib negativo ({vram_mib})")

    load_keys = entry.get("load_keys")
    shape_keys = entry.get("shape_keys")
    return BackendDescriptor(
        name=name,
        adapter=str(entry["adapter"]),
        vram_mib=vram_mib,
        priority=int(entry.get("priority", 0)),
        footprint_key=entry.get("footprint_key"),
        tool=entry.get("tool"),
        runtime=RuntimeSpec.from_dict(entry.get("runtime")),
        load_keys=frozenset(str(k) for k in load_keys) if load_keys else None,
        shape_keys=frozenset(str(k) for k in shape_keys) if shape_keys else None,
        vram=dict(entry.get("vram") or {}),
        peak_profile=dict(entry.get("peak_profile") or {}),
        calibrate_request=dict(entry.get("calibrate_request") or {}),
        calibrate_load_kwargs=dict(entry.get("calibrate_load_kwargs") or {}),
    )


def load_descriptors(yaml_path: str | None = None) -> dict[str, BackendDescriptor]:
    """Carrega descriptors, fundindo as camadas de configuração.

    Uma camada do utilizador corrupta (overlay escrito a meio, typo de YAML) é
    **saltada com warning** — não derruba o arranque do supervisor. A fonte
    empacotada (primeira) e um ``yaml_path`` explícito continuam a ser fatais.

    Raises:
        FileNotFoundError: ``yaml_path`` explícito não existe.
        ValueError: YAML malformado (fonte base/explicita), ou entrada fundida
            sem campos obrigatórios.
    """
    sources = descriptor_sources(yaml_path)
    if yaml_path and not sources:
        # Path explícito inexistente: erro do caller, não silenciar.
        raise FileNotFoundError(yaml_path)

    layers: list[list[dict[str, Any]]] = []
    for idx, path in enumerate(sources):
        try:
            layers.append(_read_entries(path))
        except (ValueError, OSError) as e:
            if idx == 0 or yaml_path:
                raise
            # Overlay do utilizador envenenado: saltar a camada em vez de
            # brickar o arranque — os descriptors da base mantêm-se usáveis.
            print(f"[vramd] WARNING: source de backends ignorada ({e})", file=sys.stderr)
    merged = merge_entries(layers)
    label = ", ".join(sources)
    return {name: _to_descriptor(entry, source=label) for name, entry in merged.items()}


class Registry:
    """Registry de backends com resolução lazy de adapters.

    Mantém os ``BackendDescriptor`` (data estática do YAML) e instancia a classe
    ``BackendAdapter`` de cada backend sob procura (lazy import do módulo da tool).
    """

    def __init__(
        self, descriptors: dict[str, BackendDescriptor] | None = None, *, yaml_path: str | None = None
    ) -> None:
        self._descriptors = descriptors if descriptors is not None else load_descriptors(yaml_path)
        self._adapter_instances: dict[str, object] = {}
        # O server despacha em thread-per-connection: dois generates simultâneos
        # no mesmo backend faziam lazy-import duplo e corrida no cache.
        self._adapter_lock = threading.Lock()

    @property
    def names(self) -> list[str]:
        """Nomes de todos os backends registados."""
        return list(self._descriptors)

    def descriptor(self, name: str) -> BackendDescriptor:
        """Retorna o descriptor de um backend.

        Raises:
            KeyError: Backend não registado.
        """
        if name not in self._descriptors:
            raise KeyError(f"Backend desconhecido: {name!r}. Disponíveis: {sorted(self._descriptors)}")
        return self._descriptors[name]

    def has(self, name: str) -> bool:
        """True se o backend ``name`` está registado."""
        return name in self._descriptors

    def adapter(self, name: str) -> object:
        """Retorna a instância do ``BackendAdapter`` para ``name`` (lazy import + cache).

        O módulo do adapter só é importado na primeira invocação. A instância é
        cacheada — adapters são stateless quanto ao modelo (o modelo vive no
        BackendManager, não no adapter).

        Raises:
            KeyError: Backend não registado.
            ImportError: Módulo do adapter não encontrável (deps da tool em falta).
        """
        with self._adapter_lock:
            if name in self._adapter_instances:
                return self._adapter_instances[name]

            desc = self.descriptor(name)
            module = import_module(desc.adapter)
            # Convenção: cada módulo adapter exporta uma classe sem argumentos que
            # implementa o contrato (load/generate/unload). Instanciamos sem estado.
            cls = getattr(module, "Adapter", None)
            if cls is None:
                raise ImportError(f"Adapter {desc.adapter} não exporta a classe 'Adapter'")
            instance = cls()
            self._adapter_instances[name] = instance
            return instance

    def __iter__(self) -> Iterator[BackendDescriptor]:
        return iter(self._descriptors.values())

    def __len__(self) -> int:
        return len(self._descriptors)
