"""Lint da configuração: descritores de backends e ``hooks.yaml``.

O registry é tolerante de propósito — um overlay corrompido é saltado com um
warning para não impedir o arranque. O reverso é que um typo (``vram_mb``,
``prority``) é uma chave desconhecida ignorada em silêncio, e um
``runtime.command`` que aponta para um venv apagado só rebenta no primeiro job.

``vramd validate`` lê as mesmas camadas que o supervisor e diz, antes de
arrancar, o que está errado e onde. Sem GPU, sem importar adapters, sem
arrancar workers.
"""

from __future__ import annotations

import difflib
import importlib.util
import math
import os
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import registry as R

LEVEL_ERROR = "error"
LEVEL_WARNING = "warning"
LEVEL_INFO = "info"
_LEVEL_RANK = {LEVEL_ERROR: 0, LEVEL_WARNING: 1, LEVEL_INFO: 2}

KNOWN_BACKEND_KEYS = frozenset(
    {
        "name",
        "adapter",
        "vram_mib",
        "priority",
        "footprint_key",
        "tool",
        "runtime",
        "load_keys",
        "shape_keys",
        "vram",
        "peak_profile",
        "calibrate_request",
        "calibrate_load_kwargs",
    }
)
KNOWN_RUNTIME_KEYS = frozenset(
    {"command", "cwd", "monorepo_tool", "env", "load_timeout_sec", "event_timeout_sec", "max_jobs"}
)
KNOWN_TOP_LEVEL_KEYS = frozenset({"version", "backends"})


@dataclass(frozen=True)
class Finding:
    """Um problema (ou nota) encontrado na configuração."""

    level: str
    message: str
    backend: str | None = None
    source: str | None = None
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ValidationReport:
    """Resultado do lint: fontes lidas, backends fundidos e achados."""

    sources: list[str]
    backends: list[str]
    findings: list[Finding]

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == LEVEL_ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == LEVEL_WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "sources": self.sources,
            "backends": self.backends,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "findings": [f.to_dict() for f in self.findings],
        }


def _suggest(key: str, known: frozenset[str]) -> str | None:
    match = difflib.get_close_matches(key, sorted(known), n=1, cutoff=0.6)
    return f"querias dizer {match[0]!r}?" if match else None


def _is_int_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _positive_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def _check_unknown_keys(
    entry: Mapping[str, Any], *, name: str, source: str, findings: list[Finding], check_runtime: bool = True
) -> None:
    for key in entry:
        if str(key) not in KNOWN_BACKEND_KEYS:
            findings.append(
                Finding(
                    LEVEL_WARNING,
                    f"chave desconhecida {key!r} — é ignorada pelo supervisor",
                    backend=name,
                    source=source,
                    hint=_suggest(str(key), KNOWN_BACKEND_KEYS),
                )
            )
    runtime = entry.get("runtime")
    if check_runtime and isinstance(runtime, Mapping):
        for key in runtime:
            if str(key) not in KNOWN_RUNTIME_KEYS:
                findings.append(
                    Finding(
                        LEVEL_WARNING,
                        f"chave desconhecida runtime.{key} — é ignorada pelo supervisor",
                        backend=name,
                        source=source,
                        hint=_suggest(str(key), KNOWN_RUNTIME_KEYS),
                    )
                )


def _read_layer(path: str, findings: list[Finding]) -> list[dict[str, Any]] | None:
    """Lê uma camada com o mesmo parser do registry; erros viram achados."""
    try:
        entries = R._read_entries(path)
    except (ValueError, OSError) as e:
        findings.append(
            Finding(
                LEVEL_ERROR,
                str(e),
                source=path,
                hint="o supervisor salta esta camada inteira — os overrides dela não têm efeito",
            )
        )
        return None
    try:
        import yaml

        with open(os.path.expanduser(path), encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
    except Exception:
        doc = None
    if isinstance(doc, Mapping):
        for key in doc:
            if str(key) not in KNOWN_TOP_LEVEL_KEYS:
                findings.append(
                    Finding(
                        LEVEL_WARNING,
                        f"chave de topo desconhecida {key!r}",
                        source=path,
                        hint=_suggest(str(key), KNOWN_TOP_LEVEL_KEYS),
                    )
                )
    seen: set[str] = set()
    for entry in entries:
        name = str(entry["name"])
        if name in seen:
            findings.append(
                Finding(
                    LEVEL_WARNING,
                    "backend declarado duas vezes no mesmo ficheiro — a segunda entrada sobrepõe a primeira",
                    backend=name,
                    source=path,
                )
            )
        seen.add(name)
        _check_unknown_keys(entry, name=name, source=path, findings=findings)
    return entries


def _command_problem(argv: list[str], cwd: str | None) -> str | None:
    """Razão pela qual ``argv[0]`` não arranca (``None`` = parece executável)."""
    exe = argv[0]
    has_sep = os.sep in exe or (os.altsep is not None and os.altsep in exe)
    if has_sep:
        candidate = Path(exe)
        if not candidate.is_absolute() and cwd:
            candidate = Path(cwd) / candidate
        if not candidate.exists():
            return f"executável não existe: {candidate}"
        if not candidate.is_file():
            return f"não é um ficheiro: {candidate}"
        if os.name != "nt" and not os.access(candidate, os.X_OK):
            return f"sem permissão de execução: {candidate}"
        return None
    if shutil.which(exe) is None:
        return f"{exe!r} não está no PATH"
    return None


def _check_runtime(entry: Mapping[str, Any], *, name: str, source: str, findings: list[Finding]) -> None:
    raw_runtime = entry.get("runtime")
    tool = entry.get("tool")
    if raw_runtime is not None and not isinstance(raw_runtime, Mapping):
        findings.append(Finding(LEVEL_ERROR, "runtime deve ser um mapa", backend=name, source=source))
        return

    if isinstance(raw_runtime, Mapping):
        for key in ("load_timeout_sec", "event_timeout_sec"):
            value = raw_runtime.get(key)
            if value is not None and not _positive_number(value):
                findings.append(
                    Finding(LEVEL_ERROR, f"runtime.{key}={value!r} deve ser um número > 0", backend=name, source=source)
                )
        max_jobs = raw_runtime.get("max_jobs")
        if max_jobs is not None and (not _is_int_like(max_jobs) or int(max_jobs) < 0):
            findings.append(
                Finding(
                    LEVEL_ERROR,
                    f"runtime.max_jobs={max_jobs!r} deve ser um inteiro ≥ 0 (0 = nunca reciclar)",
                    backend=name,
                    source=source,
                )
            )
        env = raw_runtime.get("env")
        if env is not None and not isinstance(env, Mapping):
            findings.append(Finding(LEVEL_ERROR, "runtime.env deve ser um mapa", backend=name, source=source))
            return
        command = raw_runtime.get("command")
        if command is not None and (isinstance(command, str) or not isinstance(command, (list, tuple))):
            findings.append(
                Finding(
                    LEVEL_ERROR,
                    "runtime.command deve ser uma lista (argv), não uma string",
                    backend=name,
                    source=source,
                    hint='ex.: command: ["/opt/venv/bin/python", "-m", "pkg.worker"]',
                )
            )
            return

    try:
        spec = R.RuntimeSpec.from_dict(raw_runtime if isinstance(raw_runtime, Mapping) else None)
    except (TypeError, ValueError) as e:
        findings.append(Finding(LEVEL_ERROR, f"bloco runtime inválido: {e}", backend=name, source=source))
        return

    if spec is None and not tool:
        # Backend in-process: o adapter tem de ser importável NO venv do supervisor.
        adapter = str(entry.get("adapter") or "")
        top = adapter.split(".", 1)[0]
        try:
            importable = bool(top) and importlib.util.find_spec(top) is not None
        except (ImportError, ValueError):
            importable = False
        if adapter and not importable:
            findings.append(
                Finding(
                    LEVEL_WARNING,
                    f"sem runtime/tool e o adapter {adapter!r} não é importável neste ambiente",
                    backend=name,
                    source=source,
                    hint="declara runtime.command para correr o modelo no seu próprio venv",
                )
            )
        return

    cwd = spec.resolve_cwd() if spec is not None else None
    if cwd and not Path(cwd).is_dir():
        findings.append(Finding(LEVEL_ERROR, f"runtime.cwd não existe: {cwd}", backend=name, source=source))
        cwd = None

    argv = spec.resolve_command(tool=str(tool) if tool else None) if spec is not None else None
    if argv is None and spec is None and tool:
        argv = R._monorepo_worker_cmd(str(tool))
    if argv is None:
        unresolved = [
            str(part)
            for part in ((spec.command if spec is not None else None) or ())
            if R._expand_token(str(part), tool=str(tool) if tool else None) is None
        ]
        if unresolved:
            message = f"não há como arrancar o worker (por resolver: {', '.join(unresolved)})"
            hint = "define a variável de ambiente referida ou um runtime.command absoluto"
        else:
            # Só ``tool:`` (ou ``monorepo_tool``): o venv da tool não foi encontrado.
            target = (spec.monorepo_tool if spec is not None else None) or tool
            message = f"não há como arrancar o worker — venv da tool {target!r} não encontrado"
            hint = "aponta VRAMD_TOOLS_ROOT para o checkout das tools ou declara runtime.command"
        findings.append(Finding(LEVEL_ERROR, message, backend=name, source=source, hint=hint))
        return
    if not argv:
        findings.append(Finding(LEVEL_ERROR, "runtime.command vazio", backend=name, source=source))
        return
    problem = _command_problem(argv, cwd)
    if problem:
        findings.append(
            Finding(
                LEVEL_ERROR,
                f"worker não arranca — {problem}",
                backend=name,
                source=source,
                hint="o primeiro job deste backend falharia no spawn",
            )
        )


def _check_merged(
    name: str,
    entry: Mapping[str, Any],
    *,
    source: str,
    gpu_total_mib: int | None,
    findings: list[Finding],
) -> None:
    missing = [k for k in ("adapter", "vram_mib") if entry.get(k) is None]
    if missing:
        findings.append(
            Finding(
                LEVEL_ERROR,
                f"falta {', '.join(missing)}",
                backend=name,
                source=source,
                hint="um overlay parcial só funciona se uma camada de base declarar o backend completo",
            )
        )
    vram_mib = entry.get("vram_mib")
    if vram_mib is not None:
        if not _is_int_like(vram_mib):
            findings.append(Finding(LEVEL_ERROR, f"vram_mib={vram_mib!r} não é inteiro", backend=name, source=source))
        elif int(vram_mib) < 0:
            findings.append(Finding(LEVEL_ERROR, f"vram_mib negativo ({vram_mib})", backend=name, source=source))
        elif gpu_total_mib and int(vram_mib) > gpu_total_mib:
            findings.append(
                Finding(
                    LEVEL_WARNING,
                    f"vram_mib={int(vram_mib)} excede a VRAM total da GPU ({gpu_total_mib} MiB) — nunca será admitido",
                    backend=name,
                    source=source,
                    hint="mede com `vramd calibrate` ou usa um preset de quantização",
                )
            )
    priority = entry.get("priority")
    if priority is not None and not _is_int_like(priority):
        findings.append(Finding(LEVEL_ERROR, f"priority={priority!r} não é inteiro", backend=name, source=source))

    for key in ("load_keys", "shape_keys"):
        value = entry.get(key)
        if value is not None and (isinstance(value, (str, Mapping)) or not isinstance(value, (list, tuple, set))):
            findings.append(Finding(LEVEL_ERROR, f"{key} deve ser uma lista de nomes", backend=name, source=source))
    load_keys, shape_keys = entry.get("load_keys"), entry.get("shape_keys")
    if isinstance(load_keys, (list, tuple, set)) and isinstance(shape_keys, (list, tuple, set)):
        extra = sorted({str(k) for k in shape_keys} - {str(k) for k in load_keys})
        if extra:
            findings.append(
                Finding(
                    LEVEL_WARNING,
                    f"shape_keys fora de load_keys: {', '.join(extra)} — nunca chegam ao load, "
                    "logo nunca forçam reload",
                    backend=name,
                    source=source,
                )
            )

    for key in ("vram", "peak_profile", "calibrate_request", "calibrate_load_kwargs"):
        value = entry.get(key)
        if value is not None and not isinstance(value, Mapping):
            findings.append(Finding(LEVEL_ERROR, f"{key} deve ser um mapa", backend=name, source=source))

    vram = entry.get("vram")
    if isinstance(vram, Mapping):
        peak = vram.get("peak_mib")
        if peak is not None and _is_int_like(peak) and gpu_total_mib and int(peak) > gpu_total_mib:
            findings.append(
                Finding(
                    LEVEL_WARNING,
                    f"vram.peak_mib={int(peak)} medido excede a VRAM total ({gpu_total_mib} MiB)",
                    backend=name,
                    source=source,
                    hint="a calibração foi feita noutra GPU? recalibra nesta máquina",
                )
            )

    _check_runtime(entry, name=name, source=source, findings=findings)


def _check_hooks(hooks_path: Path | str | None, findings: list[Finding]) -> None:
    from . import hooks as H

    raw = hooks_path or os.environ.get(H.ENV_HOOKS_FILE, "") or H.DEFAULT_HOOKS_PATH
    path = Path(os.path.expanduser(str(raw)))
    if not path.is_file():
        if hooks_path is not None:
            findings.append(Finding(LEVEL_ERROR, "hooks.yaml não existe", source=str(path)))
        return
    try:
        specs = H.load_hooks(path)
    except (ValueError, OSError) as e:
        findings.append(
            Finding(
                LEVEL_ERROR,
                str(e),
                source=str(path),
                hint="o supervisor recusa arrancar com hooks malformados",
            )
        )
        return
    for spec in specs:
        problem = _command_problem(list(spec.command), None)
        if problem:
            findings.append(
                Finding(
                    LEVEL_WARNING,
                    f"hook {spec.name!r}: {problem}",
                    source=str(path),
                    hint="o hook falharia em silêncio a cada evento",
                )
            )


def validate_config(
    yaml_path: str | None = None,
    *,
    hooks_path: Path | str | None = None,
    gpu_total_mib: int | None = None,
    check_hooks: bool = True,
) -> ValidationReport:
    """Valida as camadas de descritores (e os hooks) sem arrancar nada.

    Args:
        yaml_path: Se dado, valida só este ficheiro (em vez das camadas activas).
        hooks_path: ``hooks.yaml`` explícito (default: o que o supervisor leria).
        gpu_total_mib: VRAM total da GPU, para avisar de footprints impossíveis.
        check_hooks: ``False`` salta o ``hooks.yaml``.

    Returns:
        Relatório com os achados ordenados por gravidade.
    """
    findings: list[Finding] = []
    if yaml_path and not Path(os.path.expanduser(yaml_path)).is_file():
        findings.append(Finding(LEVEL_ERROR, "ficheiro não existe", source=yaml_path))
        return ValidationReport(sources=[], backends=[], findings=findings)

    sources = R.descriptor_sources(yaml_path)
    env_file = os.environ.get(R.ENV_BACKENDS_FILE, "").strip()
    if not yaml_path and env_file:
        for raw in (p.strip() for p in env_file.split(os.pathsep)):
            if raw and not Path(os.path.expanduser(raw)).is_file():
                findings.append(
                    Finding(
                        LEVEL_ERROR,
                        f"{R.ENV_BACKENDS_FILE} aponta para um ficheiro que não existe",
                        source=raw,
                        hint="o supervisor ignora-o e arranca sem esses backends",
                    )
                )

    layers: list[list[dict[str, Any]]] = []
    origin: dict[str, str] = {}
    for path in sources:
        entries = _read_layer(path, findings)
        if entries is None:
            continue
        layers.append(entries)
        for entry in entries:
            origin[str(entry["name"])] = path  # última camada que tocou no backend

    merged = R.merge_entries(layers)
    if not merged:
        findings.append(
            Finding(
                LEVEL_WARNING,
                "nenhum backend configurado",
                hint=f"define {R.ENV_BACKENDS_FILE} ou cria ficheiros em {R.DEFAULT_BACKENDS_DIR}",
            )
        )
    for name in sorted(merged):
        _check_merged(
            name,
            merged[name],
            source=origin.get(name, ", ".join(sources)),
            gpu_total_mib=gpu_total_mib,
            findings=findings,
        )

    if check_hooks:
        _check_hooks(hooks_path, findings)

    findings.sort(key=lambda f: (_LEVEL_RANK.get(f.level, 9), f.backend or "", f.message))
    return ValidationReport(sources=list(sources), backends=sorted(merged), findings=findings)
