"""Kwargs de load vindos do ``hw-auto`` da própria tool.

Calibrar com kwargs escritos à mão mede um caminho que a produção não usa. Os
adapters **não aplicam hw-auto sozinhos** — em produção é o CLI da tool que
injeta ``memory_efficient``/``sdnq_preset`` no payload
(``with_ums_peak_opts``). Quem chama o worker diretamente, como o calibrador,
tem de fazer o mesmo.

Foi exatamente isto que fez o paint3d falhar na primeira calibração: sem
``memory_efficient`` o adapter carregou em precisão cheia com 6 vistas e OOMou
numa placa onde a produção corre sem problemas.

O perfil é lido **no venv da tool** (subprocesso), porque é lá que o módulo
``<tool>.hardware`` e o torch existem — o supervisor não os importa.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

# Campos do perfil que são kwargs de load; o resto (nome do perfil, VRAM total,
# limites de resolução) é informativo e não entra no ``adapter.load``.
_LOAD_FIELDS = (
    "memory_efficient",
    "sdnq_preset",
    "quant_preset",
    "model_id",
    "model",
    "offload_text_encoder",
    "cpu_offload",
    "allow_group_offload",
    "validation_steps",
    "half",
    "chunked_vae",
    "max_views",
    "view_resolution",
    "render_size",
    "texture_size",
    "volume_decoder",
)

# Renomes: o perfil e o adapter nem sempre usam o mesmo nome.
_RENAMES = {
    "cpu_offload": "cpu_offload",
    "half": "half_precision",
    "max_views": "max_num_view",
}

_PROBE = """
import dataclasses, importlib, json, sys

tool = sys.argv[1]
try:
    hw = importlib.import_module(f"{tool}.hardware")
except Exception as exc:
    print(json.dumps({"error": f"sem modulo hardware: {exc}"}))
    raise SystemExit(0)

fn = getattr(hw, "detect_hardware_profile", None) or getattr(hw, "detect_profile", None)
if fn is None:
    print(json.dumps({"error": "sem detect_hardware_profile"}))
    raise SystemExit(0)

profile = fn()
data = dataclasses.asdict(profile) if dataclasses.is_dataclass(profile) else dict(vars(profile))
print(json.dumps({"profile": {k: v for k, v in data.items() if not k.startswith("_")}}, default=str))
"""


def probe_tool_profile(tool: str, python: str, *, timeout_sec: float = 300.0) -> dict[str, Any]:
    """Corre o ``hw-auto`` da tool no venv dela e devolve o perfil.

    Args:
        tool: Nome da tool (``text3d``, ``paint3d``…).
        python: Interpretador do venv da tool.
        timeout_sec: Limite — importar torch e sondar a GPU não é instantâneo.

    Returns:
        Dict do perfil, ou ``{"error": …}`` quando a tool não expõe hw-auto.
    """
    try:
        out = subprocess.run(
            [python, "-c", _PROBE, tool],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"probe falhou: {exc}"}

    line = next((ln for ln in reversed((out.stdout or "").splitlines()) if ln.strip().startswith("{")), "")
    if not line:
        return {"error": f"probe sem saída utilizável (rc={out.returncode})"}
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        return {"error": f"probe devolveu JSON inválido: {exc}"}
    return payload.get("profile", payload)


def load_kwargs_from_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Extrai do perfil só o que é kwarg de ``adapter.load``.

    Valores ``None`` são omitidos: no contrato dos adapters, "chave ausente"
    e "chave a None" não são equivalentes — o paint3d, por exemplo, lê
    ``memory_efficient`` só se a chave existir.
    """
    kwargs: dict[str, Any] = {}
    for field in _LOAD_FIELDS:
        if field not in profile:
            continue
        value = profile[field]
        if value is None:
            continue
        kwargs[_RENAMES.get(field, field)] = value
    return kwargs


def resolve_hw_auto_kwargs(tool: str | None, *, python: str | None = None) -> tuple[dict[str, Any], str | None]:
    """``(kwargs, erro)`` do hw-auto da tool.

    Args:
        tool: Tool do backend (``None`` para backends externos → sem hw-auto).
        python: Interpretador; se ``None``, descobre o venv da tool no checkout.

    Returns:
        Kwargs prontos para juntar aos explícitos, e uma mensagem de erro
        legível quando não foi possível obter o perfil (o caller decide se
        continua com o que tem).
    """
    if not tool:
        return {}, "backend sem tool: hw-auto não aplicável"

    interpreter = python
    if interpreter is None:
        from ..toolchain import resolve_tool_python

        interpreter = resolve_tool_python(tool)
    if not interpreter:
        return {}, f"venv da tool {tool!r} não encontrado"

    profile = probe_tool_profile(tool, interpreter)
    if "error" in profile:
        return {}, str(profile["error"])
    return load_kwargs_from_profile(profile), None
