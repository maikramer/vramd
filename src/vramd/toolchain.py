"""Descoberta do interpretador de um backend declarado só com ``tool:``.

O caminho canónico para dizer ao ``vramd`` como arrancar um worker é o bloco
``runtime.command`` do descriptor. Este módulo cobre o atalho: projetos que
guardam cada modelo numa pasta com o seu próprio venv podem escrever apenas
``tool: whisper`` e deixar o caminho ser derivado.

O layout esperado é o mais comum em monorepos de ML::

    <raiz>/<Tool>/.venv/bin/python

A raiz vem de ``VRAMD_TOOLS_ROOT`` (vários caminhos separados por ``os.pathsep``).
Sem essa variável **não há descoberta** — e é de propósito: adivinhar caminhos
no disco de quem instala o pacote produz erros piores que um "não sei arrancar
este worker, declara ``runtime.command``".
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_TOOLS_ROOT = "VRAMD_TOOLS_ROOT"


# Nomes de pasta tentados para cada ``tool`` (o mesmo nome, e a forma
# capitalizada que muitos projetos usam para a pasta do componente).
def _camel_title(tool: str) -> str:
    """``text2icon`` → ``Text2Icon``; ``skymap2d`` → ``Skymap2D``.

    O ``str.capitalize()`` só capitaliza a primeira letra — pastas com
    camelCase após um dígito (Text2Icon, Paint3D, Skymap2D…) não eram
    encontradas. Capitaliza o início de cada segmento alfanumérico.
    """
    import re

    return "".join(part.capitalize() if not part.isdigit() else part for part in re.split(r"(\d+)", tool))


def _candidate_dirs(tool: str) -> list[str]:
    """Variações de nome de pasta a tentar para ``tool``."""
    seen: list[str] = []
    for name in (
        tool,
        tool.capitalize(),
        _camel_title(tool),
        tool.upper(),
        tool.replace("_", "-"),
    ):
        if name and name not in seen:
            seen.append(name)
    return seen


def _venv_python(venv: Path) -> str | None:
    """``python`` dentro de um venv, sem resolver symlinks.

    Resolver o symlink partiria o venv: ``<venv>/bin/python`` aponta para o
    interpretador base, mas é o symlink que faz o ``sys.path`` apontar para os
    site-packages do venv.
    """
    scripts = venv / ("Scripts" if os.name == "nt" else "bin")
    for candidate in (scripts / "python", scripts / "python.exe"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def tools_roots() -> list[Path]:
    """Raízes onde procurar venvs de tools (vazio = descoberta desligada)."""
    raw = os.environ.get(ENV_TOOLS_ROOT, "").strip()
    if not raw:
        return []
    return [Path(os.path.expanduser(p)) for p in raw.split(os.pathsep) if p.strip()]


def resolve_tool_python(tool: str, *, roots: list[Path] | None = None) -> str | None:
    """Interpretador do venv de ``tool``, ou ``None`` se não for encontrável.

    Args:
        tool: Nome do backend/tool (ex. ``whisper``).
        roots: Raízes a procurar; se ``None``, usa :func:`tools_roots`.

    Returns:
        Caminho absoluto do ``python``, ou ``None`` — o caller trata isso como
        "declara ``runtime.command``", que é uma mensagem acionável.
    """
    if not tool:
        return None
    for root in roots if roots is not None else tools_roots():
        for name in _candidate_dirs(tool):
            found = _venv_python(root / name / ".venv")
            if found:
                return found
    return None
