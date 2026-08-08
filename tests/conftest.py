"""Fixtures partilhados dos testes.

O registry empacotado (``vramd/data/backends.yaml``) é um **exemplo genérico**.
Os testes precisam de um registry realista — nomes, footprints e perfis de pico
de um caso real — para verificar a matemática de admissão contra números que
foram medidos em hardware. Esse registry vive em ``tests/data/backends.test.yaml``
e entra por ``VRAMD_BACKENDS_FILE``, que é o mesmo mecanismo que um utilizador
usa para instalar o seu.
"""

from __future__ import annotations

import os
import pathlib
import time

import pytest

FIXTURE = pathlib.Path(__file__).parent / "data" / "backends.test.yaml"


@pytest.fixture(autouse=True)
def _test_registry(monkeypatch):
    """Aponta o registry para o fixture em todos os testes."""
    monkeypatch.setenv("VRAMD_BACKENDS_FILE", str(FIXTURE))
    # Sem isto, uma pasta ~/.config/vramd/backends.d na máquina de quem corre
    # os testes alteraria os números e as falhas seriam impossíveis de explicar.
    monkeypatch.setenv("VRAMD_BACKENDS_DIR", str(pathlib.Path(os.devnull).parent / "vramd-sem-overlay"))


# ``time.monotonic()`` no Linux é o tempo desde o arranque. Vários testes
# fabricam "este backend está idle há 200 s" subtraindo a esse valor — e num
# runner de CI acabado de criar (uptime ~30 s) o resultado é **negativo**, que o
# ``idle_candidates`` descarta por tratar ``last_used > 0`` como "nunca usado".
# Resultado: quatro testes que passavam na máquina de quem os escreveu (uptime
# de dias) e falhavam em CI. O piso abaixo torna-os independentes do uptime;
# como é um deslocamento constante, nenhum delta medido muda.
_UPTIME_FLOOR_SEC = 86_400.0


@pytest.fixture(autouse=True)
def _monotonic_uptime_floor(monkeypatch):
    """Garante ``time.monotonic() >= 1 dia``, venha a máquina de onde vier."""
    real = time.monotonic
    offset = _UPTIME_FLOOR_SEC - real()
    if offset > 0:
        monkeypatch.setattr(time, "monotonic", lambda: real() + offset)
