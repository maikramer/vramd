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

import pytest

FIXTURE = pathlib.Path(__file__).parent / "data" / "backends.test.yaml"


@pytest.fixture(autouse=True)
def _test_registry(monkeypatch):
    """Aponta o registry para o fixture em todos os testes."""
    monkeypatch.setenv("VRAMD_BACKENDS_FILE", str(FIXTURE))
    # Sem isto, uma pasta ~/.config/vramd/backends.d na máquina de quem corre
    # os testes alteraria os números e as falhas seriam impossíveis de explicar.
    monkeypatch.setenv("VRAMD_BACKENDS_DIR", str(pathlib.Path(os.devnull).parent / "vramd-sem-overlay"))
