"""Adapter in-process trivial, usado pelo registry de testes.

O fixture ``tests/data/backends.test.yaml`` tem de apontar ``adapter:`` para um
módulo importável que exporte ``Adapter``. Os testes que exercitam o modo
subprocesso nunca o instanciam; os do modo in-process usam-no como duplo.
"""

from __future__ import annotations

from typing import Any

from vramd.adapter import BackendAdapter


class Adapter(BackendAdapter):
    """Não carrega nada e devolve sempre ok — o que se testa é o supervisor."""

    name = "fake"

    def load(self, **kwargs: Any) -> Any:
        return {"loaded": True, "kwargs": dict(kwargs)}

    def generate(self, model: Any, request: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok", "output": request.get("output", ""), "seconds": 0.0}

    def unload(self, model: Any) -> None:
        return None
