#!/usr/bin/env python3
"""Backend mínimo — o contrato inteiro em três métodos.

Não carrega modelo nenhum: serve para ver o ciclo completo (spawn → load →
generate → unload) sem GPU. Trocar o corpo dos três métodos por um modelo real
é tudo o que separa isto de um backend de produção.

    python worker.py            # fala JSONL por stdin/stdout
"""

from __future__ import annotations

import time
from typing import Any

from vramd.worker import WorkerAdapter, run_worker_loop


class Adapter(WorkerAdapter):
    name = "echo"

    def load(self, **kwargs: Any) -> Any:
        """Carrega o modelo. Devolve o objeto que o ``generate`` vai receber."""
        time.sleep(0.2)  # onde um modelo real leria pesos do disco
        return {"device": kwargs.get("device", "cuda"), "loaded_at": time.time()}

    def generate(self, model: Any, request: dict[str, Any]) -> dict[str, Any]:
        """Corre um job. ``_progress`` e ``_abort`` chegam dentro do request."""
        steps = int(request.get("steps", 5))
        for i in range(1, steps + 1):
            if self.should_abort(request):
                return self.cancelled_response("cancelado a meio")
            self.report_progress(request, i / steps, f"passo {i}/{steps}")
            time.sleep(0.1)
        return {"status": "ok", "output": request.get("prompt", "").upper(), "seconds": steps * 0.1}

    def unload(self, model: Any) -> None:
        """Liberta a VRAM. Idempotente e à prova de exceções."""
        model.clear()


if __name__ == "__main__":
    run_worker_loop(Adapter, backend_name="echo")
