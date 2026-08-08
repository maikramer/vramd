"""SDK do worker — o lado do modelo.

Embrulhar um modelo qualquer são três métodos::

    from vramd.worker import WorkerAdapter, run_worker_loop

    class Adapter(WorkerAdapter):
        name = "meu-modelo"

        def load(self, **kw): ...          # devolve o objeto do modelo
        def generate(self, model, req): ...  # corre um job
        def unload(self, model): ...         # liberta a VRAM

    if __name__ == "__main__":
        run_worker_loop(Adapter, backend_name="meu-modelo")

O loop fala JSONL por stdin/stdout com o supervisor: ``load``/``generate``/
``unload``/``abort``/``ping``/``shutdown`` ↔ ``ready``/``progress``/``done``/
``error``/``unloaded``/``pong``.
"""

from __future__ import annotations

from .adapter import WorkerAdapter
from .protocol import (
    CMD_ABORT,
    CMD_GENERATE,
    CMD_LOAD,
    CMD_PING,
    CMD_SHUTDOWN,
    CMD_UNLOAD,
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_PROGRESS,
    EVENT_READY,
    EVENT_UNLOADED,
)
from .serve import run_worker_loop

__all__ = [
    "CMD_ABORT",
    "CMD_GENERATE",
    "CMD_LOAD",
    "CMD_PING",
    "CMD_SHUTDOWN",
    "CMD_UNLOAD",
    "EVENT_DONE",
    "EVENT_ERROR",
    "EVENT_PROGRESS",
    "EVENT_READY",
    "EVENT_UNLOADED",
    "WorkerAdapter",
    "run_worker_loop",
]
