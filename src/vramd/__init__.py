"""vramd — controlo de admissão de VRAM para inferência generativa.

Um processo detém a GPU e decide **quem entra**: admite pelo pico real
(pesos + activação + margem), põe em fila com prioridade e afinidade, evicta
por peso+LRU, e corre cada modelo num processo/venv próprio.

Feito para inferência que dura segundos a minutos numa GPU de consumo — não
para throughput de tokens em datacenter.

Três peças:

- **supervisor** (:mod:`vramd.server`, :mod:`vramd.cli`) — fila, admissão, evicção;
- **worker SDK** (:mod:`vramd.worker`) — 3 métodos para embrulhar qualquer modelo;
- **cliente** (:mod:`vramd.client`) — submeter, esperar, cancelar.

E um calibrador (:mod:`vramd.calibrate`) que **mede** o footprint em vez de o
adivinhar: o número que o admit usa vem do driver, não de uma estimativa.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
