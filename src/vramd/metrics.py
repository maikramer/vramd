"""Exposição de métricas no formato de texto do Prometheus.

Função pura: recebe as respostas ``status``/``stats`` do supervisor (dicts) e
devolve o texto. Sem servidor HTTP e sem dependências — ``vramd metrics`` imprime
ou escreve um ficheiro ``.prom`` para o *textfile collector* do node_exporter,
que é como uma máquina com uma GPU de consumo costuma ser monitorizada.

Campos ausentes são omitidos em vez de exportados como 0: um ``0`` inventado é
indistinguível de uma medição real num gráfico.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from . import __version__

_PREFIX = "vramd_"


def _escape_label(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _number(value: Any) -> float | None:
    """Converte para float finito; ``None`` se não for um número utilizável."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _format_value(value: float) -> str:
    return str(int(value)) if value == int(value) and abs(value) < 1e15 else repr(value)


class _Exposition:
    """Acumula amostras agrupadas por métrica (HELP/TYPE emitidos uma vez)."""

    def __init__(self) -> None:
        self._order: list[str] = []
        self._meta: dict[str, tuple[str, str]] = {}
        self._samples: dict[str, list[str]] = {}

    def add(
        self,
        name: str,
        value: Any,
        *,
        help_text: str,
        kind: str = "gauge",
        labels: Mapping[str, object] | None = None,
    ) -> None:
        number = _number(value)
        if number is None:
            return
        full = _PREFIX + name
        if full not in self._meta:
            self._meta[full] = (help_text, kind)
            self._samples[full] = []
            self._order.append(full)
        label_txt = ""
        if labels:
            label_txt = "{" + ",".join(f'{k}="{_escape_label(v)}"' for k, v in labels.items()) + "}"
        self._samples[full].append(f"{full}{label_txt} {_format_value(number)}")

    def render(self) -> str:
        lines: list[str] = []
        for full in self._order:
            help_text, kind = self._meta[full]
            lines.append(f"# HELP {full} {help_text}")
            lines.append(f"# TYPE {full} {kind}")
            lines.extend(self._samples[full])
        return "\n".join(lines) + "\n"


def _backend_rows(status: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for row in status.get("backends") or []:
        if isinstance(row, Mapping) and row.get("name"):
            yield row


def render_prometheus(
    status: Mapping[str, Any] | None,
    stats: Mapping[str, Any] | None = None,
    *,
    gpu: Mapping[str, Any] | None = None,
) -> str:
    """Texto de exposição Prometheus.

    Args:
        status: Resposta de ``{"cmd": "status"}`` (``None`` = supervisor em baixo).
        stats: Resposta de ``{"cmd": "stats"}`` (contadores por backend); opcional.
        gpu: ``{"free_mib": …, "total_mib": …}`` lido localmente; opcional.

    Returns:
        Texto terminado em newline. Com o supervisor em baixo só sai ``vramd_up 0``
        (e a GPU, se dada) — séries antigas não ficam congeladas no último valor.
    """
    exp = _Exposition()
    exp.add("up", 1 if status else 0, help_text="1 se o supervisor respondeu ao status.")
    if gpu:
        exp.add("gpu_free_mib", gpu.get("free_mib"), help_text="VRAM livre na GPU (MiB).")
        exp.add("gpu_total_mib", gpu.get("total_mib"), help_text="VRAM total da GPU (MiB).")
    if not status:
        return exp.render()

    exp.add(
        "info",
        1,
        help_text="Metadados do supervisor.",
        labels={
            "version": __version__,
            "transport": status.get("transport") or "?",
            "platform": status.get("platform") or "?",
        },
    )
    exp.add(
        "requests_served_total", status.get("requests_served"), help_text="Jobs servidos com sucesso.", kind="counter"
    )
    exp.add("max_inflight", status.get("max_inflight"), help_text="Gerações simultâneas permitidas.")

    queue = status.get("queue") or {}
    exp.add("queue_depth", queue.get("queue_depth"), help_text="Jobs à espera na fila.")
    exp.add("inflight", queue.get("inflight"), help_text="Jobs em execução.")
    exp.add("queue_eta_seconds", status.get("eta_sec"), help_text="ETA estimado para esvaziar a fila.")

    qm = status.get("queue_metrics") or {}
    for key, name, help_text in (
        ("enqueued", "jobs_enqueued_total", "Jobs enfileirados."),
        ("completed", "jobs_completed_total", "Jobs terminados (done ou failed)."),
        ("cancelled", "jobs_cancelled_total", "Jobs cancelados."),
        ("queue_full_count", "queue_full_total", "Pedidos recusados por fila cheia."),
        ("affinity_cuts_total", "affinity_cuts_total", "Vezes que a afinidade passou à frente da cabeça da fila."),
    ):
        exp.add(name, qm.get(key), help_text=help_text, kind="counter")
    exp.add("queue_max_depth_seen", qm.get("max_depth_seen"), help_text="Profundidade máxima observada.")
    for quantile, key in (("0.5", "queue_wait_p50_sec"), ("0.95", "queue_wait_p95_sec")):
        exp.add(
            "queue_wait_seconds",
            qm.get(key),
            help_text="Tempo de espera na fila (janela recente).",
            labels={"quantile": quantile},
        )

    exp.add("loaded_backends", status.get("loaded_count"), help_text="Backends com pesos em VRAM.")
    exp.add("loaded_declared_vram_mib", status.get("loaded_vram_mib"), help_text="Soma do vram_mib dos carregados.")
    exp.add("supervisor_vram_mib", status.get("process_vram_mib"), help_text="VRAM no PID do supervisor.")
    exp.add("worker_vram_mib", status.get("worker_vram_mib"), help_text="VRAM somada dos workers.")

    strays = status.get("strays") or {}
    exp.add("stray_processes", strays.get("count"), help_text="Processos vramd órfãos.")
    exp.add("stray_vram_mib", strays.get("vram_mib"), help_text="VRAM presa por órfãos.")

    for row in _backend_rows(status):
        labels = {"backend": row["name"]}
        exp.add("backend_loaded", bool(row.get("loaded")), help_text="1 se o backend está carregado.", labels=labels)
        exp.add("backend_declared_vram_mib", row.get("vram_mib"), help_text="Footprint declarado.", labels=labels)
        exp.add("backend_peak_vram_mib", row.get("peak_mib"), help_text="Pico usado na admissão.", labels=labels)
        exp.add("backend_inflight", row.get("ref_count"), help_text="Gerações em curso no backend.", labels=labels)

    per_backend = (stats or {}).get("backends") or {}
    if isinstance(per_backend, Mapping):
        for name in sorted(per_backend):
            row = per_backend[name]
            if not isinstance(row, Mapping):
                continue
            labels = {"backend": name}
            for key, metric, help_text in (
                ("load_count", "backend_loads_total", "Loads do backend."),
                ("generate_count", "backend_generations_total", "Gerações concluídas."),
                ("evict_count", "backend_evictions_total", "Evicções."),
                ("error_count", "backend_errors_total", "Erros de geração/load."),
            ):
                exp.add(metric, row.get(key), help_text=help_text, kind="counter", labels=labels)
            exp.add(
                "backend_generate_seconds_avg",
                row.get("avg_generate_time_sec"),
                help_text="Duração média do generate.",
                labels=labels,
            )
            exp.add(
                "backend_load_seconds_avg",
                row.get("avg_load_time_sec"),
                help_text="Duração média do load.",
                labels=labels,
            )
            exp.add(
                "backend_idle_seconds", row.get("idle_sec"), help_text="Segundos desde o último uso.", labels=labels
            )

    learn = status.get("learn") or {}
    learned = learn.get("backends") or {}
    if isinstance(learned, Mapping):
        for name in sorted(learned):
            row = learned[name]
            if not isinstance(row, Mapping):
                continue
            labels = {"backend": name}
            exp.add(
                "learn_observed_p95_mib", row.get("observed_p95_mib"), help_text="Pico p95 observado.", labels=labels
            )
            exp.add("learn_samples", row.get("samples"), help_text="Jobs observados pelo learn.", labels=labels)
            if row.get("verdict"):
                exp.add(
                    "learn_verdict",
                    1,
                    help_text="Veredicto de drift corrente (1 na label activa).",
                    labels={"backend": name, "verdict": row["verdict"]},
                )

    workers = (status.get("debug") or {}).get("workers") or {}
    if isinstance(workers, Mapping):
        for name in sorted(workers):
            row = workers[name]
            if not isinstance(row, Mapping):
                continue
            labels = {"backend": name}
            exp.add("worker_alive", bool(row.get("alive")), help_text="1 se o worker está vivo.", labels=labels)
            exp.add(
                "worker_spawns_total",
                row.get("spawn_count"),
                help_text="Spawns do worker.",
                kind="counter",
                labels=labels,
            )
            exp.add(
                "worker_recycles_total",
                row.get("recycle_count"),
                help_text="Reciclagens por max_jobs.",
                kind="counter",
                labels=labels,
            )
            exp.add("worker_jobs_served", row.get("jobs_served"), help_text="Jobs servidos neste spawn.", labels=labels)

    hooks = status.get("hooks") or {}
    if isinstance(hooks, Mapping):
        exp.add("hooks_configured", hooks.get("configured"), help_text="Hooks configurados.")
        for key, metric, help_text in (
            ("fired", "hooks_fired_total", "Hooks disparados."),
            ("failed", "hooks_failed_total", "Hooks que falharam ou excederam o timeout."),
            ("throttled", "hooks_throttled_total", "Disparos suprimidos por throttle."),
        ):
            exp.add(metric, hooks.get(key), help_text=help_text, kind="counter")

    return exp.render()
