"""``vramd top`` — dashboard TUI live da GPU e da fila.

Um ``htop`` para o supervisor: VRAM da GPU com barra, processos que a seguram,
jobs running com progresso em tempo real, fila com ETA, backends com countdown
para idle-evict e os veredictos do learn. Só de leitura — com um batch a
correr, olhar é a única operação segura (a regra «não mates GPU» aplica-se ao
autor do dashboard também).

Poll por RPC (``status`` + ``queue``) + NVML; o supervisor não sabe que o top
existe (nenhuma alteração de protocolo só para a UI).
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

from rich import box
from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from . import __version__
from . import protocol as P


def _short_job_id(job_id: object, n: int = 12) -> str:
    jid = str(job_id or "")
    if not jid:
        return "?"
    return jid if len(jid) <= n else f"{jid[:n]}…"


def _mib(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{int(v):,} MiB".replace(",", " ")
    except (TypeError, ValueError):
        return str(v)


def _fmt_sec(v: Any) -> str:
    if v is None:
        return "—"
    try:
        s = float(v)
    except (TypeError, ValueError):
        return str(v)
    if s < 0:
        return "—"
    if s < 60:
        return f"{s:.0f}s"
    m, sec = divmod(int(s), 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


class TopDataSource:
    """Tudo o que o dashboard lê, num sítio — injectável em testes.

    Distingue **down** (socket/PID mortos) de **hung** (vivo mas sem resposta):
    mostrar "não está ativo" para um supervisor preso convidava o operador a
    arrancar um segundo vramd — dois supervisores a competir pela mesma GPU.
    """

    def fetch(self) -> dict[str, Any]:
        from .client import is_server_running, send_request

        out: dict[str, Any] = {"running": False}
        if not is_server_running(P.DEFAULT_SOCKET_PATH):
            return out
        status = send_request({"cmd": P.CMD_STATUS}, P.DEFAULT_SOCKET_PATH, timeout_sec=5.0)
        queue = send_request({"cmd": P.CMD_QUEUE}, P.DEFAULT_SOCKET_PATH, timeout_sec=5.0)
        if not status and not queue:
            # Probe OK mas RPCs calados: supervisor vivo mas wedged ( deadlock,
            # driver preso). NUNCA "não está ativo".
            out["hung"] = True
            return out
        out.update({"running": True, "status": status or {}, "queue": queue or {}})

        with contextlib.suppress(Exception):
            from .gpu import query_gpu_snapshot

            out["gpu"] = query_gpu_snapshot()
        with contextlib.suppress(Exception):
            from .process_guard import gpu_vram_by_pid

            out["procs"] = sorted(gpu_vram_by_pid().items(), key=lambda kv: -(kv[1] or 0))
        return out


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _render_header(data: dict[str, Any]) -> Panel:
    status = data.get("status") or {}
    queue = data.get("queue") or {}
    eta = queue.get("eta_sec") or status.get("eta_sec")
    q = status.get("queue") or {}
    pid = status.get("pid", "?")
    line1 = Text()
    line1.append("vramd", style="bold blue")
    line1.append(f"  v{__version__}", style="dim")
    line1.append(f"  ·  PID {pid}", style="cyan")
    line1.append(f"  ·  {status.get('requests_served', 0)} pedidos", style="green")
    line1.append(f"  ·  fila {q.get('queue_depth', 0)}q/{q.get('inflight', 0)}run", style="yellow")
    if eta is not None:
        line1.append(f"  ·  ETA {_fmt_sec(eta)}", style="magenta")
    line2 = Text()
    line2.append(str(status.get("socket", P.DEFAULT_SOCKET_PATH)), style="dim")
    line2.append("  ·  Ctrl+C para sair (só leitura — nada é alterado)", style="dim italic")
    return Panel(Group(line1, line2), box=box.SIMPLE, border_style="blue")


def _render_gpu(data: dict[str, Any]) -> Panel:
    snap = data.get("gpu")
    if snap is None:
        return Panel(Text("GPU indisponível (NVML/nvidia-smi sem leitura)", style="dim"), box=box.SIMPLE, title="GPU")
    total = getattr(snap, "total_mib", None) or 0
    free = getattr(snap, "free_mib", None) or 0
    used = max(0, total - free) if total and free is not None else None
    name = getattr(snap, "name", "?")

    body: list[Any] = []
    if total and used is not None:
        pct = used / total if total else 0.0
        style = "green" if pct < 0.7 else "yellow" if pct < 0.9 else "red"
        gt = Table(box=None, pad_edge=False, show_header=False)
        gt.add_column(width=22, no_wrap=True)
        gt.add_column(width=30)
        gt.add_column(no_wrap=True)
        gt.add_row(
            Text(name, style="cyan"),
            ProgressBar(total=total, completed=used, width=28),
            Text(f"{used/1024:.1f}/{total/1024:.1f} GiB ({pct:.0%})  ·  livres {_mib(free)}", style=style),
        )
        body.append(gt)
    else:
        body.append(Text(f"{name} — leitura de memória indisponível", style="dim"))

    procs = data.get("procs") or []
    if procs:
        pt = Table(box=None, pad_edge=False, show_header=False)
        pt.add_column("pid", justify="right", style="dim", width=7)
        pt.add_column("proc", style="cyan", no_wrap=True, max_width=28)
        pt.add_column("mib", justify="right")
        for pid, mib in procs[:6]:
            who = _proc_name(pid)
            pt.add_row(str(pid), who, _mib(mib))
        body.append(pt)
    return Panel(Group(*body), box=box.SIMPLE, title="[bold]GPU", border_style="green")


def _proc_name(pid: int) -> str:
    with contextlib.suppress(Exception):
        import psutil

        return psutil.Process(int(pid)).name()[:28]
    return "?"


def _render_jobs(data: dict[str, Any]) -> Panel:
    queue = data.get("queue") or {}
    running = list(queue.get("running") or [])
    queued = list(queue.get("queued") or [])

    rows: list[Any] = []
    if running:
        t = Table(box=None, pad_edge=False, show_header=False)
        t.add_column("job", style="cyan", width=13)
        t.add_column("backend", style="bold", no_wrap=True, max_width=16)
        t.add_column("prog", width=24)
        t.add_column("pct", width=5)
        t.add_column("msg", style="dim", no_wrap=True, max_width=30)
        t.add_column("t", justify="right", width=7)
        for j in running:
            pct = j.get("progress_pct")
            if isinstance(pct, (int, float)):
                prog: Any = ProgressBar(total=100, completed=max(0.0, min(100.0, pct * 100)), width=22)
                pct_s = Text(f"{pct:.0%}", style="green" if pct >= 1.0 else "cyan")
            else:
                prog = Text("···", style="dim")
                pct_s = Text("—", style="dim")
            t.add_row(
                _short_job_id(j.get("job_id")),
                str(j.get("backend") or "?"),
                prog,
                pct_s,
                str(j.get("progress_msg") or ""),
                _fmt_sec(j.get("generate_sec")) if j.get("generate_sec") is not None else "run",
            )
        rows.append(Text("[bold]running", style="green"))
        rows.append(t)
    else:
        rows.append(Text("running: (nenhum job)", style="dim"))

    if queued:
        t = Table(box=None, pad_edge=False, show_header=False)
        t.add_column("job", style="cyan", width=13)
        t.add_column("backend", no_wrap=True, max_width=16)
        t.add_column("pri", width=4, style="dim")
        t.add_column("cuts", justify="right", width=4)
        t.add_column("wait", justify="right", width=7)
        for j in queued[:8]:
            t.add_row(
                _short_job_id(j.get("job_id")),
                str(j.get("backend") or "?"),
                str(j.get("priority") or "?"),
                str(j.get("affinity_cuts", 0)),
                _fmt_sec(j.get("queue_wait_sec")),
            )
        if len(queued) > 8:
            t.add_row(f"+{len(queued) - 8}", "…", "", "", "")
        rows.append(Text("[bold]queued", style="yellow"))
        rows.append(t)

    return Panel(Group(*rows), box=box.SIMPLE, title="[bold]Fila", border_style="cyan")


def _render_backends(data: dict[str, Any]) -> Panel:
    status = data.get("status") or {}
    backends = list(status.get("backends") or [])
    learn = (status.get("learn") or {}).get("backends") or {}
    idle_timeout = float(status.get("idle_evict_timeout_sec") or 0)

    t = Table(box=None, pad_edge=False, show_header=False)
    t.add_column("st", width=1)
    t.add_column("backend", style="cyan", no_wrap=True, max_width=16)
    t.add_column("peak", justify="right", width=10)
    t.add_column("learn", justify="right", width=10)
    t.add_column("refs", justify="right", width=4)
    t.add_column("evict-in", justify="right", width=9)

    for b in backends:
        name = b.get("name", "?")
        loaded = bool(b.get("loaded"))
        st = Text("●", style="green") if loaded else Text("○", style="dim")
        peak = b.get("peak_mib")
        lr = learn.get(name) or {}
        obs = lr.get("observed_p95_mib")
        verdict = str(lr.get("verdict") or "")
        learn_s = Text()
        if obs is not None:
            learn_s.append(_mib(obs))
            if verdict == "underprovisioned":
                learn_s.append(" !", style="bold red")
            elif verdict == "overprovisioned":
                learn_s.append(" ↓", style="yellow")
        evict_in = "—"
        if loaded and idle_timeout > 0:
            refs = int(b.get("ref_count") or 0)
            if refs > 0:
                evict_in = "in use"
            else:
                last_used = float(b.get("last_used") or 0)
                if last_used > 0:
                    idle = time.monotonic() - last_used
                    remain = idle_timeout - idle
                    evict_in = _fmt_sec(remain) if remain > 0 else "now"
        t.add_row(st, name, _mib(peak), learn_s, str(b.get("ref_count", 0)), evict_in)

    title = "[bold]Backends"
    hint = Text("learn = p95 observado (vramd learn) · ! subdimensionado · ↓ sobredimensionado", style="dim")
    return Panel(Group(t, hint), box=box.SIMPLE, title=title, border_style="magenta")


def render_dashboard(data: dict[str, Any]) -> Any:
    """Composição completa de um frame do dashboard."""
    if data.get("hung"):
        return Panel(
            Text(
                "vramd ATIVO mas sem resposta (hung?) — socket existe, RPCs calados.\n"
                "NÃO arranques um segundo supervisor: `vramd queue` / `kill -0 <pid>`\n"
                "para confirmar; investiga antes de qualquer acção destrutiva.",
                style="bold red",
                justify="center",
            ),
            box=box.ROUNDED,
            border_style="red",
            title="[bold]vramd top",
        )
    if not data.get("running"):
        return Panel(
            Text(
                "vramd não está ativo.  Arranca com:  vramd top fica à espera…\n"
                "(este painel faz refresh sozinho quando o supervisor aparecer)",
                style="yellow",
                justify="center",
            ),
            box=box.ROUNDED,
            border_style="yellow",
            title="[bold]vramd top",
        )
    left = Group(_render_gpu(data), _render_jobs(data))
    right = _render_backends(data)
    return Group(_render_header(data), Columns([left, right], expand=True))


def run_top(
    *,
    interval_sec: float = 1.0,
    source: TopDataSource | None = None,
    once: bool = False,
    console: Any = None,
) -> int:
    """Loop do dashboard. ``once=True`` renderiza um frame (usado em testes/CI)."""
    src = source or TopDataSource()
    if once:
        target = console.print if console is not None else _default_print
        target(render_dashboard(src.fetch()))
        return 0

    from rich.console import Console

    con = console if console is not None else Console()
    # SIGTERM: sem handler, o Live(screen=True) ficava sem __exit__ e o
    # terminal do utilizador preso no alt-screen até um `reset`.
    import signal
    import sys as _sys

    with contextlib.suppress(Exception):

        def _on_term(signum: int, frame: Any) -> None:
            _sys.exit(0)  # passa pelo __exit__ do Live — terminal restaurado

        signal.signal(signal.SIGTERM, _on_term)

    last_good: Any = None
    with Live(console=con, refresh_per_second=2, screen=True) as live:
        while True:
            try:
                frame = render_dashboard(src.fetch())
                last_good = frame
            except Exception as e:
                # Um campo malformado (version skew, JSON parcial) não pode
                # matar o dashboard: manter o último frame bom + erro inline.
                from rich.text import Text as _Text

                frame = (
                    Group(last_good, _Text(f"[render error: {e}]", style="bold red"))
                    if last_good is not None
                    else _Text(f"render error: {e}", style="bold red")
                )
            live.update(frame)
            time.sleep(max(0.2, interval_sec))
    return 0


def _default_print(renderable: Any) -> None:
    from rich.console import Console

    Console().print(renderable)
