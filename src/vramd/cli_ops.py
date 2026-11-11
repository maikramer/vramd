"""Comandos de operação do CLI: history, logs, metrics, validate, completion.

Vivem fora de ``cli.py`` (já com >2000 linhas) e registam-se no grupo raiz por
:func:`register`. Todos são só de leitura sobre o supervisor — seguros com um
batch a correr — e, tirando ``metrics``, funcionam com ele parado: lêem
ficheiros, que é o que resta quando se quer saber porque é que parou.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

try:
    import rich_click as click
except ImportError:  # pragma: no cover
    import click  # type: ignore[no-redef]

from . import host
from . import protocol as P

console = Console()

_STATE_STYLE = {P.JOB_DONE: "green", P.JOB_FAILED: "red", P.JOB_CANCELLED: "yellow"}
_DURATION_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def complete_backend(ctx: Any, param: Any, incomplete: str) -> list[str]:
    """Completion de nomes de backend a partir do registry (nunca lança)."""
    try:
        from .registry import Registry

        return [name for name in Registry().names if name.startswith(incomplete)]
    except Exception:
        return []


def parse_duration(text: str) -> float:
    """``"90"`` → 90 s, ``"15m"`` → 900 s, ``"2h"``, ``"1d"``.

    Raises:
        click.BadParameter: formato não reconhecido ou valor ≤ 0.
    """
    raw = text.strip().lower()
    unit = 1.0
    if raw and raw[-1] in _DURATION_UNITS:
        unit = _DURATION_UNITS[raw[-1]]
        raw = raw[:-1]
    try:
        value = float(raw) * unit
    except ValueError:
        raise click.BadParameter(f"duração inválida: {text!r} (ex.: 90, 15m, 2h, 1d)") from None
    if not math.isfinite(value):
        raise click.BadParameter(f"duração deve ser finita: {text!r}")
    if value <= 0:
        raise click.BadParameter(f"duração deve ser > 0: {text!r}")
    return value


def _safe_backend(name: str) -> str:
    """Nome de backend sem componentes de path (``logs ../../x``)."""
    text = str(name or "").strip()
    if not text or text in (".", "..") or Path(text).name != text or "\\" in text or ":" in text:
        raise click.BadParameter(f"backend inválido: {name!r}")
    return text


def _fmt_sec(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    if value >= 3600:
        return f"{value / 3600:.1f}h"
    if value >= 60:
        return f"{int(value // 60)}m{int(value % 60):02d}s"
    return f"{value:.1f}s"


def _fmt_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"  # pragma: no cover


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


def _history_cmd(
    limit: int,
    backend: str | None,
    state: str | None,
    failed: bool,
    since: str | None,
    summary: bool,
    clear: bool,
    as_json: bool,
) -> None:
    """Jobs terminados (persistido; funciona com o supervisor parado).

    Mostra backend, estado, tempos de fila/geração e o erro dos que falharam.
    ``--summary`` agrega por backend: taxa de sucesso, média e p95 de geração.
    """
    from .history import JobHistory, default_history_path, read_history, summarize

    path = default_history_path()
    if clear:
        count = JobHistory(path).clear()
        if as_json:
            console.print_json(json.dumps({"status": "ok", "cleared": count, "path": str(path)}))
        else:
            console.print(f"[bold green]✓[/bold green] histórico limpo ({count} entradas) — {path}")
        return

    if failed:
        state = P.JOB_FAILED
    since_sec = parse_duration(since) if since else None
    # O resumo agrega tudo o que passa nos filtros; o --limit só corta a listagem.
    entries = read_history(path, limit=None if summary else limit, backend=backend, state=state, since_sec=since_sec)

    if summary:
        rows = summarize(entries)
        if as_json:
            console.print_json(json.dumps({"path": str(path), "jobs": len(entries), "backends": rows}))
            return
        if not rows:
            console.print(f"[dim]Sem histórico ({path}).[/dim]")
            return
        table = Table(title=f"[bold]histórico — {len(entries)} jobs", box=box.SIMPLE)
        table.add_column("Backend", style="cyan", no_wrap=True)
        table.add_column("Jobs", justify="right")
        table.add_column("OK", justify="right", style="green")
        table.add_column("Falha", justify="right", style="red")
        table.add_column("Canc", justify="right", style="yellow")
        table.add_column("%OK", justify="right")
        table.add_column("Gen média", justify="right")
        table.add_column("Gen p95", justify="right")
        table.add_column("GPU total", justify="right")
        table.add_column("Fila p95", justify="right")
        for name in sorted(rows):
            row = rows[name]
            rate = row.get("success_rate")
            table.add_row(
                name,
                str(row["jobs"]),
                str(row["done"]),
                str(row["failed"]),
                str(row["cancelled"]),
                f"{rate:.0%}" if isinstance(rate, (int, float)) else "—",
                _fmt_sec(row.get("generate_avg_sec")),
                _fmt_sec(row.get("generate_p95_sec")),
                _fmt_sec(row.get("generate_total_sec")),
                _fmt_sec(row.get("queue_wait_p95_sec")),
            )
        console.print(table)
        return

    if as_json:
        console.print_json(json.dumps({"path": str(path), "jobs": entries}, ensure_ascii=False, default=str))
        return
    if not entries:
        console.print(f"[dim]Sem histórico que corresponda ({path}).[/dim]")
        return
    table = Table(title=f"[bold]últimos {len(entries)} jobs", box=box.SIMPLE)
    table.add_column("Quando", style="dim", no_wrap=True)
    table.add_column("Job", no_wrap=True)
    table.add_column("Backend", style="cyan", no_wrap=True)
    table.add_column("Estado", no_wrap=True)
    table.add_column("Fila", justify="right")
    table.add_column("Geração", justify="right")
    table.add_column("Detalhe", overflow="fold")
    for entry in entries:
        ts = entry.get("ts")
        when = datetime.fromtimestamp(float(ts)).strftime("%m-%d %H:%M:%S") if isinstance(ts, (int, float)) else "—"
        st = str(entry.get("state") or "?")
        style = _STATE_STYLE.get(st, "white")
        detail = ""
        if entry.get("error"):
            code = f"[{entry['error_code']}] " if entry.get("error_code") else ""
            detail = f"{code}{str(entry['error'])[:120]}"
        elif entry.get("output") is not None:
            detail = str(entry["output"])[:120]
        retries = int(entry.get("vram_retries") or 0)
        if retries:
            detail = f"{detail} (retries VRAM: {retries})".strip()
        table.add_row(
            when,
            str(entry.get("job_id") or "?")[:8],
            str(entry.get("backend") or "?"),
            f"[{style}]{st}[/{style}]",
            _fmt_sec(entry.get("queue_wait_sec")),
            _fmt_sec(entry.get("generate_sec")),
            escape(detail),  # erros de modelos trazem `[...]` que o rich leria como markup
        )
    console.print(table)


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------


def supervisor_log_path() -> Path | None:
    """Log mais recente do supervisor (``None`` se ainda não há nenhum)."""
    from .logging import default_log_dir

    override = os.environ.get("VRAMD_LOG_FILE", "").strip()
    if override:
        path = Path(os.path.expanduser(override))
        return path if path.is_file() else None
    log_dir = default_log_dir()
    try:
        candidates = [p for p in log_dir.glob("*.log") if p.is_file()]
    except OSError:
        return None
    if not candidates:
        return None
    # Preferir os que são do supervisor; o directório é partilhado com tools.
    named = [p for p in candidates if p.name.startswith(("vramd", "ums"))]
    return max(named or candidates, key=lambda p: p.stat().st_mtime)


def available_logs() -> list[tuple[str, Path]]:
    """``(etiqueta, path)`` de todos os logs conhecidos, existentes em disco."""
    out: list[tuple[str, Path]] = []
    sup = supervisor_log_path()
    if sup is not None:
        out.append(("supervisor", sup))
    auto = host.autostart_log_path()
    if auto.is_file():
        out.append(("autostart", auto))
    prefix, suffix = "vramd-worker-", ".log"
    try:
        workers = sorted(host.cache_dir().glob(f"{prefix}*{suffix}"))
    except OSError:
        workers = []
    for path in workers:
        if path.is_file():
            out.append((f"worker/{path.name[len(prefix) : -len(suffix)]}", path))
    return out


def tail_lines(path: Path, lines: int, *, chunk_size: int = 65536) -> list[str]:
    """Últimas ``lines`` linhas, lendo o ficheiro de trás para a frente."""
    if lines <= 0:
        return []
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        pos = fh.tell()
        buf = b""
        while pos > 0 and buf.count(b"\n") <= lines:
            step = min(chunk_size, pos)
            pos -= step
            fh.seek(pos)
            buf = fh.read(step) + buf
    return buf.decode("utf-8", errors="replace").splitlines()[-lines:]


def follow(path: Path, *, poll_sec: float = 0.3, stop: Any = None, out: Any = None) -> None:
    """``tail -f``: escreve o que for acrescentado; sobrevive a truncagem/rotação."""
    write = out or (lambda text: (sys.stdout.write(text), sys.stdout.flush()))
    try:
        offset = path.stat().st_size
    except OSError:
        offset = 0
    while stop is None or not stop():
        try:
            size = path.stat().st_size
        except OSError:
            time.sleep(poll_sec)
            continue
        if size < offset:  # truncado / substituído
            offset = 0
        if size > offset:
            with path.open("rb") as fh:
                fh.seek(offset)
                chunk = fh.read(size - offset)
            offset = size
            write(chunk.decode("utf-8", errors="replace"))
        else:
            time.sleep(poll_sec)


def _logs_cmd(backend: str | None, lines: int, follow_flag: bool, list_flag: bool, show_path: bool) -> None:
    """Log do supervisor ou do worker de um BACKEND (stderr do modelo).

    Sem argumento mostra o log do supervisor. ``-f`` segue em tempo real;
    ``--list`` mostra que logs existem e onde.
    """
    if list_flag:
        logs = available_logs()
        if not logs:
            console.print(f"[dim]Sem logs em {host.cache_dir()}.[/dim]")
            return
        table = Table(title="[bold]logs", box=box.SIMPLE)
        table.add_column("Origem", style="cyan")
        table.add_column("Tamanho", justify="right")
        table.add_column("Modificado", style="dim")
        table.add_column("Path", overflow="fold")
        for label, path in logs:
            st = path.stat()
            table.add_row(
                label,
                _fmt_size(st.st_size),
                datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                str(path),
            )
        console.print(table)
        return

    if backend:
        backend = _safe_backend(backend)
        path: Path | None = host.worker_log_path(backend)
        label = f"worker {backend}"
    else:
        path = supervisor_log_path()
        label = "supervisor"
    if path is None or not path.is_file():
        where = path if path is not None else host.cache_dir() / "logs"
        console.print(f"[yellow]Sem log do {label} ({where}).[/yellow]")
        known = [lbl for lbl, _ in available_logs()]
        if known:
            console.print(f"[dim]Disponíveis: {', '.join(known)} — `vramd logs --list`.[/dim]")
        sys.exit(1)
    if show_path:
        click.echo(str(path))
        return
    for line in tail_lines(path, lines):
        click.echo(line)
    if follow_flag:
        with contextlib.suppress(KeyboardInterrupt):
            follow(path)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _local_gpu() -> dict[str, Any] | None:
    try:
        from .gpu import _nvml_memory_mib

        mem = _nvml_memory_mib(0)
    except Exception:
        return None
    if mem is None:
        return None
    free_mib, total_mib, _used = mem
    return {"free_mib": free_mib, "total_mib": total_mib}


def _metrics_cmd(textfile: str | None) -> None:
    """Métricas no formato de texto do Prometheus (só leitura).

    Sem flags imprime no stdout. ``--textfile`` escreve atomicamente um ``.prom``
    para o textfile collector do node_exporter (um cron/timer a cada 15 s chega).
    Com o supervisor parado sai ``vramd_up 0`` e exit code 1.
    """
    from .cli import _atomic_write_text, _send
    from .metrics import render_prometheus

    status = _send({"cmd": P.CMD_STATUS}, timeout=10.0)
    stats = _send({"cmd": P.CMD_STATS}, timeout=10.0) if status else None
    text = render_prometheus(status, stats, gpu=_local_gpu())
    if textfile:
        _atomic_write_text(textfile, text)
    else:
        click.echo(text, nl=False)
    if not status:
        sys.exit(1)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _validate_cmd(yaml_path: str | None, hooks_path: str | None, strict: bool, as_json: bool) -> None:
    """Lint dos descritores de backends e do hooks.yaml — sem GPU, sem workers.

    Apanha o que o registry tolera em silêncio: chaves com typo, comandos de
    worker que não existem, variáveis ``${env:…}`` por definir, footprints
    maiores que a GPU. Exit code 1 com erros (``--strict``: também com avisos).
    """
    from .validate import LEVEL_ERROR, LEVEL_WARNING, validate_config

    gpu = _local_gpu()
    report = validate_config(
        yaml_path,
        hooks_path=hooks_path,
        gpu_total_mib=int(gpu["total_mib"]) if gpu and gpu.get("total_mib") else None,
    )
    failed = bool(report.errors) or (strict and bool(report.warnings))
    if as_json:
        console.print_json(json.dumps(report.to_dict(), ensure_ascii=False))
        if failed:
            sys.exit(1)
        return

    for src in report.sources:
        console.print(f"[dim]fonte: {src}[/dim]")
    icons = {LEVEL_ERROR: "[bold red]✗ erro  [/bold red]", LEVEL_WARNING: "[yellow]! aviso [/yellow]"}
    for finding in report.findings:
        where = f"[cyan]{escape(finding.backend)}[/cyan]: " if finding.backend else ""
        console.print(f"{icons.get(finding.level, '[dim]· nota  [/dim]')} {where}{escape(finding.message)}")
        if finding.source:
            console.print(f"         [dim]{escape(finding.source)}[/dim]")
        if finding.hint:
            console.print(f"         [dim]→ {escape(finding.hint)}[/dim]")
    summary = f"{len(report.backends)} backend(s), {len(report.errors)} erro(s), {len(report.warnings)} aviso(s)"
    if failed:
        console.print(f"[bold red]✗ {summary}[/bold red]")
        sys.exit(1)
    console.print(f"[bold green]✓ {summary}[/bold green]")


# ---------------------------------------------------------------------------
# completion
# ---------------------------------------------------------------------------


def _completion_cmd(shell: str | None) -> None:
    """Script de completion para bash/zsh/fish (comandos, flags e backends).

    \b
    bash:  eval "$(vramd completion bash)"
    zsh:   eval "$(vramd completion zsh)"
    fish:  vramd completion fish > ~/.config/fish/completions/vramd.fish
    """
    from click.shell_completion import get_completion_class

    name = shell or Path(os.environ.get("SHELL", "")).name
    if not name:
        raise click.UsageError("não consegui detectar a shell — indica bash, zsh ou fish")
    comp_cls = get_completion_class(name)
    if comp_cls is None:
        raise click.UsageError(f"shell não suportada: {name!r} (bash, zsh, fish)")
    root = click.get_current_context().find_root().command
    click.echo(comp_cls(root, {}, "vramd", "_VRAMD_COMPLETE").source())


# ---------------------------------------------------------------------------
# registo
# ---------------------------------------------------------------------------


def register(group: Any) -> None:
    """Regista os comandos de operação no grupo raiz do CLI."""
    history = click.option("--json", "as_json", is_flag=True, help="Dump JSON.")(_history_cmd)
    history = click.option("--clear", is_flag=True, help="Apaga o histórico.")(history)
    history = click.option("--summary", is_flag=True, help="Agregado por backend (sucesso, média, p95).")(history)
    history = click.option("--since", default=None, metavar="DUR", help="Só os últimos DUR (ex.: 30m, 2h, 1d).")(
        history
    )
    history = click.option("--failed", is_flag=True, help="Atalho para --state failed.")(history)
    history = click.option(
        "--state",
        type=click.Choice([P.JOB_DONE, P.JOB_FAILED, P.JOB_CANCELLED]),
        default=None,
        help="Filtra por estado.",
    )(history)
    history = click.option("--backend", default=None, shell_complete=complete_backend, help="Filtra um backend.")(
        history
    )
    history = click.option(
        "-n",
        "--limit",
        default=20,
        show_default=True,
        type=click.IntRange(min=0),
        help="Entradas a mostrar (0 = todas).",
    )(history)
    group.command("history")(history)

    logs = click.option("--path", "show_path", is_flag=True, help="Só imprime o path do log.")(_logs_cmd)
    logs = click.option("--list", "list_flag", is_flag=True, help="Lista os logs existentes.")(logs)
    logs = click.option("-f", "--follow", "follow_flag", is_flag=True, help="Segue o log (Ctrl+C para sair).")(logs)
    logs = click.option("-n", "--lines", default=50, show_default=True, type=int, help="Linhas finais a mostrar.")(logs)
    logs = click.argument("backend", required=False, shell_complete=complete_backend)(logs)
    group.command("logs")(logs)

    metrics = click.option(
        "--textfile",
        type=click.Path(dir_okay=False),
        default=None,
        help="Escreve um .prom (textfile collector do node_exporter) em vez do stdout.",
    )(_metrics_cmd)
    group.command("metrics")(metrics)

    validate = click.option("--json", "as_json", is_flag=True, help="Dump JSON do relatório.")(_validate_cmd)
    validate = click.option("--strict", is_flag=True, help="Avisos também dão exit code 1.")(validate)
    validate = click.option(
        "--hooks", "hooks_path", type=click.Path(dir_okay=False), default=None, help="hooks.yaml a validar."
    )(validate)
    validate = click.argument("yaml_path", required=False, type=click.Path(dir_okay=False))(validate)
    group.command("validate")(validate)

    completion = click.argument("shell", required=False, type=click.Choice(["bash", "zsh", "fish"]))(_completion_cmd)
    group.command("completion")(completion)
    register_routing(group)


# ---------------------------------------------------------------------------
# route / devices / peers — o que o router vê e o que decidiria
# ---------------------------------------------------------------------------

_ROUTE_KIND_LABEL = {"local": "GPU local", "split": "split", "remote": "rede"}


def _fmt_mib(value: Any) -> str:
    return f"{int(value)} MiB" if isinstance(value, (int, float)) else "—"


def _render_lanes(rows: list[dict[str, Any]]) -> None:
    table = Table(title="[bold]GPUs (lanes)", box=box.SIMPLE)
    table.add_column("Lane", style="cyan", no_wrap=True)
    table.add_column("GPUs")
    table.add_column("Livre", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Carregados")
    table.add_column("Estado")
    table.add_column("Jobs", justify="right")
    for row in rows:
        if row.get("busy"):
            remaining = row.get("remaining_sec")
            eta = f" ~{remaining:.0f}s" if isinstance(remaining, (int, float)) else ""
            state = f"[yellow]a correr {escape(str(row.get('backend') or '?'))}{eta}[/yellow]"
        else:
            state = "[green]livre[/green]"
        table.add_row(
            escape(str(row.get("lane"))),
            ",".join(str(d) for d in row.get("devices") or []) or "—",
            _fmt_mib(row.get("free_mib")),
            _fmt_mib(row.get("total_mib")),
            escape(", ".join(row.get("loaded") or []) or "—"),
            state,
            str(row.get("jobs_run", 0)),
        )
    console.print(table)


def _render_peers(rows: list[dict[str, Any]]) -> None:
    if not rows:
        console.print(
            "[dim]Sem peers configurados (VRAMD_PEERS=nome=host:porta ou peers.yaml no directório de config).[/dim]"
        )
        return
    table = Table(title="[bold]peers", box=box.SIMPLE)
    table.add_column("Peer", style="cyan", no_wrap=True)
    table.add_column("Endereço")
    table.add_column("Estado")
    table.add_column("RTT", justify="right")
    table.add_column("Banda", justify="right")
    table.add_column("Fila", justify="right")
    table.add_column("Quentes")
    table.add_column("Enviados", justify="right")
    for row in rows:
        if row.get("healthy"):
            state = "[green]ok[/green]"
        else:
            state = f"[red]{escape(str(row.get('last_error') or 'em baixo'))[:40]}[/red]"
        rtt = row.get("rtt_ms")
        band = f"{row.get('bandwidth_mib_s', 0):.0f} MiB/s" + ("" if row.get("bandwidth_measured") else "*")
        table.add_row(
            escape(str(row.get("name"))),
            escape(str(row.get("address"))),
            state,
            f"{rtt:.1f} ms" if isinstance(rtt, (int, float)) else "—",
            band,
            f"{row.get('inflight', 0)}+{row.get('queue_depth', 0)}",
            escape(", ".join(row.get("loaded") or []) or "—"),
            f"{row.get('forwarded_total', 0)} ({row.get('failed_total', 0)} falhas)",
        )
    console.print(table)
    if any(not r.get("bandwidth_measured") for r in rows):
        console.print("[dim]* largura de banda ainda não medida (declarada ou default).[/dim]")


def _status_or_exit() -> dict[str, Any]:
    from .cli import _send

    status = _send({"cmd": P.CMD_STATUS}, timeout=15.0)
    if status is None:
        console.print("[yellow]vramd não está ativo.[/yellow]")
        sys.exit(1)
    return status


def _devices_cmd(as_json: bool) -> None:
    """GPUs locais e remotas: VRAM, modelos carregados e ocupação.

    As locais vêm das lanes do supervisor; as remotas são os peers configurados
    (``VRAMD_PEERS``/``peers.yaml``), com RTT, banda estimada, fila e backends
    quentes. ``vramd route <backend>`` mostra o custo de cada rota.
    """
    status = _status_or_exit()
    rows = status.get("lanes") or []
    peers = status.get("peers") or []
    if as_json:
        console.print_json(
            json.dumps(
                {"lanes": rows, "peers": peers, "routing": status.get("routing")},
                default=str,
            )
        )
        return
    _render_lanes(rows)
    if peers:
        console.print()
        console.print("[bold]GPUs remotas (peers)[/bold]")
        _render_peers(peers)
    decisions = (status.get("routing") or {}).get("decisions") or {}
    if any(decisions.values()):
        console.print("[dim]rotas até agora: " + ", ".join(f"{k}={v}" for k, v in decisions.items() if v) + "[/dim]")


def _peers_cmd(as_json: bool) -> None:
    """Outros supervisores vramd para onde este pode encaminhar jobs."""
    status = _status_or_exit()
    rows = status.get("peers") or []
    if as_json:
        console.print_json(json.dumps({"peers": rows}, default=str))
        return
    _render_peers(rows)


def _route_cmd(
    backend: str,
    prompt: str | None,
    output_path: str | None,
    params: tuple[str, ...],
    request_json: str | None,
    as_json: bool,
) -> None:
    """Onde correria este job AGORA, e porquê — sem o submeter.

    Mostra todas as rotas que o router considerou (cada GPU, o modelo dividido,
    cada peer) com o custo estimado decomposto: espera, evicção, load, geração,
    transferência. Aceita os mesmos parâmetros do ``submit``.
    """
    from .cli import _print_ums_error, _send, build_submit_request

    request = build_submit_request(
        backend, prompt=prompt, output_path=output_path, params=params, request_json=request_json
    )
    request["cmd"] = P.CMD_ROUTE
    request.pop("priority", None)
    resp = _send(request, timeout=30.0)
    if resp is None:
        console.print("[yellow]vramd não está ativo.[/yellow]")
        sys.exit(1)
    if as_json:
        console.print_json(json.dumps(resp, ensure_ascii=False, default=str))
        if resp.get("status") != P.STATUS_OK:
            sys.exit(1)
        return
    if resp.get("status") != P.STATUS_OK:
        _print_ums_error(resp)
        sys.exit(1)

    profile = resp.get("profile") or {}
    tiers = profile.get("tiers") or []
    crit = profile.get("criticality")
    notes = []
    if crit and crit != "normal":
        notes.append(f"criticidade {crit}")
    if tiers:
        notes.append("tiers: " + ", ".join(f"{t.get('label')} ≥{t.get('min_vram_mib')} MiB" for t in tiers))
    if notes:
        console.print("[dim]" + escape(" · ".join(notes)) + "[/dim]")
    console.print(
        f"[bold]{escape(str(resp.get('backend')))}[/bold]: pico {profile.get('peak_mib')} MiB, "
        f"load ~{profile.get('load_sec')}s, geração ~{profile.get('gen_sec')}s"
        + (f", inputs {profile.get('input_mib')} MiB" if profile.get("input_mib") else "")
    )
    table = Table(box=box.SIMPLE)
    table.add_column("", no_wrap=True)
    table.add_column("Rota", style="cyan", no_wrap=True)
    table.add_column("Tipo")
    table.add_column("Total", justify="right")
    table.add_column("Espera", justify="right")
    table.add_column("Evict", justify="right")
    table.add_column("Load", justify="right")
    table.add_column("Geração", justify="right")
    table.add_column("Rede", justify="right")
    table.add_column("Tier")
    table.add_column("Qualid.", justify="right")
    table.add_column("Nota", overflow="fold")
    best_target = (resp.get("best") or {}).get("target")
    for route in resp.get("routes") or []:
        kind = _ROUTE_KIND_LABEL.get(str(route.get("kind")), str(route.get("kind")))
        if not route.get("feasible"):
            table.add_row(
                "[red]✗[/red]",
                escape(str(route.get("target"))),
                kind,
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
                escape(str(route.get("reason") or "")),
            )
            continue
        note = "quente" if route.get("hot") else ("réplica nova" if route.get("replica") else "a frio")
        mark = "[bold green]→[/bold green]" if route.get("target") == best_target else " "
        table.add_row(
            mark,
            escape(str(route.get("target"))),
            kind,
            _fmt_sec(route.get("total_sec")),
            _fmt_sec(route.get("wait_sec")),
            _fmt_sec(route.get("evict_sec")),
            _fmt_sec(route.get("load_sec")),
            _fmt_sec(route.get("gen_sec")),
            _fmt_sec(route.get("transfer_sec")),
            escape(str(route.get("tier") or "—")),
            _fmt_sec(route.get("quality_sec")) if route.get("quality_sec") else "—",
            note,
        )
    console.print(table)
    action = resp.get("action")
    style = {"run": "bold green", "wait": "yellow", "reject": "bold red"}.get(str(action), "white")
    console.print(f"[{style}]{action}[/{style}]: {escape(str(resp.get('explanation') or ''))}")
    if not resp.get("routed"):
        console.print("[dim]Uma GPU e sem peers: o router não tem escolha a fazer (despacho clássico).[/dim]")


def register_routing(group: Any) -> None:
    """Regista ``route``/``devices``/``peers`` no grupo raiz."""
    route = click.option("--json", "as_json", is_flag=True, help="Dump JSON da decisão.")(_route_cmd)
    route = click.option("--request-json", default=None, metavar="JSON|@FICHEIRO|-", help="Request completo em JSON.")(
        route
    )
    route = click.option("-p", "--param", "params", multiple=True, metavar="K=V", help="Parâmetro do job (repetível).")(
        route
    )
    route = click.option("--output", "output_path", default=None, help="Atalho para -p output=…")(route)
    route = click.option("--prompt", default=None, help="Atalho para -p prompt=…")(route)
    route = click.argument("backend", shell_complete=complete_backend)(route)
    group.command("route")(route)

    devices = click.option("--json", "as_json", is_flag=True, help="Dump JSON.")(_devices_cmd)
    group.command("devices")(devices)

    peers = click.option("--json", "as_json", is_flag=True, help="Dump JSON.")(_peers_cmd)
    group.command("peers")(peers)
