#!/usr/bin/env python3
"""vramd — CLI principal.

Comandos (alias ``ums`` = ``vramd``):
  start|stop|status|submit|cancel|flush|queue|wait|backends|preload|evict|reap|
  respawn|zero|stats|debug|bench|doctor|calibrate|recalibrate|learn|top|mcp|
  history|logs|metrics|validate|completion|route|devices|peers

Agentes / humanos: se a GPU estiver ocupada, usa ``status`` / ``queue`` /
``debug`` / ``top`` — **não** mates processos GPU enquanto houver jobs.
``stats --reset`` só limpa contadores (não para o vramd).
``bench`` mede RTT IPC (não submete GPU). Para limpar fila stale:
``vramd flush`` ou ``vramd cancel --all``.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    import rich_click as click
except ImportError:  # pragma: no cover
    import click  # type: ignore[no-redef]

from vramd.client import (
    UMS_DO_NOT_KILL_TIP,
    format_ums_holding_summary,
    get_server_pid,
    is_server_running,
    send_request,
    send_request_stream,
)

from . import __version__, host, transport
from . import protocol as P
from .cli_ops import complete_backend
from .cli_ops import register as _register_ops
from .registry import Registry


def _parse_listen(spec: str) -> tuple[str, int]:
    """``HOST:PORT`` → ``(host, port)``; aceita ``[::1]:8790``."""
    text = spec.strip()
    if text.startswith("["):
        host_text, _, rest = text[1:].partition("]")
        port_text = rest.lstrip(":")
    elif text.count(":") == 1:
        host_text, _, port_text = text.partition(":")
    else:
        raise click.BadParameter(f"--listen espera HOST:PORT (ex.: 0.0.0.0:8790), recebi {spec!r}")
    if not host_text:
        raise click.BadParameter("--listen sem host")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise click.BadParameter(f"--listen com porta inválida: {port_text!r}") from exc
    if not (1 <= port <= 65535):
        raise click.BadParameter(f"--listen com porta fora de 1..65535: {port}")
    return host_text, port


def _load_or_create_token(token_file: str | None, *, remote: bool) -> str | None:
    """Token estável do ficheiro (criado 0600 se faltar) ou ``None`` (local)."""
    if not token_file:
        return None
    path = Path(token_file).expanduser()
    if path.is_file():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token + "\n", encoding="utf-8")
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)
    if remote:
        console.print(f"[dim]Token remoto criado em {path} (0600).[/dim]")
    return token


console = Console()


def _atomic_write_text(path: Path | str, text: str) -> None:
    """Escrita atómica (tmp + ``os.replace``) — nunca deixa meio ficheiro.

    Um ``write_text`` directo truncava o destino primeiro: OOM/SIGKILL/disco
    cheio a meio deixava um ``backends.yaml``/relatório truncado — no caso do
    overlay de calibração, a fonte de verdade de admissão do daemon corrompida
    sem backup. O ``replace`` é atómico no mesmo filesystem.
    """
    import os

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # tmp ÚNICO por processo: dois `vramd learn --apply` / calibrates
    # concorrentes a escrever o mesmo destino interoperlavam o mesmo .tmp.
    tmp = target.with_suffix(f"{target.suffix}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    # Windows: um overlay aberto por outro processo (editor/antivírus) pode
    # fazer o replace falhar durante uns ms — tentar antes de desistir.
    for attempt in range(8):
        try:
            os.replace(tmp, target)
            break
        except PermissionError:
            if attempt == 7:
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.05)


def _send(request: dict, *, timeout: float = 30.0) -> dict | None:
    """Envia um pedido ao vramd no socket canónico. Retorna None se down."""
    if not is_server_running(P.DEFAULT_SOCKET_PATH):
        return None
    return send_request(request, P.DEFAULT_SOCKET_PATH, timeout_sec=timeout)


def _print_json(resp: dict[str, Any]) -> None:
    console.print_json(json.dumps(resp, ensure_ascii=False, default=str))


def _print_ums_error(resp: dict[str, Any]) -> None:
    """Erro CLI com error_code / hint / ums_debug."""
    code = resp.get("error_code", "?")
    console.print(f"[bold red]✗ [{code}][/bold red] {resp.get('error', resp)}")
    if resp.get("hint"):
        console.print(f"[dim]hint: {resp['hint']}[/dim]")
    dbg = resp.get("ums_debug")
    if dbg:
        console.print(f"[dim]ums_debug: {json.dumps(dbg, ensure_ascii=False, default=str)}[/dim]")


def _print_do_not_kill_tip(*, inflight: int = 0, depth: int = 0) -> None:
    """Aviso estável quando há (ou pode haver) carga GPU via UMS."""
    busy = inflight > 0 or depth > 0
    style = "yellow" if busy else "dim"
    console.print(f"[{style}]{UMS_DO_NOT_KILL_TIP}[/{style}]")


def _describe_stray(proc: dict[str, Any]) -> str:
    """Uma linha por órfão: ``PID 123 worker/text3d 3470 MiB``."""
    who = str(proc.get("kind") or "?")
    if proc.get("backend"):
        who += f"/{proc['backend']}"
    vram = f" {proc['vram_mib']} MiB" if proc.get("vram_mib") else ""
    return f"PID {proc.get('pid')} {who}{vram}"


def _print_strays(strays: dict[str, Any]) -> None:
    """Avisa sobre supervisores/workers UMS órfãos e a VRAM que seguram."""
    count = int(strays.get("count") or 0)
    if not count:
        return
    vram = strays.get("vram_mib") or 0
    st = Table(title="[bold red]Processos vramd órfãos", box=box.SIMPLE)
    st.add_column("PID", justify="right", style="red")
    st.add_column("Tipo", style="cyan")
    st.add_column("Backend")
    st.add_column("VRAM", justify="right")
    for proc in strays.get("processes") or []:
        st.add_row(
            str(proc.get("pid")),
            str(proc.get("kind")),
            str(proc.get("backend") or "—"),
            f"{proc.get('vram_mib')} MiB" if proc.get("vram_mib") else "—",
        )
    console.print(st)
    console.print(f"[yellow]{count} processo(s) órfão(s) a segurar ~{vram} MiB — limpa com `vramd reap`.[/yellow]")


def _short_job_id(job_id: object, *, n: int = 12) -> str:
    jid = str(job_id or "")
    if not jid:
        return "?"
    return jid if len(jid) <= n else f"{jid[:n]}…"


@click.group()
@click.version_option(version=__version__, prog_name="vramd")
def cli() -> None:
    """vramd — controlo de admissão de VRAM."""


@cli.command("start")
@click.option(
    "--socket",
    "socket_path",
    type=click.Path(),
    default=None,
    help="Path do endpoint (Unix socket no POSIX; descritor TCP no Windows).",
)
@click.option(
    "--idle-timeout",
    "idle_timeout_min",
    default=P.DEFAULT_IDLE_TIMEOUT_MIN,
    show_default=True,
    type=int,
    help="Minutos de idle antes de encerrar.",
)
@click.option(
    "--idle-evict-sec",
    default=P.IDLE_EVICT_SEC,
    show_default=True,
    type=float,
    help="Segundos sem uso antes de descarregar os pesos de um backend.",
)
@click.option(
    "--worker-shutdown-sec",
    default=P.WORKER_IDLE_SHUTDOWN_SEC,
    show_default=True,
    type=float,
    help="Segundos sem uso antes de terminar o subprocesso worker (0 desliga).",
)
@click.option("--verbose", "-v", is_flag=True, help="Logs detalhados")
@click.option(
    "--listen",
    "listen_spec",
    default=None,
    envvar="VRAMD_LISTEN",
    help="Modo remoto: HOST:PORT para o listener TCP (ex.: 0.0.0.0:8790).",
)
@click.option(
    "--token-file",
    "token_file",
    type=click.Path(dir_okay=False),
    default=None,
    envvar="VRAMD_TOKEN_FILE",
    help=(
        "Token estável para clientes remotos (lido do ficheiro; criado se faltar). "
        "Sem isto o token é novo a cada arranque."
    ),
)
@click.option(
    "--allow-empty",
    is_flag=True,
    help="Arranca mesmo com zero backends (tests/dev — cada delegação falhará).",
)
@click.option(
    "--allow-client-code",
    is_flag=True,
    envvar="VRAMD_ALLOW_CLIENT_CODE",
    help=(
        "Aceita sessões com código do cliente (bundle tar + workers). "
        "Só numa rede de confiança: o token passa a equivaler a execução de código."
    ),
)
@click.option(
    "--allow-client-env",
    is_flag=True,
    envvar="VRAMD_ALLOW_CLIENT_ENV",
    help="Permite construir envs Python a partir do freeze enviado pelo cliente (uv).",
)
@click.option(
    "--config",
    "config_file",
    type=click.Path(),
    default=None,
    envvar="VRAMD_CONFIG_FILE",
    help="config.yml do servidor (ver `vramd config`).",
)
def start_cmd(
    socket_path: str | None,
    idle_timeout_min: int,
    idle_evict_sec: float,
    worker_shutdown_sec: float,
    verbose: bool,
    listen_spec: str | None,
    token_file: str | None,
    allow_empty: bool,
    allow_client_code: bool,
    allow_client_env: bool,
    config_file: str | None,
) -> None:
    """Arranca o vramd (foreground)."""
    from vramd.logging import configure_logging

    from .config import apply_env, config_paths, load_config

    # Só aplicar o config.yml quando existe (ou foi pedido): um `vramd start`
    # sem config mantém o Unix socket e o cache por omissão.
    has_config = bool(config_file) or any(path.is_file() for path in config_paths())
    if has_config:
        try:
            cfg = load_config(config_file, strict=bool(config_file))
            apply_env(cfg)
        except (OSError, ValueError) as exc:
            console.print(f"[bold red]✗ config.yml inválido:[/bold red] {exc}")
            sys.exit(1)
        listen_spec = listen_spec or cfg.listen
        token_file = token_file or cfg.token_file
        allow_client_code = allow_client_code or cfg.allow_client_code
        allow_client_env = allow_client_env or cfg.allow_client_env
        if socket_path is None:
            socket_path = str(Path(cfg.cache_dir).expanduser() / "vramd.sock")

    from .server import VramdServer

    log_path = configure_logging("ums")
    sock = Path(socket_path) if socket_path else P.DEFAULT_SOCKET_PATH
    if is_server_running(sock):
        console.print("[yellow]vramd já está ativo neste socket.[/yellow]")
        sys.exit(1)

    listen: tuple[str, int] | None = None
    if listen_spec:
        listen = _parse_listen(listen_spec)
    token = _load_or_create_token(token_file, remote=listen is not None)

    registry = Registry()
    if not registry.names and not allow_empty and not allow_client_code:
        # Um vramd sem backends é uma armadilha: agarra o socket, o auto-start
        # das tools passa a aceitá-lo como "ativo" e TODA a delegação falha
        # com "backend desconhecido". Recusar aqui é o diagnóstico. Um servidor
        # de sessões (`--allow-client-code`) é a excepção: os backends vêm do
        # bundle do cliente e não há registry fixo.
        console.print(
            Panel.fit(
                "[bold red]✗ Nenhum backend configurado — o vramd não arranca.[/bold red]\n\n"
                "O registry empacotado é vazio de propósito (nunca serve exemplos "
                "fictícios). Aponta o vramd para o teu registry:\n\n"
                "  • [cyan]VRAMD_BACKENDS_FILE[/cyan]=ficheiro.yaml (um ou mais, separados por ':')\n"
                "  • overlays em [cyan]backends.d/*.yaml[/cyan] do directório de configuração\n\n"
                "Um descriptor calibrado sai de [cyan]vramd calibrate <backend> --out[/cyan].\n"
                "No monorepo AiGameKit as tools auto-arrancam o vramd já configurado — "
                "basta [cyan]vramd stop[/cyan] se uma instância vazia estiver agarrada ao socket.\n\n"
                "[dim]Para arrancar na mesma (tests): vramd start --allow-empty[/dim]\n"
                "[dim]Para um servidor de sessões: vramd start --allow-client-code[/dim]",
                border_style="red",
            )
        )
        sys.exit(1)
    # Topologia: que GPUs e que outros nós o router pode usar. Detectado aqui
    # (e não no construtor do servidor) para ser uma decisão do arranque.
    from .lanes import resolve_gpu_devices
    from .peers import load_peers

    gpu_devices = resolve_gpu_devices()
    try:
        peer_specs = load_peers()
    except ValueError as exc:
        console.print(f"[bold red]✗ peers.yaml inválido:[/bold red] {exc}")
        sys.exit(1)
    topo_parts = []
    if gpu_devices:
        topo_parts.append("GPUs " + ", ".join(f"gpu{d}" for d in gpu_devices))
    if peer_specs:
        topo_parts.append("peers " + ", ".join(f"{s.name} ({s.address})" for s in peer_specs))
    topo_line = f"Router: [green]{'; '.join(topo_parts)}[/green]\n" if topo_parts else ""
    log_line = f"Log: [cyan]{log_path}[/cyan]\n" if log_path else ""
    endpoint_label = f"tcp://{listen[0]}:{listen[1]}" if listen else f"{sock} ({transport.transport_mode()})"
    console.print(
        Panel.fit(
            f"[bold]vramd[/bold]\n"
            f"Endpoint: [cyan]{endpoint_label}[/cyan]\n"
            f"Descritor: [cyan]{sock}[/cyan]\n"
            f"Backends: [green]{', '.join(registry.names)}[/green]\n"
            f"Idle timeout: [green]{idle_timeout_min} min[/green]\n"
            f"Idle evict: [green]{idle_evict_sec:.0f}s[/green] (unload) / "
            f"[green]{worker_shutdown_sec:.0f}s[/green] (worker)\n"
            f"{topo_line}"
            f"{log_line}\n"
            f"[dim]Os backends carregam sob procura (lazy). Use 'preload' para "
            f"pré-aquecer um backend específico.[/dim]",
            border_style="blue",
        )
    )
    if listen and listen[0] not in ("127.0.0.1", "::1", "localhost"):
        console.print(
            "[yellow]Listener de rede ativo — o token do descritor dá controlo total "
            "do daemon (jobs, zero, shutdown). Mantém-no fora de git e usa WireGuard "
            "se a rede não for de confiança.[/yellow]"
        )

    srv = VramdServer(
        registry=registry,
        socket_path=sock,
        idle_timeout_min=idle_timeout_min,
        idle_evict_sec=idle_evict_sec,
        worker_shutdown_sec=worker_shutdown_sec,
        verbose=verbose,
        listen=listen,
        token=token,
        gpu_devices=gpu_devices,
        peers=peer_specs,
        allow_client_code=allow_client_code,
        allow_client_env=allow_client_env,
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]vramd interrompido.[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]✗ Erro no vramd:[/bold red] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command("stop")
def stop_cmd() -> None:
    """Para o vramd (graceful shutdown)."""
    # Ler o PID ANTES do shutdown: o ficheiro desaparece no cleanup e sem ele
    # não há como esperar que o processo saia. Sem esta espera, um `start`
    # imediato falhava com ALREADY_RUNNING durante a janela de cleanup (mais
    # visível no Windows, onde o fecho de workers é mais lento).
    pid = get_server_pid(P.DEFAULT_SOCKET_PATH)
    resp = _send({"cmd": P.CMD_SHUTDOWN}, timeout=5.0)
    if resp is None:
        console.print("[yellow]vramd não está ativo.[/yellow]")
        sys.exit(0)
    if resp.get("status") != "ok":
        console.print(f"[bold red]✗ Falha ao parar o vramd:[/bold red] {resp.get('error', resp)}")
        sys.exit(1)
    if pid and host.pid_exists(pid):
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and host.pid_exists(pid):
            time.sleep(0.1)
        if host.pid_exists(pid):
            console.print(
                f"[yellow]✓ shutdown aceite, mas o PID {pid} ainda está vivo "
                f"(cleanup de workers?) — `vramd status` para confirmar.[/yellow]"
            )
            return
    console.print("[bold green]✓ vramd encerrado.[/bold green]")


@cli.command("status")
@click.option("--json", "as_json", is_flag=True, help="Dump JSON completo (inclui debug).")
def status_cmd(as_json: bool) -> None:
    """Mostra o estado do vramd e backends carregados."""
    resp = _send({"cmd": P.CMD_STATUS}, timeout=5.0)
    if resp is None:
        console.print("[yellow]vramd não está ativo.[/yellow]")
        console.print("[dim]Arranca com: vramd start[/dim]")
        sys.exit(1)

    if as_json:
        _print_json(resp)
        return

    t = Table(title="[bold blue]vramd", box=box.ROUNDED)
    t.add_column("Campo", style="cyan", no_wrap=True)
    t.add_column("Valor", style="green")
    t.add_row("PID", str(resp.get("pid", "?")))
    t.add_row("Socket", str(resp.get("socket", "?")))
    if resp.get("endpoint"):
        t.add_row("Endpoint", f"{resp['endpoint']} ({resp.get('transport', '?')})")
    if resp.get("vramd_version"):
        t.add_row("Versão", f"{resp['vramd_version']} (protocolo {resp.get('protocol', '?')})")
    sessions = resp.get("sessions") or {}
    if sessions:
        t.add_row(
            "Sessões",
            f"{sessions.get('active', 0)} activa(s)"
            f" (código={'on' if sessions.get('allow_client_code') else 'off'},"
            f" env={'on' if sessions.get('allow_client_env') else 'off'})",
        )
    models = resp.get("models") or {}
    if models:
        pinned = models.get("pinned") or []
        t.add_row(
            "Modelos HF",
            f"{models.get('home')} (policy={models.get('policy')}, {len(pinned)} pinado(s))",
        )
    t.add_row("Backends carregados", f"{resp.get('loaded_count', 0)} ({resp.get('loaded_vram_mib', 0)} MiB)")
    worker_vram = resp.get("worker_vram_mib")
    if worker_vram:
        t.add_row("VRAM nos workers", f"{worker_vram} MiB")
    t.add_row("Pedidos servidos", str(resp.get("requests_served", 0)))
    t.add_row(
        "Idle evict",
        f"{float(resp.get('idle_evict_timeout_sec') or 0):.0f}s unload / "
        f"{float(resp.get('worker_shutdown_sec') or 0):.0f}s worker",
    )
    q = resp.get("queue") or {}
    t.add_row(
        "Fila", f"{q.get('queue_depth', 0)} queued / {q.get('inflight', 0)} inflight (max {q.get('max_depth', '?')})"
    )
    t.add_row("Affinity cuts", str(resp.get("max_affinity_cuts", "?")))
    t.add_row("Max inflight", str(resp.get("max_inflight", "?")))
    console.print(t)

    qresp = _send({"cmd": P.CMD_QUEUE}, timeout=5.0)
    if qresp:
        console.print(f"[bold]{format_ums_holding_summary(qresp)}[/bold]")
        _print_do_not_kill_tip(
            inflight=int(qresp.get("inflight") or 0),
            depth=int(qresp.get("queue_depth") or 0),
        )
    else:
        _print_do_not_kill_tip(
            inflight=int(q.get("inflight") or 0),
            depth=int(q.get("queue_depth") or 0),
        )

    _print_strays(resp.get("strays") or {})

    workers = (resp.get("debug") or {}).get("workers") or {}
    if workers:
        alive = [
            f"{n} pid={w.get('pid')} {w.get('jobs_served', 0)}/{w.get('max_jobs', '?')} jobs"
            for n, w in sorted(workers.items())
            if w.get("alive")
        ]
        dead = []
        for n, w in sorted(workers.items()):
            if w.get("alive"):
                continue
            code = w.get("last_exit_code")
            why = w.get("last_abort_reason") or (f"exit={code}" if code is not None else "morto")
            dead.append(f"{n} ({why})")
        if alive:
            console.print("[dim]workers: " + " · ".join(alive) + "[/dim]")
        if dead:
            console.print("[yellow]workers sem processo vivo: " + " · ".join(dead) + "[/yellow]")
    probe = (resp.get("learn") or {}).get("probe") or {}
    if probe and not probe.get("ok"):
        console.print(
            f"[yellow]probe de VRAM: {probe.get('last_reason') or '?'} "
            f"(falhas={probe.get('consecutive_failures', 0)}, "
            f"backoff={probe.get('backoff_sec_remaining', 0)}s)[/yellow]"
        )

    backends = resp.get("backends", [])
    if backends:
        bt = Table(title="[bold]Backends", box=box.SIMPLE)
        bt.add_column("Backend", style="cyan")
        bt.add_column("YAML", justify="right")
        bt.add_column("Peak", justify="right")
        bt.add_column("Act+", justify="right")
        bt.add_column("Priority", justify="right")
        bt.add_column("Carregado")
        bt.add_column("Refs", justify="right")
        for b in backends:
            loaded = "[green]✓[/green]" if b.get("loaded") else "[dim]✗[/dim]"
            bt.add_row(
                b["name"],
                str(b["vram_mib"]),
                str(b.get("peak_mib", "?")),
                str(b.get("activation_headroom_mib", "?")),
                str(b["priority"]),
                loaded,
                str(b.get("ref_count", 0)),
            )
        console.print(bt)
        console.print(
            "[dim]Peak = pesos(fp16)+activação+safety (admit/refuse). "
            "Act+ = livre necessário com pesos já carregados.[/dim]"
        )

    dbg = resp.get("debug") or {}
    last_errors = dbg.get("last_errors") or {}
    if last_errors:
        et = Table(title="[bold yellow]Últimos erros (debug)", box=box.SIMPLE)
        et.add_column("Backend", style="cyan")
        et.add_column("last_error", style="yellow")
        for name, err in last_errors.items():
            et.add_row(str(name), str(err)[:120])
        console.print(et)
    elif dbg:
        console.print(
            f"[dim]debug: loaded={dbg.get('loaded_backends', [])} "
            f"depth={dbg.get('queue_depth', 0)} inflight={dbg.get('inflight', 0)}[/dim]"
        )


# Chaves do protocolo que um parâmetro de job não pode usurpar.
_RESERVED_REQUEST_KEYS = frozenset({"cmd", "stream", "backend", "auth", "job_id"})
_SMOKE_REQUEST = {"prompt": "smoke", "output": "/tmp/ums-smoke-out.bin"}


def _load_request_json(raw: str | None) -> dict[str, Any]:
    """``--request-json``: JSON inline, ``@ficheiro`` ou ``-`` (stdin).

    Raises:
        click.BadParameter: JSON inválido, ficheiro ilegível ou não-objecto.
    """
    if not raw:
        return {}
    text = raw
    if raw == "-":
        text = sys.stdin.read()
    elif raw.startswith("@"):
        try:
            text = Path(raw[1:]).expanduser().read_text(encoding="utf-8")
        except OSError as e:
            raise click.BadParameter(f"não consegui ler {raw[1:]}: {e}") from e
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise click.BadParameter(f"--request-json não é JSON válido: {e}") from e
    if not isinstance(data, dict):
        raise click.BadParameter("--request-json deve ser um objecto JSON")
    return data


def build_submit_request(
    backend: str,
    *,
    prompt: str | None = None,
    output_path: str | None = None,
    params: tuple[str, ...] = (),
    request_json: str | None = None,
    priority: str | None = None,
) -> dict[str, Any]:
    """Monta o request de ``submit``.

    Precedência (o mais específico ganha): ``--request-json`` < ``-p K=V`` <
    ``--prompt``/``--output``/``--priority``. Sem nenhum parâmetro cai no smoke-test histórico
    (``prompt=smoke``), para ``vramd submit <backend> --wait`` continuar a servir
    de teste de fumo.

    Raises:
        click.BadParameter: parâmetro com nome reservado pelo protocolo.
    """
    body: dict[str, Any] = {}
    body.update(_load_request_json(request_json))
    body.update(_parse_kv(params))
    if prompt is not None:
        body["prompt"] = prompt
    if output_path is not None:
        body["output"] = output_path
    bad = sorted(k for k in body if k in _RESERVED_REQUEST_KEYS or str(k).startswith("_"))
    if bad:
        raise click.BadParameter(f"parâmetro(s) reservado(s) pelo protocolo: {', '.join(bad)}")
    if not body:
        body = dict(_SMOKE_REQUEST)
    if priority is not None:
        body["priority"] = priority
    body.setdefault("priority", P.DEFAULT_PRIORITY)
    return {**body, "cmd": P.CMD_SUBMIT, "backend": backend}


def _wait_with_progress(job_id: str, *, timeout: float, quiet: bool) -> dict[str, Any] | None:
    """Espera o job por stream NDJSON, mostrando fila/fase/percentagem ao vivo.

    Devolve o resultado final (linha com ``status``) ou ``None`` se a ligação
    caiu sem resultado.
    """
    request = {"cmd": P.CMD_WAIT, "job_id": job_id, "stream": True, "timeout_sec": timeout}
    events = send_request_stream(request, P.DEFAULT_SOCKET_PATH, timeout_sec=timeout + 15.0)
    final: dict[str, Any] | None = None
    live = contextlib.nullcontext() if quiet else console.status("[dim]à espera…[/dim]", spinner="dots")
    with live as status:
        try:
            for event in events:
                if "event" not in event:
                    final = event
                    continue
                if status is None:
                    continue
                kind = event.get("event")
                if kind == P.EVENT_QUEUED:
                    status.update(f"[dim]na fila — posição {event.get('queue_position', '?')}[/dim]")
                elif kind == P.EVENT_STARTED:
                    status.update("[cyan]a correr…[/cyan]")
                elif kind == P.EVENT_PROGRESS:
                    pct = event.get("pct")
                    pct_txt = f"{float(pct):.0%} " if isinstance(pct, (int, float)) else ""
                    status.update(f"[cyan]{pct_txt}{str(event.get('msg') or '')[:80]}[/cyan]")
        except OSError:
            return final
    return final


@cli.command("submit")
@click.argument("backend", shell_complete=complete_backend)
@click.option("--prompt", default=None, help="Atalho para -p prompt=…")
@click.option("--output", "output_path", default=None, help="Atalho para -p output=…")
@click.option("-p", "--param", "params", multiple=True, metavar="K=V", help="Parâmetro do job (repetível).")
@click.option(
    "--request-json",
    default=None,
    metavar="JSON|@FICHEIRO|-",
    help="Request completo em JSON (inline, @ficheiro ou - para stdin).",
)
@click.option(
    "--priority",
    type=click.Choice(["interactive", "batch"]),
    default=None,
    help="Prioridade na fila (default: interactive).",
)
@click.option("--wait/--no-wait", default=False, help="Esperar conclusão (progresso ao vivo).")
@click.option("--timeout", default=600.0, show_default=True, type=float, help="Segundos máximos com --wait.")
@click.option("--json", "as_json", is_flag=True, help="Dump JSON.")
def submit_cmd(
    backend: str,
    prompt: str | None,
    output_path: str | None,
    params: tuple[str, ...],
    request_json: str | None,
    priority: str | None,
    wait: bool,
    timeout: float,
    as_json: bool,
) -> None:
    """Submete um job a BACKEND (e opcionalmente espera pelo resultado).

    \b
    vramd submit text2img --prompt "a red fox" -p steps=8 -p seed=42 --wait
    vramd submit text3d --request-json @job.json --priority batch
    echo '{"prompt": "x"}' | vramd submit my-model --request-json - --wait --json

    Sem parâmetros envia o smoke-test (``prompt=smoke``). Exit code 0 só se o
    job terminar em ``done`` (com ``--wait``) ou for aceite (sem).
    """
    request = build_submit_request(
        backend,
        prompt=prompt,
        output_path=output_path,
        params=params,
        request_json=request_json,
        priority=priority,
    )
    resp = _send(request, timeout=30.0)
    if resp is None:
        console.print("[yellow]vramd não está ativo.[/yellow]")
        sys.exit(1)
    if as_json and not wait:
        _print_json(resp)
        if resp.get("status") != "ok":
            sys.exit(1)
        return
    if resp.get("status") != "ok":
        _print_ums_error(resp)
        sys.exit(1)
    job_id = str(resp.get("job_id", ""))
    if not as_json:
        console.print(
            f"[bold green]✓[/bold green] submit {backend} job={job_id[:8]}… "
            f"pri={resp.get('priority')} pos={resp.get('queue_position', '?')}"
        )
    if not wait:
        return
    try:
        final = _wait_with_progress(job_id, timeout=timeout, quiet=as_json)
    except KeyboardInterrupt:
        # Ctrl+C larga o cliente, não o job: matar trabalho de GPU por reflexo
        # é exactamente o que a fila existe para evitar.
        console.print(f"\n[yellow]a deixar de esperar — o job continua.[/yellow] `vramd cancel {job_id[:8]}` cancela.")
        sys.exit(130)
    if final is None:
        console.print("[yellow]vramd caiu durante wait.[/yellow]")
        sys.exit(1)
    ok = final.get("status") == P.STATUS_OK
    if as_json:
        _print_json(final)
    elif ok:
        dbg = final.get("ums_debug") or {}
        took = dbg.get("generate_sec")
        took_txt = f" em {took:.1f}s" if isinstance(took, (int, float)) else ""
        console.print(f"[bold green]✓ done[/bold green]{took_txt} {final.get('output') or ''}".rstrip())
    else:
        _print_ums_error(final)
    sys.exit(0 if ok else 1)


@cli.command("cancel")
@click.argument("job_id", required=False, default=None)
@click.option("--all", "cancel_all_flag", is_flag=True, help="Cancela todos (queued + running).")
@click.option(
    "--queued-only",
    is_flag=True,
    help="Com --all: só queued (não pede cancel aos running).",
)
@click.option("--json", "as_json", is_flag=True, help="Dump JSON da resposta.")
def cancel_cmd(
    job_id: str | None,
    cancel_all_flag: bool,
    queued_only: bool,
    as_json: bool,
) -> None:
    """Cancela job (UUID ou prefixo) ou ``--all`` / ``all`` / ``*``."""
    want_all = cancel_all_flag or (job_id is not None and job_id.strip().lower() in ("all", "*"))
    if want_all:
        resp = _send(
            {"cmd": P.CMD_CANCEL, "all": True, "queued_only": queued_only},
            timeout=30.0,
        )
    else:
        if not job_id:
            console.print("[red]Uso: ums cancel <job_id|prefixo> | ums cancel --all[/red]")
            sys.exit(2)
        resp = _send({"cmd": P.CMD_CANCEL, "job_id": job_id}, timeout=10.0)
    if resp is None:
        console.print("[yellow]vramd não está ativo.[/yellow]")
        sys.exit(1)
    if as_json:
        _print_json(resp)
        if resp.get("status") != "ok":
            sys.exit(1)
        return
    if resp.get("status") == "ok":
        if want_all or "count" in resp:
            console.print(
                f"[bold green]✓[/bold green] flush: {resp.get('message', resp)} (count={resp.get('count', '?')})"
            )
        else:
            jid = str(resp.get("job_id") or job_id or "")
            console.print(
                f"[bold green]✓[/bold green] job {_short_job_id(jid)} → "
                f"{resp.get('state', '?')} {resp.get('message', '')}"
            )
        if resp.get("ums_debug"):
            console.print(f"[dim]ums_debug: {json.dumps(resp['ums_debug'], ensure_ascii=False)}[/dim]")
    else:
        _print_ums_error(resp)
        sys.exit(1)


@cli.command("flush")
@click.option(
    "--queued-only",
    is_flag=True,
    help="Só cancela queued (não pede cancel aos running).",
)
@click.option("--json", "as_json", is_flag=True, help="Dump JSON da resposta.")
def flush_cmd(queued_only: bool, as_json: bool) -> None:
    """Limpa a fila (alias de ``cancel --all``)."""
    resp = _send({"cmd": P.CMD_FLUSH, "queued_only": queued_only}, timeout=30.0)
    if resp is None:
        console.print("[yellow]vramd não está ativo.[/yellow]")
        sys.exit(1)
    if as_json:
        _print_json(resp)
        if resp.get("status") != "ok":
            sys.exit(1)
        return
    if resp.get("status") == "ok":
        console.print(f"[bold green]✓[/bold green] {resp.get('message', 'fila limpa')} (count={resp.get('count', 0)})")
    else:
        _print_ums_error(resp)
        sys.exit(1)


@cli.command("queue")
@click.option("--json", "as_json", is_flag=True, help="Dump JSON completo (inclui debug).")
def queue_cmd(as_json: bool) -> None:
    """Lista jobs na fila e em execução."""
    resp = _send({"cmd": P.CMD_QUEUE}, timeout=5.0)
    if resp is None:
        console.print("[yellow]vramd não está ativo.[/yellow]")
        sys.exit(1)

    if as_json:
        _print_json(resp)
        return

    dbg = resp.get("debug") or {}
    depth = int(resp.get("queue_depth") or 0)
    inflight = int(resp.get("inflight") or 0)
    console.print(
        Panel.fit(
            f"[bold]Fila vramd[/bold] — {depth} queued, "
            f"{inflight} inflight, max_depth={resp.get('max_depth', '?')}\n"
            f"[bold]{format_ums_holding_summary(resp)}[/bold]\n"
            f"[dim]loaded={dbg.get('loaded_backends', [])} "
            f"max_cuts={dbg.get('max_affinity_cuts', '?')} "
            f"max_inflight={dbg.get('max_inflight', '?')}[/dim]",
            border_style="blue",
        )
    )
    _print_do_not_kill_tip(inflight=inflight, depth=depth)

    def _print_jobs(title: str, jobs: list) -> None:
        if not jobs:
            console.print(f"[dim]{title}: (vazio)[/dim]")
            return
        jt = Table(title=f"[bold]{title}", box=box.SIMPLE)
        jt.add_column("job_id", style="cyan")
        jt.add_column("backend")
        jt.add_column("priority")
        jt.add_column("state")
        jt.add_column("cuts", justify="right")
        jt.add_column("wait_s", justify="right")
        jt.add_column("gen_s", justify="right")
        jt.add_column("progress")
        for j in jobs:
            pct = j.get("progress_pct")
            msg = j.get("progress_msg") or ""
            prog = f"{pct:.0%}" if isinstance(pct, (int, float)) else "—"
            if msg:
                prog = f"{prog} {msg}"[:28]
            jt.add_row(
                _short_job_id(j.get("job_id")),
                str(j.get("backend", "")),
                str(j.get("priority", "")),
                str(j.get("state", "")),
                str(j.get("affinity_cuts", 0)),
                str(j.get("queue_wait_sec") if j.get("queue_wait_sec") is not None else "—"),
                str(j.get("generate_sec") if j.get("generate_sec") is not None else "—"),
                prog,
            )
        console.print(jt)
        if jobs:
            console.print(
                "[dim]job_id truncado acima — `cancel` / `wait` aceitam prefixo ou UUID completo "
                f"(ex.: cancel {_short_job_id(jobs[0].get('job_id'), n=8)})[/dim]"
            )

    _print_jobs("Running", resp.get("running") or [])
    _print_jobs("Queued", resp.get("queued") or [])


@cli.command("wait")
@click.argument("job_id")
@click.option("--timeout", default=600.0, show_default=True, type=float, help="Segundos máximos.")
@click.option("--json", "as_json", is_flag=True, help="Dump JSON da resposta final.")
def wait_cmd(job_id: str, timeout: float, as_json: bool) -> None:
    """Bloqueia até o job UMS terminar (ou timeout)."""
    from vramd.client import wait_ums_job

    console.print(f"[dim]À espera do job {job_id}… ({UMS_DO_NOT_KILL_TIP})[/dim]")
    resp = wait_ums_job(job_id, timeout_sec=timeout)
    if resp is None:
        console.print("[yellow]vramd não está ativo ou job desconhecido.[/yellow]")
        sys.exit(1)
    if as_json:
        _print_json(resp)
        if resp.get("status") != "ok":
            sys.exit(1)
        return
    if resp.get("status") == "ok":
        console.print(f"[bold green]✓[/bold green] job {_short_job_id(job_id)} concluído")
        if resp.get("output"):
            console.print(f"[cyan]{resp['output']}[/cyan]")
    else:
        _print_ums_error(resp)
        sys.exit(1)


@cli.command("backends")
def backends_cmd() -> None:
    """Lista os backends registados (não precisa do vramd a correr)."""
    registry = Registry()
    t = Table(title="[bold blue]Backends registados", box=box.SIMPLE)

    # Estado loaded se o vramd estiver up.
    loaded_set: set[str] = set()
    resp = _send({"cmd": P.CMD_LIST_BACKENDS}, timeout=5.0)
    if resp and resp.get("status") == "ok":
        loaded_set = {b["name"] for b in resp.get("backends", []) if b.get("loaded")}

    t.add_column("Backend", style="cyan")
    t.add_column("Adapter")
    t.add_column("VRAM (MiB)", justify="right")
    t.add_column("Priority", justify="right")
    t.add_column("Estado")
    for desc in registry:
        estado = "[green]carregado[/green]" if desc.name in loaded_set else "[dim]—[/dim]"
        t.add_row(desc.name, desc.adapter, str(desc.vram_mib), str(desc.priority), estado)
    console.print(t)


@cli.command("preload")
@click.argument("name")
@click.option("--json", "as_json", is_flag=True, help="Dump JSON da resposta.")
def preload_cmd(name: str, as_json: bool) -> None:
    """Pré-carrega um backend (ex: text2icon)."""
    resp = _send({"cmd": P.CMD_PRELOAD, "backend": name}, timeout=600.0)
    if resp is None:
        console.print("[yellow]vramd não está ativo. Arranca com: vramd start[/yellow]")
        sys.exit(1)
    if as_json:
        _print_json(resp)
        if resp.get("status") != "ok":
            sys.exit(1)
        return
    if resp.get("status") == "ok":
        console.print(f"[bold green]✓ {resp.get('message', 'pré-carregado')}[/bold green]")
        if resp.get("ums_debug"):
            console.print(f"[dim]ums_debug: {json.dumps(resp['ums_debug'], ensure_ascii=False)}[/dim]")
    else:
        _print_ums_error(resp)
        sys.exit(1)


@cli.command("evict")
@click.argument("name", required=False)
def evict_cmd(name: str | None) -> None:
    """Evicta um backend específico ou todos (sem argumento).

    Descarrega os pesos do modelo mas MANTÉM o worker vivo — NÃO apanha código
    novo da tool. Para recarregar código editado, usa ``vramd respawn``.
    """
    request: dict = {"cmd": P.CMD_RELEASE}
    if name:
        request["backend"] = name
    resp = _send(request, timeout=60.0)
    if resp is None:
        console.print("[yellow]vramd não está ativo.[/yellow]")
        sys.exit(0)
    if resp.get("status") == "ok":
        console.print(f"[bold green]✓ {resp.get('message', 'evicted')}[/bold green]")
    else:
        console.print(f"[bold red]✗ {resp.get('error', resp)}[/bold red]")
        sys.exit(1)


@cli.command("reap")
@click.option("--dry-run", is_flag=True, help="Só lista o que seria terminado.")
@click.option("--json", "as_json", is_flag=True, help="Dump JSON da resposta.")
def reap_cmd(dry_run: bool, as_json: bool) -> None:
    """Termina supervisores/workers UMS órfãos que seguram VRAM.

    Com o vramd ativo, o pedido é delegado nele (protege-se a si e aos seus
    workers). Sem UMS, o reap corre localmente e limpa toda a família — é o caso
    típico de um supervisor zombie que já não responde no socket.
    """
    resp = _send({"cmd": P.CMD_REAP, "dry_run": dry_run}, timeout=30.0)
    if resp is None:
        # None = down OU sem resposta. Reap local num supervisor VIVO mas lento
        # (hung/timeout) matava-o com jobs a meio — re-verificar antes de agir.
        from .client import is_server_running

        if is_server_running(P.DEFAULT_SOCKET_PATH):
            console.print("[bold red]✗ vramd ativo mas sem resposta ao reap — não vou reaper localmente.[/bold red]")
            console.print("Confirma com `vramd status`; se o supervisor está wedged, investiga antes do reap.")
            sys.exit(1)
        from .process_guard import reap_strays as _reap

        resp = {"status": P.STATUS_OK, "local": True, **_reap(dry_run=dry_run)}
    if as_json:
        _print_json(resp)
        return
    count = int(resp.get("count") or 0)
    if not count:
        console.print("[green]✓ Nenhum processo vramd órfão.[/green]")
        return
    freed = resp.get("vram_mib_freed")
    if dry_run:
        for line in resp.get("would_reap") or []:
            console.print(f"[yellow]• {line}[/yellow]")
        console.print(f"[yellow]{count} órfão(s), ~{freed} MiB — corre sem --dry-run para terminar.[/yellow]")
        return
    for entry in resp.get("reaped") or []:
        console.print(f"[green]✓[/green] PID {entry.get('pid')} ({entry.get('kind')}) — {entry.get('signal')}")
    console.print(f"[bold green]✓ {count} processo(s) terminado(s); ~{freed} MiB de VRAM recuperados.[/bold green]")


@cli.command("respawn")
@click.argument("name", required=False)
@click.option(
    "--lazy/--hot",
    "lazy",
    default=True,
    help="lazy (default): mata o worker vivo; o próximo generate arranca-o com código novo. "
    "hot: mata e recarrega já o modelo (fica quente).",
)
def respawn_cmd(name: str | None, lazy: bool) -> None:
    """Reinicia SÓ o worker de um backend para apanhar código novo da tool.

    Depois de editar código de uma tool (ex.: Text3D ``utils/export.py``), o
    worker persistente em ``<Tool>/.venv`` ainda tem o módulo antigo em memória
    — ``evict`` só larga os pesos. ``respawn`` mata o subprocesso e arranca um
    novo no venv da tool, pelo que o próximo ``generate`` já corre o código
    atualizado, SEM reiniciar o supervisor UMS.

    Sem argumento: reinicia todos os backends com worker subprocesso.
    """
    request: dict = {"cmd": P.CMD_RESPAWN, "lazy": lazy}
    if name:
        request["backend"] = name
    resp = _send(request, timeout=120.0)
    if resp is None:
        console.print("[yellow]vramd não está ativo — nada para respawnar.[/yellow]")
        sys.exit(0)
    if resp.get("status") != "ok":
        _print_ums_error(resp)
        _print_do_not_kill_tip()
        sys.exit(1)

    results = resp.get("results", []) or []
    mode = "lazy" if resp.get("lazy", lazy) else "hot"
    killed = sum(1 for r in results if r.get("respawned"))
    total = len(results)
    console.print(f"[bold green]✓ {killed}/{total} worker(s) reiniciado(s) ({mode}) — supervisor intacto.[/bold green]")
    for r in results:
        rname = r.get("name", "?")
        if r.get("respawned"):
            tag = f"reiniciado ({mode})"
        elif r.get("mode") in ("in-process", "no-pool"):
            tag = f"[dim]no-op: {r.get('reason', r.get('mode'))}[/dim]"
        else:
            tag = "[dim]não estava vivo[/dim]"
        shape = r.get("load_shape") or {}
        shape_str = f" shape={shape}" if shape else ""
        console.print(f"  • {rname}: {tag}{shape_str}")
    # Recomendar preload se foi hot e a shape não chegou (worker novo precisa de load).
    if mode == "hot":
        console.print(
            "[dim]Próximo generate está quente se o load_shape foi preservado; "
            "caso contrário usa `vramd preload <backend>`.[/dim]"
        )
    else:
        console.print("[dim]Worker arranca no próximo generate/preload — já com código atualizado.[/dim]")


@cli.command("zero")
def zero_cmd() -> None:
    """Zera TODA a VRAM segurada pelo vramd SEM parar o supervisor.

    ``evict`` só larga os pesos — os workers ficam vivos a segurar o contexto
    CUDA (~0.3-1 GiB cada). ``zero`` termina todos os workers (o próximo
    generate arranca-os frescos), evicta resíduos e scrubba caches. Recusa com
    fila ocupada — nunca mata um worker a meio de um job.
    """
    resp = _send({"cmd": P.CMD_ZERO}, timeout=120.0)
    if resp is None:
        console.print("[yellow]vramd não está ativo — nada para zerar.[/yellow]")
        sys.exit(0)
    if resp.get("status") != "ok":
        _print_ums_error(resp)
        _print_do_not_kill_tip()
        sys.exit(1)

    for r in resp.get("results", []) or []:
        rname = r.get("name", "?")
        if r.get("killed"):
            tag = "terminado (contexto CUDA libertado)"
        elif r.get("was_alive"):
            tag = "[dim]skip (load/generate em curso)[/dim]"
        else:
            tag = "[dim]não estava vivo[/dim]"
        model_str = " (tinha modelo)" if r.get("had_model") else ""
        console.print(f"  • {rname}: {tag}{model_str}")
    fb, fa = resp.get("free_mib_before"), resp.get("free_mib_after")
    free_str = f" — VRAM livre: {fb} → {fa} MiB" if isinstance(fa, int) else ""
    console.print(f"[bold green]✓ {resp.get('message', 'VRAM zerada')}{free_str}[/bold green]")


def _print_queue_metrics(qm: dict[str, Any], *, affinity_hits: object = None) -> None:
    """Tabela compacta de métricas de fila."""
    if not qm and affinity_hits is None:
        return
    t = Table(title="[bold]Fila (métricas)", box=box.SIMPLE)
    t.add_column("Métrica", style="cyan")
    t.add_column("Valor", justify="right")
    rows = [
        ("enqueued", qm.get("enqueued", 0)),
        ("completed", qm.get("completed", 0)),
        ("cancelled", qm.get("cancelled", 0)),
        ("queue_full", qm.get("queue_full_count", 0)),
        ("affinity_cutsΣ", qm.get("affinity_cuts_total", 0)),
        ("max_depth_seen", qm.get("max_depth_seen", 0)),
        ("wait_p50_s", qm.get("queue_wait_p50_sec", "—")),
        ("wait_p95_s", qm.get("queue_wait_p95_sec", "—")),
        ("wait_samples", qm.get("queue_wait_samples", 0)),
    ]
    if affinity_hits is not None:
        rows.append(("affinity_hits", affinity_hits))
    for k, v in rows:
        t.add_row(k, str(v))
    console.print(t)


def _budget_short(budget: dict[str, Any] | None) -> str:
    if not budget:
        return "—"
    parts: list[str] = []
    for key in ("num_chunks", "max_num_view", "tiles", "octree_resolution", "free_vram_mib"):
        if key in budget and budget[key] is not None:
            parts.append(f"{key}={budget[key]}")
    if not parts:
        # fallback: primeiras 2 keys
        for i, (k, v) in enumerate(budget.items()):
            if i >= 2:
                break
            parts.append(f"{k}={v}")
    return ", ".join(parts)[:48] or "—"


@cli.command("stats")
@click.option("--reset", is_flag=True, help="Limpa contadores in-process (NÃO para UMS / NÃO cancela jobs).")
@click.option("--json", "as_json", is_flag=True, help="Dump JSON completo.")
def stats_cmd(reset: bool, as_json: bool) -> None:
    """Estatísticas por backend + métricas de fila (loads/gens/timings/budget)."""
    resp = _send({"cmd": P.CMD_STATS, "reset": bool(reset)}, timeout=5.0)
    if resp is None:
        console.print("[yellow]vramd não está ativo.[/yellow]")
        sys.exit(1)

    if reset:
        if resp.get("reset") or resp.get("message"):
            if as_json:
                _print_json(resp)
            else:
                console.print(
                    f"[bold green]✓[/bold green] {resp.get('message', 'stats reset')} "
                    f"(PID {resp.get('pid', '?')}) — jobs/backends intactos"
                )
            if not resp.get("reset"):
                console.print(
                    "[dim]Nota: vramd antigo pode ignorar reset — reinicia o supervisor "
                    "quando puderes (não agora se houver jobs).[/dim]"
                )
            return
        console.print("[yellow]Reset não confirmado pelo vramd (supervisor antigo?).[/yellow]")
        sys.exit(1)

    if as_json:
        _print_json(resp)
        return

    q = resp.get("queue") or {}
    console.print(
        Panel.fit(
            f"[bold]vramd Stats[/bold] — PID {resp.get('pid', '?')}, "
            f"{resp.get('requests_served', 0)} pedidos, "
            f"idle-evict {resp.get('idle_evict_timeout_sec', '?')}s\n"
            f"fila {q.get('queue_depth', 0)}q / {q.get('inflight', 0)}run · "
            f"inflight≤{resp.get('max_inflight', '?')} · cuts≤{resp.get('max_affinity_cuts', '?')}",
            border_style="blue",
        )
    )
    _print_do_not_kill_tip(
        inflight=int(q.get("inflight") or 0),
        depth=int(q.get("queue_depth") or 0),
    )

    _print_queue_metrics(
        dict(resp.get("queue_metrics") or {}),
        affinity_hits=resp.get("affinity_hits"),
    )

    backends = resp.get("backends", {})
    if not backends:
        console.print("[dim]Sem atividade registada (nenhum backend usado ainda).[/dim]")
        return

    t = Table(title="[bold blue]Backends", box=box.SIMPLE)
    t.add_column("Backend", style="cyan")
    t.add_column("Loads", justify="right")
    t.add_column("Gens", justify="right")
    t.add_column("Evicts", justify="right")
    t.add_column("Err", justify="right")
    t.add_column("AvgLoad", justify="right")
    t.add_column("AvgGen", justify="right")
    t.add_column("LastGen", justify="right")
    t.add_column("Idle", justify="right")
    t.add_column("Budget", style="dim")

    for name in sorted(backends):
        s = backends[name]
        idle = s.get("idle_sec")
        idle_s = f"{idle:.0f}s" if isinstance(idle, (int, float)) else "—"
        t.add_row(
            name,
            str(s.get("load_count", 0)),
            str(s.get("generate_count", 0)),
            str(s.get("evict_count", 0)),
            str(s.get("error_count", 0)),
            f"{float(s.get('avg_load_time_sec') or 0):.1f}s",
            f"{float(s.get('avg_generate_time_sec') or 0):.1f}s",
            f"{float(s.get('last_generate_time_sec') or 0):.1f}s",
            idle_s,
            _budget_short(s.get("last_runtime_budget")),
        )
    console.print(t)

    dbg = resp.get("debug") or {}
    last_errors = dbg.get("last_errors") or {n: s.get("last_error") for n, s in backends.items() if s.get("last_error")}
    if last_errors:
        et = Table(title="[bold yellow]last_error", box=box.SIMPLE)
        et.add_column("Backend", style="cyan")
        et.add_column("Erro", style="yellow")
        for name, err in last_errors.items():
            et.add_row(str(name), str(err)[:140])
        console.print(et)


@cli.command("debug")
@click.option("--json", "as_json", is_flag=True, help="Dump JSON agregado (status+queue+stats).")
@click.option(
    "--watch",
    "watch_sec",
    type=float,
    default=0.0,
    help="Re-imprimir a cada N segundos (0=uma vez). Só leitura — não para jobs.",
)
def debug_cmd(as_json: bool, watch_sec: float) -> None:
    """Snapshot debug read-only: HOLDING, fila, erros, budgets, GPU.

    Nunca faz stop/flush/cancel/evict. Ideal enquanto batch corre.
    """

    def _once() -> int:
        status = _send({"cmd": P.CMD_STATUS}, timeout=5.0)
        if status is None:
            console.print("[yellow]vramd não está ativo.[/yellow]")
            return 1
        queue = _send({"cmd": P.CMD_QUEUE}, timeout=5.0) or {}
        stats = _send({"cmd": P.CMD_STATS}, timeout=5.0) or {}

        if as_json:
            _print_json({"status": status, "queue": queue, "stats": stats})
            return 0

        q = status.get("queue") or {}
        depth = int(q.get("queue_depth") or queue.get("queue_depth") or 0)
        inflight = int(q.get("inflight") or queue.get("inflight") or 0)
        dbg = status.get("debug") or {}
        hold = format_ums_holding_summary(queue if queue else q)

        free_s = "?"
        with contextlib.suppress(Exception):
            from vramd.gpu import query_gpu_free_mib

            free = query_gpu_free_mib()
            if free is not None:
                free_s = f"{free} MiB"

        console.print(
            Panel.fit(
                f"[bold]vramd Debug[/bold] — PID {status.get('pid', '?')}\n"
                f"[bold]{hold}[/bold]\n"
                f"GPU free≈{free_s} · loaded={dbg.get('loaded_backends', status.get('loaded', []))}\n"
                f"affinity_hits={dbg.get('affinity_hits', stats.get('affinity_hits', '—'))} · "
                f"eta={status.get('eta_sec', queue.get('eta_sec', '—'))}s",
                border_style="magenta",
            )
        )
        _print_do_not_kill_tip(inflight=inflight, depth=depth)

        _print_queue_metrics(
            dict(status.get("queue_metrics") or stats.get("queue_metrics") or {}),
            affinity_hits=dbg.get("affinity_hits", stats.get("affinity_hits")),
        )

        last_errors = dbg.get("last_errors") or (stats.get("debug") or {}).get("last_errors") or {}
        if last_errors:
            et = Table(title="[bold yellow]last_errors", box=box.SIMPLE)
            et.add_column("Backend", style="cyan")
            et.add_column("Erro")
            for name, err in last_errors.items():
                et.add_row(str(name), str(err)[:140])
            console.print(et)

        budgets = (stats.get("debug") or {}).get("last_runtime_budgets") or {}
        if not budgets:
            for name, s in (stats.get("backends") or {}).items():
                if s.get("last_runtime_budget"):
                    budgets[name] = s["last_runtime_budget"]
        if budgets:
            bt = Table(title="[bold]last_runtime_budget", box=box.SIMPLE)
            bt.add_column("Backend", style="cyan")
            bt.add_column("Budget", style="dim")
            for name, b in sorted(budgets.items()):
                bt.add_row(str(name), _budget_short(b if isinstance(b, dict) else None))
            console.print(bt)

        workers = dbg.get("workers") or {}
        if workers:
            wt = Table(title="[bold]workers", box=box.SIMPLE)
            wt.add_column("Backend", style="cyan")
            wt.add_column("PID", justify="right")
            wt.add_column("Estado")
            wt.add_column("Jobs", justify="right")
            wt.add_column("Spawn/Rec", justify="right")
            wt.add_column("Exit", justify="right")
            wt.add_column("Motivo")
            wt.add_column("Progresso", style="dim")
            for name, w in sorted(workers.items()):
                pct = w.get("last_progress_pct")
                pct_s = f"{float(pct):.0%}" if isinstance(pct, (int, float)) else "—"
                age = w.get("last_event_age_sec")
                age_s = f" há {age:.0f}s" if isinstance(age, (int, float)) else ""
                st = ("● " if w.get("loaded") else "○ ") + ("vivo" if w.get("alive") else "morto")
                wt.add_row(
                    str(name),
                    str(w.get("pid") or "—"),
                    st,
                    f"{w.get('jobs_served', 0)}/{w.get('max_jobs', '?')}",
                    f"{w.get('spawn_count', 0)}/{w.get('recycle_count', 0)}",
                    str(w.get("last_exit_code") if w.get("last_exit_code") is not None else "—"),
                    str(w.get("last_abort_reason") or "—"),
                    f"{pct_s} {str(w.get('last_progress_msg') or '—')[:28]}{age_s}",
                )
            console.print(wt)

        probe = (status.get("learn") or {}).get("probe") or {}
        if probe:
            ok = bool(probe.get("ok"))
            console.print(
                f"[{'green' if ok else 'yellow'}]probe VRAM: {probe.get('last_reason') or 'ok'}"
                f" | falhas={probe.get('consecutive_failures', 0)}"
                f" | backoff={probe.get('backoff_sec_remaining', 0)}s"
                f" | pid={probe.get('last_pid', '—')}[/]"
            )

        running = list(queue.get("running") or [])
        queued = list(queue.get("queued") or [])
        if running or queued:
            jt = Table(title="[bold]Jobs", box=box.SIMPLE)
            jt.add_column("state")
            jt.add_column("job_id", style="cyan")
            jt.add_column("backend")
            jt.add_column("pri")
            jt.add_column("pct", justify="right")
            for j in running:
                pct = j.get("progress_pct")
                pct_s = f"{pct:.0%}" if isinstance(pct, (int, float)) else "—"
                jt.add_row(
                    "RUN",
                    _short_job_id(j.get("job_id")),
                    str(j.get("backend") or "?"),
                    str(j.get("priority") or "?"),
                    pct_s,
                )
            for j in queued[:12]:
                jt.add_row(
                    "Q",
                    _short_job_id(j.get("job_id")),
                    str(j.get("backend") or "?"),
                    str(j.get("priority") or "?"),
                    "—",
                )
            if len(queued) > 12:
                jt.add_row("…", f"+{len(queued) - 12} queued", "", "", "")
            console.print(jt)

        console.print("[dim]Só leitura. Para parar jobs: ums cancel / flush — nunca kill GPU enquanto HOLDING.[/dim]")
        return 0

    if watch_sec and watch_sec > 0:
        try:
            while True:
                console.clear()
                code = _once()
                if code != 0:
                    # Sem isto o watch morria com exit 0 quando o daemon
                    # desaparecia — scripts de monitorização não notavam.
                    sys.exit(code)
                time.sleep(watch_sec)
        except KeyboardInterrupt:
            console.print("\n[dim]debug watch parado (vramd intacto).[/dim]")
            return
    rc = _once()
    if rc:
        sys.exit(rc)


def _percentile(samples: list[float], p: float) -> float | None:
    if not samples:
        return None
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, max(0, round((p / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


@cli.command("bench")
@click.option("--rounds", default=20, show_default=True, type=int, help="Rounds por comando IPC.")
@click.option("--json", "as_json", is_flag=True, help="Dump JSON com amostras.")
@click.option(
    "--cmds",
    default="status,queue,stats",
    show_default=True,
    help="Comandos RPC a medir (vírgula). Só leitura — sem generate/submit.",
)
def bench_cmd(rounds: int, as_json: bool, cmds: str) -> None:
    """Benchmark RTT do socket do vramd (IPC). Não submete jobs GPU.

    Seguro com batch a correr: só ``status`` / ``queue`` / ``stats``.
    """
    if rounds < 1:
        raise click.ClickException("--rounds deve ser ≥ 1")

    allowed = {"status": P.CMD_STATUS, "queue": P.CMD_QUEUE, "stats": P.CMD_STATS}
    wanted = [c.strip().lower() for c in cmds.split(",") if c.strip()]
    for c in wanted:
        if c not in allowed:
            raise click.ClickException(f"cmd não permitido no bench: {c} (só {sorted(allowed)})")

    # Probe busy (read-only) — aviso, não aborta.
    q0 = _send({"cmd": P.CMD_QUEUE}, timeout=5.0)
    if q0 is None:
        console.print("[yellow]vramd não está ativo.[/yellow]")
        sys.exit(1)
    depth = int(q0.get("queue_depth") or 0)
    inflight = int(q0.get("inflight") or 0)
    if depth or inflight:
        console.print(
            f"[yellow]vramd ocupado ({format_ums_holding_summary(q0)}) — bench IPC continua; NÃO submete GPU.[/yellow]"
        )
        _print_do_not_kill_tip(inflight=inflight, depth=depth)

    results: dict[str, Any] = {"rounds": rounds, "busy": bool(depth or inflight), "cmds": {}}

    console.print(
        Panel.fit(
            f"[bold]vramd Bench IPC[/bold] — {rounds} rounds · cmds={wanted}\n"
            f"[dim]Sem generate/submit/preload/evict — jobs a correr ficam intactos.[/dim]",
            border_style="cyan",
        )
    )

    t = Table(title="[bold blue]RTT (ms)", box=box.ROUNDED)
    t.add_column("cmd", style="cyan")
    t.add_column("n", justify="right")
    t.add_column("min", justify="right")
    t.add_column("p50", justify="right")
    t.add_column("avg", justify="right")
    t.add_column("p95", justify="right")
    t.add_column("max", justify="right")
    t.add_column("err", justify="right")

    for name in wanted:
        cmd = allowed[name]
        samples_ms: list[float] = []
        errors = 0
        for _ in range(rounds):
            t0 = time.perf_counter()
            resp = _send({"cmd": cmd}, timeout=5.0)
            dt = (time.perf_counter() - t0) * 1000.0
            if resp is None:
                errors += 1
            else:
                samples_ms.append(dt)
        if samples_ms:
            avg = sum(samples_ms) / len(samples_ms)
            p50 = _percentile(samples_ms, 50)
            p95 = _percentile(samples_ms, 95)
            row = {
                "n": len(samples_ms),
                "min_ms": round(min(samples_ms), 2),
                "p50_ms": round(p50, 2) if p50 is not None else None,
                "avg_ms": round(avg, 2),
                "p95_ms": round(p95, 2) if p95 is not None else None,
                "max_ms": round(max(samples_ms), 2),
                "errors": errors,
            }
            results["cmds"][name] = row
            t.add_row(
                name,
                str(row["n"]),
                f"{row['min_ms']:.2f}",
                f"{row['p50_ms']:.2f}" if row["p50_ms"] is not None else "—",
                f"{row['avg_ms']:.2f}",
                f"{row['p95_ms']:.2f}" if row["p95_ms"] is not None else "—",
                f"{row['max_ms']:.2f}",
                str(errors),
            )
        else:
            results["cmds"][name] = {"n": 0, "errors": errors}
            t.add_row(name, "0", "—", "—", "—", "—", "—", str(errors))

    if as_json:
        _print_json(results)
    else:
        console.print(t)
        console.print("[dim]Valores = round-trip Unix socket (não tempo de generate GPU).[/dim]")


@cli.command("doctor")
@click.option("--fix", is_flag=True, help="Corrige o que é seguro (reap de processos UMS órfãos).")
def doctor_cmd(fix: bool) -> None:
    """Diagnostica: deps de backends, GPU, socket, fila, peak VRAM, órfãos, legacy."""
    import importlib

    from rich.panel import Panel

    from vramd import host as _host

    console.print(Panel.fit("[bold]vramd Doctor[/bold] — diagnóstico de ambiente", border_style="blue"))

    checks: list[tuple[str, bool, str]] = []

    # 1. Socket do vramd ativo?
    from vramd.client import UMS_SOCKET, discover_active_sockets, is_ums_running

    ums_up = is_ums_running()
    checks.append(("vramd ativo", ums_up, "Socket presente e respondendo" if ums_up else "Arrancar: vramd start"))
    checks.append(
        (
            "Transporte",
            True,
            f"{transport.transport_mode()} — {transport.endpoint_label(P.DEFAULT_SOCKET_PATH)}",
        )
    )

    free_mib: int | None = None
    with contextlib.suppress(Exception):
        from vramd.gpu import query_gpu_free_mib

        free_mib = query_gpu_free_mib()

    qresp: dict | None = None
    # Fila / scheduler / peak (se UMS up).
    if ums_up:
        qresp = _send({"cmd": P.CMD_STATUS}, timeout=5.0)
        if qresp:
            q = qresp.get("queue") or {}
            depth = q.get("queue_depth", 0)
            inflight = q.get("inflight", 0)
            qm = qresp.get("queue_metrics") or q.get("metrics") or {}
            eta = qresp.get("eta_sec")
            dbg = qresp.get("debug") or {}
            affinity_hits = dbg.get("affinity_hits", qm.get("affinity_hits", "—"))
            detail = (
                f"depth={depth}/{q.get('max_depth', '?')}, inflight={inflight}, "
                f"affinity_cuts≤{qresp.get('max_affinity_cuts', '?')}, "
                f"affinity_hits={affinity_hits}, "
                f"eta={eta if eta is not None else '—'}s, "
                f"fulls={qm.get('queue_full_count', 0)}, "
                f"wait_p95={qm.get('queue_wait_p95_sec', '—')}"
            )
            max_d = int(q.get("max_depth") or 32)
            ok_q = depth < max_d
            checks.append(("Fila vramd", ok_q, detail if ok_q else f"SATURADA — {detail}"))

            # Peak vs free por backend carregado.
            backends = qresp.get("backends") or []
            loaded = [b for b in backends if b.get("loaded")]
            if loaded:
                parts = []
                for b in loaded:
                    peak = b.get("peak_mib") or b.get("vram_mib") or "?"
                    parts.append(f"{b.get('name')} peak={peak} MiB")
                free_s = f"{free_mib} MiB livres" if free_mib is not None else "free=?"
                # Aviso informativo: free baixo com modelos já loaded é normal —
                # só aparece como falha quando há vários loaded e quase nada livre.
                low_free = [
                    b.get("name")
                    for b in loaded
                    if free_mib is not None
                    and isinstance(b.get("peak_mib"), (int, float))
                    and free_mib < int(b["peak_mib"]) * 0.15
                ]
                peak_ok = not low_free
                extra = f"; free baixo para {', '.join(low_free)}" if low_free else ""
                checks.append(
                    (
                        "Backends carregados",
                        peak_ok,
                        f"{free_s}; " + "; ".join(parts) + extra,
                    )
                )
            else:
                free_s = f"{free_mib} MiB livres" if free_mib is not None else "free=?"
                checks.append(("Backends carregados", True, f"nenhum — {free_s}"))

            if inflight or depth:
                checks.append(
                    (
                        "Não matar GPU",
                        True,
                        "vramd tem jobs na fila — usa `vramd queue` / cancel / wait; NÃO kill processos GPU",
                    )
                )

    # Órfãos: supervisores/workers de runs anteriores a segurar VRAM.
    from .process_guard import reap_strays as _reap
    from .process_guard import stray_report as _stray_report

    strays: dict[str, Any] = {}
    if ums_up and qresp:
        strays = qresp.get("strays") or {}
    else:
        with contextlib.suppress(Exception):
            strays = _stray_report()
    stray_count = int(strays.get("count") or 0)
    if stray_count:
        detail = ", ".join(_describe_stray(p) for p in strays.get("processes") or [])
        if fix:
            if ums_up:
                # CRÍTICO: com o daemon vivo, o reap tem de correr NELE — a
                # protecção de self/ancestrais/descendentes só existe no lado
                # do supervisor. O reap local classificava o daemon VIVO como
                # órfão e SIGTERM/SIGKILL'ava supervisor + workers com jobs a
                # meio (o snapshot filtrado acima não protege a execução local).
                report = _send({"cmd": P.CMD_REAP, "dry_run": False}, timeout=30.0)
                report = report if isinstance(report, dict) else {}
            else:
                with contextlib.suppress(Exception):
                    from .client import is_server_running

                    if is_server_running(P.DEFAULT_SOCKET_PATH):
                        console.print("[red]✗ vramd tornou-se ativo a meio do doctor — reap delegado:[/red]")
                        sys.exit(1)
                report = _reap()
            checks.append(
                (
                    "Processos vramd órfãos",
                    True,
                    f"{report.get('count')} terminado(s) (~{report.get('vram_mib_freed')} MiB) — {detail}",
                )
            )
        else:
            checks.append(
                (
                    "Processos vramd órfãos",
                    False,
                    f"{stray_count} a segurar ~{strays.get('vram_mib')} MiB ({detail}) — corre `vramd reap`",
                )
            )
    else:
        checks.append(("Processos vramd órfãos", True, "nenhum"))

    # Legacy per-tool sockets (conflito potencial com UMS).
    try:
        legacy = [s for s in discover_active_sockets() if Path(s).resolve() != Path(UMS_SOCKET).resolve()]
    except Exception:
        legacy = []
    if legacy:
        names = ", ".join(Path(s).name for s in legacy)
        checks.append(
            (
                "Sockets legacy",
                False,
                f"Activos: {names} — conflito com UMS; para ou VRAMD_ALLOW_LEGACY_SERVER=1 só se preciso",
            )
        )
    else:
        checks.append(("Sockets legacy", True, "nenhum per-tool activo"))

    # Hooks: config malformada só rebentaria no arranque do supervisor — o
    # doctor dá o feedback antes disso.
    from .hooks import load_hooks

    try:
        hook_specs = load_hooks()
        if hook_specs:
            events = ", ".join(sorted({e for spec in hook_specs for e in spec.events}))
            checks.append(("Hooks", True, f"{len(hook_specs)} configurado(s): {events}"))
        else:
            checks.append(("Hooks", True, "nenhum (~/.config/vramd/hooks.yaml)"))
    except (ValueError, OSError) as e:
        checks.append(("Hooks", False, f"hooks.yaml inválido — o vramd não vai arrancar: {e}"))

    # 2. GPU disponível? (NVML preferido; fallback nvidia-smi via Shared)
    from vramd.gpu import check_nvidia_driver_match, list_gpu_snapshots, nvml_available

    driver_ok, driver_detail = check_nvidia_driver_match()
    checks.append(("NVIDIA driver match", driver_ok, driver_detail))

    snaps = list_gpu_snapshots()
    if snaps:
        s0 = snaps[0]
        extra = f" (+{len(snaps) - 1} GPU)" if len(snaps) > 1 else ""
        gpu_detail = f"{s0.name} — {s0.total_mib} MiB total, {s0.free_mib} MiB livres via {s0.source}{extra}"
    elif nvml_available():
        gpu_detail = "NVML ok mas sem devices"
    elif _host.nvidia_smi_path() is not None:
        gpu_detail = "nvidia-smi presente mas sem leitura de memória"
    else:
        gpu_detail = "NVML/nvidia-smi indisponível"
    checks.append(("GPU NVIDIA", bool(snaps), gpu_detail))

    # 3. hf_xet (downloads HF acelerados via hub >=1.5)
    try:
        importlib.import_module("hf_xet")
        checks.append(("hf_xet", True, "Downloads HF acelerados (Xet / hub >=1.5)"))
    except ImportError:
        checks.append(("hf_xet", False, "pip install 'hf-xet>=1.2' — downloads HF acelerados"))

    # 4. Deps de cada backend (verificar se o módulo da tool importa).
    from .registry import Registry

    registry = Registry()
    tool_modules = {
        "text2icon": "text2icon.generator",
        "texture2d": "texture2d.generator",
        "text2d": "text2d.generator",
        "skymap2d": "skymap2d.generator",
        "text3d": "text3d.generator",
        "paint3d": "paint3d.painter",
        "text2sound": "text2sound.generator",
        "terrain3d": "terrain3d.generator",
        "part3d": "part3d.pipeline",
    }
    for backend_name in sorted(registry.names):
        mod_path = tool_modules.get(backend_name)
        if mod_path is None:
            checks.append((f"Backend {backend_name}", False, "mapping em falta"))
            continue
        try:
            importlib.import_module(mod_path)
            checks.append((f"Backend {backend_name}", True, "deps OK"))
        except ImportError as e:
            checks.append((f"Backend {backend_name}", False, f"ImportError: {e.name or e}"))

    # Renderizar tabela.
    t = Table(title="[bold blue]Diagnóstico", box=box.ROUNDED)
    t.add_column("Check", style="cyan", no_wrap=True)
    t.add_column("Estado")
    t.add_column("Detalhe", style="dim")

    all_ok = True
    for name, passed, detail in checks:
        status = "[green]✓ OK[/green]" if passed else "[red]✗ FALHA[/red]"
        if not passed:
            all_ok = False
        t.add_row(name, status, detail)

    console.print(t)
    _print_do_not_kill_tip()
    if ums_up and qresp and (qresp.get("queue") or {}).get("queue_depth"):
        console.print(
            "[yellow]Hint:[/yellow] há jobs na fila — [bold]não mates GPU[/bold]; "
            "usa [cyan]vramd queue[/cyan] / cancel."
        )
    if all_ok:
        console.print("[bold green]✓ Todos os checks passaram.[/bold green]")
    else:
        console.print("[yellow]Alguns checks falharam — ver detalhes acima.[/yellow]")
        # CI/health scripts viam sempre exit 0: o doctor dizia «falhou» e
        # ninguém abaixo dele notava.
        sys.exit(1)


def _coerce_scalar(raw: str) -> Any:
    """``"4"`` → 4, ``"0.7"`` → 0.7, ``"true"`` → True, ``"none"`` → None, resto string."""
    text = raw.strip()
    low = text.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null", ""):
        return None
    with contextlib.suppress(ValueError):
        return int(text)
    with contextlib.suppress(ValueError):
        return float(text)
    return text


def _parse_kv(pairs: tuple[str, ...]) -> dict[str, Any]:
    """``("a=1", "b=x")`` → ``{"a": 1, "b": "x"}``.

    Raises:
        click.BadParameter: par sem ``=``.
    """
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise click.BadParameter(f"esperado K=V, recebido {pair!r}")
        key, _, value = pair.partition("=")
        out[key.strip()] = _coerce_scalar(value)
    return out


def _render_calibration(cal: Any) -> None:
    """Tabela com a decomposição medida + avisos."""
    colour = {"high": "green", "medium": "yellow", "low": "red"}.get(cal.confidence, "white")
    console.print(
        Panel.fit(
            f"[bold]{cal.backend}[/bold] — pico medido [bold]{cal.peak_mib} MiB[/bold] "
            f"(admit {cal.admit_peak_mib} MiB) · confiança [{colour}]{cal.confidence}[/{colour}]",
            border_style=colour,
        )
    )
    t = Table(box=box.SIMPLE, show_header=True)
    t.add_column("Componente")
    t.add_column("MiB", justify="right")
    t.add_column("GiB", justify="right")
    t.add_column("Origem", style="dim")
    t.add_row("contexto CUDA", str(cal.context_mib), f"{cal.context_gib:.2f}", "residual pós-unload")
    t.add_row("pesos", str(cal.weights_mib), f"{cal.weights_gib:.2f}", "residente pós-load - contexto")
    t.add_row("activação", str(cal.activation_mib), f"{cal.activation_gib:.2f}", "pico generate - residente")
    t.add_row("pico do load", str(cal.load_peak_mib), "", "transiente de carregamento")
    t.add_row("pico do generate", str(cal.generate_peak_mib), "", f"máx de {cal.repeats} repetições")
    t.add_row("[bold]pico[/bold]", f"[bold]{cal.peak_mib}[/bold]", f"{cal.peak_gib:.2f}", "max(load, generate)")
    t.add_row("safety recomendado", str(cal.recommended_safety_mib), "", "dispersão entre repetições")
    console.print(t)

    q = Table(box=box.SIMPLE, show_header=True, title="Qualidade da medição")
    q.add_column("Métrica")
    q.add_column("Valor", justify="right")
    q.add_row("amostras", str(cal.samples_n))
    q.add_row("maior gap", f"{cal.max_gap_sec:.3f}s (alvo {cal.interval_sec:.3f}s)")
    q.add_row("amostras sem dados", f"{cal.missed_ratio:.1%}")
    q.add_row("VRAM de terceiros", f"{cal.foreign_baseline_mib} → {cal.foreign_max_mib} MiB")
    q.add_row("load / generate", f"{cal.load_sec:.1f}s / {cal.generate_sec_median:.1f}s (mediana)")
    console.print(q)

    for warning in cal.warnings:
        console.print(f"[yellow]![/yellow] {warning}")


def _render_comparison(rows: list[Any]) -> int:
    """Tabela medido vs declarado; devolve o número de métricas subdimensionadas."""
    from .calibrate.compare import VERDICT_OK, VERDICT_OVER, VERDICT_UNDER

    t = Table(box=box.SIMPLE, show_header=True, title="Declarado vs medido")
    t.add_column("Métrica")
    t.add_column("Declarado", justify="right")
    t.add_column("Medido", justify="right")
    t.add_column("Δ", justify="right")
    t.add_column("Rácio", justify="right")
    t.add_column("Veredicto")
    styles = {
        VERDICT_OK: "[green]ok[/green]",
        VERDICT_UNDER: "[red]subdimensionado[/red]",
        VERDICT_OVER: "[yellow]sobredimensionado[/yellow]",
    }
    under = 0
    for row in rows:
        if row.verdict == VERDICT_UNDER:
            under += 1
        delta = "—" if row.delta_mib is None else f"{row.delta_mib:+d}"
        ratio = "—" if row.ratio is None else f"{row.ratio:.2f}x"
        t.add_row(
            row.metric,
            "—" if row.declared_mib is None else str(row.declared_mib),
            str(row.measured_mib),
            delta,
            ratio,
            styles.get(row.verdict, row.verdict),
        )
    console.print(t)
    return under


def _resolve_calibrate_assets(request: dict[str, Any]) -> None:
    """Resolve inputs de teste do calibrador para o package data do vramd.

    Um descriptor declara ``calibrate_request: {mesh_path: test-mesh.glb, …}``
    com um NOME curto em vez de um path. Se o nome corresponde a um ficheiro
    empacotado em ``vramd/data/`` (test-mesh.glb, test-image.png), é
    substituído pelo path absoluto do pacote — o utilizador comum não precisa
    de ter nem gerar um modelo de teste.
    """
    from importlib import resources

    for key, value in list(request.items()):
        if not isinstance(value, str) or not value:
            continue
        if Path(value).is_absolute() or "/" in value or "\\" in value:
            continue  # path real — não mexer
        bundled = resources.files("vramd").joinpath("data", value)
        if bundled.is_file():
            request[key] = str(bundled)


@cli.command("calibrate")
@click.argument("backend")
@click.option("--prompt", default=None, help="Prompt do job de calibração (atalho para --request-json).")
@click.option("--output", "output_path", default=None, help="Path de output do job.")
@click.option(
    "--request-json",
    "request_json",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Ficheiro JSON com o request completo (backends sem prompt: paint3d, part3d…).",
)
@click.option("--load-kwarg", "load_kwargs_raw", multiple=True, metavar="K=V", help="Kwarg de load (repetível).")
@click.option(
    "--hw-auto/--no-hw-auto",
    "use_hw_auto",
    default=True,
    show_default=True,
    help="Obter os kwargs de load do hw-auto da tool (o que a produção usa). --load-kwarg tem precedência.",
)
@click.option("--quant", default=None, help="Atalho para --load-kwarg sdnq_preset=… (etiqueta no relatório).")
@click.option("--repeats", default=3, show_default=True, help="Repetições do generate (≥2 separa warmup).")
@click.option("--cycles", default=1, show_default=True, help="Pares load/unload extra (isola contexto CUDA).")
@click.option("--interval", default=0.05, show_default=True, help="Intervalo de amostragem (s).")
@click.option("--settle", default=1.5, show_default=True, help="Espera para o driver assentar (s).")
@click.option("--baseline", default=1.0, show_default=True, help="Janela de silêncio antes do spawn (s).")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=None, help="Escreve o YAML calibrado.")
@click.option("--report", "report_path", type=click.Path(dir_okay=False), default=None, help="Escreve o JSON.")
@click.option("--compare/--no-compare", default=True, show_default=True, help="Comparar com o declarado.")
@click.option(
    "--raw/--no-raw",
    "keep_raw",
    default=True,
    show_default=True,
    help="Guardar as amostras cruas no --report (permite `vramd recalibrate` sem GPU).",
)
@click.option("--zero/--no-zero", default=True, show_default=True, help="`vramd zero` antes de medir (GPU limpa).")
@click.option(
    "--wait-free",
    "wait_free",
    default=120.0,
    show_default=True,
    help="Segundos à espera que a VRAM de terceiros drene antes de medir (0 = não esperar).",
)
@click.option("--force", is_flag=True, help="Ignora o preflight (medição pode ficar contaminada).")
@click.option("--json", "as_json", is_flag=True, help="Dump JSON do relatório.")
def calibrate_cmd(
    backend: str,
    prompt: str | None,
    output_path: str | None,
    request_json: str | None,
    load_kwargs_raw: tuple[str, ...],
    use_hw_auto: bool,
    quant: str | None,
    repeats: int,
    cycles: int,
    interval: float,
    settle: float,
    baseline: float,
    out_path: str | None,
    report_path: str | None,
    compare: bool,
    keep_raw: bool,
    zero: bool,
    wait_free: float,
    force: bool,
    as_json: bool,
) -> None:
    """Mede o footprint VRAM real de um backend e emite o descriptor YAML.

    Corre um job real com amostragem por processo a ~20 Hz, separa contexto
    CUDA / pesos / activação, e compara com o que está declarado.

    A GPU tem de estar livre: com jobs em curso a medição mede-os a eles.
    """
    from .calibrate import CalibrationRunner, CalibrationSpec
    from .calibrate.compare import compare_to_declared, declared_parts_from_registry
    from .calibrate.emit import calibration_to_report, calibration_to_yaml
    from .subprocess_pool import SubprocessWorkerPool

    registry = Registry()
    try:
        desc = registry.descriptor(backend)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(2)

    explicit_kwargs = _parse_kv(load_kwargs_raw)
    load_kwargs: dict[str, Any] = {}
    if use_hw_auto:
        from .calibrate.hw_auto import resolve_hw_auto_kwargs

        auto_kwargs, hw_error = resolve_hw_auto_kwargs(desc.tool)
        if hw_error:
            console.print(f"[yellow]hw-auto indisponível:[/yellow] {hw_error} — a medir só com os kwargs dados.")
        elif auto_kwargs:
            console.print(f"[dim]hw-auto ({desc.tool}): {auto_kwargs}[/dim]")
        load_kwargs.update(auto_kwargs)
    # ``calibrate_load_kwargs`` do descriptor (config real de produção do
    # backend) ganha ao hw-auto; o explícito do CLI ganha a ambos.
    load_kwargs.update(desc.calibrate_load_kwargs)
    load_kwargs.update(explicit_kwargs)
    if quant:
        load_kwargs.setdefault("sdnq_preset", quant)
    # quant_preset é o alias Text2D de sdnq_preset — sem isto uma calibração
    # com quant_preset ficava etiquetada quant_mode=none e o admit descartava
    # a medição (quant mismatch) quando a produção pedia o preset.
    quant_mode = str(
        load_kwargs.get("sdnq_preset") or load_kwargs.get("quant_preset") or load_kwargs.get("quant_mode") or "none"
    )

    if request_json:
        request = json.loads(Path(request_json).read_text(encoding="utf-8"))
    else:
        # Default do descriptor (prompt/output com a extensão certa por
        # backend) — sem isto ``calibrate <backend>`` falha em backends que
        # exigem inputs (mesh_path/output) ou formatos específicos.
        request = dict(desc.calibrate_request)
        request.setdefault("prompt", prompt or "calibration job")
        if output_path:
            request["output"] = output_path
        request.setdefault("output", f"/tmp/vramd-calibrate-{backend}.bin")
    _resolve_calibrate_assets(request)

    pool = SubprocessWorkerPool()
    runner = CalibrationRunner(pool)

    if wait_free > 0:
        remaining = runner.wait_until_drained(timeout_sec=wait_free)
        console.print(f"[dim]VRAM de terceiros: {remaining} MiB[/dim]")

    blockers = runner.preflight()
    if blockers:
        for reason in blockers:
            console.print(f"[red]✗[/red] {reason}")
        if not force:
            console.print("[dim]Usa --force para medir na mesma (resultado marcado como pouco fiável).[/dim]")
            sys.exit(2)

    if zero:
        with contextlib.suppress(Exception):
            from vramd.client import zero_ums_vram

            if zero_ums_vram() is not None:
                console.print("[dim]vramd zero: workers idle terminados antes de medir.[/dim]")

    spec = CalibrationSpec(
        backend=backend,
        tool=desc.tool,
        request=request,
        load_kwargs=load_kwargs,
        repeats=repeats,
        cycles=cycles,
        baseline_sec=baseline,
        settle_sec=settle,
        interval_sec=interval,
        quant_mode=quant_mode,
    )

    console.print(f"[dim]A calibrar {backend} ({repeats}x generate, amostragem {interval:.3f}s)…[/dim]")
    try:
        cal = runner.run(spec)
    except Exception as exc:
        console.print(f"[red]Calibração falhou:[/red] {exc}")
        sys.exit(1)
    finally:
        with contextlib.suppress(Exception):
            pool.shutdown_all()

    report = calibration_to_report(cal, windows=runner.last_windows if keep_raw else None)
    if as_json:
        _print_json(report)
    else:
        _render_calibration(cal)

    exit_code = 0
    if compare:
        weights, activation, vram_mib = declared_parts_from_registry(backend, registry=registry, quant_mode=quant_mode)
        rows = compare_to_declared(
            cal,
            declared_weights_mib=weights,
            declared_activation_mib=activation,
            declared_vram_mib=vram_mib,
        )
        report["comparison"] = [row.as_dict() for row in rows]
        if not as_json:
            under = _render_comparison(rows)
            if under:
                console.print(f"[red]{under} métrica(s) subdimensionada(s)[/red] — risco de OOM em admissão.")
                exit_code = 1

    if out_path:
        # Preservar runtime/load_keys/shape_keys do descriptor atual — o emit
        # regenera o bloco runtime quando não o recebe, apagando command/cwd/
        # env/timeouts de backends externos (bug: calibrate --out perdia a
        # configuração de arranque do backend).
        meta = {
            backend: {
                "adapter": desc.adapter,
                "priority": desc.priority,
                "footprint_key": desc.footprint_key,
                "runtime": desc.runtime.to_dict() if desc.runtime is not None else None,
                "load_keys": list(desc.load_keys) if desc.load_keys else None,
                "shape_keys": list(desc.shape_keys) if desc.shape_keys else None,
            }
        }
        _atomic_write_text(out_path, calibration_to_yaml(cal, descriptors=meta))
        console.print(f"[green]YAML escrito:[/green] {out_path}")
    if report_path:
        _atomic_write_text(report_path, json.dumps(report, indent=2, ensure_ascii=False))
        console.print(f"[green]Relatório escrito:[/green] {report_path}")

    sys.exit(exit_code)


@cli.command("recalibrate")
@click.argument("report_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=None, help="Escreve o YAML recalculado.")
@click.option("--report", "new_report", type=click.Path(dir_okay=False), default=None, help="Escreve o JSON novo.")
@click.option("--compare/--no-compare", default=True, show_default=True, help="Comparar com o declarado.")
@click.option("--json", "as_json", is_flag=True, help="Dump JSON do relatório.")
def recalibrate_cmd(
    report_path: str, out_path: str | None, new_report: str | None, compare: bool, as_json: bool
) -> None:
    """Re-deriva uma calibração a partir das amostras cruas de um relatório.

    Medir custa minutos de GPU exclusiva; derivar custa microssegundos. Quando a
    análise melhora, isto refaz os números dos relatórios já existentes **sem
    voltar a ocupar a placa**. Exige um ``--report`` gravado com ``--raw``.
    """
    from .calibrate.compare import compare_to_declared, declared_parts_from_registry
    from .calibrate.emit import calibration_to_report, calibration_to_yaml
    from .calibrate.serde import derive_from_report

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    try:
        cal = derive_from_report(report)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(2)

    old_peak = ((report.get("vram_mib") or {}).get("peak")) or 0
    console.print(
        f"[dim]{cal.backend}: pico {old_peak} → {cal.peak_mib} MiB "
        f"({(report.get('quality') or {}).get('confidence', '?')} → {cal.confidence})[/dim]"
    )

    fresh = calibration_to_report(cal, windows=None)
    # As amostras cruas viajam com o relatório novo: senão, recalibrar uma vez
    # tornava o ficheiro não-recalibrável outra vez.
    fresh["raw_samples"] = report["raw_samples"]

    if as_json:
        _print_json(fresh)
    else:
        _render_calibration(cal)

    exit_code = 0
    if compare:
        weights, activation, vram_mib = declared_parts_from_registry(cal.backend, quant_mode=cal.quant_mode)
        rows = compare_to_declared(
            cal, declared_weights_mib=weights, declared_activation_mib=activation, declared_vram_mib=vram_mib
        )
        fresh["comparison"] = [row.as_dict() for row in rows]
        if not as_json and _render_comparison(rows):
            exit_code = 1

    if out_path:
        registry = Registry()
        meta = {}
        with contextlib.suppress(KeyError):
            desc = registry.descriptor(cal.backend)
            meta = {
                cal.backend: {
                    "adapter": desc.adapter,
                    "priority": desc.priority,
                    "footprint_key": desc.footprint_key,
                    # Preservar runtime do descriptor — ver calibrate_cmd.
                    "runtime": desc.runtime.to_dict() if desc.runtime is not None else None,
                    "load_keys": list(desc.load_keys) if desc.load_keys else None,
                    "shape_keys": list(desc.shape_keys) if desc.shape_keys else None,
                }
            }
        _atomic_write_text(out_path, calibration_to_yaml(cal, descriptors=meta))
        console.print(f"[green]YAML escrito:[/green] {out_path}")
    if new_report:
        _atomic_write_text(new_report, json.dumps(fresh, indent=2, ensure_ascii=False))
        console.print(f"[green]Relatório escrito:[/green] {new_report}")

    sys.exit(exit_code)


_VERDICT_STYLE = {
    "ok": "[green]ok[/green]",
    "underprovisioned": "[bold red]subdimensionado[/bold red]",
    "overprovisioned": "[yellow]sobredimensionado[/yellow]",
    "no_data": "[dim]sem dados[/dim]",
}


def _render_learn(resp: dict[str, Any]) -> int:
    """Tabela de drift declarado vs observado; devolve nº de backends em risco."""
    console.print(
        Panel.fit(
            f"[bold]vramd Learn[/bold] — pico observado na produção vs declarado\n"
            f"[dim]amostragem {resp.get('interval_sec', '?')}s · "
            f"{'ativa' if resp.get('enabled') else '[red]desativada (VRAMD_LEARN_INTERVAL_SEC=0)'}[/dim]",
            border_style="magenta",
        )
    )
    rows = list(resp.get("backends") or [])
    if not rows:
        console.print("[dim]Sem observações ainda — corre jobs; o tracker aprende com tráfego real.[/dim]")
        return 0
    t = Table(box=box.SIMPLE, title="Declarado vs observado")
    t.add_column("Backend", style="cyan")
    t.add_column("Declarado", justify="right")
    t.add_column("p95 obs.", justify="right")
    t.add_column("Máx obs.", justify="right")
    t.add_column("Amostras", justify="right")
    t.add_column("Veredicto")
    t.add_column("Sugestão", justify="right")
    actionable = 0
    for r in rows:
        verdict = str(r.get("verdict") or "no_data")
        if verdict in ("underprovisioned", "overprovisioned") and not r.get("has_measured_block"):
            actionable += 1
        t.add_row(
            str(r.get("backend")),
            "—" if r.get("declared_peak_mib") is None else f"{r['declared_peak_mib']} MiB",
            "—" if r.get("observed_p95_mib") is None else f"{r['observed_p95_mib']} MiB",
            "—" if r.get("observed_max_mib") is None else f"{r['observed_max_mib']} MiB",
            str(r.get("samples", 0)),
            _VERDICT_STYLE.get(verdict, verdict),
            "—" if not r.get("suggested_mib") else f"{r['suggested_mib']} MiB",
        )
    console.print(t)
    if any(r.get("has_measured_block") for r in rows):
        console.print("[dim]Backends com bloco vram: calibrado mantêm a calibração — o learn não a sobrepõe.[/dim]")
    return actionable


@cli.command("learn")
@click.option("--json", "as_json", is_flag=True, help="Dump JSON do relatório.")
@click.option(
    "--apply",
    "apply_fix",
    is_flag=True,
    help="Escreve as correcções como overlay YAML em <config>/backends.d/learned.yaml.",
)
@click.option("--reset", is_flag=True, help="Apaga as observações (todas ou --backend X).")
@click.option("--backend", default=None, help="Filtra um backend (com --reset: só esse).")
def learn_cmd(as_json: bool, apply_fix: bool, reset: bool, backend: str | None) -> None:
    """Drift do pico de VRAM: observado na produção vs declarado na admissão.

    O supervisor amostra a VRAM real de cada job (~2 Hz) e compara com o pico
    que usou para admitir. ``--apply`` fecha o loop: escreve um overlay YAML
    com os vram_mib corrigidos, sem tocar no package — e sem sobrepôr
    calibrações feitas com ``vramd calibrate``.
    """
    if reset:
        req: dict = {"cmd": P.CMD_LEARN, "reset": True}
        if backend:
            req["backend"] = backend
        resp = _send(req, timeout=10.0)
        if resp is None:
            console.print("[yellow]vramd não está ativo.[/yellow]")
            sys.exit(1)
        if resp.get("status") != P.STATUS_OK:
            _print_ums_error(resp)
            sys.exit(1)
        console.print(f"[bold green]✓[/bold green] {resp.get('message', 'observações limpas')}")
        return

    resp = _send({"cmd": P.CMD_LEARN}, timeout=10.0)
    if resp is None:
        console.print("[yellow]vramd não está ativo.[/yellow]")
        sys.exit(1)
    if resp.get("status") != P.STATUS_OK:
        _print_ums_error(resp)
        sys.exit(1)
    if as_json:
        _print_json(resp)
        return

    actionable = _render_learn(resp)
    probe = resp.get("probe") or {}
    if probe:
        ok = bool(probe.get("ok"))
        detail = (
            f"probe VRAM: {probe.get('last_reason') or 'ok'}"
            f" (pid={probe.get('last_pid', '—')}, falhas={probe.get('consecutive_failures', 0)}"
            f", backoff={probe.get('backoff_sec_remaining', 0)}s)"
        )
        if probe.get("last_error"):
            detail += f" — {probe['last_error']}"
        console.print(f"[{'green' if ok else 'yellow'}]{detail}[/]")
    if not apply_fix:
        if actionable:
            console.print(
                f"[yellow]{actionable} backend(s) com drift accionável —[/yellow] "
                "corre com [cyan]--apply[/cyan] para escrever o overlay."
            )
        return

    rows = [r for r in (resp.get("backends") or []) if r.get("suggested_mib") and not r.get("has_measured_block")]
    if not rows:
        console.print("[green]✓ Nada a corrigir — declarado e observado concordam.[/green]")
        return

    from .learn import DriftReport, learn_overlay_yaml
    from .registry import DEFAULT_BACKENDS_DIR

    overlay = learn_overlay_yaml([DriftReport.from_dict(r) for r in rows])
    out_dir = DEFAULT_BACKENDS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "learned.yaml"
    _atomic_write_text(out_path, overlay)
    fixed = ", ".join(f"{r['backend']}→{r['suggested_mib']} MiB" for r in rows)
    console.print(f"[bold green]✓ Overlay escrito:[/bold green] {out_path}")
    console.print(f"  [dim]{fixed}[/dim]")
    console.print("[dim]Efectivo no próximo arranque do supervisor (vramd stop && vramd start).[/dim]")


@cli.command("top")
@click.option("--interval", default=1.0, show_default=True, type=float, help="Segundos entre refresh.")
@click.option("--once", is_flag=True, help="Renderiza um frame e sai (CI/scripts).")
def top_cmd(interval: float, once: bool) -> None:
    """Dashboard TUI live: GPU, fila com progresso, backends e drift.

    Só de leitura — seguro com batch a correr. Ctrl+C para sair.
    """
    from .top import run_top

    try:
        sys.exit(run_top(interval_sec=interval, once=once))
    except KeyboardInterrupt:
        console.print("\n[dim]top parado (vramd intacto — só leitura).[/dim]")


@cli.command("mcp")
def mcp_cmd() -> None:
    """Servidor MCP sobre stdio — a fila de VRAM como tools de agentes de IA.

    Registo num cliente MCP (Claude Desktop, etc.)::

        {"mcpServers": {"vramd": {"command": "vramd", "args": ["mcp"]}}}
    """
    from .mcp_server import serve_stdio

    sys.exit(serve_stdio())


@cli.command("config")
@click.option(
    "--file",
    "config_file",
    type=click.Path(),
    default=None,
    help="Onde escrever o config.yml (default: /etc/vramd/config.yml se root).",
)
@click.option("--show", is_flag=True, help="Mostra a configuração efectiva e sai.")
@click.option("--json", "as_json", is_flag=True, help="Com --show, imprime JSON.")
@click.option("--non-interactive", is_flag=True, help="Não faz perguntas; usa flags/defaults.")
@click.option("--listen", default=None, help="HOST:PORT do listener.")
@click.option("--cache-dir", default=None, help="Cache do vramd (sockets, sandboxes, envs).")
@click.option("--token-file", default=None, help="Token estável dos clientes remotos.")
@click.option("--hf-home", default=None, help="HF_HOME (cache normal do Hugging Face).")
@click.option("--hf-token-file", default=None, help="Ficheiro com o HF_TOKEN (modelos gated).")
@click.option(
    "--models-policy",
    type=click.Choice(["auto", "hub", "off"]),
    default=None,
    help="auto: tenta o Hub; hub: exige rede; off: só cache local.",
)
@click.option("--models-max-gib", type=int, default=None, help="Tecto do cache HF para o GC.")
@click.option("--gpus", default=None, help="GPUs locais: auto, 0,1, off.")
@click.option("--allow-client-code/--no-allow-client-code", default=None, help="Aceitar sessões de código.")
@click.option("--allow-client-env/--no-allow-client-env", default=None, help="Construir envs com uv.")
@click.option("--session-ttl-hours", type=int, default=None, help="TTL das sessões sem uso.")
@click.option("--service-user", default=None, help="Utilizador da unidade systemd.")
@click.option("--install-service/--no-install-service", default=None, help="Instalar/activar systemd.")
@click.option("--move-models/--no-move-models", default=None, help="Mover o cache HF antigo.")
@click.option("--keep-source/--no-keep-source", default=None, help="Manter a cópia antiga do cache.")
@click.option("--yes", "-y", is_flag=True, help="Não pedir confirmação para mover/instalar.")
def config_cmd(
    config_file: str | None,
    show: bool,
    as_json: bool,
    non_interactive: bool,
    listen: str | None,
    cache_dir: str | None,
    token_file: str | None,
    hf_home: str | None,
    hf_token_file: str | None,
    models_policy: str | None,
    models_max_gib: int | None,
    gpus: str | None,
    allow_client_code: bool | None,
    allow_client_env: bool | None,
    session_ttl_hours: int | None,
    service_user: str | None,
    install_service: bool | None,
    move_models: bool | None,
    keep_source: bool | None,
    yes: bool,
) -> None:
    """Configura o servidor (wizard ou flags) e instala o serviço systemd.

    Exemplos::

        vramd config                       # wizard
        vramd config --show                # config efectiva
        vramd config --non-interactive --listen 0.0.0.0:8790 \\
            --hf-home /home/vramd/.cache/huggingface \\
            --allow-client-code --allow-client-env --install-service -y
    """
    import yaml

    from .config import (
        chown_for_user,
        detect_old_cache,
        ensure_token,
        load_config,
        migrate_models,
        resolve_path_for_user,
        validate_user,
        write_config,
        write_secret,
    )
    from .config import (
        install_service as install_service_fn,
    )

    cfg = load_config(config_file)
    if show:
        data = cfg.to_dict()
        if as_json:
            console.print_json(json.dumps(data, ensure_ascii=False))
        else:
            console.print(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), markup=False)
        return

    interactive = not non_interactive and sys.stdin.isatty() and not yes
    if interactive:
        cfg.listen = click.prompt("Listener (HOST:PORT)", default=cfg.listen)
        cfg.cache_dir = click.prompt("Cache do vramd", default=cfg.cache_dir)
        cfg.models_home = click.prompt("HF_HOME (cache de modelos)", default=cfg.models_home)
        cfg.models_policy = click.prompt(
            "Política de modelos",
            type=click.Choice(["auto", "hub", "off"]),
            default=cfg.models_policy,
        )
        cfg.models_max_gib = click.prompt("Tamanho máximo do cache HF (GiB)", type=int, default=cfg.models_max_gib)
        cfg.allow_client_code = click.confirm("Aceitar código do cliente (sessões)?", default=cfg.allow_client_code)
        cfg.allow_client_env = click.confirm("Construir envs Python no servidor (uv)?", default=cfg.allow_client_env)
        cfg.gpus = click.prompt("GPUs locais (auto/0,1/off)", default=cfg.gpus)
        cfg.session_ttl_hours = click.prompt("TTL das sessões sem uso (h)", type=int, default=cfg.session_ttl_hours)
        cfg.service_user = click.prompt("Utilizador do serviço", default=cfg.service_user)
        cfg.install_service = click.confirm("Instalar/activar o serviço systemd?", default=cfg.install_service)
        hf_token = click.prompt(
            "HF_TOKEN (opcional, Enter para ignorar)",
            default="",
            hide_input=True,
            show_default=False,
        )
        if hf_token.strip():
            write_secret(cfg.hf_token_file, hf_token)
        old = detect_old_cache(cfg)
        if old is not None:
            cfg.move_models = click.confirm(
                f"Mover o cache existente {old} para {cfg.models_home} (sem re-download)?",
                default=True,
            )
            if cfg.move_models:
                cfg.keep_source = click.confirm("Manter a cópia antiga?", default=False)

    # Flags explícitas ganham ao wizard.
    for attr, value in (
        ("listen", listen),
        ("cache_dir", cache_dir),
        ("token_file", token_file),
        ("models_home", hf_home),
        ("hf_token_file", hf_token_file),
        ("models_policy", models_policy),
        ("models_max_gib", models_max_gib),
        ("gpus", gpus),
        ("allow_client_code", allow_client_code),
        ("allow_client_env", allow_client_env),
        ("session_ttl_hours", session_ttl_hours),
        ("service_user", service_user),
        ("install_service", install_service),
        ("move_models", move_models),
        ("keep_source", keep_source),
    ):
        if value is not None:
            setattr(cfg, attr, value)

    try:
        validate_user(cfg.service_user)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)
    for attr in ("cache_dir", "token_file", "models_home", "hf_token_file"):
        setattr(cfg, attr, resolve_path_for_user(getattr(cfg, attr), cfg.service_user))

    target = write_config(cfg, config_file)
    token = ensure_token(cfg.token_file)
    Path(cfg.cache_dir).expanduser().mkdir(parents=True, exist_ok=True)
    Path(cfg.models_home).expanduser().mkdir(parents=True, exist_ok=True)
    for path in (
        target,
        token,
        Path(cfg.cache_dir),
        Path(cfg.models_home),
        Path(cfg.hf_token_file),
    ):
        chown_for_user(path, cfg.service_user)

    console.print(f"[green]config[/green] {target}")
    console.print(f"[green]token[/green]  {token} (0600)")

    old = detect_old_cache(cfg)
    if old is not None and cfg.move_models:
        console.print(
            "[yellow]Para o container vramd antigo antes de mover o cache "
            "(se ainda estiver a correr).[/yellow]"
        )
        plan = migrate_models(cfg, dry_run=True)
        if plan:
            console.print(
                f"[bold]Cache HF[/bold]: {plan['source']} → {plan['dest']} "
                f"({plan['files']} ficheiros, {plan['symlinks']} links, "
                f"{plan['bytes'] / (1024**3):.1f} GiB, "
                f"{'mesmo' if plan['same_filesystem'] else 'outro'} filesystem)"
            )
        if not yes and interactive:
            cfg.move_models = click.confirm("Mover agora?", default=True)
        if cfg.move_models:
            result = migrate_models(cfg)
            if result:
                # A migração corre como root; o serviço tem de ler o cache.
                chown_for_user(cfg.models_home, cfg.service_user)
                console.print(f"[green]modelos movidos[/green] {result['source']} → {result['dest']}")

    if cfg.install_service:
        if not yes and interactive:
            cfg.install_service = click.confirm("Instalar o serviço systemd agora?", default=True)
        if cfg.install_service:
            result = install_service_fn(cfg, config_file=target)
            console.print(f"[green]unidade[/green] {result['unit_path']}")
            for command in result["commands"]:
                console.print(f"  [dim]{command}[/dim]")

    console.print(
        "\n[bold]Próximos passos[/bold]\n"
        f"  servidor:  [cyan]vramd start --config {target}[/cyan]\n"
        f"  cliente:   [cyan]VRAMD_REMOTE=<host>:{cfg.listen.rsplit(':', 1)[-1]} "
        f"VRAMD_REMOTE_TOKEN=$(cat {token})[/cyan]\n"
        "  O cliente envia o bundle/env no início do batch; não é preciso checkout no servidor."
    )


@cli.group("models")
def models_group() -> None:
    """Cache de modelos HF (listar, garantir, GC) — local ou via VRAMD_REMOTE."""


def _models_request(request: dict[str, Any], *, timeout_sec: float = 300.0) -> dict[str, Any] | None:
    from .config import apply_env, load_config

    cfg = load_config()
    apply_env(cfg)
    sock = Path(cfg.cache_dir).expanduser() / "vramd.sock"
    return send_request(request, sock, timeout_sec=timeout_sec)


@models_group.command("list")
@click.option("--json", "as_json", is_flag=True, help="Dump JSON.")
def models_list_cmd(as_json: bool) -> None:
    """Lista os modelos no cache HF (repos, revisions, tamanho)."""
    response = _models_request({"cmd": P.CMD_MODELS}, timeout_sec=120.0)
    if response is None:
        console.print("[red]vramd não respondeu.[/red]")
        sys.exit(1)
    if response.get("status") != P.STATUS_OK:
        console.print(f"[red]{response.get('error') or response}[/red]")
        sys.exit(1)
    if as_json:
        console.print_json(json.dumps(response, ensure_ascii=False))
        return
    console.print(f"HF_HOME: [cyan]{response.get('home')}[/cyan]")
    total = int(response.get("bytes") or 0)
    console.print(f"Total: [green]{total / (1024**3):.1f} GiB[/green]")
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("repo")
    table.add_column("GiB", justify="right")
    table.add_column("revs", justify="right")
    table.add_column("pinned", justify="center")
    pins = set(response.get("pinned") or ())
    for repo in response.get("repos") or []:
        is_pinned = any(
            f"{repo['repo']}@{ref}" in pins for rev in repo.get("revisions") or () for ref in rev.get("refs") or ()
        )
        table.add_row(
            str(repo["repo"]),
            f"{int(repo.get('size') or 0) / (1024**3):.2f}",
            str(len(repo.get("revisions") or ())),
            "✓" if is_pinned else "",
        )
    console.print(table)


@models_group.command("ensure")
@click.argument("repo")
@click.option("--revision", default="main", show_default=True)
@click.option("--allow", multiple=True, help="Padrão allow (repetível).")
@click.option("--ignore", multiple=True, help="Padrão ignore (repetível).")
@click.option("--json", "as_json", is_flag=True, help="Dump JSON.")
def models_ensure_cmd(repo: str, revision: str, allow: tuple[str, ...], ignore: tuple[str, ...], as_json: bool) -> None:
    """Garante um modelo no cache (descarrega só o que faltar)."""
    response = _models_request(
        {
            "cmd": P.CMD_MODELS_ENSURE,
            "models": [
                {
                    "repo": repo,
                    "revision": revision,
                    "allow": list(allow),
                    "ignore": list(ignore),
                }
            ],
        },
        timeout_sec=float(os.environ.get("VRAMD_MODELS_TIMEOUT_SEC", 86400)),
    )
    if response is None or response.get("status") != P.STATUS_OK:
        console.print(f"[red]{(response or {}).get('error') or 'vramd não respondeu'}[/red]")
        sys.exit(1)
    if as_json:
        console.print_json(json.dumps(response, ensure_ascii=False))
    else:
        for model in response.get("models") or []:
            console.print(f"[green]{model['repo']}@{model['revision']}[/green] {model['path']}")


@models_group.command("gc")
@click.option("--json", "as_json", is_flag=True, help="Dump JSON.")
def models_gc_cmd(as_json: bool) -> None:
    """Remove revisions antigas do cache HF até ao tecto configurado."""
    response = _models_request({"cmd": P.CMD_MODELS_GC}, timeout_sec=600.0)
    if response is None or response.get("status") != P.STATUS_OK:
        console.print(f"[red]{(response or {}).get('error') or 'vramd não respondeu'}[/red]")
        sys.exit(1)
    if as_json:
        console.print_json(json.dumps(response, ensure_ascii=False))
    else:
        console.print(
            f"removidas [bold]{response.get('removed', 0)}[/bold] revisions, "
            f"libertados [bold]{(response.get('freed_bytes') or 0) / (1024**3):.2f} GiB[/green]"
        )


@cli.group("sessions")
def sessions_group() -> None:
    """Sessões de código abertas no servidor (`session-open`/`session-close`)."""


@sessions_group.command("list")
@click.option("--json", "as_json", is_flag=True, help="Dump JSON.")
def sessions_list_cmd(as_json: bool) -> None:
    """Lista sessões activas, bundles e envs."""
    response = _models_request({"cmd": P.CMD_SESSIONS}, timeout_sec=60.0)
    if response is None or response.get("status") != P.STATUS_OK:
        console.print(f"[red]{(response or {}).get('error') or 'vramd não respondeu'}[/red]")
        sys.exit(1)
    if as_json:
        console.print_json(json.dumps(response, ensure_ascii=False))
        return
    sessions = response.get("sessions") or []
    if not sessions:
        console.print("[dim]nenhuma sessão activa[/dim]")
        return
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("session")
    table.add_column("bundle")
    table.add_column("backends")
    table.add_column("idade", justify="right")
    for session in sessions:
        table.add_row(
            str(session.get("session")),
            str(session.get("bundle_sha", ""))[:12],
            ", ".join(session.get("backends") or ()),
            f"{float(session.get('age_sec') or 0):.0f}s",
        )
    console.print(table)


@sessions_group.command("close")
@click.argument("session")
@click.option("--force", is_flag=True, help="Mata workers mesmo com jobs (perigoso).")
def sessions_close_cmd(session: str, force: bool) -> None:
    """Fecha uma sessão (mata os workers e liberta a VRAM)."""
    response = _models_request(
        {"cmd": P.CMD_SESSION_CLOSE, "session": session, "force": force},
        timeout_sec=120.0,
    )
    if response is None or response.get("status") != P.STATUS_OK:
        console.print(f"[red]{(response or {}).get('error') or 'vramd não respondeu'}[/red]")
        sys.exit(1)
    console.print(f"[green]sessão {session} fechada[/green] (workers parados: {response.get('workers_stopped')})")


@cli.group("env")
def env_group() -> None:
    """Ambientes Python content-addressed (builds com uv)."""


@env_group.command("list")
@click.option("--json", "as_json", is_flag=True, help="Dump JSON.")
def env_list_cmd(as_json: bool) -> None:
    """Lista as envs construídas no servidor."""
    from .config import apply_env, load_config
    from .envs import EnvManager

    apply_env(load_config())
    items = EnvManager().list()
    if as_json:
        console.print_json(json.dumps(items, ensure_ascii=False))
        return
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("hash")
    table.add_column("python")
    table.add_column("path")
    for item in items:
        table.add_row(str(item.get("hash", ""))[:12], str(item.get("python", "")), str(item.get("path", "")))
    console.print(table)


@env_group.command("build")
@click.argument("lock", type=click.Path(exists=True))
@click.option("--python", default="3.13", show_default=True, help="Versão de Python do uv venv.")
@click.option("--index-url", default=None)
@click.option("--extra-index-url", default=None)
def env_build_cmd(lock: str, python: str, index_url: str | None, extra_index_url: str | None) -> None:
    """Constrói/reutiliza uma env a partir de um freeze (servidor ou cliente)."""
    from .config import apply_env, load_config
    from .envs import EnvError, EnvManager

    apply_env(load_config())
    try:
        info = EnvManager().build_lock(
            lock,
            python=python,
            index_url=index_url,
            extra_index_url=extra_index_url,
        )
    except EnvError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)
    console.print(f"[green]env[/green] {info['hash'][:12]} → {info['python_path']}")


_register_ops(cli)


def main() -> None:
    """Entry point para ``vramd`` / ``ums``."""
    cli()


if __name__ == "__main__":
    main()
