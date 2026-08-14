"""Servidor MCP do vramd — a GPU como ferramenta de agentes de IA.

Model Context Protocol sobre stdio (JSON-RPC 2.0, uma mensagem por linha —
sem dependências novas: o protocolo que um agente precisa é pequeno e o vramd
já fala JSON o dia todo). Corre com::

    vramd mcp

e transforma a fila de VRAM em tools que um agente pode chamar: consultar
estado, submeter jobs, esperar resultados, cancelar, evictar — com as
mutações destrutivas guardadas por um argumento ``confirm`` explícito (um
agente deve ser deliberado sobre libertar VRAM, tal como um humano).

Princípio herdado do CLI: as tools preferem ler a vós mexer. ``status``,
``queue``, ``stats``, ``learn`` e ``doctor`` são read-only; ``submit`` passa
pela fila de admissão como qualquer job; ``evict``/``zero`` recusam sem
``confirm: true`` — e continuam a recusar com jobs em curso (busy-guard do
supervisor), porque NÃO MATAR GPU a meio de um job é a regra nº 1 da casa.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from . import __version__
from . import protocol as P

# Subset do protocolo MCP que este servidor implementa (2025-06-18).
PROTOCOL_VERSION = "2025-06-18"

_JSONRPC = "2.0"

# Códigos de erro JSON-RPC usados.
_CODE_PARSE = -32700
_CODE_METHOD = -32601
_CODE_INVALID = -32602
_CODE_INTERNAL = -32603


def _text_result(text: str, *, is_error: bool = False, structured: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    }
    if structured is not None:
        out["structuredContent"] = structured
    return out


def _json_text_result(payload: Any, *, is_error: bool = False, preamble: str = "") -> dict[str, Any]:
    text = (preamble + "\n" if preamble else "") + json.dumps(payload, ensure_ascii=False, default=str, indent=2)
    return _text_result(text, is_error=is_error, structured=payload if isinstance(payload, dict) else None)


def _object_schema(properties: dict[str, Any], *, required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


_STR = {"type": "string"}
_NUM = {"type": "number"}
_BOOL = {"type": "boolean"}

# ---------------------------------------------------------------------------
# Tools — definição + handlers
# ---------------------------------------------------------------------------


def _tool_definitions() -> list[dict[str, Any]]:
    """``tools/list`` — descrições escritas PARA agentes (a UI é o prompt)."""
    return [
        {
            "name": "vramd_status",
            "description": (
                "Estado do supervisor de VRAM: backends carregados, VRAM usada, fila, "
                "ETA, processos órfãos. Read-only e seguro com jobs a correr. Começa por aqui."
            ),
            "inputSchema": _object_schema({}),
        },
        {
            "name": "vramd_queue",
            "description": "Jobs running (com progresso) e queued, com ETA estimado. Read-only.",
            "inputSchema": _object_schema({}),
        },
        {
            "name": "vramd_backends",
            "description": "Backends registados: footprint declarado, prioridade, estado loaded. Read-only.",
            "inputSchema": _object_schema({}),
        },
        {
            "name": "vramd_stats",
            "description": "Estatísticas por backend (loads/gerações/erros/timings) e métricas de fila. Read-only.",
            "inputSchema": _object_schema({}),
        },
        {
            "name": "vramd_learn",
            "description": (
                "Relatório de aprendizagem de picos: pico VRAM observado na produção vs "
                "declarado, por backend (veredictos under/overprovisioned). Read-only."
            ),
            "inputSchema": _object_schema({}),
        },
        {
            "name": "vramd_submit",
            "description": (
                "Submete um job de geração à fila de admissão. O campo 'request' é o payload "
                "do backend (prompt, output, kwargs…). Com wait=true bloqueia até terminar "
                "(default) e devolve o output; com wait=false devolve job_id para vramd_wait."
            ),
            "inputSchema": _object_schema(
                {
                    "backend": _STR,
                    "request": {"type": "object", "description": "Payload do backend (prompt, output, …)."},
                    "priority": {"type": "string", "enum": ["interactive", "batch"]},
                    "wait": _BOOL,
                    "timeout_sec": _NUM,
                },
                required=["backend"],
            ),
        },
        {
            "name": "vramd_wait",
            "description": "Espera um job terminar (devolve resultado/erro). Usa o job_id do vramd_submit.",
            "inputSchema": _object_schema({"job_id": _STR, "timeout_sec": _NUM}, required=["job_id"]),
        },
        {
            "name": "vramd_cancel",
            "description": "Cancela um job (queued: imediato; running: cooperativo entre fases).",
            "inputSchema": _object_schema(
                {"job_id": _STR, "all": _BOOL, "queued_only": _BOOL},
            ),
        },
        {
            "name": "vramd_preload",
            "description": "Pré-carrega um backend (pesos em VRAM) para o próximo job ser quente.",
            "inputSchema": _object_schema({"backend": _STR, "confirm": _BOOL}, required=["backend", "confirm"]),
        },
        {
            "name": "vramd_evict",
            "description": (
                "Descarrega os pesos de um backend (ou de todos, sem argumento) para libertar "
                "VRAM. Destrutivo para a cache quente — exige confirm=true."
            ),
            "inputSchema": _object_schema({"backend": _STR, "confirm": _BOOL}, required=["confirm"]),
        },
        {
            "name": "vramd_zero",
            "description": (
                "Liberta TODA a VRAM do supervisor (termina workers) sem o parar. Exige "
                "confirm=true; recusa com jobs em curso."
            ),
            "inputSchema": _object_schema({"confirm": _BOOL}, required=["confirm"]),
        },
        {
            "name": "vramd_doctor",
            "description": (
                "Diagnóstico read-only: driver NVIDIA, GPUs, sockets legacy, órfãos. "
                "Não corrige nada — para reparar, sugere ao utilizador os comandos."
            ),
            "inputSchema": _object_schema({}),
        },
    ]


def _ums(request: dict[str, Any], *, timeout_sec: float = 30.0, auto_start: bool = False) -> dict[str, Any]:
    from .client import send_to_ums

    return send_to_ums(request, timeout_sec=timeout_sec, auto_start=auto_start) or {}


_DOWN_HINT = "O vramd não está ativo (ou não respondeu). Arranca com `vramd start` e repete."


def _handle_status(_: dict[str, Any]) -> dict[str, Any]:
    resp = _ums({"cmd": P.CMD_STATUS}, timeout_sec=10.0)
    if not resp:
        return _text_result(_DOWN_HINT, is_error=True)
    return _json_text_result(resp)


def _handle_queue(_: dict[str, Any]) -> dict[str, Any]:
    resp = _ums({"cmd": P.CMD_QUEUE}, timeout_sec=10.0)
    if not resp:
        return _text_result(_DOWN_HINT, is_error=True)
    return _json_text_result(resp)


def _handle_backends(_: dict[str, Any]) -> dict[str, Any]:
    resp = _ums({"cmd": P.CMD_LIST_BACKENDS}, timeout_sec=10.0)
    if not resp:
        return _text_result(_DOWN_HINT, is_error=True)
    return _json_text_result(resp)


def _handle_stats(_: dict[str, Any]) -> dict[str, Any]:
    resp = _ums({"cmd": P.CMD_STATS}, timeout_sec=10.0)
    if not resp:
        return _text_result(_DOWN_HINT, is_error=True)
    return _json_text_result(resp)


def _handle_learn(_: dict[str, Any]) -> dict[str, Any]:
    resp = _ums({"cmd": P.CMD_LEARN}, timeout_sec=10.0)
    if not resp:
        return _text_result(_DOWN_HINT, is_error=True)
    return _json_text_result(resp)


def _clamp_timeout(value: Any, default: float = 600.0) -> float:
    """``timeout_sec`` do agente → float >= 1s (negativos passavam crus ao socket)."""
    try:
        return max(1.0, float(value))
    except (TypeError, ValueError):
        return default


def _handle_submit(args: dict[str, Any]) -> dict[str, Any]:
    from .client import wait_ums_job

    backend = str(args.get("backend") or "").strip()
    if not backend:
        return _text_result("backend é obrigatório.", is_error=True)
    request = args.get("request") or {}
    if not isinstance(request, dict):
        return _text_result("request deve ser um objecto JSON.", is_error=True)
    # Payload do agente PRIMEIRO, chaves reservadas re-afirmadas DEPOIS: com
    # ``{..., **request}`` um request malicioso/confuso com ``"cmd": "shutdown"``
    # sobrepunha o comando e passava por cima de TODOS os guards de confirm.
    payload = {**request, "cmd": P.CMD_SUBMIT, "backend": backend}
    payload.pop("job_id", None)
    payload.pop("stream", None)
    if args.get("priority"):
        payload["priority"] = str(args["priority"])
    resp = _ums(payload, timeout_sec=30.0, auto_start=True)
    if resp.get("status") != P.STATUS_OK:
        return _json_text_result(resp or {"error": _DOWN_HINT}, is_error=True)
    job_id = str(resp.get("job_id") or "")
    if not args.get("wait", True):
        return _json_text_result(resp, preamble=f"Job {job_id} submetido (não esperado).")
    wait_resp = wait_ums_job(job_id, timeout_sec=_clamp_timeout(args.get("timeout_sec")))
    if wait_resp is None:
        return _text_result(
            f"Job {job_id} submetido mas o wait falhou (vramd down a meio?). "
            "Verifica com vramd_queue antes de repetir — NÃO resubmetas às cegas.",
            is_error=True,
        )
    return _json_text_result(wait_resp, is_error=wait_resp.get("status") not in (P.STATUS_OK, None))


def _handle_wait(args: dict[str, Any]) -> dict[str, Any]:
    from .client import wait_ums_job

    job_id = str(args.get("job_id") or "").strip()
    if not job_id:
        return _text_result("job_id é obrigatório.", is_error=True)
    resp = wait_ums_job(job_id, timeout_sec=_clamp_timeout(args.get("timeout_sec")))
    if resp is None:
        return _text_result(_DOWN_HINT, is_error=True)
    return _json_text_result(resp, is_error=resp.get("status") not in (P.STATUS_OK, None))


def _handle_cancel(args: dict[str, Any]) -> dict[str, Any]:
    request: dict[str, Any] = {"cmd": P.CMD_CANCEL}
    if args.get("all"):
        request["all"] = True
        request["queued_only"] = bool(args.get("queued_only"))
    elif args.get("job_id"):
        request["job_id"] = str(args["job_id"])
    else:
        return _text_result("Indica job_id ou all=true.", is_error=True)
    resp = _ums(request, timeout_sec=30.0)
    if not resp:
        return _text_result(_DOWN_HINT, is_error=True)
    # Sem is_error, um agente via um cancel recusado (job running) como sucesso
    # e seguia como se o job já não existisse.
    return _json_text_result(resp, is_error=resp.get("status") != P.STATUS_OK)


def _handle_preload(args: dict[str, Any]) -> dict[str, Any]:
    if not args.get("confirm"):
        return _text_result("preload carrega pesos para VRAM — chama com confirm=true.", is_error=True)
    backend = str(args.get("backend") or "").strip()
    if not backend:
        return _text_result("backend é obrigatório.", is_error=True)
    resp = _ums({"cmd": P.CMD_PRELOAD, "backend": backend}, timeout_sec=600.0)
    if not resp:
        return _text_result(_DOWN_HINT, is_error=True)
    return _json_text_result(resp, is_error=resp.get("status") != P.STATUS_OK)


def _handle_evict(args: dict[str, Any]) -> dict[str, Any]:
    if not args.get("confirm"):
        return _text_result(
            "evict descarrega pesos (perde a cache quente). Chama de novo com confirm=true quando tiveres a certeza.",
            is_error=True,
        )
    request: dict[str, Any] = {"cmd": P.CMD_RELEASE}
    if args.get("backend"):
        request["backend"] = str(args["backend"])
    resp = _ums(request, timeout_sec=60.0)
    if not resp:
        return _text_result(_DOWN_HINT, is_error=True)
    # Evict recusado (backend em uso) reportado como erro — senão o agente
    # assume VRAM libertada e faz double-load.
    return _json_text_result(resp, is_error=resp.get("status") != P.STATUS_OK)


def _handle_zero(args: dict[str, Any]) -> dict[str, Any]:
    if not args.get("confirm"):
        return _text_result(
            "zero termina TODOS os workers (liberta o contexto CUDA). Chama com confirm=true.",
            is_error=True,
        )
    resp = _ums({"cmd": P.CMD_ZERO}, timeout_sec=120.0)
    if not resp:
        return _text_result(_DOWN_HINT, is_error=True)
    return _json_text_result(resp, is_error=resp.get("status") != P.STATUS_OK)


def _handle_doctor(_: dict[str, Any]) -> dict[str, Any]:
    """Resumo read-only do ambiente (driver, GPUs, órfãos, legacy sockets)."""
    try:
        from .client import UMS_SOCKET, discover_active_sockets, is_ums_running
        from .gpu import check_nvidia_driver_match, list_gpu_snapshots
        from .process_guard import stray_report

        driver_ok, driver_detail = check_nvidia_driver_match()
        snaps = list_gpu_snapshots()
        legacy = [str(s) for s in discover_active_sockets() if str(s) != str(UMS_SOCKET)]
        strays = stray_report() if not is_ums_running() else {}
        return _json_text_result(
            {
                "vramd_running": is_ums_running(),
                "driver_ok": driver_ok,
                "driver_detail": driver_detail,
                "gpus": [{"name": s.name, "total_mib": s.total_mib, "free_mib": s.free_mib} for s in snaps],
                "legacy_sockets": legacy,
                "strays": strays,
                "hint": "Para reparar órfãos: `vramd reap`. Para diagnóstico completo: `vramd doctor`.",
            }
        )
    except Exception as e:
        return _text_result(f"diagnóstico falhou: {e}", is_error=True)


_TOOL_HANDLERS = {
    "vramd_status": _handle_status,
    "vramd_queue": _handle_queue,
    "vramd_backends": _handle_backends,
    "vramd_stats": _handle_stats,
    "vramd_learn": _handle_learn,
    "vramd_submit": _handle_submit,
    "vramd_wait": _handle_wait,
    "vramd_cancel": _handle_cancel,
    "vramd_preload": _handle_preload,
    "vramd_evict": _handle_evict,
    "vramd_zero": _handle_zero,
    "vramd_doctor": _handle_doctor,
}


# ---------------------------------------------------------------------------
# JSON-RPC dispatch
# ---------------------------------------------------------------------------


def handle_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Processa uma mensagem JSON-RPC; ``None`` para notifications.

    Função pura (I/O fica no loop) — é isto que os testes exercitam.
    """
    method = str(msg.get("method") or "")
    msg_id = msg.get("id")
    is_notification = "id" not in msg

    if method == "initialize":
        return {
            "jsonrpc": _JSONRPC,
            "id": msg_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "vramd", "version": __version__},
                "instructions": (
                    "Supervisor de VRAM para inferência generativa. Fluxo típico: "
                    "vramd_status → vramd_submit (wait=true) → output. A GPU é partilhada: "
                    "espera a fila em vez de lançar jobs paralelos. Nunca mates processos GPU."
                ),
            },
        }

    if method.startswith("notifications/"):
        return None

    if is_notification:
        return None

    if method == "ping":
        return {"jsonrpc": _JSONRPC, "id": msg_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": _JSONRPC, "id": msg_id, "result": {"tools": _tool_definitions()}}

    if method == "tools/call":
        params = msg.get("params") or {}
        name = str(params.get("name") or "")
        handler = _TOOL_HANDLERS.get(name)
        if handler is None:
            return {
                "jsonrpc": _JSONRPC,
                "id": msg_id,
                "error": {"code": _CODE_INVALID, "message": f"tool desconhecida: {name}"},
            }
        try:
            result = handler(params.get("arguments") or {})
        except Exception as e:
            result = _text_result(f"erro interno na tool {name}: {e}", is_error=True)
        return {"jsonrpc": _JSONRPC, "id": msg_id, "result": result}

    return {
        "jsonrpc": _JSONRPC,
        "id": msg_id,
        "error": {"code": _CODE_METHOD, "message": f"method não suportado: {method}"},
    }


def handle_message_safe(msg: dict[str, Any]) -> dict[str, Any] | None:
    """``handle_message`` que nunca levanta — erros viram JSON-RPC internal error.

    Notificações continuam a devolver ``None`` (não têm resposta, por protocolo).
    """
    try:
        return handle_message(msg)
    except Exception as e:
        return {
            "jsonrpc": _JSONRPC,
            "id": msg.get("id"),
            "error": {"code": _CODE_INTERNAL, "message": str(e)},
        }


def serve_stdio(
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    """Loop NDJSON sobre stdio. Retorna exit code (0 no EOF gracioso).

    Blindagem:
    - **stdout é protocolo**: o ``sys.stdout`` global é trocado por stderr
      durante o serve — qualquer print/console que fuja (logger do client,
      warnings de libs) corrompia o framing NDJSON e derrubava a conexão MCP
      exactamente nos paths de falha.
    - **dispatch em threads**: um ``vramd_submit wait=true`` (até 600s) deixava
      de responder a ``ping``/``tools/list`` e mantinha ``notifications/
      cancelled`` por ler — o host matava o server com o job ainda na GPU.
    - Linhas >16 MiB e BrokenPipe/Ctrl+C tratados (exit limpo, sem traceback).
    """
    import concurrent.futures
    import queue as _queue

    sin = stdin if stdin is not None else sys.stdin
    real_stdout = stdout if stdout is not None else sys.stdout
    serr = stderr if stderr is not None else sys.stderr

    responses: _queue.Queue[dict[str, Any] | None] = _queue.Queue()

    def _worker(msg: dict[str, Any]) -> None:
        resp = handle_message_safe(msg)
        if resp is not None:
            responses.put(resp)

    import threading

    stop = threading.Event()

    def _writer() -> None:
        while True:
            item = responses.get()
            if item is None:
                return
            try:
                real_stdout.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
                real_stdout.flush()
            except (BrokenPipeError, OSError):
                stop.set()  # cliente foi-se; drenar e sair
                return

    writer = threading.Thread(target=_writer, name="vramd-mcp-writer", daemon=True)
    writer.start()

    real_sys_stdout = sys.stdout
    sys.stdout = serr  # TUDO o que fuja vai para stderr, nunca para o protocolo
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="vramd-mcp")
    try:
        for line in sin:
            if stop.is_set():
                break
            line = line.strip()
            if not line:
                continue
            if len(line) > 16 * 1024 * 1024:
                responses.put(
                    {
                        "jsonrpc": _JSONRPC,
                        "id": None,
                        "error": {"code": _CODE_PARSE, "message": "linha excede 16 MiB"},
                    }
                )
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                responses.put(
                    {
                        "jsonrpc": _JSONRPC,
                        "id": None,
                        "error": {"code": _CODE_PARSE, "message": f"JSON inválido: {e}"},
                    }
                )
                continue
            if not isinstance(msg, dict):
                print(f"[vramd-mcp] mensagem não é objecto: {line[:80]}", file=serr, flush=True)
                continue
            executor.submit(_worker, msg)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout = real_sys_stdout
        executor.shutdown(wait=True)  # deixará os waits em curso terminarem
        responses.put(None)
        writer.join(timeout=5.0)
    return 0
