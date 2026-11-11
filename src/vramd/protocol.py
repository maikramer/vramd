"""Protocolo do vramd.

JSON sobre Unix domain socket. Comandos curtos: 1 linha request → 1 linha response.
``generate`` / ``wait`` com ``stream: true``: várias linhas NDJSON (eventos + resultado).

O vramd escuta num único socket canónico (``~/.cache/vramd/vramd.sock``)
e roteia pedidos para backends via fila inteligente (afinidade VRAM + prioridades).

Comandos suportados:

  Request:
    {"cmd": "generate", "backend": "text2icon", ...kwargs}
        Enfileira + espera resultado (sync). Opcional: priority, stream.
    {"cmd": "submit", "backend": "...", ...}
        Enfileira; devolve job_id de imediato.
    {"cmd": "poll", "job_id": "..."}
        Estado actual do job.
    {"cmd": "wait", "job_id": "..."}
        Bloqueia até o job terminar (opcional stream).
    {"cmd": "cancel", "job_id": "..."}
        Cancela se queued; se running, best-effort.
    {"cmd": "queue"}
        Snapshot da fila (jobs queued/running).
    {"cmd": "release"} / {"cmd": "release", "backend": "X"}
    {"cmd": "status"} / {"cmd": "stats"} / {"cmd": "list-backends"}
    {"cmd": "preload", "backend": "X"}
    {"cmd": "ensure-vram", "needed_mib": N, "backend"?: "..."}
        Evicta até N MiB livres; com backend usa max(N, peak=pesos+activação+safety).
    {"cmd": "respawn", "backend": "X"} / {"cmd": "respawn"}
        Reinicia SÓ o worker subprocesso de um backend (código novo da tool,
        sem reiniciar o supervisor). Sem backend: todos os backends.
        Com ``lazy=true`` (default): mata o worker vivo mas NÃO recarrega — o
        próximo generate arranca-o já com o código atualizado.
    {"cmd": "flush", "queued_only": bool}
        Cancela jobs da fila (com ou sem os em curso).
    {"cmd": "reap", "dry_run": bool}
        Limpa processos GPU órfãos.
    {"cmd": "zero"}
        Liberta toda a VRAM ociosa sem parar o supervisor.
    {"cmd": "learn"}
        Relatório de drift: pico observado na produção vs declarado
        (``reset: true`` limpa as observações; ``backend`` filtra).
    {"cmd": "topology"}
        GPUs, peers, capacidade paralela e tamanhos de GPU (para ``hw_tiers``).
    {"cmd": "route", "backend": "X", ...kwargs}
        Dry-run do router: onde este job correria agora (GPU local, réplica,
        split, outro nó) e o custo estimado de cada rota. Não enfileira.
    {"cmd": "shutdown"}

  Response:
    {"status": "ok"|"error"|"status"|"queue_full", ...}
"""

from __future__ import annotations

import os

from .host import default_socket_path

# Endpoint canónico do vramd (mesmo directório dos per-tool legacy servers).
SOCKET_FILENAME = "vramd.sock"
WAL_FILENAME = "vramd-jobs.jsonl"
DEFAULT_SOCKET_PATH = default_socket_path()

# Comandos do protocolo.
CMD_GENERATE = "generate"
CMD_SUBMIT = "submit"
CMD_POLL = "poll"
CMD_WAIT = "wait"
CMD_CANCEL = "cancel"
CMD_FLUSH = "flush"
CMD_QUEUE = "queue"
CMD_RELEASE = "release"
CMD_STATUS = "status"
CMD_SHUTDOWN = "shutdown"
CMD_STATS = "stats"
CMD_LIST_BACKENDS = "list-backends"
CMD_PRELOAD = "preload"
CMD_ENSURE_VRAM = "ensure-vram"
CMD_RESPAWN = "respawn"
CMD_REAP = "reap"
CMD_ZERO = "zero"
CMD_LEARN = "learn"
# Modo remoto: sandbox por job + transferência de ficheiros (chunks base64).
CMD_ROUTE = "route"
# Topologia para clientes: GPUs (lanes), peers, capacidade paralela e tamanhos
# de GPU — leve, para um batch decidir quantos jobs manter em voo e para que
# tamanhos calcular os ``hw_tiers``.
CMD_TOPOLOGY = "topology"
# Tiers suspensos por OOM (``reset`` limpa).
CMD_TIERS = "tiers"
# Previsão de procura (``backend``, ``count``, ``eta_sec``, ``request`` de
# exemplo): o supervisor pré-carrega o modelo quando a chegada está a menos de
# um load de distância.
CMD_FORECAST = "forecast"
CMD_STAGE = "stage"
CMD_PUT = "put"
CMD_GET = "get"
CMD_LIST = "list"
CMD_CLEAN = "clean"
# Protocolo v2: handshake de versão/capabilities, blobs content-addressed e
# frames binários (ver ``frames.py``).
CMD_HELLO = "hello"
CMD_BLOB_HAS = "blob-has"
CMD_BLOB_PUT = "blob-put"
CMD_BLOB_GET = "blob-get"
CMD_BLOB_DELETE = "blob-delete"
CMD_BLOBS = "blobs"
CMD_BLOBS_GC = "blobs-gc"
# Download binário de um ficheiro do sandbox (outputs de jobs).
CMD_JOB_GET = "job-get"
# Sessões de código do cliente: bundle content-addressed + envs + backends
# efémeros. Ver ``sessions.py``.
CMD_SESSION_OPEN = "session-open"
CMD_SESSION_CLOSE = "session-close"
CMD_SESSIONS = "sessions"
# Cache de modelos HF (`vramd models`).
CMD_MODELS = "models"
CMD_MODELS_ENSURE = "models-ensure"
CMD_MODELS_GC = "models-gc"

# Versão do protocolo binário e features que o servidor anuncia no ``hello``.
PROTOCOL_VERSION = 1
FEATURE_BINARY = "binary"
FEATURE_BLOBS = "blobs"
FEATURE_PERSISTENT = "persistent"
FEATURE_SESSIONS = "sessions"
# Router multi-GPU/peers; ``hw_tiers`` resolvidos na GPU de destino; campo
# ``criticality`` no custo da rota.
FEATURE_ROUTING = "routing"
FEATURE_HW_TIERS = "hw_tiers"
FEATURE_CRITICALITY = "criticality"
FEATURE_FORECAST = "forecast"
FEATURE_TIER_HEALTH = "tier_health"
FEATURES = frozenset(
    {
        FEATURE_BINARY,
        FEATURE_BLOBS,
        FEATURE_PERSISTENT,
        FEATURE_SESSIONS,
        FEATURE_ROUTING,
        FEATURE_HW_TIERS,
        FEATURE_CRITICALITY,
        FEATURE_FORECAST,
        FEATURE_TIER_HEALTH,
    }
)

# Comandos válidos (para validação no servidor).
KNOWN_COMMANDS = frozenset(
    {
        CMD_GENERATE,
        CMD_SUBMIT,
        CMD_POLL,
        CMD_WAIT,
        CMD_CANCEL,
        CMD_FLUSH,
        CMD_QUEUE,
        CMD_RELEASE,
        CMD_STATUS,
        CMD_SHUTDOWN,
        CMD_LIST_BACKENDS,
        CMD_PRELOAD,
        CMD_ENSURE_VRAM,
        CMD_STATS,
        CMD_RESPAWN,
        CMD_REAP,
        CMD_ZERO,
        CMD_LEARN,
        CMD_ROUTE,
        CMD_TOPOLOGY,
        CMD_TIERS,
        CMD_FORECAST,
        CMD_STAGE,
        CMD_PUT,
        CMD_GET,
        CMD_LIST,
        CMD_CLEAN,
        CMD_HELLO,
        CMD_BLOB_HAS,
        CMD_BLOB_PUT,
        CMD_BLOB_GET,
        CMD_BLOB_DELETE,
        CMD_BLOBS,
        CMD_BLOBS_GC,
        CMD_JOB_GET,
        CMD_SESSION_OPEN,
        CMD_SESSION_CLOSE,
        CMD_SESSIONS,
        CMD_MODELS,
        CMD_MODELS_ENSURE,
        CMD_MODELS_GC,
    }
)

# Valores de "status" nas respostas.
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_STATUS = "status"
STATUS_QUEUE_FULL = "queue_full"

# Códigos de erro estáveis (campo ``error_code`` nas respostas) — úteis para debug/CI.
ERR_BACKEND_UNKNOWN = "BACKEND_UNKNOWN"
ERR_BACKEND_AMBIGUOUS = "BACKEND_AMBIGUOUS"
ERR_QUEUE_FULL = "QUEUE_FULL"
ERR_GENERATE_FAILED = "GENERATE_FAILED"
ERR_WORKER_DEAD = "WORKER_DEAD"
ERR_CANCELLED = "CANCELLED"
ERR_TIMEOUT = "TIMEOUT"
ERR_JOB_UNKNOWN = "JOB_UNKNOWN"
ERR_INVALID_REQUEST = "INVALID_REQUEST"
ERR_PRELOAD_FAILED = "PRELOAD_FAILED"
ERR_VRAM_INSUFFICIENT = "VRAM_INSUFFICIENT"
ERR_RESPAWN_FAILED = "RESPAWN_FAILED"
ERR_RESPAWN_BUSY = "RESPAWN_BUSY"
ERR_ZERO_BUSY = "ZERO_BUSY"
ERR_ALREADY_RUNNING = "ALREADY_RUNNING"
ERR_SHAPE_BUSY = "SHAPE_BUSY"
# Saturação do teto de conexões concorrentes do supervisor (VRAMD_MAX_CLIENTS).
ERR_SERVER_BUSY = "SERVER_BUSY"
# Endpoint TCP (Windows) sem o token do descritor — recusado antes do dispatch.
ERR_UNAUTHORIZED = "UNAUTHORIZED"
# Sandbox remoto: job/path inválido, quota excedida.
ERR_REMOTE_PATH = "REMOTE_PATH_INVALID"
ERR_REMOTE_QUOTA = "REMOTE_QUOTA"
ERR_REMOTE_JOB = "REMOTE_JOB_INVALID"
# Blobs content-addressed / protocolo binário.
ERR_BLOB_UNKNOWN = "BLOB_UNKNOWN"
ERR_BLOB_INVALID = "BLOB_INVALID"
ERR_BLOB_QUOTA = "BLOB_QUOTA"
ERR_BLOB_HASH = "BLOB_HASH_MISMATCH"
ERR_BLOB_SIZE = "BLOB_SIZE_MISMATCH"
ERR_PROTOCOL = "PROTOCOL_ERROR"
# Sessões de código do cliente.
ERR_SESSION_UNKNOWN = "SESSION_UNKNOWN"
ERR_SESSION_DISABLED = "SESSION_DISABLED"
ERR_SESSION_INVALID = "SESSION_INVALID"
ERR_SESSION_BUSY = "SESSION_BUSY"
ERR_SESSION_ENV = "SESSION_ENV"
ERR_SESSION_ENV_DISABLED = "SESSION_ENV_DISABLED"
ERR_SESSION_BUNDLE = "SESSION_BUNDLE"
# Cache de modelos HF.
ERR_MODEL = "MODEL_ERROR"
ERR_MODEL_MISSING = "MODEL_MISSING"

# Prioridades de pedido (menor rank = atende primeiro).
PRIORITY_INTERACTIVE = "interactive"
PRIORITY_BATCH = "batch"
PRIORITY_RANK: dict[str, int] = {
    PRIORITY_INTERACTIVE: 0,
    PRIORITY_BATCH: 1,
}
DEFAULT_PRIORITY = PRIORITY_INTERACTIVE

# Estados de job.
JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_DONE = "done"
JOB_FAILED = "failed"
JOB_CANCELLED = "cancelled"

# Eventos NDJSON (stream).
EVENT_QUEUED = "queued"
EVENT_STARTED = "started"
EVENT_PROGRESS = "progress"
EVENT_DONE = "done"
EVENT_ERROR = "error"
EVENT_CANCELLED = "cancelled"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Fila / scheduler (env overrideáveis).
MAX_AFFINITY_CUTS = _env_int("VRAMD_MAX_AFFINITY_CUTS", 3)
MAX_QUEUE_DEPTH = _env_int("VRAMD_MAX_QUEUE_DEPTH", 32)
MAX_INFLIGHT = _env_int("VRAMD_MAX_INFLIGHT", 1)
# 0 = desactivado. Se >0, job queued há mais de N segundos força pick (anti-starve).
STARVATION_TIMEOUT_SEC = float(_env_int("VRAMD_STARVATION_TIMEOUT_SEC", 0))

# VRAM transitória (processo externo / fragmentação CUDA): requeue em vez de
# falhar o batch inteiro. Pico > VRAM total da GPU → sem retry (impossível).
MAX_VRAM_RETRIES = _env_int("VRAMD_MAX_VRAM_RETRIES", 8)
# Worker subprocesso morreu entre load e generate (IdleEvictor race, OOM kill,
# crash) — requeue curto em vez de falhar o asset no batch (ex. scorpion_nest).
MAX_WORKER_DEAD_RETRIES = _env_int("VRAMD_MAX_WORKER_DEAD_RETRIES", 2)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


VRAM_RETRY_BASE_SEC = _env_float("VRAMD_VRAM_RETRY_BASE_SEC", 2.0)
VRAM_RETRY_MAX_SEC = _env_float("VRAMD_VRAM_RETRY_MAX_SEC", 30.0)
# Retries consecutivos SEM progresso (VRAM livre plana ±slack e nada evictável)
# antes de falhar rápido — evita o loop histórico de 8x30s sem saída possível.
VRAM_FLAT_RETRY_MAX = _env_int("VRAMD_VRAM_FLAT_RETRY_MAX", 2)
# Slack (MiB) para considerar a VRAM livre «plana» entre retries.
VRAM_FLAT_SLACK_MIB = _env_int("VRAMD_VRAM_FLAT_SLACK_MIB", 32)
# Espera curta dentro de ensure_loaded antes de recusar (evict+clear já feitos).
VRAM_ADMIT_WAIT_SEC = _env_float("VRAMD_VRAM_ADMIT_WAIT_SEC", 8.0)
VRAM_ADMIT_POLL_SEC = _env_float("VRAMD_VRAM_ADMIT_POLL_SEC", 0.5)
# Limiar para o aviso de residual no PID do vramd com ``loaded=[]`` (contexto
# CUDA / cache que sobrevive ao scrub). O residual é baseline do worker: o
# admit in-process credita o cache reutilizável (reserved-allocated) — sem
# acção destrutiva sobre o processo.
DEAD_VRAM_MIB = _env_int("VRAMD_DEAD_VRAM_MIB", 256)

# Descarregar pesos de um backend após este tempo sem uso. 120s equilibra
# "VRAM livre para o resto do sistema" com o custo do cold start (text3d/paint3d
# levam dezenas de segundos); num batch contínuo o last_used renova-se e o
# modelo fica quente.
IDLE_EVICT_SEC = _env_float("VRAMD_IDLE_EVICT_SEC", 120.0)
IDLE_EVICT_CHECK_SEC = _env_float("VRAMD_IDLE_EVICT_CHECK_SEC", 15.0)
# Terminar o subprocesso worker após este tempo sem uso. ``unload`` só liberta
# pesos — o contexto CUDA do processo (~0.3-1 GiB) só sai com o processo.
WORKER_IDLE_SHUTDOWN_SEC = _env_float("VRAMD_WORKER_IDLE_SHUTDOWN_SEC", 300.0)
# Health-check (ping/pong) aos workers vivos; sem pong ⇒ mata e marca para respawn.
WORKER_HEALTH_CHECK_SEC = _env_float("VRAMD_WORKER_HEALTH_CHECK_SEC", 60.0)
# Reap de supervisores/workers órfãos no arranque (0 desliga).
REAP_ON_START = _env_int("VRAMD_REAP_ON_START", 1)

# Default cmd quando ausente no request (retrocompat com per-tool: gerar).
DEFAULT_CMD = CMD_GENERATE

# Timeout default para pedidos de geração (segundos).
DEFAULT_GENERATE_TIMEOUT_SEC = 600.0

# Tamanho máximo de um request (1 linha JSON) — proteção contra reads sem
# newline. 8 MiB: os chunks de transferência remota (`put`, base64) têm de
# caber com folga sobre os 2 MiB de dados úteis.
MAX_REQUEST_BYTES = 8 * 1024 * 1024  # 8 MiB

# Minutos de idle antes de self-shutdown do vramd (0 = desativado). 30 min: os
# clientes fazem auto-start quando precisam, logo um supervisor parado só está
# a arriscar ficar zombie. (Contexto CUDA do supervisor: não existe em modo
# subprocesso — `clear_cuda_memory` salta torch sem `is_initialized()`; para
# VRAM presa em workers idle vivos, `vramd zero` liberta sem parar o supervisor.)
DEFAULT_IDLE_TIMEOUT_MIN = _env_int("VRAMD_IDLE_TIMEOUT_MIN", 30)


def normalize_priority(value: object | None) -> str:
    """Normaliza priority do request; default ``interactive``."""
    if value is None:
        return DEFAULT_PRIORITY
    text = str(value).strip().lower()
    if text in PRIORITY_RANK:
        return text
    return DEFAULT_PRIORITY
