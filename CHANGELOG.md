# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/1.1.0/).
Versioning: [SemVer](https://semver.org/).

## [0.7.6] — 2026-11-11

### Corrigido

- **Restart do `--listen` com porta fixa.** O listener TCP em POSIX passa a
  ligar `SO_REUSEADDR`; sem isso, um restart imediato (supervisor morto com
  ligações em `TIME_WAIT`) falhava com `EADDRINUSE` mesmo sem `LISTEN`.

## [0.7.5] — 2026-11-11

### Corrigido

- **Locks com CUDA 12 são recusados no servidor.** O pipeline é CUDA 13
  (`torch +cu130`, `nvidia-*-cu13`); `EnvManager.describe/build_lock` falham
  com um hint acionável se o freeze trouxer `-cu12`/`+cu12`. O Hi3D (torch
  2.5+cu124) é local e não entra em sessões.

## [0.7.4] — 2026-11-11

### Corrigido

- **Libs CUDA 13 por cima das CUDA 12 nos envs.** Os pacotes
  `nvidia-*-cu12` e `nvidia-*-cu13` escrevem no mesmo
  `site-packages/nvidia/<lib>/lib`; quando a cu12 ficava por cima, o torch
  `+cu130` falhava com `ncclCommResume` em falta ou
  `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`. O build reinstala os pins
  `nvidia-*-cu13` do lock no fim (com `--no-deps`).

## [0.7.3] — 2026-11-11

### Corrigido

- **Build de env com PyPI + índice PyTorch.** O `uv pip install` do freeze
  passou a usar `--index-strategy unsafe-best-match`; sem isto o `uv` fixava-se
  no primeiro índice onde via o pacote e recusava a versão do outro
  (`certifi==…` existia no índice PyTorch noutra versão e o build falhava).

## [0.7.2] — 2026-11-11

### Corrigido

- **`vramd models list/gc` com huggingface_hub 2.4.** O
  `CachedRevisionInfo` passou a expor `last_modified` (datetime) em vez de
  `last_accessed`; o inventário e o GC usam um timestamp tolerante às duas
  APIs. Testado contra um cache real de 143 GB (19 repos).

## [0.7.1] — 2026-11-11

### Corrigido

- **Servidor só de sessões arranca sem backends fixos.** `vramd start
  --allow-client-code` deixou de exigir `--allow-empty`: os backends vêm do
  bundle do cliente via `session-open`. Sem sessões nem registry, a recusa
  continua (e agora o hint menciona as duas saídas).

## [0.7.0] — 2026-11-11

### Adicionado

- **Router: o vramd decide onde cada job corre.** Para cada job estima, em
  segundos, quando o resultado estaria pronto em cada rota possível
  (`espera + evicção + load + geração + transferência`) e escolhe a menor
  (`src/vramd/router.py`, puro e testável sem GPU). Do mesmo modelo saem:
  - **réplicas** — o mesmo backend em várias GPUs quando a fila acumulada na
    GPU quente ultrapassa o custo de um load a frio numa GPU livre (um job
    isolado nunca paga um segundo load; `VRAMD_ROUTE_REPLICA_MARGIN_SEC`);
  - **split** — o modelo dividido por várias GPUs, só quando o pico não cabe
    em nenhuma sozinha e o backend declara `placement.split`;
  - **rede** — encaminhamento para outro supervisor quando a latência, a
    transferência à largura de banda medida e a fila do peer ainda acabam
    mais cedo.
- **Multi-GPU com lanes.** Cada GPU tem o seu gestor de backends e os seus
  workers fixados por `CUDA_VISIBLE_DEVICES` (`src/vramd/lanes.py`) — a
  admissão por pico, a evicção e a calibração valem por dispositivo sem
  alterações. Activo automaticamente com mais de uma GPU; `VRAMD_GPUS=0,2`
  escolhe um subconjunto, `VRAMD_GPUS=legacy` repõe a lane única. `gpu_ids`
  explícito num pedido é respeitado como pin. `evict`/`zero`/`respawn`
  alcançam todas as lanes.
- **Peers** (`src/vramd/peers.py`): `peers.yaml` ou `VRAMD_PEERS`; monitor com
  RTT por EWMA, fila, modelos quentes e GPUs do peer; largura de banda
  aprendida com transferências reais. O `RemoteExecutor` envia os inputs para
  o sandbox do peer, faz stream do progresso, traz os outputs para os paths
  locais e propaga o cancelamento. Um job encaminhado leva `_route.hops` e
  nunca é reencaminhado; se o peer cai antes de aceitar, o job volta à fila
  com esse peer excluído.
- Bloco `placement:` nos descritores (`split`, `split_overhead`,
  `split_slowdown`, `max_replicas`, `remote`, `input_keys`, `output_keys`,
  `load_sec`, `gen_sec`, `output_mib`), validado pelo `vramd validate`
  (que passa também a validar o `peers.yaml`).
- **`vramd route <backend>`** (dry-run: todas as rotas consideradas, com custo
  decomposto e o motivo das inviáveis), **`vramd devices`** e **`vramd peers`**.
- `status` ganha os blocos `lanes`, `peers` e `routing` (GPUs, estimativas de
  load/geração por backend — é o que um peer lê); cada job mostra a `route`
  escolhida; `vramd metrics` exporta lanes, peers e decisões por tipo de rota.

### Alterado

- `status.max_inflight` reflecte a capacidade real (uma geração por GPU mais
  os slots de encaminhamento) quando o router está activo.
- Com uma GPU e sem peers o despacho é exactamente o anterior.

### Corrigido

- `VRAMD_REMOTE` com um endpoint inválido rebentava com `NameError`
  (`_logger` inexistente em `transport.py`) em vez de ignorar o endpoint com
  aviso.

### Adicionado (protocolo remoto, blobs, sessões e config)

- **Protocolo v2: frames binários e ligações persistentes.** O primeiro byte
  `{` continua a ser uma linha JSON; `V` entra no framing `VRMD` (magic,
  versão, tipo, flags, request_id, length). Payloads `DATA` vão crus, com
  compressão zlib automática (≥1 MiB e ≥5% de ganho) e zstd opcional via
  `vramd[zstd]`. Antes de comprimir um chunk inteiro há uma sonda de 64 KiB:
  dados já comprimidos (GLB, PNG) seguem raw e não gastam CPU (benchmark local
  de 32 MiB: ~64 MiB/s no caminho base64 vs ~283 MiB/s no binário, ~4,4×).
  Um `hello` negocia `protocol`, `features` (`binary`, `blobs`, `persistent`)
  e versão do vramd; o cliente recusa executar contra um servidor antigo em
  vez de falhar a meio.
- **Blobs content-addressed** (`src/vramd/blobs.py`): `blob-has`, `blob-put`,
  `blob-get`, `blob-delete`, `blobs`, `blobs-gc`. O staging aceita
  `{"blobs": {"in/x.png": "<sha256>"}}` e materializa o blob no sandbox por
  hardlink, sem copiar bytes nem reenviar o mesmo input em jobs repetidos.
  Quotas (`VRAMD_BLOB_MAX_MIB`, `VRAMD_BLOB_CACHE_GIB`), pins por job/sessão,
  GC LRU e TTL (`VRAMD_BLOB_TTL_DAYS`).
- **Download binário de outputs** (`job-get`): os GLB/PNG/OGG voltam por
  frames `DATA` em vez de base64.
- **`vramd.transfer.TransferClient`**: cliente Python do protocolo v2 com
  `hello`, `put_blob`, `get_blob`, `get_file`, `stage` e `rpc` compatível com
  o `RemoteClient` antigo. O `aigamekit-shared` usa-o quando disponível e cai
  para JSON/base64 em servidores 0.6.
- **`vramd status`** inclui `vramd_version`, `protocol`, `features` e o estado
  do cache de blobs.
- Testes: codec de frames (leituras parciais, limites, compressão), CAS
  (dedupe, corrupção, pins, GC), transporte binário fim-a-fim e `job-get`.
- **Sessões de código do cliente** (`src/vramd/sessions.py`): o cliente envia
  um bundle tar.gz content-addressed (código + locks) via CAS; o servidor
  valida cada membro contra o manifesto (paths relativos, sem symlinks,
  tamanho/sha256), extrai para `<cache>/sessions/bundles/<sha>` e regista
  backends efémeros `<backend>@<session>`. O worker corre com
  `PYTHONPATH=<bundle>/...` e `cwd=<bundle>`; não é preciso checkout do
  pipeline no servidor. `session-open`, `session-close`, `sessions`.
- **Envs Python por lock** (`src/vramd/envs.py`): freeze exacto enviado pelo
  cliente; o vramd constrói/reutiliza `<cache>/envs/<hash>` com `uv` (cache de
  wheels, lock de build, timeout) ou usa um interpretador existente
  (`path`/`system`). `vramd env list/build`.
- **Cache HF gerido pelo vramd** (`src/vramd/models.py`): `models`,
  `models-ensure`, `models-gc`; layout normal (`HF_HOME`), pins por sessão,
  GC LRU e `move_hf_cache` para migrar um cache existente (rename no mesmo
  filesystem, cópia com symlinks noutro, verificação) sem re-download.
- **`vramd config`**: wizard interactivo ou `config.yml` não-interactivo;
  escreve listener/token/cache/HF_HOME/política de modelos/GPUs/permissões de
  sessões, cria diretórios e token 0600, dá posse ao utilizador do serviço,
  move o cache HF antigo e gera/instala a unidade systemd.
- **`vramd start --config`** lê o `config.yml`; `--allow-client-code` e
  `--allow-client-env` habilitam sessões/builds (opt-in explícito).
- **`vramd devices`** mostra as lanes locais **e** os peers remotos (RTT,
  banda estimada, fila, backends quentes); `vramd route` continua a mostrar o
  custo de cada rota.
- Testes: sessões (abertura, jobs, isolamento de duas sessões, bundle
  adulterado, close), envs, modelos (pins/GC/migração) e config/systemd.

### Alterado

- `MAX_REQUEST_BYTES` continua 8 MiB para o caminho JSON; os frames `DATA`
  usam `frames.MAX_DATA_BYTES` (16 MiB) e o cliente envia em chunks de 8 MiB.
- O servidor fecha a ligação após cada resposta JSON sem `keepalive`, como
  antes; com `keepalive` (ou frames), mantém a ligação até ao idle timeout
  (`VRAMD_CLIENT_IDLE_TIMEOUT_SEC`, default 300s).

## [0.6.0] — 2026-11-11

### Adicionado

- **Modo remoto: `vramd start --listen HOST:PORT`.** O supervisor pode servir
  GPUs de outra máquina na rede. `--token-file` dá um token estável (criado
  0600 se faltar); o descritor em `<cache>/vramd.sock` publica host, porta e
  token. No cliente, `VRAMD_REMOTE=host:port` (CSV para failover) e
  `VRAMD_REMOTE_TOKEN` ligam ao daemon remoto — `vramd status`, `queue`,
  `cancel`, etc. passam a funcionar contra o servidor.
- **Sandbox de transferência por job** (`src/vramd/remote.py`) com os comandos
  `stage`, `put`, `get`, `list` e `clean`: chunks base64 até 2 MiB, validação
  canónica de paths (sem `..` nem symlink para fora), quotas por ficheiro/job
  (`VRAMD_REMOTE_MAX_FILE_MIB`, `VRAMD_REMOTE_MAX_JOB_MIB`) e TTL
  (`VRAMD_REMOTE_TTL_HOURS`) com janitor. Raiz: `VRAMD_REMOTE_ROOT`
  (default `<cache>/remote`).
- `MAX_REQUEST_BYTES` sobe para 8 MiB para acomodar os chunks.
- **`vramd submit` deixa de ser só um smoke-test.** Aceita parâmetros reais do
  job: `-p K=V` (repetível, com coerção de int/float/bool), `--request-json`
  (inline, `@ficheiro` ou `-` para stdin) e `--timeout`. Precedência:
  JSON < `-p` < `--prompt`/`--output`/`--priority`. Chaves do protocolo (`cmd`,
  `backend`, `stream`, `auth`, `_*`) são recusadas — `-p cmd=shutdown` não
  transforma um submit noutro comando. Sem parâmetros continua a enviar o
  `prompt=smoke` de sempre.
- **`submit --wait` mostra o progresso ao vivo** (posição na fila, fase,
  percentagem) por stream NDJSON em vez de poll a 5 Hz; Ctrl+C larga o cliente
  e deixa o job a correr.
- **`vramd history`** — jobs terminados, persistidos em
  `<cache>/vramd-history.jsonl` (teto `VRAMD_HISTORY_MAX`, default 2000; 0
  desliga). Filtros `--backend`/`--state`/`--failed`/`--since 2h`; `--summary`
  agrega por backend (taxa de sucesso, média/p95 de geração, tempo de GPU).
  Lê o ficheiro directamente: funciona com o supervisor parado. O request do
  job não é guardado (prompts podem ser privados).
- **`vramd logs [BACKEND]`** — tail do log do supervisor ou do stderr do worker
  de um backend, com `-f` (segue, sobrevive a truncagem), `--list` e `--path`.
- **`vramd metrics`** — exposição no formato de texto do Prometheus (fila,
  backends, workers, learn, hooks, GPU). `--textfile` escreve atomicamente um
  `.prom` para o textfile collector do node_exporter. Com o supervisor parado
  sai `vramd_up 0` e exit code 1.
- **`vramd validate [FICHEIRO]`** — lint dos descritores e do `hooks.yaml` sem
  GPU nem workers: chaves com typo (com sugestão), `runtime.command` que não
  existe ou não resolve (`${env:…}` por definir), `cwd` inexistente, timeouts
  inválidos, `shape_keys` fora de `load_keys`, footprint maior que a GPU,
  overlay parcial sem base, camada corrompida. `--strict` falha também com
  avisos; `--json` para CI.
- **`vramd completion bash|zsh|fish`** — completion de comandos, flags e nomes
  de backend.
- Tool MCP `vramd_history` (read-only; funciona com o supervisor parado).

### Corrigido

- **Backends só com `runtime.command` corriam in-process.** O supervisor só
  tratava como subprocesso os backends com `tool:`; o descriptor documentado no
  README e o `examples/echo-backend` (sem `tool:`) eram despachados in-process
  e o primeiro job falhava com `No module named 'worker'`. Um bloco `runtime:`
  com `command` ou `monorepo_tool` passa a contar. A calibração, que fala
  directamente com o pool, já funcionava — por isso o bug só aparecia ao
  submeter.

### Segurança

- `SECURITY.md`: o token do descritor é credencial de **controlo total**
  (jobs, `zero`, `shutdown`). Um listener fora de loopback deve ficar numa
  rede de confiança ou atrás de WireGuard. Os paths de `put`/`get` ficam
  confinados ao sandbox do job.

## [0.5.1] — 2026-11-11

### Corrigido

- **A reciclagem do worker repetia-se a cada job depois do limite.** O contador
  ``jobs_served`` não era resetado no spawn: depois dos primeiros 12 jobs, cada
  worker novo servia **um** job e era reciclado outra vez, pagando um load de
  30-55 s por job (o log de hoje mostra 43 reciclagens de text2d). Agora o
  contador começa em 0 em cada spawn; há teste de regressão.
- **540 WARN de "torch não está instalado" por dia.** O supervisor em modo
  subprocesso não tem (nem deve ter) torch; ``_clear_cache`` salta o scrub e
  avisa uma única vez em INFO, em vez de um WARN por evicção.
- **Learn cego durante reciclagem/respawn.** O PID do worker era fixado no
  início do job; depois de um respawn o probe falhava sempre contra um PID
  morto e entrava em backoff global de 30 s. O tracker refresca o PID a cada
  tick e distingue "PID ainda não aparece no driver" (transitório, sem
  backoff) de "driver/probe em baixo" (com backoff e erro no log).
- **Estado `loaded` dessincronizado após a reciclagem.** O manager continuava
  a considerar o backend carregado com o processo já morto, o que produzia 42
  WARN "marcado loaded mas worker morto" e um `status` que mentia. No fim do
  `generate`, se o pool já não tem processo vivo, o manager marca o backend
  como descarregado.
- **O filho `_xatlas_worker` ficava órfão quando o worker morria.** O
  `terminate()` do Popen matava só o worker; o xatlas continuava a 299% de CPU
  e VRAM. No POSIX o worker é líder de sessão (`start_new_session`), por isso a
  terminação passa a fazer `killpg` (TERM→KILL) e alcança filhos/netos; no
  Windows continua a árvore do `psutil`. O shutdown gracioso também mata
  descendentes que o pai deixou vivos. Se o worker morrer sozinho (crash/OOM),
  o PGID guardado no spawn limpa o grupo quando o supervisor o deteta. Se algum
  sobreviver, fica um WARN com os PIDs.

### Adicionado

- **Diagnóstico de workers em `status`/`stats`/`debug`:** PID, vivo/carregado,
  jobs servidos/limite, spawns/reciclagens, último exit code, motivo do último
  abort, último progresso (pct+stage), idade do último evento e caminho do log.
- **`learn.probe`:** última razão do probe, erro, PID/job, falhas consecutivas
  e backoff restante; `vramd learn` imprime o estado do probe e `vramd debug`
  mostra a tabela de workers.
- **Contexto nos erros do worker:** `job_id`, PID, stage/pct e idade do evento
  entram em `idle timeout`, `abort timeout`, `OOM-spin` e `worker fechou
  stdout`; o resultado do job leva o bloco `worker` e as últimas 8 linhas do
  stderr (`worker_log_tail`, truncadas).
- **`cancel_reason`** no job/`ums_debug` (`client cancel` vs
  `flush/cancel --all`), para saber quem pediu o abort.
- **`runtime.max_jobs`** por backend no YAML (limite de reciclagem próprio);
  o default global continua 12.

## [0.5.0] — 2026-11-10

### Adicionado

- **Suporte Windows.** O supervisor corre com o mesmo CLI e o mesmo protocolo:
  - transporte **TCP loopback com token** no Windows (o path canónico
    `%LOCALAPPDATA%\vramd\vramd.sock` passa a ser um descritor JSON com
    porta+token; pedidos sem token são recusados). `VRAMD_TRANSPORT=tcp` força
    o mesmo caminho em Linux para CI;
  - stdout dos workers drenado por **thread leitora** (o `select()` do Windows
    só aceita sockets), com timeout/EOF no supervisor;
  - singleton por `msvcrt.locking`; processos via `psutil` (sem `/proc`);
    `os.kill(pid, 0)` deixa de ser usado no Windows — a stdlib chama
    `TerminateProcess` e matava o supervisor;
  - workers/auto-start com `CREATE_NO_WINDOW`/`CREATE_NEW_PROCESS_GROUP`;
  - árvore de processos do worker terminada explicitamente (`psutil`), porque
    não há grupos de processos POSIX;
  - paths `%LOCALAPPDATA%\vramd` (cache) e `%APPDATA%\vramd` (config);
  - `nvidia-smi` descoberto em `System32`/`NVSMI` quando não está no PATH;
    comparação kernel↔userspace desactivada no Windows (driver é um pacote só);
  - watchdog do worker compara o instante de criação do pai (o PPID não muda
    quando o supervisor morre no Windows);
  - `RuntimeSpec`/toolchain já aceitavam `Scripts\python.exe`; o cliente e o
    worker do monorepo passam a usar o mesmo transporte.
- Job `test-windows` no CI (Python 3.11/3.13) e testes do caminho TCP em Linux.
- `status` inclui `transport`/`endpoint`/`platform`; `status --json` mantém
  `socket` para clientes antigos.

### Alterado

- Detecção do cliente passa a ser pelo conteúdo do endpoint (descritor JSON ⇒
  TCP+token; socket ⇒ Unix), não por plataforma — um cliente Linux fala com um
  supervisor Windows e vice-versa.
- `vramd stop` espera que o processo saia (até 15 s) em vez de só aceitar o
  shutdown: um `start` imediato deixa de falhar com `ALREADY_RUNNING` durante o
  cleanup (mais visível no Windows).
- `vramd start`/`vramd doctor` mostram o transporte e o endpoint efectivo.
- `reap`/`doctor` só classificam um argv como supervisor quando ele nomeia
  mesmo `vramd`/`ums` (ou `-m vramd`); antes, o path do checkout
  (`D:\a\vramd\...`) fazia qualquer `python -m pytest` parecer supervisor.

### Corrigido

- `vramd stop`/`zero`/`respawn` continuam por RPC no Windows (SIGTERM não é
  entregue como sinal); o auto-start destacado sobrevive ao processo chamador.
- Protocolo JSONL forçado a UTF-8 nos dois lados (os pipes herdavam cp1252 no
  Windows e mensagens não-ASCII partiam o worker).
- O lock do Windows fica num byte fora da região do PID: os byte-range locks
  são obrigatórios e impediam `owner_pid()` de ler o ficheiro.

## [0.4.0] — 2026-11-04

### Adicionado

- **Presets SDNQ 0.2.9 na tabela de fatores VRAM.** `sdnq-uint4-hq` (0.38)
  entra como tier de qualidade opt-in. Sem esta entrada, o preset novo caía
  no fator 1.0 (fp16) e era recusado em GPUs de 6 GB, apesar de medir
  ~5.6 b/peso.
- `sdnq-int3`/`sdnq-int2` passam a 0.24/0.18 (antes 0.28/0.25): a receita
  0.2.9 (uint3/uint2 + Hadamard + escalas double-quant, **sem** codebook)
  mede 3.5/2.5 b/peso. O codebook antigo a group 32 gastava 11/6 b/peso —
  acima do int4 e sem poupança real.

### Alterado

- As estimativas de pico para os degraus finos deixam de sobrestimar 28%/39%
  e alinham-se com o que as tools quantizam de facto.

## [0.3.6] — 2026-09-08

### Corrigido

- **Watcher de OOM-spin matava jobs saudáveis mas silenciosos.** O pipeline
  de paint não emite progresso dentro do xatlas/bake/inpaint (minutos) —
  8 linhas de OOM transitórias espalhadas por 266s disparavam o abort
  "worker wedged" num job a trabalhar (village_house a 135k: xatlas 147s +
  bake, morto aos ~4.5 min). Agora a contagem é por **janela rolante de 120s**
  (``_OOM_SPIN_WINDOW_SEC``): só densidade recente conta — o spin real
  (~0.7 linhas/s) dispara em ~40s; OOMs espaçados expiram sem matar ninguém.
  O adapter paint3d do worker também passou a ligar os hooks ``_step`` do
  pipeline ao progresso do vramd, tornando as fases visíveis ao watchdog.

## [0.3.5] — 2026-09-08

### Corrigido

- **0.3.4 recusava TODO o paint3d numa GPU do tamanho exacto da calibração.**
  Honrar ``admit_peak_mib`` (= peak + safety) no ``peak_vram_mib`` fazia o
  ``can_admit`` comparar 6144 contra um free que nunca passa disso numa 6 GiB
  (6141 MiB) — admit permanente recusado e fallback in-process. Agora:
  ``vram.peak_mib`` **medido** vence (5760 — o pico real, a safety fica
  implícita na folga do free); ``admit_peak_mib`` só como fallback, devolvido
  **sem** a safety declarada para não duplicar a margem.

## [0.3.4] — 2026-09-08

### Corrigido

- **Admit sub-dimensionado do paint3d na 6 GB (duplo desconto mem-eff).** O
  fator 0.65 de activação memory-efficient era aplicado sobre uma activação
  **medida já em modo mem-eff** (a calibração regista
  ``peak_profile.load_kwargs.memory_efficient``) — o admit pedia ~4721 MiB
  quando o pico real é ~5760 e a recomendação do calibrador era 6144. Jobs
  condenados eram admitidos e entravam em OOM-spin. Agora: (a) o desconto só
  vale quando a medição foi feita **sem** mem-eff; (b) ``vram.admit_peak_mib``
  da calibração vence a soma analítica em ``peak_vram_mib``.

- **OOM-spin do worker agora é detetado em ~40s, não 600s.** O pool lê o
  stderr do worker durante o job: 8+ linhas de ``expandable_segments: memory
  mapping failed``/``CUDA out of memory`` sem progresso há 30s → abort com
  ``worker wedged — OOM-spin``. O idle-timeout passou a emitir ``worker
  wedged — idle …`` e o dispatcher trata ambos como worker morto transitório
  → **requeue automático** (antes: GENERATE_FAILED sem retry e o servidor
  mentia "timeout do cliente" — agora o hint aponta para o log do worker).

- **Reciclagem do worker persistente.** ``VRAMD_WORKER_MAX_JOBS`` (default 12):
  após N jobs o worker morre e o próximo load re-spawna fresco — a
  fragmentação/estado CUDA acumulada entre jobs deixa de herdar para o batch
  inteiro. ``RuntimeSpec.event_timeout_sec`` do YAML v2 passa a ser aplicado
  no pool (antes era parseado e ignorado).

## [0.3.3] — 2026-09-07

### Corrigido

- **`vramd --version` reportava 0.3.1 dentro do pacote 0.3.2.** O literal
  `__version__` em `__init__.py` não acompanhou o bump do `pyproject.toml`
  (o guard do release valida o TAG contra o nome do sdist, não o literal).
  Novo teste fixa `__version__` == `pyproject.toml` para não voltar a
  dessincronizar.

## [0.3.2] — 2026-09-07

### Corrigido

- **`vramd start` sem configuração servia backends de EXEMPLO.** O
  `data/backends.yaml` empacotado trazia `example-whisper`/`example-diffusion`
  "ilustrativos": um start manual (sem `VRAMD_BACKENDS_FILE`) arrancava feliz
  com esses dois, agarrava o socket — e o auto-start das tools passava a
  aceitar a instância como "ativa". Resultado: TODA a delegação falhava com
  "backend desconhecido" sem pista do porquê. O registry empacotado passa a
  ser **vazio de propósito** e o `start` **recusa arrancar com zero backends**,
  com um painel que diz como configurar (`VRAMD_BACKENDS_FILE`,
  `~/.config/vramd/backends.d/*.yaml`, `vramd calibrate --out`). Escape para
  tests/dev: `vramd start --allow-empty`. Os exemplos de descriptor mudaram-se
  para `examples/backends.yaml` (referência, nunca carregados).

## [0.3.1] — 2026-08-24

### Corrigido

- **Admit recusava text2d calibrado na própria GPU que o mediu (regressão
  0.3.0).** Os callers (``ensure_loaded``, ``_backend_peak_mib``, ``preload``,
  ``ensure-vram``) dobravam ``streams_on_load`` em ``group_offload`` — e o ramo
  medido de ``footprint_parts_mib`` só aplica o desconto 0.65 de activação com
  ``memory_efficient and not group_offload``. Resultado na RTX 4050 6 GB com o
  catálogo ``backends-6g.yaml``: peak=6117 MiB (pesos 429 + activação medida
  5304 + safety) contra ~5736 MiB livres → TODO o job text2d recusado
  (``VRAM_INSUFFICIENT``), fallback in-process pior (10055 MiB fp16) e o batch
  GameAssets em retry-loop. O load streaming (diffusers model_cpu offload,
  text2d) É o caminho memory-efficient: ``streams_on_load`` passa a viajar
  separado no novo parâmetro de ``footprint_parts_mib``/``peak_vram_mib`` e o
  desconto aplica; ``group_offload`` real (text3d SDNQ) mantém activação
  completa (chunks dinâmicos). No caminho estimado, ``streams_on_load`` usa
  maior-módulo + activação com desconto (warmup do offload por módulos).
  Headroom pós-load (``activation_headroom_mib``) inalterado: stream/offload
  basta a margem de um bloco de activação.

## [0.3.0] — 2026-08-14

### Adicionado

- **Aprendizagem contínua de picos (`vramd learn`)** — o loop fechado da admissão.
  O supervisor amostra a VRAM real de cada job em curso (~2 Hz, por PID do worker)
  e compara o pico observado com o pico que a admissão usou **nesse job**.
  Veredictos: subdimensionado (risco de OOM), sobredimensionado (headroom
  desperdiçado), ok. `vramd learn --apply` escreve as correcções como overlay
  YAML (`backends.d/learned.yaml`) sem tocar no package — e sem sobrepôr
  calibrações do `vramd calibrate` (o bloco `vram:` medido continua a mandar).
  Observações persistem em `~/.cache/vramd/learn/` (sobrevivem a restarts);
  `VRAMD_LEARN_INTERVAL_SEC=0` desliga. Só backends subprocesso são observados
  (em in-process a VRAM do supervisor inclui vizinhos e a observação seria
  mentirosa).
- **Hooks de eventos (`~/.config/vramd/hooks.yaml`)** — o vramd reage e
  integra-se sem editar código: `on_job_done/failed/cancelled`, `on_evict`,
  `on_zero`, `on_drift`, `on_shutdown`. Payload JSON no stdin + env
  (`VRAMD_EVENT`, `VRAMD_HOOK`) + interpolação `${campo}` no argv. Threads
  daemon com timeout e throttle — hooks nunca derrubam nem atrasam a fila.
  Config malformada falha no arranque (silêncio seria pior).
- **Servidor MCP (`vramd mcp`)** — a GPU como ferramenta de agentes de IA.
  Model Context Protocol sobre stdio (JSON-RPC NDJSON, zero dependências
  novas): 12 tools (status/queue/stats/learn/submit/wait/cancel/preload/
  evict/zero/doctor/backends). As mutações destrutivas exigem `confirm: true`
  explícito e continuam sujeitas aos busy-guards do supervisor — a regra
  «não matar GPU a meio de um job» aplica-se a agentes como a humanos.
- **Dashboard TUI (`vramd top`)** — um `htop` para o supervisor: VRAM da GPU
  com barra e por-processo, jobs running com progresso em tempo real, fila
  com ETA, backends com countdown para idle-evict e os veredictos do learn.
  Só de leitura, seguro com batch a correr; arranca sozinho quando o
  supervisor aparece.
- Protocolo: comando `learn` (+ blocos `learn`/`hooks` no `status`);
  `JobQueue` ganha `on_finish` (callback de transição terminal — cobre done,
  failed E cancelado-em-fila); `BackendManager` ganha `on_evict` (cobre
  release manual, evicção por admissão e idle-evict).

## [0.2.4] — 2026-08-08

### Corrigido

- **`vramd calibrate` etiquetava `quant_mode: none` quando o load usava
  `quant_preset`** (alias Text2D de `sdnq_preset`) — o admit descartava a
  medição por quant mismatch e voltava às estimativas.

## [0.2.3] — 2026-08-08

### Adicionado

- **Assets de teste empacotados** (`data/test-mesh.glb`, `data/test-image.png`):
  o calibrador resolve nomes curtos no `calibrate_request` (ex.:
  `mesh_path: test-mesh.glb`) para o package data do vramd — backends que
  exigem um mesh/imagem de entrada calibram sem o utilizador ter de fornecer
  nada.

## [0.2.2] — 2026-08-08

### Added

- **`calibrate_request` / `calibrate_load_kwargs` on the descriptor**: the YAML
  can declare the generation request and the load kwargs that calibration
  should use by default — `vramd calibrate <backend>` works without flags even
  for backends that require inputs (mesh_path/output) or specific formats.
  Load-kwargs precedence: hw-auto < descriptor < explicit.

## [0.2.1] — 2026-08-08

### Fixed

- **camelCase venv discovery** (`toolchain._candidate_dirs`):
  `text2icon` couldn't find the `Text2Icon` folder (nor `skymap2d`→`Skymap2D`,
  `paint3d`→`Paint3D`…) because `str.capitalize()` only capitalizes the first
  letter. The AiGameKit layout (folder capitalized per segment) is now covered
  by `_camel_title`. Without this the tool workers never spawned and clients
  fell back to the in-process fallback.

## [0.2.0] — 2026-08-08

### Fixed

- **Ref-count leaked on cancel-after-load**: `generate` with a cancel firing
  after `ensure_loaded` left the backend pinned (`ref_count > 0`) forever —
  never evictable, and invisible to the IdleEvictor/`ensure_vram`.
- **Idle self-shutdown no longer kills in-flight generates**: auto-shutdown
  after `idle_timeout` didn't consult the queue; a job > 30 min with a client
  in `wait`/stream made the supervisor kill the worker mid-way.
- **`max_inflight` validated atomically in `take`**: the check outside the
  lock allowed exceeding the cap by 1 with `max_inflight > 1` (with a cap of 2,
  3 jobs could run).
- **Memory-efficient factor applied once in the headroom**:
  `activation_headroom_mib` re-applied the 0.65 that `footprint_parts_mib` had
  already applied (0.42× effective) — the free-VRAM check passed with less
  than intended.
- **`vramd calibrate --out` preserves the `runtime:` block** (command/cwd/env/
  timeouts) and `load_keys`/`shape_keys` of the descriptor — it used to
  regenerate `monorepo_tool` and lose the startup configuration of external
  backends.
- **Cooperative abort no longer lost in the worker queue**: resetting the flag
  on dequeue wiped an abort arriving while generate waited; now the job
  doesn't even start (answers `cancelled before start`) and the flag is
  consumed at the end of generate.
- **EOF on worker stdin does `unload` before exiting** (adapter cleanup), as
  the loop docstring always promised.
- **`round_up_mib` always rounds up** — `round()` (banker's) returned
  multiples below the input (e.g.: `64.4` → `64`).

### Configuration

- Overlay directory aligned with the documentation:
  `~/.config/vramd/backends.d` (was `~/.config/ums/backends.d`, an AiGameKit
  leftover — whoever followed the README put files that were never read).

### Internal

- WAL lock order fixed (`_lock` → `_wal_lock` everywhere):
  `_rewrite_wal_from_queue` inverted the order and the comment claimed the
  opposite — a latent ABBA deadlock if the call graph changed.
- `footprints.py`: removed duplicate block (the second definition shadowed the
  first — dead code that could diverge).
- `${monorepo:python}` reachable again (the generic `${monorepo:*}` branch
  swallowed it); user-facing strings and docs "UMS" → "vramd"; `doctor`
  flags low free VRAM with loaded models again.
- Subprocess tests inherit `PYTHONPATH` with the repo's `src` — the suite runs
  without `pip install -e .` (before: 7 local failures that passed in CI).

## [0.1.0] — 2026-08-08

First public release. Extracted from the [AiGameKit][origin], where it was
born to have ten generative models share a 6 GB RTX 4050.

### Supervisor

- Admission by the **real peak** (weights + activation + margin), not just
  weights.
- Priority queue (`interactive` > `batch`) with **VRAM affinity**: skips the
  head up to 3 times if a later job uses an already-hot backend.
- Weight+LRU eviction, escalating to terminating the worker when calibration
  proved that backend's `unload` doesn't return VRAM.
- Cooperative cancellation between phases — no CUDA kernels killed mid-flight.
- Job WAL, `zero`, `respawn`, `reap`, `doctor`.

### Isolation

- Each backend runs in its own process and venv, speaking JSONL over
  stdin/stdout. Models with incompatible dependencies coexist.
- Worker SDK (`vramd.worker`): three methods to wrap any model.

### Calibration

- `vramd calibrate` measures the footprint with per-process VRAM sampling at
  20 Hz and splits CUDA context / weights / activation at phase boundaries.
- Detects: load peak above inference peak, lazy loading, `unload` that doesn't
  free, leak by repetition, 1st-inference warmup, contamination by neighboring
  processes, and sampler blindness.
- Raw samples kept in the report: `vramd recalibrate` recomputes the numbers
  when the analysis improves, without re-occupying the GPU.

### Configuration

- `backends.yaml` v2 with per-key overlay:
  `data/backends.yaml` → `$VRAMD_BACKENDS_FILE` → `~/.config/vramd/backends.d/*.yaml`.
- `runtime:` block (command/cwd/env/timeouts) with `${env:VAR}`; `load_keys`
  and `shape_keys` per backend; declarative `peak_profile:`.
- The measured `vram:` block **wins** over the estimate at admit, as long as
  the requested quantization is the one it was measured under.

### Packaging

- Torch-free core: the supervisor installs in ~9 MB and never creates a CUDA
  context.
- 760 tests, no GPU, on Python 3.11 / 3.12 / 3.13.

[origin]: https://github.com/maikramer
[Unreleased]: https://github.com/maikramer/vramd/compare/v0.3.6...HEAD
[0.3.6]: https://github.com/maikramer/vramd/compare/v0.3.5...v0.3.6
[0.3.5]: https://github.com/maikramer/vramd/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/maikramer/vramd/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/maikramer/vramd/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/maikramer/vramd/compare/v0.3.1...v0.3.2
[0.3.0]: https://github.com/maikramer/vramd/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/maikramer/vramd/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/maikramer/vramd/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/maikramer/vramd/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/maikramer/vramd/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/maikramer/vramd/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/maikramer/vramd/releases/tag/v0.1.0
