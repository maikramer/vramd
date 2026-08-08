# Changelog

Formato: [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).
Versionamento: [SemVer](https://semver.org/lang/pt-BR/).

## [0.2.2] — 2026-08-08

### Adicionado

- **`calibrate_request` / `calibrate_load_kwargs` no descriptor**: o YAML pode
  declarar o request de geração e os kwargs de load que a calibração deve usar
  por default — `vramd calibrate <backend>` funciona sem flags mesmo para
  backends que exigem inputs (mesh_path/output) ou formatos específicos.
  Ordem de precedência dos kwargs de load: hw-auto < descriptor < explícito.

## [0.2.1] — 2026-08-08

### Corrigido

- **Descoberta de venvs com camelCase** (`toolchain._candidate_dirs`):
  `text2icon` não encontrava a pasta `Text2Icon` (nem `skymap2d`→`Skymap2D`,
  `paint3d`→`Paint3D`…) porque `str.capitalize()` só capitaliza a primeira
  letra. O layout do AiGameKit (pasta capitalizada por segmento) é agora
  coberto por `_camel_title`. Sem isto os workers das tools nunca spawnavam e
  os clientes caíam no fallback in-process.

## [0.2.0] — 2026-08-08

### Corrigido

- **Ref-count vazava no cancel pós-load**: `generate` com cancel a disparar
  depois do `ensure_loaded` deixava o backend pinado (`ref_count > 0`) para
  sempre — nunca evictável, e invisível para o IdleEvictor/`ensure_vram`.
- **Idle self-shutdown já não mata generates em curso**: o auto-encerramento
  após `idle_timeout` não consultava a fila; um job > 30 min com cliente em
  `wait`/stream fazia o supervisor matar o worker a meio.
- **`max_inflight` validado atomicamente no `take`**: o check fora do lock
  permitia exceder o cap em 1 com `max_inflight > 1` (com cap 2 podiam correr
  3 jobs).
- **Fator memory-efficient aplicado uma só vez no headroom**: `activation_headroom_mib`
  reaplicava o 0.65 que `footprint_parts_mib` já aplicara (0.42× efetivo) —
  o check de VRAM livre passava com menos do que o pretendido.
- **`vramd calibrate --out` preserva o bloco `runtime:`** (command/cwd/env/
  timeouts) e `load_keys`/`shape_keys` do descriptor — antes regenerava
  `monorepo_tool` e perdia a configuração de arranque de backends externos.
- **Abort cooperativo não se perde na fila do worker**: o reset do flag na
  dequeue apagava um abort que chegasse enquanto o generate esperava; agora o
  job nem arranca (responde `cancelled before start`) e o flag é consumido
  no fim do generate.
- **EOF no stdin do worker faz `unload` antes de sair** (cleanup do adapter),
  como a docstring do loop sempre prometeu.
- **`round_up_mib` arredonda sempre para cima** — o `round()` (banker's)
  devolvia múltiplos abaixo do input (ex.: `64.4` → `64`).

### Configuração

- Diretório de overlays alinhado com a documentação:
  `~/.config/vramd/backends.d` (era `~/.config/ums/backends.d`, legado do
  AiGameKit — quem seguia o README punha ficheiros que nunca eram lidos).

### Interno

- Ordem de locks do WAL corrigida (`_lock` → `_wal_lock` em todo o lado):
  `_rewrite_wal_from_queue` invertia a ordem e o comentário afirmava o
  contrário — deadlock ABBA latente se o call graph mudasse.
- `footprints.py`: removido bloco duplicado (a segunda definição sombreava a
  primeira — código morto que podia divergir).
- `${monorepo:python}` voltou a ser alcançável (o branch genérico `${monorepo:*}`
  engolia-o); strings de utilizador e docs "UMS" → "vramd";
  `doctor` volta a sinalizar free baixo com modelos carregados.
- Testes de subprocesso herdam `PYTHONPATH` com o `src` do repo — a suite
  corre sem `pip install -e .` (antes: 7 falhas locais que no CI passavam).

## [0.1.0] — 2026-08-08

Primeira versão pública. Extraído do [AiGameKit][origem], onde nasceu para pôr
dez modelos generativos a partilhar uma RTX 4050 de 6 GB.

### Supervisor

- Admissão pelo **pico real** (pesos + activação + margem), não só pesos.
- Fila com prioridade (`interactive` > `batch`) e **afinidade de VRAM**: salta a
  cabeça até 3 vezes se um job mais atrás usa um backend já quente.
- Evicção peso+LRU, com escalada para terminar o worker quando a calibração
  provou que o `unload` desse backend não devolve VRAM.
- Cancelamento cooperativo entre fases — sem matar kernels CUDA a meio.
- WAL de jobs, `zero`, `respawn`, `reap`, `doctor`.

### Isolamento

- Cada backend corre num processo e venv próprios, a falar JSONL por
  stdin/stdout. Modelos com dependências incompatíveis coexistem.
- Worker SDK (`vramd.worker`): três métodos para embrulhar qualquer modelo.

### Calibração

- `vramd calibrate` mede o footprint com amostragem de VRAM **por processo** a
  20 Hz e separa contexto CUDA / pesos / activação pelas fronteiras de fase.
- Deteta: pico no load acima do da inferência, carga preguiçosa, `unload` que
  não liberta, fuga por repetição, warmup da 1.ª inferência, contaminação por
  processos vizinhos e cegueira do amostrador.
- Amostras cruas guardadas no relatório: `vramd recalibrate` refaz os números
  quando a análise melhora, sem voltar a ocupar a GPU.

### Configuração

- `backends.yaml` v2 com sobreposição **por chave**:
  `data/backends.yaml` → `$VRAMD_BACKENDS_FILE` → `~/.config/vramd/backends.d/*.yaml`.
- Bloco `runtime:` (command/cwd/env/timeouts) com `${env:VAR}`; `load_keys` e
  `shape_keys` por backend; `peak_profile:` declarativo.
- O bloco `vram:` medido **vence** a estimativa no admit, desde que a
  quantização pedida seja aquela sob a qual foi medido.

### Empacotamento

- Núcleo sem torch: o supervisor instala em ~9 MB e nunca cria contexto CUDA.
- 760 testes, sem GPU, em Python 3.11 / 3.12 / 3.13.

[origem]: https://github.com/maikramer
[Unreleased]: https://github.com/maikramer/vramd/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/maikramer/vramd/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/maikramer/vramd/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/maikramer/vramd/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/maikramer/vramd/releases/tag/v0.1.0
