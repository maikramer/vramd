# Changelog

Formato: [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).
Versionamento: [SemVer](https://semver.org/lang/pt-BR/).

## [Unreleased]

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
[Unreleased]: https://github.com/maikramer/vramd/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/maikramer/vramd/releases/tag/v0.1.0
