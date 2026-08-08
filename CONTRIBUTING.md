# Contribuir

## Arrancar

```bash
git clone https://github.com/maikramer/vramd
cd vramd
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q            # 760 testes, ~27 s, sem GPU
```

A suite é **CPU-only de propósito**. O supervisor não importa torch, e os
workers dos testes são duplos que falam o protocolo real. Se um teste teu
precisa de GPU, provavelmente está a testar o modelo e não o `vramd`.

## Antes de abrir PR

```bash
ruff check . && ruff format --check .
pytest -q
```

## O que este projeto valoriza

**Medir em vez de estimar.** A razão de existir do calibrador é que footprints
escritos à mão erram — em dez modelos reais, entre −3154 e +22448 MiB. Se mudas
números de admissão, diz como os verificaste. Uma corrida de `vramd calibrate`
antes/depois vale mais que um argumento.

**Falhar cedo e explicar.** Recusar um job em 0.2 s com uma mensagem acionável é
melhor que aceitá-lo e morrer com OOM a 80%. Quando algo não é fiável, o código
baixa a confiança e diz porquê — não arredonda para um número plausível.

**Um teste por defeito, nomeado pelo caso real.** Os testes de regressão deste
repositório citam o que os revelou (`texture2d marcava 19% "sem dados"`,
`text2icon: 82 MiB de 4764`). Isso torna óbvio, dois anos depois, porque é que a
condição existe.

**Comentários explicam o porquê, não o quê.** O código já diz o quê.

## Estilo

- Português nos comentários e docstrings (o projeto é escrito assim).
- Docstrings Google-style; `from __future__ import annotations` primeiro.
- 120 colunas, aspas duplas — tudo aplicado pelo `ruff` (`ruff.toml`).
- Tipos em código novo. `Any` é aceitável para objetos de modelo.

## Estrutura

```
src/vramd/
  server.py cli.py             supervisor e CLI
  job_queue.py scheduler.py    fila: prioridade + afinidade VRAM
  backend_manager.py           carga, admissão, evicção
  vram_planner.py              plano de evicção (puro, sem GPU)
  subprocess_pool.py           workers persistentes (JSONL por stdin/stdout)
  registry.py                  descriptors + camadas de configuração
  calibrate/                   medição de footprint
  worker/                      SDK do lado do modelo
  client.py                    submit/wait/cancel
```

`vram_planner.py`, `job_queue.py` e `calibrate/analysis.py` são **puros** — sem
GPU, sem sockets, sem threads. Mantê-los assim é o que torna a suite rápida.

## Releases

Versionamento semântico. Publicar é: actualizar `CHANGELOG.md` e a versão no
`pyproject.toml`, e criar a tag `vX.Y.Z` — o workflow trata do resto (build,
PyPI por Trusted Publishing, GitHub release).
