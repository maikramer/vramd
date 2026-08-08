# vramd

[![CI](https://github.com/maikramer/vramd/actions/workflows/ci.yml/badge.svg)](https://github.com/maikramer/vramd/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/vramd.svg)](https://pypi.org/project/vramd/)
[![Python](https://img.shields.io/pypi/pyversions/vramd.svg)](https://pypi.org/project/vramd/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Controlo de admissão de VRAM para inferência generativa em GPUs de consumo.**

Um processo detém a GPU e decide quem entra. Admite pelo **pico real** — pesos +
activação + margem, não só pesos — põe em fila com prioridade e afinidade,
evicta por peso+LRU, e corre cada modelo num processo e venv próprios.

Feito para inferência que dura **segundos a minutos** numa placa que não chega
para tudo. Não é um servidor de LLM: não optimiza throughput de tokens, optimiza
*caber*.

```bash
pip install vramd          # 9 MB — o supervisor não importa torch
vramd start &
vramd submit meu-modelo --prompt "…" --wait
```

## O problema

Tens 6 GB de VRAM e cinco modelos que, somados, pedem 40. Cada um corre bem
sozinho. Juntos, o segundo job entra a meio do primeiro e ambos morrem com
`CUDA out of memory` — depois de já terem carregado os pesos.

As soluções habituais assumem o que aqui não se verifica: que o modelo cabe
(vLLM, TGI), que o ambiente é homogéneo (Ray Serve, Triton), ou que a unidade de
trabalho é um token e não um job de dois minutos.

## O que o vramd faz de diferente

**Admite pelo pico, não pelos pesos.** A pergunta não é "o modelo cabe" — é
"cabe o pico da inferência". É a diferença entre recusar em 0.2 s e morrer a 80%
do job com os pesos já carregados.

**Cada modelo no seu venv.** Um backend é um processo com o seu próprio
interpretador. Modelos com dependências incompatíveis — torch 2.x contra 2.y,
wheels CUDA diferentes — coexistem sem se verem.

**Afinidade na fila.** Se a cabeça precisa de um modelo frio e mais atrás há um
job cujo modelo já está em VRAM, o scheduler salta a cabeça (até 3 vezes, depois
força). Onde um load custa 60 s, isto muda um batch de 40 minutos para 10.

**Cancelamento cooperativo.** Jobs longos reportam progresso por fase e param
*entre* fases — sem matar kernels CUDA a meio.

**Mede em vez de adivinhar.** `vramd calibrate` corre um job real, amostra a
VRAM por processo a 20 Hz e escreve o footprint medido. Sobre dez modelos reais
numa RTX 4050, os valores escritos à mão erravam entre −3154 e +22448 MiB.

## Integrar um modelo

Três métodos:

```python
from vramd.worker import WorkerAdapter, run_worker_loop

class Adapter(WorkerAdapter):
    name = "meu-modelo"

    def load(self, **kw):
        import torch, meulib
        return meulib.load(device=kw.get("device", "cuda"))

    def generate(self, model, request):
        if self.should_abort(request):
            return self.cancelled_response()
        self.report_progress(request, 0.0, "a gerar")
        return {"status": "ok", "output": model(request["prompt"])}

    def unload(self, model):
        del model

if __name__ == "__main__":
    run_worker_loop(Adapter, backend_name="meu-modelo")
```

E registá-lo — sem tocar no código do vramd:

```yaml
# ~/.config/vramd/backends.d/meu-modelo.yaml
version: 2
backends:
  - name: meu-modelo
    adapter: meu_pacote.adapter
    vram_mib: 4200
    priority: 20
    runtime:
      command: ["/opt/meu-modelo/venv/bin/python", "-m", "meu_pacote.worker"]
      env: { HF_HOME: ~/hf-cache }
    load_keys: [device, compute_type]
    shape_keys: [device]
```

Exemplo completo e executável: [`examples/echo-backend/`](examples/echo-backend/).

## Calibração

O atrito de qualquer sistema destes é a pergunta "que números meto no
descriptor?". A resposta do vramd é: nenhum — mede-se.

```bash
vramd calibrate meu-modelo --repeats 3 --out ~/.config/vramd/backends.d/medido.yaml
```

Corre o job, separa **contexto CUDA / pesos / activação** pelas fronteiras de
fase, e escreve o descriptor. O que apanha, e que uma estimativa não apanha:

| Sinal | Porque importa |
|---|---|
| pico no **load** acima do da inferência | carregar fp16 e quantizar depois OOMa antes de gerar |
| activação ≫ pesos | o modelo carrega outro modelo dentro do `generate` |
| nada residente após o load | carga preguiçosa: não há o que evictar |
| `unload` que não devolve VRAM | evictar este backend não liberta nada — o plano de evicção seria ficção |
| fuga por repetição | o residente cresce a cada job |
| warmup na 1.ª inferência | calibrar com `--repeats 1` inflaciona o número |

Cada medição guarda as amostras cruas: `vramd recalibrate relatorio.json` refaz
os números quando a análise melhora, sem voltar a ocupar a GPU.

## Comandos

```
start stop status queue wait cancel flush backends preload evict reap
respawn zero stats debug bench doctor calibrate recalibrate
```

- `vramd status` / `queue` — quem tem a GPU e o que espera
- `vramd zero` — liberta toda a VRAM ociosa sem parar o supervisor
- `vramd respawn <backend>` — reinicia só um worker (código novo) sem parar a fila
- `vramd doctor` — diagnóstico de ambiente

**Nunca é preciso `kill`.** Matar processos GPU corre contra a fila e mata o
workload errado.

## Configuração

```
data/backends.yaml (exemplo)  →  $VRAMD_BACKENDS_FILE  →  ~/.config/vramd/backends.d/*.yaml
```

Sobreposição **por chave**: um ficheiro com `{name: x, vram_mib: 5632}` corrige
só esse campo e herda o resto. É assim que um descriptor calibrado entra em
vigor sem editar o pacote.

Variáveis: `VRAMD_BACKENDS_FILE`, `VRAMD_BACKENDS_DIR`, `VRAMD_TOOLS_ROOT`,
`VRAMD_MAX_INFLIGHT`, `VRAMD_MAX_QUEUE_DEPTH`, `VRAMD_VRAM_SAFETY_MIB`,
`VRAMD_PRIORITY`.

## Limites conhecidos

Vale a pena saber antes de adoptar:

- **`MAX_INFLIGHT=1` por omissão** — uma geração de cada vez. É a escolha certa
  para 6 GB e subutiliza uma A100. Há suporte para >1 com verificação de VRAM,
  mas falta *packing* a sério.
- **Multi-GPU sem placement central.** `gpu_ids` é passado ao worker; o
  supervisor não decide colocação nem contabiliza por device.
- **POSIX.** A leitura dos pipes usa `select`/`O_NONBLOCK`. Windows precisa de
  uma camada de IO diferente.
- **Sem autenticação.** Socket unix com permissões de utilizador. Local, não
  partilhado.
- **A calibração não faz milagres.** Mede o que o teu pipeline faz. Um modelo
  que carrega tudo em fp16 de uma vez não passa a caber por ser medido — só
  passas a saber que não cabe, e em 0.2 s em vez de a meio do job.

## Origem

Extraído do [AiGameKit](https://github.com/maikramer), onde nasceu para pôr dez
modelos generativos (texto→imagem, →3D, →áudio, →movimento) a partilhar uma RTX
4050 de 6 GB sem intervenção manual. Os números deste README são medições dessa
placa.

## Contribuir

[`CONTRIBUTING.md`](CONTRIBUTING.md) — arrancar, estilo, e o que este projeto
valoriza. A suite corre em ~27 s sem GPU.

## Licença

MIT — ver [LICENSE](LICENSE).
