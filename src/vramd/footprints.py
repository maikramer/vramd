"""Registry de pegadas de modelo — pesos, activação e quantização.

Extraído do ``vramd.footprints`` com **só a parte pura**: a matemática
de footprint não precisa de torch, e é ela que o admit usa. O planner de
offload (que precisa de diffusers) ficou de fora de propósito — o ``vramd``
admite e agenda; quem decide como carregar é o worker.

Uma pegada declarada aqui é um **ponto de partida**. O número que vale é o
medido: ``vramd calibrate <backend>`` escreve `vram:` no descriptor e esse
vence a estimativa.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

QUANT_WEIGHT_FACTOR: dict[str, float] = {
    "none": 1.0,
    "fp8": 0.55,
    "fp8-layerwise": 0.55,  # diffusers.hooks layerwise casting (storage fp8, compute bf16)
    "sdnq-fp8": 0.55,
    "int8": 0.55,
    "sdnq-uint8": 0.55,
    "sdnq-int8": 0.55,
    "int4": 0.32,
    "sdnq-int4": 0.32,
}

# Ordem de preferência (qualidade desce, poupança sobe). "none" primeiro; int4 por
# último. fp8-layerwise antes de SDNQ (melhor qualidade, sem needing Triton/kernels);
# SDNQ-first para int8/int4 (uint8 é o preset mais testado; int4 só quando é preciso caber).
_QUANT_LADDER: tuple[str, ...] = ("none", "fp8-layerwise", "sdnq-uint8", "sdnq-int8", "sdnq-int4")

# Offload por ordem de agressividade. "none" = tudo na GPU.
OFFLOAD_NONE = "none"
OFFLOAD_GROUP_STREAM = "group_stream"  # group offload + CUDA streams (preferido)
OFFLOAD_MODEL = "model_cpu"  # módulos inteiros migram 1 a 1 (rápido)
OFFLOAD_SEQUENTIAL = "sequential_cpu"  # sub-módulos migram (lento, mínimo VRAM)


@dataclass(frozen=True)
class ModelFootprint:
    """Pegada de memória estimada de um modelo, em GiB.

    Args:
        fp16_weights_gib: Peso dos pesos do modelo em fp16 (sem quantização).
        activation_gib: Overhead de ativação/runtime no pico, à resolução-alvo.
            Para difusão de imagem ~1.0-2.0; para 3D/DiT pode ser maior.
        largest_module_gib: Maior sub-módulo individual (define o pico em
            ``model_cpu`` offload, onde um módulo de cada vez está na GPU). Se 0,
            estima-se como 40% dos pesos fp16.
        architecture: Nome da arquitetura (ex: ``"flux"``, ``"hunyuan3d"``) para
            ligar ao ``no_split_module_classes`` do registry multi-GPU. ``None``
            se irrelevante (offload only, sem multi-GPU).
    """

    fp16_weights_gib: float
    activation_gib: float = 1.5
    largest_module_gib: float = 0.0
    architecture: str | None = None

    def weights_gib(self, quant_mode: str) -> float:
        return self.fp16_weights_gib * QUANT_WEIGHT_FACTOR.get(quant_mode, 1.0)

    def largest_gib(self, quant_mode: str) -> float:
        base = self.largest_module_gib or (self.fp16_weights_gib * 0.4)
        return base * QUANT_WEIGHT_FACTOR.get(quant_mode, 1.0)


# Registry centralizado de pegadas por modelo/família. Cada tool consulta aqui
# em vez de inline literais dispersos. Valores calibrados das tools de produção.
# "flux-dev-uint4": fp16_weights_gib reflete o tamanho JÁ quantizado (uint4 ~2.2 GiB);
# usar com allow_quant=("none",) para não duplicar a redução.
# (Antes 7.4 GiB — o peak recusava o skymap2d em GPUs ~6 GB apesar do modelo
# SDNQ uint4 real caber. Calibrado do checkpoint Disty0/FLUX.1-dev-SDNQ-uint4.)
FOOTPRINTS: dict[str, ModelFootprint] = {
    "flux-klein-4b": ModelFootprint(14.0, 1.5, 5.0, architecture="flux"),
    "flux-klein-9b": ModelFootprint(26.0, 1.5, 9.0, architecture="flux"),
    "flux-dev-uint4": ModelFootprint(2.2, 2.0, 3.0, architecture="flux"),
    "hunyuan3d-2.1-dit": ModelFootprint(6.5, 1.5, 5.0, architecture="hunyuan3d"),
    # Hunyuan3D-Omni (~3.3B): DiT + ShapeVAE + OmniEncoder/DINOv2; README ~10 GB fp16.
    "hunyuan3d-omni": ModelFootprint(10.0, 2.0, 6.0, architecture="hunyuan3d"),
    # Hunyuan3D-Part: DiT ~3.3 + conditioner ~0.9 + ShapeVAE ~0.3 + P3-SAM ~0.2 + overhead.
    "hunyuan3d-part": ModelFootprint(4.75, 1.5, 5.2, architecture="dit"),
    "hunyuan-paint": ModelFootprint(6.0, 2.0, 5.0, architecture="unet"),
    "stable-audio-open": ModelFootprint(3.5, 1.5, 2.0, architecture="stable-audio"),
    # HY-Motion staged load (Text2D-like): GPU holds DiT OR text encode, not both.
    # Text encoder (Qwen3-8B) runs on CPU when mem-eff; DiT (+optional SDNQ) on GPU.
    # Pegadas = DiT residente + act (não soma Qwen+DiT ~24 GiB).
    "hy-motion-lite": ModelFootprint(1.2, 1.2, 1.0, architecture="dit"),
    "hy-motion-full": ModelFootprint(2.5, 1.5, 2.0, architecture="dit"),
    # Legacy Motius key (retired path) — keep for old payloads.
    "motius-t2mgpt": ModelFootprint(1.5, 0.8, 2.5, architecture="dit"),
    # Sana Sprint 600M transformer + Gemma 2B encoder (~7.3 GiB fp16 total).
    "sana-sprint-600m": ModelFootprint(7.3, 1.5, 3.0, architecture="sana"),
}

# Footprint genérico de fallback (modelo médio ~8 GiB) quando a chave é desconhecida.
_DEFAULT_FOOTPRINT = ModelFootprint(8.0, 1.5, 3.2)


@dataclass(frozen=True)
class ModelFootprint:
    """Pegada de memória estimada de um modelo, em GiB.

    Args:
        fp16_weights_gib: Peso dos pesos do modelo em fp16 (sem quantização).
        activation_gib: Overhead de ativação/runtime no pico, à resolução-alvo.
            Para difusão de imagem ~1.0-2.0; para 3D/DiT pode ser maior.
        largest_module_gib: Maior sub-módulo individual (define o pico em
            ``model_cpu`` offload, onde um módulo de cada vez está na GPU). Se 0,
            estima-se como 40% dos pesos fp16.
        architecture: Nome da arquitetura (ex: ``"flux"``, ``"hunyuan3d"``) para
            ligar ao ``no_split_module_classes`` do registry multi-GPU. ``None``
            se irrelevante (offload only, sem multi-GPU).
    """

    fp16_weights_gib: float
    activation_gib: float = 1.5
    largest_module_gib: float = 0.0
    architecture: str | None = None

    def weights_gib(self, quant_mode: str) -> float:
        return self.fp16_weights_gib * QUANT_WEIGHT_FACTOR.get(quant_mode, 1.0)

    def largest_gib(self, quant_mode: str) -> float:
        base = self.largest_module_gib or (self.fp16_weights_gib * 0.4)
        return base * QUANT_WEIGHT_FACTOR.get(quant_mode, 1.0)


# Registry centralizado de pegadas por modelo/família. Cada tool consulta aqui
# em vez de inline literais dispersos. Valores calibrados das tools de produção.
# "flux-dev-uint4": fp16_weights_gib reflete o tamanho JÁ quantizado (uint4 ~2.2 GiB);
# usar com allow_quant=("none",) para não duplicar a redução.
# (Antes 7.4 GiB — o peak recusava o skymap2d em GPUs ~6 GB apesar do modelo
# SDNQ uint4 real caber. Calibrado do checkpoint Disty0/FLUX.1-dev-SDNQ-uint4.)

FOOTPRINTS: dict[str, ModelFootprint] = {
    "flux-klein-4b": ModelFootprint(14.0, 1.5, 5.0, architecture="flux"),
    "flux-klein-9b": ModelFootprint(26.0, 1.5, 9.0, architecture="flux"),
    "flux-dev-uint4": ModelFootprint(2.2, 2.0, 3.0, architecture="flux"),
    "hunyuan3d-2.1-dit": ModelFootprint(6.5, 1.5, 5.0, architecture="hunyuan3d"),
    # Hunyuan3D-Omni (~3.3B): DiT + ShapeVAE + OmniEncoder/DINOv2; README ~10 GB fp16.
    "hunyuan3d-omni": ModelFootprint(10.0, 2.0, 6.0, architecture="hunyuan3d"),
    # Hunyuan3D-Part: DiT ~3.3 + conditioner ~0.9 + ShapeVAE ~0.3 + P3-SAM ~0.2 + overhead.
    "hunyuan3d-part": ModelFootprint(4.75, 1.5, 5.2, architecture="dit"),
    "hunyuan-paint": ModelFootprint(6.0, 2.0, 5.0, architecture="unet"),
    "stable-audio-open": ModelFootprint(3.5, 1.5, 2.0, architecture="stable-audio"),
    # HY-Motion staged load (Text2D-like): GPU holds DiT OR text encode, not both.
    # Text encoder (Qwen3-8B) runs on CPU when mem-eff; DiT (+optional SDNQ) on GPU.
    # Pegadas = DiT residente + act (não soma Qwen+DiT ~24 GiB).
    "hy-motion-lite": ModelFootprint(1.2, 1.2, 1.0, architecture="dit"),
    "hy-motion-full": ModelFootprint(2.5, 1.5, 2.0, architecture="dit"),
    # Legacy Motius key (retired path) — keep for old payloads.
    "motius-t2mgpt": ModelFootprint(1.5, 0.8, 2.5, architecture="dit"),
    # Sana Sprint 600M transformer + Gemma 2B encoder (~7.3 GiB fp16 total).
    "sana-sprint-600m": ModelFootprint(7.3, 1.5, 3.0, architecture="sana"),
}

# Footprint genérico de fallback (modelo médio ~8 GiB) quando a chave é desconhecida.
_DEFAULT_FOOTPRINT = ModelFootprint(8.0, 1.5, 3.2)


def get_footprint(key: str) -> ModelFootprint:
    """Consulta o registry de pegadas por chave canónica.

    Args:
        key: Chave do modelo (ex: ``"flux-klein-9b"``, ``"hunyuan3d-2.1-dit"``).

    Returns:
        :class:`ModelFootprint` do registry, ou um footprint genérico de fallback
        com um warning (para que tools novas não partam se a chave não existir).
    """
    fp = FOOTPRINTS.get(key)
    if fp is not None:
        return fp

    logging.getLogger("vramd.footprints").warning(
        "Footprint '%s' não registry — a usar footprint genérico de fallback.", key
    )
    return _DEFAULT_FOOTPRINT
