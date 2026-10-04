from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import OmegaConf

from ..models.dit_lipsync import LipSyncModelConfig


@dataclass
class DataConfig:
    pairs_index: str | None = None  # jsonl of pseudo-paired clips (MEAD)
    arbitrary_index: str | None = None  # jsonl of arbitrary clips
    tds_enabled: bool = True  # timestep-dependent sampling (paper Eq. 3)
    tds_threshold: float = 0.85  # paper: t > 850
    sigma_mode: str = "shifted_uniform"
    sigma_shift: float = 3.0
    null_text: str | None = None  # safetensors with key "text": trimmed embedding of ""
    num_workers: int = 2
    synthetic: bool = False  # random tensors (smoke tests)
    synthetic_shape: list[int] = field(default_factory=lambda: [16, 3, 8, 8])


@dataclass
class TrainConfig:
    model: LipSyncModelConfig = field(default_factory=LipSyncModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tiny_base: dict[str, Any] | None = None  # build a random tiny Wan instead of loading weights (tests)

    train_mode: str = "lora"  # adapter | lora | full
    lora_rank: int = 64
    lora_alpha: int | None = None

    lr: float = 1e-5  # pretrained weights / LoRA (paper: 1e-5)
    lr_new: float = 1e-4  # newly added layers
    weight_decay: float = 0.0
    optimizer: str = "adamw"  # adamw | adamw8bit
    max_grad_norm: float = 1.0
    batch_size: int = 1
    grad_accum: int = 8
    max_steps: int = 80_000
    warmup_steps: int = 500
    mixed_precision: str = "bf16"  # bf16 | no
    gradient_checkpointing: bool = True
    ema_decay: float = 0.0  # 0 disables EMA

    p_drop_audio: float = 0.15
    p_drop_text: float = 0.10
    p_drop_both: float = 0.05
    mouth_loss_weight: float = 0.0  # lambda for 1 + lambda * G_mouth loss weighting
    mouth_sigma_rel: float = 0.35

    output_dir: str = "runs/stage_a"
    save_every: int = 2000
    keep_last: int = 0  # >0: delete older step_* checkpoints, keep this many (saves Drive space)
    log_every: int = 20
    resume_from: str | None = None
    seed: int = 42


def load_config(path: str | None = None, overrides: list[str] | None = None) -> TrainConfig:
    cfg = OmegaConf.structured(TrainConfig)
    if path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    obj = OmegaConf.to_object(cfg)
    assert isinstance(obj, TrainConfig)
    # YAML 1.1 reads an unquoted `no` as False (e.g. `mixed_precision=no` on the command line)
    if str(obj.mixed_precision).lower() in {"false", "no", "none", "0"}:
        obj.mixed_precision = "no"
    return obj
