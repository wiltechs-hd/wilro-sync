"""LoRA on the pretrained Wan blocks (the new audio layers are trained in full, not via LoRA)."""

from __future__ import annotations

import importlib
import warnings

import torch.nn as nn

DEFAULT_TARGETS = [
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
    "ffn.net.0.proj", "ffn.net.2",
]


def _ignore_incompatible_torchao() -> None:
    """peft raises ImportError during LoRA injection when an old ``torchao`` is installed (Google Colab ships
    0.10, recent peft wants >= 0.16), even though plain LoRA never uses torchao. Treat it as absent instead."""
    try:
        from peft import import_utils

        import_utils.is_torchao_available()
    except ImportError as e:
        if "torchao" not in str(e):
            raise

        def _absent() -> bool:
            return False

        for name in ("peft.import_utils", "peft.tuners.lora.torchao", "peft.utils.quantization_utils"):
            try:
                mod = importlib.import_module(name)
            except ImportError:
                continue
            if hasattr(mod, "is_torchao_available"):
                mod.is_torchao_available = _absent
        warnings.warn(f"ignoring incompatible torchao for LoRA ({e})", stacklevel=2)


def add_lora(model: nn.Module, rank: int = 64, alpha: int | None = None, targets: list[str] | None = None,
             dropout: float = 0.0) -> nn.Module:
    """Inject LoRA adapters into ``model.base.blocks.*`` only. Returns the model (modified in place)."""
    from peft import LoraConfig, inject_adapter_in_model

    _ignore_incompatible_torchao()
    targets = targets or DEFAULT_TARGETS
    alt = "|".join(t.replace(".", r"\.") for t in targets)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha or rank,
        lora_dropout=dropout,
        target_modules=rf"blocks\.\d+\.({alt})",
        init_lora_weights=True,
    )
    inject_adapter_in_model(config, model.base)
    model.cfg.lora = {"rank": rank, "alpha": alpha or rank, "targets": list(targets)}
    return model


def count_lora_params(model: nn.Module) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
