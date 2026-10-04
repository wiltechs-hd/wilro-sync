import pytest
import torch

TINY_WAN = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=2,
    attention_head_dim=12,
    in_channels=4,
    out_channels=4,
    text_dim=16,
    freq_dim=32,
    ffn_dim=48,
    num_layers=2,
    cross_attn_norm=True,
    rope_max_seq_len=64,
)


@pytest.fixture
def tiny_base():
    from diffusers import WanTransformer3DModel

    torch.manual_seed(0)
    m = WanTransformer3DModel(**TINY_WAN).eval()
    # random (non-zero) AdaLN tables so the equivalence test is meaningful
    return m


@pytest.fixture
def tiny_cfg():
    from wilrosync.models.dit_lipsync import LipSyncModelConfig

    return LipSyncModelConfig(base_repo="tiny", audio_layers=3, audio_dim=8, audio_window=2)
