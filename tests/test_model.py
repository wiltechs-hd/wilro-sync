import copy

import torch

from wilrosync.models.dit_lipsync import WanLipSyncTransformer
from wilrosync.models.lora import add_lora, count_lora_params


def _inputs(b=2, c=4, f=3, h=8, w=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, c, f, h, w, generator=g)
    z = torch.randn(b, c, f, h, w, generator=g)
    t = torch.tensor([500.0, 120.0])[:b]
    txt = torch.randn(b, 7, 16, generator=g)
    win = torch.randn(b, 1 + 4 * (f - 1), 2, 3, 8, generator=g)
    return x, z, t, txt, win


def test_zero_init_matches_base(tiny_base, tiny_cfg):
    ref = copy.deepcopy(tiny_base)
    model = WanLipSyncTransformer(tiny_base, tiny_cfg).eval()
    x, z, t, txt, win = _inputs()
    with torch.no_grad():
        expected = ref(x, t, txt, return_dict=False)[0]
        tokens = model.encode_audio(win)
        got = model(x, z, t, txt, tokens)
        got_no_audio = model(x, z, t, txt, None)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got_no_audio, expected, atol=1e-5, rtol=1e-5)


def test_condition_and_audio_affect_output_after_training_step(tiny_base, tiny_cfg):
    model = WanLipSyncTransformer(tiny_base, tiny_cfg)
    model.set_trainable("adapter")
    x, z, t, txt, win = _inputs()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-1)
    out = model(x, z, t, txt, model.encode_audio(win))
    out.pow(2).mean().backward()
    assert model.audio_attn[0].to_out.weight.grad.abs().sum() > 0
    assert model.base.patch_embedding.weight.grad[:, 4:].abs().sum() > 0
    opt.step()
    with torch.no_grad():
        a = model(x, z, t, txt, model.encode_audio(win))
        b = model(x, z * 0, t, txt, model.encode_audio(win))
        c = model(x, z, t, txt, model.encode_audio(win, drop=torch.tensor([True, True])))
    assert (a - b).abs().max() > 1e-6
    assert (a - c).abs().max() > 1e-6


def test_trainable_modes_and_lora(tiny_base, tiny_cfg):
    model = WanLipSyncTransformer(tiny_base, tiny_cfg)
    model.set_trainable("adapter")
    frozen_ffn = model.base.blocks[0].ffn.net[2].weight
    assert not frozen_ffn.requires_grad
    add_lora(model, rank=4)
    model.set_trainable("lora")
    assert count_lora_params(model) > 0
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any("lora_" in n for n in trainable)
    assert not any(n.endswith("attn1.to_q.base_layer.weight") for n in trainable)
    # LoRA starts as identity
    x, z, t, txt, win = _inputs()
    with torch.no_grad():
        out = model(x, z, t, txt, None)
    assert torch.isfinite(out).all()


def test_checkpoint_roundtrip(tmp_path, tiny_base, tiny_cfg):
    base2 = copy.deepcopy(tiny_base)
    model = WanLipSyncTransformer(tiny_base, tiny_cfg)
    add_lora(model, rank=4)
    model.set_trainable("lora")
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.01)
    model.save_checkpoint(str(tmp_path))
    loaded = WanLipSyncTransformer.from_checkpoint(str(tmp_path), torch_dtype=torch.float32, base=base2)
    x, z, t, txt, win = _inputs()
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(x, z, t, txt, loaded.encode_audio(win)), model(x, z, t, txt, model.encode_audio(win))
        )


def test_gradient_checkpointing_same_grads(tiny_base, tiny_cfg):
    m1 = WanLipSyncTransformer(copy.deepcopy(tiny_base), tiny_cfg)
    m2 = copy.deepcopy(m1)
    m2.enable_gradient_checkpointing()
    for m in (m1, m2):
        m.set_trainable("adapter")
        with torch.no_grad():
            m.audio_attn[1].to_out.weight.fill_(0.01)
    x, z, t, txt, win = _inputs()
    for m in (m1, m2):
        m(x, z, t, txt, m.encode_audio(win)).sum().backward()
    torch.testing.assert_close(m1.audio_attn[0].to_q.weight.grad, m2.audio_attn[0].to_q.weight.grad)
