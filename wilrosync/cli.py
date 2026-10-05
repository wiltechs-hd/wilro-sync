"""Command line: ``wilro-sync infer ...`` and ``wilro-sync train ...``.

Multi-GPU training: ``accelerate launch -m wilrosync.cli train --config configs/train/stage_a.yaml``.
"""

from __future__ import annotations

import argparse
import sys


def _parse_bbox(s: str) -> tuple[float, float, float, float]:
    vals = [float(v) for v in s.replace(" ", "").split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("bbox must be x0,y0,x1,y1")
    return tuple(vals)  # type: ignore[return-value]


def cmd_infer(a: argparse.Namespace) -> None:
    import torch

    from .io.media import read_audio, read_video, write_video
    from .pipeline.lipsync import LipSyncPipeline, load_infer_config
    from .pipeline.target import TargetSignals

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    pipe = LipSyncPipeline.from_pretrained(a.ckpt, device=a.device, dtype=dtype, text_device=a.text_device)
    frames, fps = read_video(a.video)
    wav = read_audio(a.audio)
    target = None
    if a.bbox:
        t, h, w = frames.shape[:3]
        target = TargetSignals.from_bbox(t, h, w, a.bbox)
    cfg = load_infer_config(a.config)
    for key in ("steps", "tau", "seed", "max_side", "prompt"):
        if getattr(a, key) is not None:
            setattr(cfg, key, getattr(a, key))
    for key in ("omega_peak", "omega_base", "gamma"):
        if getattr(a, key) is not None:
            setattr(cfg.dscfg, key, getattr(a, key))
    if a.no_region_lock:
        cfg.region_lock = False
    out = pipe(frames, fps, wav, target=target, cfg=cfg)
    write_video(a.out, out, fps, audio_path=a.audio)
    print(f"wrote {a.out} ({len(out)} frames)")


def cmd_train(a: argparse.Namespace) -> None:
    from .train.config import load_config
    from .train.trainer import train

    train(load_config(a.config, a.overrides))


def cmd_lse(a: argparse.Namespace) -> None:
    import json

    from .eval.syncnet import SyncNet, lse

    scorer = SyncNet(device=a.device)
    rows = []
    for video in a.videos:
        r = lse(video, a.audio, bbox=a.bbox, scorer=scorer, assume_cropped=a.assume_cropped)
        rows.append({"video": video, **r})
        print(f"{video}: LSE-C {r['lse_c']:.3f}  LSE-D {r['lse_d']:.3f}  offset {r['offset']}")
    ok = [r for r in rows if r["num_frames"]]
    if len(ok) > 1:
        print(f"mean over {len(ok)}: LSE-C {sum(r['lse_c'] for r in ok) / len(ok):.3f}  "
              f"LSE-D {sum(r['lse_d'] for r in ok) / len(ok):.3f}")
    if a.out:
        with open(a.out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["sync-filter"]:  # own argument parser (also used by the notebook)
        from .data.sync_filter import main as sync_filter_main

        sync_filter_main(argv[1:])
        return
    if argv[:1] == ["data"]:
        from .data.stages import main as data_main

        data_main(argv[1:])
        return
    p = argparse.ArgumentParser("wilro-sync")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("infer", help="lip-sync a video to an audio track")
    i.add_argument("--ckpt", required=True, help="checkpoint directory (wilrosync_config.json + weights)")
    i.add_argument("--video", required=True)
    i.add_argument("--audio", required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--config", default=None, help="inference YAML (see configs/infer/default.yaml)")
    i.add_argument("--bbox", type=_parse_bbox, help="target face box x0,y0,x1,y1 in pixels (multi-face scenes)")
    i.add_argument("--steps", type=int, default=None)
    i.add_argument("--tau", type=float, default=None)
    i.add_argument("--omega-peak", type=float, default=None)
    i.add_argument("--omega-base", type=float, default=None)
    i.add_argument("--gamma", type=float, default=None)
    i.add_argument("--max-side", type=int, default=None)
    i.add_argument("--prompt", default=None)
    i.add_argument("--seed", type=int, default=None)
    i.add_argument("--no-region-lock", action="store_true")
    i.add_argument("--device", default="cuda")
    i.add_argument("--text-device", default=None, help="e.g. cpu to keep umT5 off the GPU")
    i.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    i.set_defaults(func=cmd_infer)

    t = sub.add_parser("train", help="train the lip-sync DiT")
    t.add_argument("--config", default=None)
    t.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. max_steps=1000 data.num_workers=4")
    t.set_defaults(func=cmd_train)

    sub.add_parser("data", help="build a data stage: smoke | first_model | paper_scale "
                   "(see `wilro-sync data --help`)")
    sub.add_parser("sync-filter", help="score a clip manifest with SyncNet and drop badly synced videos "
                   "(see `wilro-sync sync-filter --help`)")

    ev = sub.add_parser("lse", help="LSE-C / LSE-D lip-sync metrics (SyncNet) for one or more videos")
    ev.add_argument("videos", nargs="+")
    ev.add_argument("--audio", default=None, help="score against this audio instead of each video's own track")
    ev.add_argument("--bbox", type=_parse_bbox, help="score only the face starting in this box (multi-face)")
    ev.add_argument("--assume-cropped", action="store_true")
    ev.add_argument("--device", default=None)
    ev.add_argument("--out", default=None, help="write per-video results as jsonl")
    ev.set_defaults(func=cmd_lse)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
