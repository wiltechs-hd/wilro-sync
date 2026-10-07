"""Data stages: sources, selection, and an end-to-end stage run with stub downloader / cropper / encoders."""

import json
import os
import shutil

import pytest

from wilrosync.data import sources
from wilrosync.data.download import export_url_list, youtube_id


def _hdtf_fixture(d):
    d.mkdir()
    (d / "RD_video_url.txt").write_text("Radio1 https://www.youtube.com/watch?v=AAAAAAAAAAA\n"
                                        "Radio2 https://www.youtube.com/watch?v=BBBBBBBBBBB&t=6s\n")
    (d / "RD_annotion_time.txt").write_text("Radio1.mp4 00:20-00:23\nRadio2.mp4 00:30-00:33 1:00:01-1:00:04\n")
    (d / "WDA_video_url.txt").write_text("WDA_X https://youtu.be/CCCCCCCCCCC\n")
    (d / "WDA_annotion_time.txt").write_text("WDA_X.mp4 00:01-00:04\n")
    return str(d)


def test_hdtf_segments(tmp_path):
    segs = sources.hdtf_segments(_hdtf_fixture(tmp_path / "hdtf"))
    assert [s["id"] for s in segs] == ["hdtf_Radio1_0", "hdtf_Radio2_0", "hdtf_Radio2_1", "hdtf_WDA_X_0"]
    assert segs[2]["start"] == 3601 and segs[2]["end"] == 3604 and segs[2]["group"] == "Radio2"


def test_talkvid_filters(tmp_path):
    def row(i, dover, person, res=200.0):
        return {"id": f"v{i}", "start-time": 0.0, "end-time": 10.0, "dover_scores": dover,
                "info": {"Person ID": person, "Video Link": f"https://www.youtube.com/watch?v=v{i:010d}",
                         "Language": "English"},
                "head_detail": {"scores": {"min_orientation": 90, "min_rotation": 80, "avg_resolution": res}}}
    rows = [row(0, 9.0, "1"), row(1, 7.0, "1"), row(2, 9.5, "2", res=90.0), row(3, 9.1, "3"),
            {"id": "nometa", "start-time": 0, "end-time": 5, "info": {}}]
    p = tmp_path / "meta.json"
    p.write_text(json.dumps(rows))
    segs = sources.talkvid_segments(str(p))
    assert [s["id"] for s in segs] == ["talkvid_v0", "talkvid_v3"]


def test_holdout_and_budget():
    segs = [{"id": f"s{i}", "group": f"g{i % 10}", "start": 0.0, "end": 60.0, "url": "u"} for i in range(100)]
    tr, ho = sources.split_holdout(segs, groups=2)
    assert len({s["group"] for s in ho}) == 2 and not ({s["group"] for s in tr} & {s["group"] for s in ho})
    assert sources.split_holdout(segs, groups=2)[1] == ho  # deterministic
    sel = sources.select_budget(tr, hours=0.5, max_seconds_per_group=120)
    assert sources.total_hours(sel) <= 0.5 and len(sel) == 16  # 8 groups x 2 min cap
    assert len({s["group"] for s in sources.select_budget(tr, max_segments=8)}) == 8  # spread over identities


def test_youtube_ids_and_url_export(tmp_path):
    assert youtube_id("https://www.youtube.com/watch?v=uBWJJvynXpA&t=6s") == "uBWJJvynXpA"
    assert youtube_id("https://youtu.be/CCCCCCCCCCC") == "CCCCCCCCCCC"
    src = tmp_path / "sources"
    src.mkdir()
    (src / "AAAAAAAAAAA.mp4").write_bytes(b"x")
    segs = [{"url": "https://www.youtube.com/watch?v=AAAAAAAAAAA"}, {"url": "https://youtu.be/BBBBBBBBBBB"},
            {"url": "https://www.youtube.com/watch?v=BBBBBBBBBBB"}]
    assert export_url_list(segs, str(tmp_path / "urls.txt"), str(src)) == 1
    assert (tmp_path / "urls.txt").read_text().strip() == "https://www.youtube.com/watch?v=BBBBBBBBBBB"


def test_mead_pairs(tmp_path):
    for actor in ("M003", "W009"):
        d = tmp_path / "mead" / actor / "video" / "front" / "happy" / "level_1"
        d.mkdir(parents=True)
        for i in range(3):
            (d / f"{i:03d}.mp4").write_bytes(b"x")
    rows = sources.mead_pairs(str(tmp_path / "mead"), per_folder=4, relative_to=str(tmp_path))
    assert len(rows) == 8 and all(r["cond"] != r["target"] for r in rows)
    assert rows[0]["cond"].startswith("mead/") and {r["group"] for r in rows} == {"mead_M003", "mead_W009"}


@pytest.fixture
def stage_env(tmp_path, monkeypatch):
    from test_prepare_clips import _make_video

    clip = tmp_path / "clip.mp4"
    _make_video(str(clip), seconds=2)
    return tmp_path, str(clip)


def test_stage_end_to_end(stage_env, monkeypatch):
    from test_prepare_clips import _StubEncoders
    from test_syncnet import _StubScorer

    from wilrosync.data import stages
    from wilrosync.data.datasets import LatentClipDataset
    from wilrosync.io.media import ffmpeg
    from wilrosync.train.config import load_config

    tmp, clip = stage_env
    work = tmp / "work"
    hdtf = _hdtf_fixture(tmp / "hdtf")
    cfg = stages.load_stage("smoke", ["hdtf.holdout_groups=1", "clips.size=32", "clips.frames=9",
                                      "clips.clips_per_minute=60", "mead.enabled=true"])
    plan = stages.plan_stage(cfg, str(work), hdtf_dir=hdtf)
    assert plan["estimate"]["train_segments"] + plan["estimate"]["holdout_segments"] == 4

    calls = []

    def fake_download(url, start, end, out, cookies=None, seg_id=None):
        calls.append(url)
        if "BBBB" in url:
            raise RuntimeError("Sign in to confirm you're not a bot")
        shutil.copy(clip, out)

    def fake_crop(src, dst, detector=None, size=512):
        ffmpeg("-i", src, "-vf", f"scale={size}:{size}", "-r", "25", "-c:a", "aac", dst)
        return {"window": [0, 0, 1], "face_px": 100.0}

    # MEAD tree (local files) for pseudo pairs
    mead = work / "raw" / "mead" / "M003" / "video" / "front" / "happy" / "level_1"
    mead.mkdir(parents=True)
    for i in range(2):
        shutil.copy(clip, mead / f"{i:03d}.mp4")

    c1 = stages.acquire(plan, str(work), downloader=fake_download, cropper=fake_crop, progress=None)
    assert c1["ok"] == 2 and c1["failed"] == 2 and c1["mead"]["ok"] == 2
    c2 = stages.acquire(plan, str(work), downloader=fake_download, cropper=fake_crop, progress=None)
    assert c2["existing"] == 2 and c2["failed"] == 0 and c2["skipped"] == 2  # failed segments not retried

    man = stages.write_manifests(plan, str(work))
    assert man["counts"]["pairs"] == 2 and man["counts"]["wild"] + man["counts"]["holdout"] == 2
    scores = {}
    for line in open(man["wild"]):
        scores[os.path.basename(json.loads(line)["video"])] = (1, 7.0)
    for line in open(man["pairs"]):
        r = json.loads(line)
        scores[os.path.basename(r["target"])] = (0, 6.0)
    res = stages.sync_and_prepare(plan, str(work), man, scorer=_StubScorer(scores), encoders=_StubEncoders())
    assert res.get("pairs_clips") == 2
    overrides = stages.train_overrides(plan, str(work))
    assert overrides["data.tds_enabled"] is True and overrides["max_steps"] == 2000
    tc = load_config("configs/train/stage_a.yaml", [f"{k}={v}" for k, v in overrides.items()])
    if man["counts"]["wild"]:
        assert len(LatentClipDataset.from_index(tc.data.arbitrary_index, kind="arbitrary")) >= 1
    assert len(LatentClipDataset.from_index(tc.data.pairs_index, kind="pseudo_pair")) == 2
    # a second stage reuses the prepared clips (no new encoding work)
    res2 = stages.sync_and_prepare(plan, str(work), man, scorer=_StubScorer(scores), encoders=None)
    assert res2.get("pairs_clips") == 2


def test_download_segments_script_and_acquire_from_segments(stage_env, monkeypatch):
    """scripts/download_segments.py (fake yt-dlp on PATH) -> raw/segments -> acquire uses them, no YouTube."""
    import stat
    import subprocess
    import sys

    from wilrosync.data import stages
    from wilrosync.io.media import _ffmpeg, ffmpeg

    tmp, clip = stage_env
    bindir = tmp / "bin"
    bindir.mkdir()
    fake = bindir / "yt-dlp"
    fake.write_text(f"""#!{sys.executable}
import shutil, sys
args = sys.argv[1:]
if "BBBBBBBBBBB" in args[-1]:
    print("ERROR: Sign in to confirm you're not a bot", file=sys.stderr); sys.exit(1)
out = args[args.index("-o") + 1].replace("%(ext)s", "mp4")
shutil.copy({clip!r}, out)
""")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    os.symlink(_ffmpeg(), bindir / "ffmpeg")
    env = dict(os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}")

    work = tmp / "work"
    cfg = stages.load_stage("smoke", ["hdtf.holdout_groups=1", "clips.size=32"])
    plan = stages.plan_stage(cfg, str(work), hdtf_dir=_hdtf_fixture(tmp / "hdtf"))
    seg_dir = work / "raw" / "segments"
    cmd = [sys.executable, "scripts/download_segments.py", "--plan", str(work / "stages" / "smoke" / "plan.json"),
           "--out", str(seg_dir), "--retries", "0"]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout
    assert "2 ok, 2 failed" in out and "cookies-from-browser" in out
    assert len(list(seg_dir.glob("hdtf_*.mp4"))) == 2
    assert "2 to download" in subprocess.run(cmd, env=env, capture_output=True, text=True).stdout  # resumes

    def fake_crop(src, dst, detector=None, size=512):
        ffmpeg("-i", src, "-vf", f"scale={size}:{size}", "-c:a", "aac", dst)
        return {"window": [0, 0, 1]}

    c = stages.acquire(plan, str(work), cropper=fake_crop, allow_youtube=False, cleanup_segments=True,
                       progress=None)
    assert c["ok"] == 2 and c["failed"] == 2
    assert not list(seg_dir.glob("hdtf_*.mp4"))  # cleaned up after cropping
