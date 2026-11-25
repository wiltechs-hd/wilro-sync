"""HTTP server: request parsing, endpoints, auth, queue (FastAPI TestClient, fake inference)."""

from __future__ import annotations

import shutil
import threading
import time

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from wilrosync.server import ServerConfig, create_app, parse_params  # noqa: E402


def _mp4_bytes(tmp_path) -> bytes:
    from wilrosync.io.media import write_video

    p = tmp_path / "probe.mp4"
    write_video(str(p), np.zeros((4, 32, 32, 3), np.uint8), 25.0)
    return p.read_bytes()


def _files(tmp_path, mp4, audio=True):
    files = {"video": ("clip.mp4", mp4, "video/mp4")}
    if audio:
        files["audio"] = ("speech.wav", b"RIFF-fake-audio", "audio/wav")
    return files


def _stub(calls):
    def stub(pipe, video, audio, out, *, bbox=None, cfg=None, **face):
        calls.append(dict(pipe=pipe, video=video, audio=audio, out=out, bbox=bbox, steps=cfg.steps,
                          window=cfg.window, omega_peak=cfg.dscfg.omega_peak, mode=face.get("mode"),
                          loop=face.get("loop"), track=face.get("track"), preview=face.get("preview")))
        shutil.copyfile(video, out)
        return out

    return stub


def make_client(infer, tmp_path, load_pipe_fn=None, **cfg_kw):
    if "ckpt" not in cfg_kw:  # default: a fake checkpoint dir the stub loader can "load"
        (tmp_path / "ck").mkdir(exist_ok=True)
        (tmp_path / "ck" / "wilrosync_config.json").write_text("{}")
        cfg_kw["ckpt"] = str(tmp_path / "ck")
    cfg = ServerConfig(load_at_startup=False, **cfg_kw)
    load_pipe_fn = load_pipe_fn or (lambda ckpt, config: object())
    return TestClient(create_app(cfg, infer_fn=infer, load_pipe_fn=load_pipe_fn))


@pytest.fixture
def calls():
    return []


@pytest.fixture
def client(calls, tmp_path):
    with make_client(_stub(calls), tmp_path) as c:
        yield c


def _wait_status(client, jid, want=("done", "error")):
    for _ in range(200):
        s = client.get(f"/jobs/{jid}").json()
        if s["status"] in want:
            return s
        time.sleep(0.05)
    raise AssertionError(f"job {jid} never reached {want}: {s}")


# ----------------------------------------------------------------------------- parsing
def test_parse_defaults():
    p = parse_params({})
    assert p["bbox"] is None and not p["preview"] and p["ckpt"] is None and p["timeout"] is None
    assert p["cfg"]["steps"] == 10 and p["cfg"]["window"] == 129 and p["cfg"]["overlap"] == 2
    assert p["cfg"]["dscfg"]["omega_peak"] == 4.5 and p["cfg"]["dscfg"]["gamma"] == 1.5
    f = p["face"]
    assert f["mode"] == "lower" and f["loop"] == "reverse" and f["crop"] == "auto"
    assert f["track"] == {"width": 1.8, "bottom": 0.2, "top": 0.1, "spread": 0.2}
    assert f["preroll"] == 4 and f["postroll"] == 8 and f["reference"] == "stride2" and f["blend"] == "double"


def test_parse_and_errors():
    assert parse_params({"bbox": " 575,140,675,255 "})["bbox"] == (575.0, 140.0, 675.0, 255.0)
    assert parse_params({"track": '{"width": 2.0}'})["face"]["track"] == {"width": 2.0}
    assert parse_params({"preview": "1"})["preview"]
    assert parse_params({"preview": "no"})["preview"] is False
    assert parse_params({"ckpt": " runs/x "})["ckpt"] == "runs/x"
    for bad in ({"bbox": "1,2,3"}, {"bbox": "1,2,3,4,5"}, {"bbox": "5,5,1,1"}, {"bbox": "-1,2,3,4"},
                {"steps": "0"}, {"steps": "many"}, {"window": "100"}, {"chain": "7"},
                {"mode": "middle"}, {"oval": "round"}, {"tau": "2"},
                {"track": '{"nope": 1}'}, {"whatever": "1"}, {"size": "60"}):
        with pytest.raises(ValueError):
            parse_params(bad)


def test_parse_oval_presets():
    from wilrosync.pipeline.oval import FIXED_DEFAULT, FIXED_OLD

    assert parse_params({"oval": "fixed"})["face"]["oval"] == dict(FIXED_DEFAULT)
    assert parse_params({"oval": "fixed_old"})["face"]["oval"] == dict(FIXED_OLD)
    oval = parse_params({"oval": '{"cx": 0.5, "cy": 0.6, "rx": 0.2, "ry": 0.3}'})["face"]["oval"]
    assert oval["rx"] == 0.2


# ----------------------------------------------------------------------------- sync endpoint
def test_lipsync_sync_ok(client, calls, tmp_path):
    r = client.post("/lipsync", files=_files(tmp_path, _mp4_bytes(tmp_path)),
                    data={"bbox": "10,20,110,220", "steps": "12", "track": '{"width": 2.0}',
                          "omega_peak": "3.5"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "video/mp4"
    assert float(r.headers["x-inference-seconds"]) >= 0 and r.content
    (c,) = calls
    assert c["bbox"] == (10.0, 20.0, 110.0, 220.0)
    assert (c["steps"], c["window"], c["omega_peak"]) == (12, 129, 3.5)
    assert c["track"] == {"width": 2.0}
    assert c["mode"] == "lower" and c["preview"] is False
    assert c["pipe"] is not None and c["audio"].endswith(".wav")


def test_lipsync_validation(client, tmp_path):
    mp4 = _mp4_bytes(tmp_path)
    for data in ({"bbox": "nope"}, {"mode": "middle"}, {"steps": "0"}, {"window": "100"},
                 {"unknown_field": "1"}):
        r = client.post("/lipsync", files=_files(tmp_path, mp4), data=data)
        assert r.status_code == 400, (data, r.text)
    assert client.post("/lipsync", files={"video": ("v.mp4", mp4)}, data={}).status_code == 400  # no audio
    assert client.post("/lipsync", files={"audio": ("a.wav", b"x")}, data={}).status_code == 400
    r = client.post("/lipsync", files={"video": ("v.gif", mp4), "audio": ("a.wav", b"x")}, data={})
    assert r.status_code == 400  # bad video suffix


def test_lipsync_preview(client, calls, tmp_path):
    r = client.post("/lipsync", files={"video": ("clip.mp4", _mp4_bytes(tmp_path)),
                                       "audio": ("a.wav", b"RIFF")}, data={"preview": "true"})
    assert r.status_code == 200
    (c,) = calls
    assert c["preview"] is True and c["pipe"] is None


def test_server_error_is_500(tmp_path):
    def boom(pipe, video, audio, out, **kw):
        raise RuntimeError("gpu exploded")

    with make_client(boom, tmp_path) as c:
        r = c.post("/lipsync", files=_files(tmp_path, _mp4_bytes(tmp_path)))
        assert r.status_code == 500 and "gpu exploded" in r.json()["detail"]


# ----------------------------------------------------------------------------- auth
def test_api_key(tmp_path):
    with make_client(_stub([]), tmp_path, api_key="secret") as c:
        assert c.get("/health").status_code == 200  # health is open
        assert c.post("/lipsync", files=_files(tmp_path, _mp4_bytes(tmp_path))).status_code == 401
        ok = c.post("/lipsync", files=_files(tmp_path, _mp4_bytes(tmp_path)),
                    headers={"X-API-Key": "secret"})
        assert ok.status_code == 200
        assert c.post("/jobs", files=_files(tmp_path, _mp4_bytes(tmp_path))).status_code == 401


# ----------------------------------------------------------------------------- jobs
def test_job_flow(client, tmp_path):
    mp4 = _mp4_bytes(tmp_path)
    r = client.post("/jobs", files=_files(tmp_path, mp4), data={"steps": "6"})
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    s = _wait_status(client, jid)
    assert s["status"] == "done" and s["inference_seconds"] is not None
    video = client.get(f"/jobs/{jid}/result")
    assert video.status_code == 200 and video.headers["content-type"] == "video/mp4"
    assert len(video.content) == len(mp4)
    assert client.get(f"/jobs/{jid}/result").status_code == 410  # delivered once
    assert client.get("/jobs/doesnotexist").status_code == 404
    assert client.get("/jobs/doesnotexist/result").status_code == 404


def test_job_client_error(tmp_path):
    def no_face(pipe, video, audio, out, **kw):
        raise ValueError("no face found in the video")

    with make_client(no_face, tmp_path) as c:
        r = c.post("/jobs", files=_files(tmp_path, _mp4_bytes(tmp_path)))
        jid = r.json()["job_id"]
        s = _wait_status(client=c, jid=jid, want=("error",))
        assert s["status"] == "error" and "no face" in s["error"]
        assert c.get(f"/jobs/{jid}/result").status_code == 400  # client error, not 500


def test_bad_requests_to_jobs(client, tmp_path):
    assert client.post("/jobs", files=_files(tmp_path, _mp4_bytes(tmp_path)),
                       data={"bbox": "9,9,1,1"}).status_code == 400
    assert client.post("/jobs", files={"video": ("v.mp4", b"x")}).status_code == 400


def test_queue_full(tmp_path):
    gate = threading.Event()

    def slow(pipe, video, audio, out, **kw):
        gate.wait(5)
        shutil.copyfile(video, out)
        return out

    with make_client(slow, tmp_path, max_queue=1) as c:
        mp4 = _mp4_bytes(tmp_path)
        c.post("/jobs", files=_files(tmp_path, mp4))  # the worker picks this up and blocks on the gate
        for _ in range(200):
            if c.get("/health").json()["jobs"]["running"] == 1:
                break
            time.sleep(0.05)
        c.post("/jobs", files=_files(tmp_path, mp4))  # queued (max_queue 1)
        assert c.post("/jobs", files=_files(tmp_path, mp4)).status_code == 429
        assert c.post("/lipsync", files=_files(tmp_path, mp4)).status_code == 429  # sync path too
        gate.set()


def test_lipsync_timeout_returns_202(tmp_path):
    gate = threading.Event()

    def slow(pipe, video, audio, out, **kw):
        gate.wait(5)
        shutil.copyfile(video, out)
        return out

    with make_client(slow, tmp_path) as c:
        r = c.post("/lipsync", files=_files(tmp_path, _mp4_bytes(tmp_path)), data={"timeout": "0.05"})
        assert r.status_code == 202, r.text
        jid = r.json()["job_id"]
        assert r.json()["poll"] == f"/jobs/{jid}"
        gate.set()
        s = _wait_status(c, jid)
        assert s["status"] == "done"
        assert c.get(f"/jobs/{jid}/result").status_code == 200


# ----------------------------------------------------------------------------- ckpt handling
def test_per_request_ckpt(tmp_path):
    with make_client(_stub([]), tmp_path, ckpt=None, ckpt_root=str(tmp_path / "root")) as c:
        (tmp_path / "root" / "step_0001").mkdir(parents=True)
        (tmp_path / "root" / "step_0001" / "wilrosync_config.json").write_text("{}")
        (tmp_path / "root" / "outside").mkdir()
        mp4 = _mp4_bytes(tmp_path)
        assert c.post("/lipsync", files=_files(tmp_path, mp4)).status_code == 400  # no default ckpt
        assert c.post("/lipsync", files=_files(tmp_path, mp4), data={"ckpt": "../outside"}).status_code == 400
        ok = c.post("/lipsync", files=_files(tmp_path, mp4), data={"ckpt": "step_0001"})
        assert ok.status_code == 200, ok.text


def test_resolve_ckpt_picks_newest(tmp_path):
    from wilrosync.server import InferenceService

    (tmp_path / "runs" / "step_0002").mkdir(parents=True)
    (tmp_path / "runs" / "step_0010").mkdir(parents=True)
    (tmp_path / "runs" / "step_0009").mkdir(parents=True)
    assert InferenceService(ServerConfig(ckpt=str(tmp_path / "runs"))).resolve_ckpt(None).endswith("step_0010")
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "wilrosync_config.json").write_text("{}")
    s = InferenceService(ServerConfig(ckpt=str(tmp_path / "runs"), ckpt_root=str(tmp_path)))
    assert s.resolve_ckpt("real") == str(tmp_path / "real")
    with pytest.raises(ValueError):
        s.resolve_ckpt("../elsewhere")
    with pytest.raises(ValueError):
        InferenceService(ServerConfig()).resolve_ckpt(None)


def test_health_and_info(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and "model_loaded" in h and "jobs" in h
    info = client.get("/").json()
    assert info["defaults"]["steps"] == 10 and "bbox" in info["fields"]
