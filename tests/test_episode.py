import os
import shutil

import numpy as np
import pytest

from wilrosync.io.media import read_audio, read_video, write_video, write_wav
from wilrosync.pipeline.episode import Character, lipsync_episode, match_faces, parse_character


def test_parse_character():
    c = parse_character("alice=faces/alice.jpg:voices/alice.wav")
    assert (c.name, c.face, c.audio) == ("alice", "faces/alice.jpg", "voices/alice.wav")
    with pytest.raises(ValueError):
        parse_character("alice.jpg:alice.wav")


def test_match_faces_one_to_one_threshold_and_size():
    e = np.eye(3)
    faces = [((0, 0, 100, 100), e[0]), ((200, 0, 300, 100), 0.9 * e[0] + 0.1 * e[1]),
             ((400, 0, 430, 30), e[2])]  # the third face is too small
    refs = {"alice": e[0], "bob": e[1], "carol": e[2]}
    m = match_faces(faces, refs, threshold=0.05, min_face=50)
    assert m["alice"][0] == (0, 0, 100, 100)  # best match wins; the second face can't also be alice
    assert m["bob"][0] == (200, 0, 300, 100) and "carol" not in m


class _Identifier:
    threshold = 0.5

    def __init__(self, per_shot):
        self.per_shot = list(per_shot)

    def reference(self, path):
        return {"alice": np.eye(3)[0], "bob": np.eye(3)[1], "carol": np.eye(3)[2]}[os.path.basename(path)[:-4]]

    def faces(self, image):
        return self.per_shot.pop(0)


def test_episode_syncs_who_speaks_per_shot(tmp_path):
    n = 50  # 2 s at 25 fps; shots [0, 20) [20, 25) [25, 50)
    video = str(tmp_path / "ep.mp4")
    write_video(video, np.full((n, 64, 96, 3), 120, np.uint8), 25.0)
    tone = (0.3 * np.sin(np.arange(n * 640) / 5.0)).astype(np.float32)
    voices = {"alice": np.where(np.arange(n * 640) < 20 * 640, tone, 0),   # speaks in shot 0
              "bob": np.zeros(n * 640),                                    # never
              "carol": np.where(np.arange(n * 640) >= 25 * 640, tone, 0)}  # speaks in shot 2
    chars = []
    for name, wav in voices.items():
        write_wav(str(tmp_path / f"{name}.wav"), wav.astype(np.float32))
        chars.append(Character(name, str(tmp_path / f"{name}.jpg"), str(tmp_path / f"{name}.wav")))
    e = np.eye(3)
    faces = [[((5, 5, 45, 45), e[0]), ((50, 5, 90, 45), e[1])],   # shot 0: alice + bob
             [((5, 5, 45, 45), e[2])]]                             # shot 2: carol (shot 1 is too short to look at)
    calls = []

    def fake_sync(v, a, o, bbox):
        calls.append((os.path.basename(a), bbox, len(read_audio(a))))
        shutil.copy(v, o)

    for silent, expected in (("keep", ["shot0000_alice.wav", "shot0002_carol.wav"]),
                             ("close", ["shot0000_alice.wav", "shot0000_bob.wav", "shot0002_carol.wav"])):
        calls.clear()
        out = str(tmp_path / f"out_{silent}.mp4")
        rep = lipsync_episode(None, video, chars, out, silent=silent, min_face=10, identifier=_Identifier(faces),
                              shot_detector=lambda v, k: [(0, 20), (20, 25), (25, k)], sync_fn=fake_sync,
                              report=str(tmp_path / "report.json"))
        assert sorted(c[0] for c in calls) == expected
        assert all(c[2] == (20 if c[0].startswith("shot0000") else 25) * 640 for c in calls)  # shot-length audio
        assert rep[1].note and rep[0].matched["bob"]["action"] == ("close" if silent == "close" else "keep")
        frames, _ = read_video(out)
        assert frames.shape[0] == n  # every shot back in place
        assert len(read_audio(out)) >= (n - 1) * 640  # with the mixed soundtrack
    assert os.path.isfile(tmp_path / "report.json")
