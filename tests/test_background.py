import numpy as np

from wilrosync.io.media import read_video, write_video, write_wav
from wilrosync.pipeline.background import (
    background_frames,
    composite_background,
    parse_color,
    refine_alpha,
    replace_background,
)


class _CentreSegmenter:  # "person" = the centre square
    def __call__(self, frames):
        t, h, w, _ = frames.shape
        p = np.zeros((t, h, w), np.float32)
        p[:, h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 1.0
        return p


def _video(tmp_path, t=10, h=64, w=96):
    frames = np.full((t, h, w, 3), 200, np.uint8)
    path, wav = str(tmp_path / "in.mp4"), str(tmp_path / "a.wav")
    write_wav(wav, np.zeros(int(16000 * t / 25), np.float32))
    write_video(path, frames, 25.0, audio_path=wav)
    return path


def test_colour_and_refine():
    assert parse_color("#00B140") == (0, 177, 64) and parse_color("green") == (0, 177, 64)
    assert parse_color("blur") is None
    prob = np.zeros((5, 8, 8), np.float32)
    prob[2] = 1.0  # a one-frame flash
    a = refine_alpha(prob, smooth=3, feather=0)
    assert a[2].max() < 0.5 and a[1].max() > 0  # spread over 3 frames: no full flicker


def test_replace_background_with_colour(tmp_path):
    video = _video(tmp_path)
    out = replace_background(video, str(tmp_path / "out.mp4"), "#0000ff", segmenter=_CentreSegmenter(),
                             smooth=1, feather=0, save_mask=str(tmp_path / "mask.mp4"))
    f, _ = read_video(out)
    assert f.shape == (10, 64, 96, 3)
    assert abs(int(f[5, 32, 48, 0]) - 200) < 8  # person (centre) kept
    assert f[5, 2, 2, 2] > 200 and f[5, 2, 2, 0] < 40  # background replaced by blue
    m, _ = read_video(str(tmp_path / "mask.mp4"))
    assert m[5, 32, 48, 0] > 240 and m[5, 2, 2, 0] < 15


def test_background_image_cover_and_composite(tmp_path):
    import imageio.v3 as iio

    img = np.zeros((40, 40, 3), np.uint8)
    img[:, :20] = 255  # left half white
    path = str(tmp_path / "bg.png")
    iio.imwrite(path, img)
    frames = np.zeros((2, 30, 60, 3), np.uint8)
    bg = background_frames(path, frames, 25.0)
    assert bg.shape == frames.shape and bg[0, 15, 5, 0] > 200 and bg[0, 15, 55, 0] < 50
    alpha = np.zeros((2, 30, 60), np.float32)
    alpha[:, :, 30:] = 1.0
    out = composite_background(frames, alpha, bg)
    assert out[0, 15, 5, 0] > 200 and out[0, 15, 45, 0] == 0
    blurred = background_frames("blur", frames, 25.0)
    assert blurred.shape == frames.shape
