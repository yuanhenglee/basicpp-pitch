"""Raw PCM input (any rate / channel count) must equal decoding the same samples from a WAV file."""
import pathlib
import struct
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import basicpp_pitch as bpp  # noqa: E402


def write_wav16(path, pcm16: np.ndarray, rate: int):
    """pcm16: int16 array of shape (frames, channels)."""
    channels = pcm16.shape[1]
    data = pcm16.astype("<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, channels, rate, rate * channels * 2, channels * 2, 16)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def same(a: dict, b: dict) -> bool:
    return all(np.array_equal(a[k], b[k]) for k in ("note", "onset", "contour"))


@pytest.fixture(scope="module")
def engine():
    return bpp.Engine(0)


@pytest.mark.parametrize("rate,channels,seconds", [(48000, 4, 3.0), (8000, 1, 2.5), (22050, 2, 2.0), (96000, 1, 1.7)])
def test_pcm_equals_wav(tmp_path, engine, rate, channels, seconds):
    rng = np.random.default_rng(rate + channels)
    t = np.arange(int(rate * seconds)) / rate
    tone = 0.4 * np.sin(2 * np.pi * 440 * t)[:, None] + 0.05 * rng.standard_normal((t.size, channels))
    pcm16 = np.clip(np.round(tone * 32767), -32768, 32767).astype(np.int16)
    path = tmp_path / "x.wav"
    write_wav16(path, pcm16, rate)
    audio = pcm16.astype(np.float32) / 32768  # libsndfile's scaling
    if channels == 1:
        audio = audio[:, 0]

    ref = engine.infer_file(path)
    assert same(engine.infer_pcm(audio, rate), ref)

    for chunk in (1000, 7919):
        stream = bpp.Stream(2, sample_rate=rate, channels=channels)
        for pos in range(0, len(audio), chunk):
            stream.push(audio[pos:pos + chunk])
        assert same(stream.finish(), ref)

    expected = bpp.output_to_notes(ref).midi_bytes
    assert bpp.predict(audio, sample_rate=rate).midi_bytes == expected


@pytest.mark.parametrize("rate,channels", [(0, 1), (999, 1), (1_000_001, 1), (44100, 0), (44100, -2)])
def test_invalid_format_raises(engine, rate, channels):
    with pytest.raises(RuntimeError):
        bpp.Stream(1, sample_rate=rate, channels=channels)
    if channels >= 1:
        with pytest.raises(RuntimeError):
            engine.infer_pcm(np.zeros((100, channels), np.float32), rate)


def test_channel_mismatch_raises():
    stream = bpp.Stream(1, sample_rate=44100, channels=2)
    with pytest.raises(ValueError):
        stream.push(np.zeros((100, 3), np.float32))
