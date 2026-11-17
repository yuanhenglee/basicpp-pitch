"""Malformed input and API misuse must raise errors, never crash."""
import pathlib
import struct
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import basicpp_pitch as bpp  # noqa: E402


def wav_bytes(channels=1, rate=22050, bits=16, block_align=None, fmt=1, data=b"\0" * 4000):
    block_align = channels * bits // 8 if block_align is None else block_align
    fmt_chunk = struct.pack("<HHIIHH", fmt, channels, rate, rate * block_align, block_align, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


@pytest.mark.parametrize("kwargs", [
    dict(channels=65535, bits=32, block_align=1),   # block align smaller than one frame
    dict(channels=64, bits=16, block_align=2),
    dict(rate=0),                                    # invalid sample rates
    dict(rate=1),
    dict(bits=12),                                   # unsupported encodings
    dict(fmt=2),
    dict(channels=0),
])
def test_malformed_wav_raises(tmp_path, kwargs):
    path = tmp_path / "bad.wav"
    path.write_bytes(wav_bytes(**kwargs))
    with pytest.raises(RuntimeError):
        bpp.load_audio(path)
    with pytest.raises(RuntimeError):
        bpp.predict(path)


def test_not_a_wav(tmp_path):
    (tmp_path / "x.wav").write_bytes(b"not audio at all")
    with pytest.raises(RuntimeError):
        bpp.load_audio(tmp_path / "x.wav")
    with pytest.raises(RuntimeError):
        bpp.load_audio(tmp_path / "missing.wav")


def test_truncated_data_chunk(tmp_path):
    raw = wav_bytes(data=b"\x01\x00" * 30000)
    (tmp_path / "t.wav").write_bytes(raw[: len(raw) // 2])  # header claims more data than present
    assert len(bpp.load_audio(tmp_path / "t.wav")) > 0


def test_push_after_finish_raises():
    s = bpp.Stream()
    s.push(np.zeros(1000, np.float32))
    s.finish()
    with pytest.raises(RuntimeError):
        s.push(np.zeros(10, np.float32))


def test_output_to_notes_shape_checked():
    out = {"note": np.zeros((10, 88), np.float32), "onset": np.zeros((5, 88), np.float32),
           "contour": np.zeros((10, 264), np.float32)}
    with pytest.raises(ValueError):
        bpp.output_to_notes(out)


def test_invalid_midi_tempo():
    out = bpp.Engine(1).infer(np.zeros(50000, np.float32))
    with pytest.raises(RuntimeError):
        bpp.output_to_notes(out, bpp.NoteOptions(midi_tempo=0))
