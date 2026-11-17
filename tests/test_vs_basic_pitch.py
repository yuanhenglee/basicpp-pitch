"""Bit-exactness tests of the Python binding against the official basic-pitch library.

Requires the golden corpus produced by tools/golden.py (golden/tf/<case>/) and an environment with
basic-pitch 0.4.0 + TensorFlow 2.15 for the note-option tests. Run:  python -m pytest tests
"""
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import basicpp_pitch as bpp  # noqa: E402

GOLDEN = ROOT / "golden" / "tf"
CASES = sorted(p.name for p in GOLDEN.glob("*") if (p / "notes.json").exists()) if GOLDEN.exists() else []
needs_golden = pytest.mark.skipif(not CASES, reason="no golden corpus; run tools/golden.py")


def golden_notes(case):
    return json.loads((GOLDEN / case / "notes.json").read_text())


def as_json_notes(events):
    return json.loads(json.dumps([[s, e, p, a, b] for s, e, p, a, b in events]))


@needs_golden
@pytest.mark.parametrize("case", CASES)
def test_predict_matches_golden(case):
    result = bpp.predict(GOLDEN / case / "audio.wav")
    ref = np.load(GOLDEN / case / "model.npz")
    for k in ("note", "onset", "contour"):
        assert np.array_equal(result.model_output[k], ref[k]), k
    assert as_json_notes(result.note_events) == golden_notes(case)
    assert result.midi_bytes == (GOLDEN / case / "golden.mid").read_bytes()


VARIANTS = [
    dict(onset_threshold=0.3, frame_threshold=0.2),
    dict(onset_threshold=0.7, frame_threshold=0.45, minimum_note_length=58.0),
    dict(minimum_frequency=110.0, maximum_frequency=1000.0),
    dict(minimum_frequency=30.0, maximum_frequency=8000.0),
    dict(melodia_trick=False),
    dict(multiple_pitch_bends=True),
    dict(midi_tempo=97.5),
]


@needs_golden
@pytest.mark.parametrize("variant", VARIANTS, ids=[",".join(v) for v in VARIANTS])
@pytest.mark.parametrize("case", [c for c in ("dense_poly_45s", "ex_pistachio", "vibrato_12s", "bass_extremes") if c in CASES])
def test_note_options_match_basic_pitch(case, variant):
    pytest.importorskip("basic_pitch")
    import io

    import basic_pitch.note_creation as infer
    from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP

    ref = dict(np.load(GOLDEN / case / "model.npz"))
    opt = bpp.NoteOptions(**variant)
    min_note_len = int(np.round(opt.minimum_note_length / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    midi, events = infer.model_output_to_notes(
        {k: v.copy() for k, v in ref.items()}, onset_thresh=opt.onset_threshold, frame_thresh=opt.frame_threshold,
        min_note_len=min_note_len, min_freq=opt.minimum_frequency, max_freq=opt.maximum_frequency,
        multiple_pitch_bends=opt.multiple_pitch_bends, melodia_trick=opt.melodia_trick, midi_tempo=opt.midi_tempo)
    buf = io.BytesIO()
    midi.write(buf)
    got = bpp.output_to_notes(ref, opt)
    expected = [[float(s), float(e), int(p), float(a), [int(x) for x in b] if b else None] for s, e, p, a, b in events]
    assert as_json_notes(got.note_events) == expected
    assert got.midi_bytes == buf.getvalue()


@needs_golden
@pytest.mark.parametrize("chunk", [1, 441, 22050, 100000])
def test_stream_equals_offline(chunk):
    audio = bpp.load_audio(GOLDEN / "chords_30s" / "audio.wav")
    offline = bpp.Engine(0).infer(audio)
    stream = bpp.Stream(2 if chunk > 1000 else 1)
    limit = 200000 if chunk == 1 else len(audio)  # sample-by-sample pushing is slow in Python
    parts = [stream.push(audio[i:i + chunk]) for i in range(0, limit, chunk)]
    if chunk == 1:
        parts.append(stream.push(audio[limit:]))
    final = stream.finish()
    n = offline["note"].shape[0]
    for k in ("note", "onset", "contour"):
        assert np.array_equal(final[k], offline[k]), k
        pushed = np.concatenate([p[k] for p in parts])
        m = min(len(pushed), n)
        assert np.array_equal(pushed[:m], offline[k][:m]), k  # frames are final values as soon as they appear


def test_empty_and_tiny_audio():
    assert bpp.predict(np.zeros(0, np.float32)).note_events == []
    assert bpp.predict(np.zeros(100, np.float32)).note_events == []
