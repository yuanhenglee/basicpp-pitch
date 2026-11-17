"""Create a small self-contained regression fixture for the C++ test (tests/test_fixture.cpp).

Writes tests/fixtures/<name>.wav (44.1 kHz stereo PCM16, exercises decoding + resampling) together
with basic-pitch's golden outputs: <name>.mid and <name>.notes (one note per line, hex floats).

Usage: python tools/make_fixture.py <source.wav> <start_s> <duration_s> <name>
"""
import os
import pathlib
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import soundfile as sf


def main(src, start, dur, name):
    out = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    y, sr = sf.read(src, dtype="int16", always_2d=True)
    clip = y[int(float(start) * sr): int((float(start) + float(dur)) * sr)]
    wav = out / f"{name}.wav"
    sf.write(wav, clip, sr, subtype="PCM_16")

    from basic_pitch.inference import predict

    _, midi, notes = predict(str(wav))
    midi.write(str(out / f"{name}.mid"))
    with open(out / f"{name}.notes", "w", newline="\n") as f:
        for s, e, p, a, bends in notes:
            fields = [float(s).hex(), float(e).hex(), str(int(p)), float(np.float32(a)).hex()]
            fields += [str(int(b)) for b in bends] if bends else []
            f.write(" ".join(fields) + "\n")
    print(f"{name}: {len(notes)} notes, {len(clip) / sr:.2f} s")


if __name__ == "__main__":
    main(*sys.argv[1:5])
