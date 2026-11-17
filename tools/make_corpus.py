"""Build the verification corpus: synthetic edge cases plus librosa's example recordings.

Usage: python tools/make_corpus.py <out_dir> [--no-download]
"""
import argparse
import pathlib

import numpy as np
import soundfile as sf

SR = 22050


def tone(freq, dur, sr=SR, harmonics=6, decay=1.5, vibrato=0.0, vib_rate=5.5):
    t = np.arange(int(dur * sr)) / sr
    phase_mod = vibrato * np.sin(2 * np.pi * vib_rate * t) / vib_rate if vibrato else 0.0
    y = np.zeros_like(t)
    for h in range(1, harmonics + 1):
        f = freq * h
        if f > sr / 2 - 500:
            break
        y += np.sin(2 * np.pi * f * t + h * freq * phase_mod * 2 * np.pi / freq) / h**1.2
    return y * np.exp(-decay * t)


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def piece(rng, dur, poly, note_len=(0.2, 1.5), pitch=(36, 96)):
    y = np.zeros(int(dur * SR))
    t = 0.0
    while t < dur - 0.3:
        for _ in range(rng.integers(1, poly + 1)):
            n = tone(midi_hz(rng.integers(*pitch)), rng.uniform(*note_len), decay=rng.uniform(0.5, 3))
            s = int(t * SR)
            e = min(len(y), s + len(n))
            y[s:e] += n[: e - s] * rng.uniform(0.1, 0.4)
        t += rng.uniform(0.08, 0.6)
    return y


def synthetic(rng):
    cases = {}
    cases["silence_5s"] = np.zeros(5 * SR)
    cases["tiny_100"] = rng.standard_normal(100) * 0.1
    cases["one_window_exact"] = piece(rng, 40004 / SR, 3)[:40004]
    cases["hop_boundary_32324"] = piece(rng, 2, 2)[:32324]
    cases["hop_boundary_32325"] = piece(rng, 2, 2)[:32325]
    t = np.arange(20 * SR) / SR
    f0, f1 = 27.5, 4186.0
    cases["log_sweep_20s"] = 0.5 * np.sin(2 * np.pi * f0 * 20 / np.log(f1 / f0) * (np.exp(t / 20 * np.log(f1 / f0)) - 1))
    chords = piece(rng, 30, 4)
    cases["chords_30s"] = chords
    cases["chords_quiet_1e-4"] = chords * 1e-4
    cases["chords_clipped"] = np.clip(chords * 4, -1, 1)
    cases["white_noise_10s"] = rng.standard_normal(10 * SR) * 0.1
    clicks = np.zeros(8 * SR)
    clicks[:: SR // 3] = 0.9
    cases["clicks_8s"] = clicks
    vib = np.concatenate([tone(midi_hz(m), 2.0, decay=0.2, vibrato=0.03) * 0.4 for m in (45, 57, 64, 69, 76, 81)])
    cases["vibrato_12s"] = vib
    cases["dense_poly_45s"] = piece(rng, 45, 8, note_len=(0.1, 3.0), pitch=(21, 108))
    sustained = sum(tone(midi_hz(m), 100, decay=0.0, harmonics=4) for m in (48, 55, 64)) * 0.15
    cases["sustained_100s"] = sustained
    cases["bass_extremes"] = np.concatenate([tone(midi_hz(m), 1.5, decay=1) * 0.5 for m in (21, 22, 23, 105, 106, 107, 108)])
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20261116)
    for name, y in synthetic(rng).items():
        sf.write(out / f"{name}.wav", np.asarray(y, dtype=np.float32), SR, subtype="FLOAT")
        print("wrote", name, f"{len(y) / SR:.2f}s")

    if not args.no_download:
        import librosa

            pass
        for key in ["brahms", "choice", "fishin", "humpback", "libri1", "nutcracker", "pistachio", "robin",
                    "sweetwaltz", "trumpet", "vibeace"]:
            try:
                src = librosa.ex(key)
                y, sr = sf.read(src, dtype="float32")
                dst = out / f"ex_{key}{pathlib.Path(src).suffix}"
                dst.write_bytes(pathlib.Path(src).read_bytes())
                print("fetched", key, sr, y.shape)
            except Exception as e:  # network or format issues are not fatal
                print("skip", key, e)


if __name__ == "__main__":
    main()
