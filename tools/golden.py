"""Generate golden reference outputs with the official basic-pitch Python library.

For an input audio file this writes, into <out_dir>/<stem>/:
  audio.wav     22050 Hz mono float32 audio exactly as basic-pitch's loader sees it
  model.npz     unwrapped model outputs: note (T, 88), onset (T, 88), contour (T, 264)
  notes.json    note events (start_s, end_s, pitch, amplitude, bends), floats as exact repr
  golden.mid    MIDI file written by pretty_midi, byte-for-byte the library's output

Usage: python tools/golden.py <audio>... --out <out_dir> [--backend tf|onnx|tflite]
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import soundfile as sf


def backend_model_path(backend):
    from basic_pitch import ICASSP_2022_MODEL_PATH

    base = pathlib.Path(ICASSP_2022_MODEL_PATH).parent
    return {
        "tf": base / "nmp",
        "onnx": base / "nmp.onnx",
        "tflite": base / "nmp.tflite",
    }[backend]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="tf", choices=["tf", "onnx", "tflite"])
    args = ap.parse_args()

    import librosa
    from basic_pitch.inference import Model

    model = Model(backend_model_path(args.backend))
    for path in args.audio:
        out = pathlib.Path(args.out) / pathlib.Path(path).stem
        try:
            run_one(path, out, model, args.backend, librosa)
        except Exception as e:  # e.g. basic-pitch cannot handle audio shorter than one frame
            (out / "notes.json").unlink(missing_ok=True)
            with open(out / "meta.json", "w") as f:
                json.dump({"source": str(path), "backend": args.backend, "error": repr(e)}, f)
            print(out.name, "reference error:", repr(e), flush=True)


def run_one(audio_path, out, model, backend, librosa):
    from basic_pitch.inference import predict

    out.mkdir(parents=True, exist_ok=True)
    # Re-encode the decoded audio losslessly so the C++ side consumes the exact same samples.
    audio, _ = librosa.load(audio_path, sr=22050, mono=True)
    wav_path = out / "audio.wav"
    sf.write(wav_path, audio, 22050, subtype="FLOAT")

    t0 = time.perf_counter()
    model_output, midi, notes = predict(str(wav_path), model)
    elapsed = time.perf_counter() - t0

    np.savez(out / "model.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in model_output.items()})
    with open(out / "notes.json", "w") as f:
        json.dump(
            [
                [float(s), float(e), int(p), float(a), [int(b) for b in bends] if bends else None]
                for s, e, p, a, bends in notes
            ],
            f,
        )
    midi.write(str(out / "golden.mid"))
    meta = {"source": str(audio_path), "backend": backend, "seconds": len(audio) / 22050, "predict_s": elapsed, "n_notes": len(notes)}
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f)
    print(out.name, json.dumps(meta), flush=True)


if __name__ == "__main__":
    sys.exit(main())
