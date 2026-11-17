"""Run the C++ CLI on every golden case and compare with the Python library's output.

Usage: python tools/verify.py [--bin build/bpp.exe] [--golden golden/tf] [--out out/verify]
"""
import argparse
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from compare import compare  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build/bpp.exe")
    ap.add_argument("--golden", default="golden/tf")
    ap.add_argument("--out", default="out/verify")
    ap.add_argument("--threads", default="0")
    ap.add_argument("--stream-chunk-ms", default=None, help="verify the streaming API with this push size")
    ap.add_argument("--source", action="store_true",
                    help="feed the original audio file (WAV sources only) instead of the decoded 22050 Hz copy")
    args = ap.parse_args()

    cases = sorted(p for p in pathlib.Path(args.golden).iterdir() if (p / "notes.json").exists())
    failed, checked = [], 0
    import json

    for case in cases:
        audio = case / "audio.wav"
        if args.source:
            src = pathlib.Path(json.loads((case / "meta.json").read_text()).get("source", ""))
            if src.suffix.lower() != ".wav" or not src.exists():
                continue
            audio = src
        out = pathlib.Path(args.out) / case.name
        out.mkdir(parents=True, exist_ok=True)
        cmd = [str(pathlib.Path(args.bin).resolve())]
        if args.stream_chunk_ms:
            cmd += ["stream", str(audio), "--chunk-ms", args.stream_chunk_ms]
        else:
            cmd += ["transcribe", str(audio)]
        cmd += ["--dump", str(out), "--threads", args.threads]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[FAIL] {case.name}: {r.stderr.strip()}")
            failed.append(case.name)
            continue
        checked += 1
        print(f"== {case.name}: {r.stdout.strip()}")
        ok, _ = compare(case, out)
        if not ok:
            failed.append(case.name)
    print(f"\n{checked - len(failed)}/{checked} cases bit-true" + (f"; mismatches: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
