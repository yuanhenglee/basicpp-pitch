"""Compare two result directories (golden.py layout, or C++ output with the same file names).

Usage: python tools/compare.py <ref_dir> <test_dir>
"""
import json
import pathlib
import sys

import numpy as np


def load_model(d):
    if (d / "model.npz").exists():
        return dict(np.load(d / "model.npz"))
    if (d / "note.npy").exists():
        return {k: np.load(d / f"{k}.npy") for k in ("note", "onset", "contour")}
    return None


def midi_path(d):
    for name in ("golden.mid", "out.mid"):
        if (d / name).exists():
            return d / name
    return None


def load_notes(d):
    with open(d / "notes.json") as f:
        return json.load(f)


def compare(ref, test, verbose=True):
    ref, test = pathlib.Path(ref), pathlib.Path(test)
    ok = True
    report = {}
    a, b = load_model(ref), load_model(test)
    if a is not None and b is not None:
        for k in ("contour", "note", "onset"):
            if a[k].shape != b[k].shape:
                report[k] = f"shape {a[k].shape} vs {b[k].shape}"
                ok = False
                continue
            d = np.abs(a[k].astype(np.float64) - b[k].astype(np.float64))
            ulp = np.abs(a[k].view(np.int32).astype(np.int64) - b[k].view(np.int32).astype(np.int64))
            report[k] = f"max|d|={d.max():.3g} mean|d|={d.mean():.3g} bitexact={np.mean(ulp == 0) * 100:.2f}% maxulp={ulp.max()}"
    na, nb = load_notes(ref), load_notes(test)
    # exact equality of every field: times, pitch, amplitude (float32 value), bends
    same_full = na == nb
    key = lambda n: (n[0], n[1], n[2])
    same_events = [key(n) for n in na] == [key(n) for n in nb]
    vel = lambda n: int(np.round(127 * np.float32(n[3])))
    same_vel = [vel(n) for n in na] == [vel(n) for n in nb]
    same_bends = [n[4] for n in na] == [n[4] for n in nb]
    amp_d = max((abs(x[3] - y[3]) for x, y in zip(na, nb)), default=0.0) if same_events else float("nan")
    ma, mb = midi_path(ref), midi_path(test)
    midi_same = ma.read_bytes() == mb.read_bytes() if ma and mb else None
    report["notes"] = (
        f"n={len(na)} vs {len(nb)} events_equal={same_events} velocity_equal={same_vel} "
        f"bends_equal={same_bends} all_fields_bitexact={same_full} max|d_amp|={amp_d:.3g} midi_bytes_equal={midi_same}"
    )
    ok = ok and same_events and same_vel and same_bends and midi_same is not False
    if verbose:
        for k, v in report.items():
            print(f"  {k:8s} {v}")
        if not same_events:
            sa, sb = set(map(key, na)), set(map(key, nb))
            print("  only in ref :", sorted(sa - sb)[:10])
            print("  only in test:", sorted(sb - sa)[:10])
    return ok, report


if __name__ == "__main__":
    ok, _ = compare(sys.argv[1], sys.argv[2])
    print("MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)
