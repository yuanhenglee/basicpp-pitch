"""Identify the floating-point reduction structure oneDNN uses for a TensorFlow Conv2D.

Probe: unit weights on output channel 0, all inputs 1.0 except one tap set to 2^24. The output
(minus 2^24) depends on where that tap sits in the reduction tree. The measured pattern over all
taps is compared with float32 simulations of candidate structures:

  chain:<order>          one FMA chain from 0 over the taps in the given loop order
  split:<outer>|<inner>  one chain per value of <outer> over <inner>, then partials added in order

Usage: python tools/debug/conv_structure.py H W C_in C_out KH KW stride_w row col [col ...]
"""
import itertools
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf

H, Wd, C, O, KH, KW, SW, R = map(int, sys.argv[1:9])
cols = [int(c) for c in sys.argv[9:]]
w = np.zeros((KH, KW, C, O), np.float32)
w[..., 0] = 1.0
wv, bv = tf.Variable(w), tf.Variable(np.zeros(O, np.float32))
conv = tf.function(lambda x: tf.nn.bias_add(tf.nn.conv2d(x, wv, strides=[1, 1, SW, 1], padding="SAME"), bv))
n_out = -(-Wd // SW)
pad_w = max((n_out - 1) * SW + KW - Wd, 0)
pl, ph = pad_w // 2, (KH - 1) // 2
BIG = np.float32(2.0**24)
dims = {"h": KH, "w": KW, "c": C}


def measure(col):
    taps = {}
    for kh, kw, ic in itertools.product(range(KH), range(KW), range(C)):
        t, f = R + kh - ph, col * SW + kw - pl
        if 0 <= t < H and 0 <= f < Wd:
            taps[(kh, kw, ic)] = (t, f)
    # batch all probes of this column into one call per input channel to keep it fast
    meas = {}
    for (kh, kw, ic), (t, f) in taps.items():
        x = np.ones((1, H, Wd, C), np.float32)
        x[0, t, f, ic] = BIG
        meas[(kh, kw, ic)] = float(conv(x).numpy()[0, R, col, 0]) - float(BIG)
    return meas


def simulate(struct, valid):
    """Expected probe value for each valid tap under a structure."""
    kind, spec = struct.split(":")
    out = {}
    for probe in valid:
        val = lambda k: BIG if k == probe else np.float32(1.0)
        if kind == "chain":
            order = [k for k in loop(spec) if k in valid]
            acc = np.float32(0)
            for k in order:
                acc = np.float32(acc + val(k))
        else:
            outer, inner = spec.split("|")
            acc = np.float32(0)
            for o in range(dims[outer]):
                part = np.float32(0)
                for k in loop(inner, fixed={outer: o}):
                    if k in valid:
                        part = np.float32(part + val(k))
                acc = np.float32(acc + part)
        out[probe] = float(acc) - float(BIG)
    return out


def loop(spec, fixed=None):
    fixed = fixed or {}
    names = [d for d in spec if d not in fixed]
    for combo in itertools.product(*[range(dims[d]) for d in names]):
        v = dict(zip(names, combo), **fixed)
        yield (v["h"], v["w"], v["c"])


candidates = [f"chain:{''.join(p)}" for p in itertools.permutations("hwc")]
for outer in "hwc":
    rest = [d for d in "hwc" if d != outer]
    for p in itertools.permutations(rest):
        candidates.append(f"split:{outer}|{''.join(p)}")

for col in cols:
    meas = measure(col)
    valid = set(meas)
    hits = [c for c in candidates if simulate(c, valid) == meas]
    print(f"col {col}: {len(valid)} taps -> matching structures: {hits or 'NONE'}")
    if not hits:
        sample = sorted(meas.items())[:12]
        print("   sample measurements:", sample)
