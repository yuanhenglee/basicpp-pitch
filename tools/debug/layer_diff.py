"""Layer-by-layer comparison of one analysis window: TensorFlow vs the C++ engine.

The TensorFlow side is basic-pitch's own Keras model loaded with the SavedModel's variables
(checked to reproduce the SavedModel bit for bit). Parameters stay tf.Variables so Grappler cannot
constant-fold the batch norms, which would select different kernels than the SavedModel uses.
The C++ side is tools/debug/window_dump.cpp, which writes every intermediate tensor.

Usage: python tools/debug/layer_diff.py <22050 Hz audio.wav> <window_index> <build/window_dump.exe>
"""
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import soundfile as sf
import tensorflow as tf
from basic_pitch import ICASSP_2022_MODEL_PATH, models


def window_samples(path, w):
    audio, sr = sf.read(path, dtype="float32")
    assert sr == 22050 and audio.ndim == 1, "expects the decoded 22050 Hz mono audio.wav from tools/golden.py"
    padded = np.concatenate([np.zeros(3840, np.float32), audio])
    win = padded[w * 36164: w * 36164 + 43844]
    return np.pad(win, (0, 43844 - len(win))).astype(np.float32)


def run_cpp(exe, win):
    tmp = tempfile.mkdtemp()
    win.tofile(os.path.join(tmp, "win.f32"))
    subprocess.run([os.path.abspath(exe), os.path.join(tmp, "win.f32"), tmp], check=True)
    return {f[:-4]: np.load(os.path.join(tmp, f)) for f in os.listdir(tmp) if f.endswith(".npy")}


def probe_model():
    saved = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
    sv = {v.name[:-2]: v.numpy() for v in saved.variables}
    kernel_by_shape = {sv[k].shape: k[: -len("/kernel")] for k in sv if k.endswith("/kernel")}
    bn_by_ch = {sv[k].shape[0]: k[: -len("/gamma")] for k in sv if k.endswith("/gamma")}
    model = models.model()
    taps = {}
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            src = kernel_by_shape[tuple(layer.kernel.shape)]
            layer.set_weights([sv[src + "/kernel"], sv[src + "/bias"]])
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            src = bn_by_ch[layer.gamma.shape[0]]
            layer.set_weights([sv[src + "/" + k] for k in ("gamma", "beta", "moving_mean", "moving_variance")])
        if any(s in layer.name for s in ("cqt", "batch_normalization", "harmonic", "re_lu", "conv2d", "contours")):
            taps[layer.name] = layer.output
    for k, v in model.output.items():
        taps["out_" + k] = v
    return saved, tf.keras.Model(model.input, taps)


def cmp(name, a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    d = np.abs(a.astype(np.float64) - b).max()
    print(f"{name:22s} bitexact {np.mean(a == b) * 100:8.3f}%   max|d| {d:.3g}")


def main(path, w, exe):
    win = window_samples(path, w)
    cpp = run_cpp(exe, win)
    saved, probe = probe_model()
    tfo = {k: v.numpy() for k, v in tf.function(lambda x: probe(x))(win[None, :, None]).items()}

    ref = saved.signatures["serving_default"](input_2=tf.constant(win[None, :, None]))
    for k in ("note", "onset", "contour"):
        assert np.array_equal(ref[k].numpy(), tfo["out_" + k]), "probe model diverges from the SavedModel"

    # CQT magnitude, from the C++ raw conv outputs scaled the same way TF does
    lengths = np.ceil((1.0 / (2 ** (1 / 36) - 1)) * 22050 / (27.5 * 2.0 ** (np.arange(309) / 36)))
    s = np.sqrt(lengths.astype(np.float32))
    re, im = (cpp["cqt_re"] * s).astype(np.float32), (cpp["cqt_im"] * s).astype(np.float32)
    cmp("cqt magnitude", tfo["cqt2010v2"][0], np.sqrt((re * re).astype(np.float32) + (im * im).astype(np.float32)))
    cmp("cqt batch norm", tfo["batch_normalization"][0, :, :, 0], cpp["bn"])
    cmp("harmonic stack", tfo["harmonic_stacking"][0][5:167], cpp["hs"][5:167, 19:283])
    cmp("conv2d_1+bn+relu", tfo["re_lu_1"][0].transpose(2, 0, 1)[:, 6:166], cpp["c1"][:, 6:166, 2:266])
    cmp("contour (Yp)", tfo["out_contour"][0][8:164], cpp["yp"][8:164, 2:266])
    cmp("conv2d_2+relu", tfo["re_lu_2"][0].transpose(2, 0, 1)[:, 11:161], cpp["n1"][:, 11:161, 1:89])
    cmp("note (Yn)", tfo["out_note"][0][14:158], cpp["cat"][0, 14:158, 1:89])
    cmp("conv2d_4+bn+relu", tfo["re_lu_3"][0].transpose(2, 0, 1)[:, 14:158], cpp["cat"][1:33, 14:158, 1:89])
    cmp("onset (Yo)", tfo["out_onset"][0][15:157], cpp["onset"])


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
