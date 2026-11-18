"""Python bindings for BasiCPP Pitch (ctypes over the C API in basicpp_pitch.dll / .so / .dylib).

Results are bit-identical to ``basic_pitch.inference.predict`` (basic-pitch 0.4.0, TensorFlow
backend): same model activations, note events, pitch bends and MIDI bytes.

    import basicpp_pitch as bpp
    result = bpp.predict("song.wav")
    open("song.mid", "wb").write(result.midi_bytes)
    for start_s, end_s, pitch, amplitude, bends in result.note_events:
        ...
"""
from __future__ import annotations

import ctypes
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

__all__ = ["NoteOptions", "Prediction", "Engine", "Stream", "load_audio", "predict", "output_to_notes", "SAMPLE_RATE"]

SAMPLE_RATE = 22050
_LIB_NAMES = {"win32": "basicpp_pitch.dll", "darwin": "basicpp_pitch.dylib"}


def _find_library() -> ctypes.CDLL:
    name = _LIB_NAMES.get(sys.platform, "basicpp_pitch.so")
    here = pathlib.Path(__file__).resolve().parent
    build = here.parent.parent / "build"
    candidates = [os.environ.get("BASICPP_PITCH_LIB", ""), here / name, build / name]
    candidates += [build / cfg / name for cfg in ("Release", "RelWithDebInfo", "MinSizeRel", "Debug")]
    for c in candidates:
        if c and pathlib.Path(c).exists():
            return ctypes.CDLL(str(c))
    raise OSError(f"cannot find {name}; build the project (cmake --build build) or set BASICPP_PITCH_LIB")


_lib = _find_library()
_f32p = ctypes.POINTER(ctypes.c_float)


class _COptions(ctypes.Structure):
    _fields_ = [
        ("onset_threshold", ctypes.c_double),
        ("frame_threshold", ctypes.c_double),
        ("minimum_note_length_ms", ctypes.c_double),
        ("minimum_frequency", ctypes.c_double),
        ("maximum_frequency", ctypes.c_double),
        ("infer_onsets", ctypes.c_int),
        ("melodia_trick", ctypes.c_int),
        ("include_pitch_bends", ctypes.c_int),
        ("multiple_pitch_bends", ctypes.c_int),
        ("midi_tempo", ctypes.c_double),
    ]


def _proto(name, restype, *argtypes):
    fn = getattr(_lib, name)
    fn.restype = restype
    fn.argtypes = list(argtypes)
    return fn


_c_int_p = ctypes.POINTER(ctypes.c_int)
_c_double_p = ctypes.POINTER(ctypes.c_double)
_last_error = _proto("bpp_last_error", ctypes.c_char_p)
_frame_count = _proto("bpp_frame_count", ctypes.c_int, ctypes.c_size_t)
_engine_new = _proto("bpp_engine_new", ctypes.c_void_p, ctypes.c_int)
_engine_free = _proto("bpp_engine_free", None, ctypes.c_void_p)
_engine_infer = _proto("bpp_engine_infer", ctypes.c_int, ctypes.c_void_p, _f32p, ctypes.c_size_t, _f32p, _f32p, _f32p)
_engine_infer_file = _proto("bpp_engine_infer_file", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)
_engine_infer_pcm = _proto("bpp_engine_infer_pcm", ctypes.c_void_p, ctypes.c_void_p, _f32p, ctypes.c_size_t,
                           ctypes.c_int, ctypes.c_int)
_output_frames = _proto("bpp_output_frames", ctypes.c_int, ctypes.c_void_p)
_output_copy = _proto("bpp_output_copy", None, ctypes.c_void_p, _f32p, _f32p, _f32p)
_output_free = _proto("bpp_output_free", None, ctypes.c_void_p)
_stream_new_pcm = _proto("bpp_stream_new_pcm", ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int)
_stream_free = _proto("bpp_stream_free", None, ctypes.c_void_p)
_stream_push = _proto("bpp_stream_push", ctypes.c_int, ctypes.c_void_p, _f32p, ctypes.c_size_t)
_stream_finish = _proto("bpp_stream_finish", ctypes.c_int, ctypes.c_void_p)
_stream_final = _proto("bpp_stream_final_frames", ctypes.c_int, ctypes.c_void_p)
_stream_read = _proto("bpp_stream_read", ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, _f32p, _f32p, _f32p)
_notes_decode = _proto("bpp_notes_decode", ctypes.c_void_p, _f32p, _f32p, _f32p, ctypes.c_int, ctypes.POINTER(_COptions))
_notes_free = _proto("bpp_notes_free", None, ctypes.c_void_p)
_notes_count = _proto("bpp_notes_count", ctypes.c_int, ctypes.c_void_p)
_notes_get = _proto("bpp_notes_get", ctypes.c_int, ctypes.c_void_p, ctypes.c_int, _c_double_p, _c_double_p, _c_int_p,
                    _c_int_p, _c_int_p, ctypes.POINTER(ctypes.c_float))
_notes_bends = _proto("bpp_notes_get_bends", None, ctypes.c_void_p, ctypes.c_int, _c_int_p)
_notes_midi = _proto("bpp_notes_midi", ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(_COptions),
                     ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t)
_load_audio = _proto("bpp_load_audio", ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(_f32p), ctypes.POINTER(ctypes.c_size_t))
_free_audio = _proto("bpp_free_audio", None, _f32p)


def _check(ok: bool):
    if not ok:
        raise RuntimeError(_last_error().decode("utf-8", "replace"))


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(_f32p)


@dataclass
class NoteOptions:
    """Note decoding parameters; defaults equal ``basic_pitch.inference.predict``."""

    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    minimum_note_length: float = 127.70  # ms
    minimum_frequency: Optional[float] = None
    maximum_frequency: Optional[float] = None
    multiple_pitch_bends: bool = False
    melodia_trick: bool = True
    midi_tempo: float = 120.0
    infer_onsets: bool = True
    include_pitch_bends: bool = True

    def _c(self) -> _COptions:
        return _COptions(self.onset_threshold, self.frame_threshold, self.minimum_note_length,
                         self.minimum_frequency or 0.0, self.maximum_frequency or 0.0, int(self.infer_onsets),
                         int(self.melodia_trick), int(self.include_pitch_bends), int(self.multiple_pitch_bends),
                         self.midi_tempo)


NoteEvent = Tuple[float, float, int, float, Optional[List[int]]]


@dataclass
class Prediction:
    """Same content as basic-pitch's ``(model_output, midi_data, note_events)``."""

    model_output: dict
    note_events: List[NoteEvent]
    midi_bytes: bytes
    frames: List[Tuple[int, int]] = field(default_factory=list)  # (start_frame, end_frame) per note


def load_audio(path: Union[str, os.PathLike]) -> np.ndarray:
    """Decode a WAV file to 22050 Hz mono float32 exactly like ``librosa.load(path, sr=22050, mono=True)``."""
    buf, n = _f32p(), ctypes.c_size_t()
    _check(_load_audio(os.fsencode(os.fspath(path)), ctypes.byref(buf), ctypes.byref(n)) == 0)
    try:
        return np.ctypeslib.as_array(buf, shape=(n.value,)).copy() if n.value else np.zeros(0, np.float32)
    finally:
        _free_audio(buf)


def _frames(audio: np.ndarray, channels: int) -> np.ndarray:
    """C-contiguous float32 audio as (frames, channels); 1-D input is mono."""
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or audio.shape[1] != channels:
        raise ValueError(f"expected audio of shape (n,) or (n, {channels}), got {audio.shape}")
    return audio


def _empty_output(n: int) -> dict:
    return {"note": np.empty((n, 88), np.float32), "onset": np.empty((n, 88), np.float32),
            "contour": np.empty((n, 264), np.float32)}


class Engine:
    """Neural network inference; thread-safe and reusable across calls."""

    def __init__(self, n_threads: int = 0):
        self._h = _engine_new(n_threads)
        _check(bool(self._h))

    def __del__(self):
        if getattr(self, "_h", None):
            _engine_free(self._h)
            self._h = None

    def infer(self, audio: np.ndarray) -> dict:
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        out = _empty_output(_frame_count(audio.size))
        _check(_engine_infer(self._h, _ptr(audio), audio.size, _ptr(out["note"]), _ptr(out["onset"]),
                             _ptr(out["contour"])) >= 0)
        return out

    def infer_file(self, path: Union[str, os.PathLike]) -> dict:
        """Decode a WAV file (like ``librosa.load``) and run inference, overlapping resampling with the network."""
        return self._collect(_engine_infer_file(self._h, os.fsencode(os.fspath(path))))

    def infer_pcm(self, audio: np.ndarray, sample_rate: int) -> dict:
        """Audio at any rate, shape (n,) or (n, channels): mixed to mono and resampled like ``librosa.load``."""
        audio = np.asarray(audio)
        frames = _frames(audio, audio.shape[1] if audio.ndim == 2 else 1)
        return self._collect(_engine_infer_pcm(self._h, _ptr(frames), frames.shape[0], int(sample_rate),
                                               frames.shape[1]))

    @staticmethod
    def _collect(h) -> dict:
        _check(bool(h))
        try:
            out = _empty_output(_output_frames(h))
            _output_copy(h, _ptr(out["note"]), _ptr(out["onset"]), _ptr(out["contour"]))
            return out
        finally:
            _output_free(h)


class Stream:
    """Incremental inference for live audio: float32 chunks of shape (n,) or (n, channels).

    Input at another rate than 22050 Hz, or with several channels, is converted like ``Engine.infer_pcm``.

    ``push`` returns the frames computed for each completed analysis window right away. The values are
    final, but basic-pitch trims the tail of the sequence when the audio ends, so ``finish`` returns the
    complete, trimmed output (identical to ``Engine.infer`` on all audio). ``final_frames`` is the number
    of leading frames guaranteed to survive that trim.
    """

    def __init__(self, n_threads: int = 1, sample_rate: int = 22050, channels: int = 1):
        self._h = _stream_new_pcm(n_threads, int(sample_rate), int(channels))
        _check(bool(self._h))
        self._channels = int(channels)
        self._read = 0

    def __del__(self):
        if getattr(self, "_h", None):
            _stream_free(self._h)
            self._h = None

    def _drain(self, total: int) -> dict:
        _check(total >= 0)
        out = _empty_output(total - self._read)
        if total > self._read:
            _check(_stream_read(self._h, self._read, total - self._read, _ptr(out["note"]), _ptr(out["onset"]),
                                _ptr(out["contour"])) == 0)
        self._read = total
        return out

    def push(self, audio: np.ndarray) -> dict:
        """Feeds samples; returns the newly computed frames (possibly none)."""
        frames = _frames(audio, self._channels)
        return self._drain(_stream_push(self._h, _ptr(frames), frames.shape[0]))

    @property
    def final_frames(self) -> int:
        return _stream_final(self._h)

    def finish(self) -> dict:
        """Flushes the stream and returns the complete, trimmed output."""
        total = _stream_finish(self._h)
        _check(total >= 0)
        self._read = 0
        return self._drain(total)


def output_to_notes(model_output: dict, options: Optional[NoteOptions] = None) -> Prediction:
    """Note events + MIDI from model output (``basic_pitch.note_creation.model_output_to_notes``)."""
    options = options or NoteOptions()
    note = np.ascontiguousarray(model_output["note"], np.float32)
    onset = np.ascontiguousarray(model_output["onset"], np.float32)
    contour = np.ascontiguousarray(model_output["contour"], np.float32)
    n = note.shape[0]
    if note.shape != (n, 88) or onset.shape != (n, 88) or contour.shape != (n, 264):
        raise ValueError(f"expected note/onset (n, 88) and contour (n, 264), got {note.shape}, {onset.shape}, {contour.shape}")
    copt = options._c()
    h = _notes_decode(_ptr(note), _ptr(onset), _ptr(contour), note.shape[0], ctypes.byref(copt))
    _check(bool(h))
    try:
        events, frames = [], []
        s, e = ctypes.c_double(), ctypes.c_double()
        sf, ef, p, a = ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_float()
        for i in range(_notes_count(h)):
            nb = _notes_get(h, i, ctypes.byref(s), ctypes.byref(e), ctypes.byref(sf), ctypes.byref(ef),
                            ctypes.byref(p), ctypes.byref(a))
            bends = None
            if nb:
                arr = (ctypes.c_int * nb)()
                _notes_bends(h, i, arr)
                bends = list(arr)
            events.append((s.value, e.value, p.value, float(a.value), bends))
            frames.append((sf.value, ef.value))
        size = _notes_midi(h, ctypes.byref(copt), None, 0)
        _check(size > 0)
        buf = (ctypes.c_ubyte * size)()
        _notes_midi(h, ctypes.byref(copt), buf, size)
        return Prediction(model_output={"note": note, "onset": onset, "contour": contour}, note_events=events,
                          midi_bytes=bytes(buf), frames=frames)
    finally:
        _notes_free(h)


_default_engine: Optional[Engine] = None


def predict(audio: Union[str, os.PathLike, np.ndarray], options: Optional[NoteOptions] = None,
            n_threads: int = 0, sample_rate: int = 22050) -> Prediction:
    """Transcribe a WAV path or an array (``basic_pitch.inference.predict``).

    Arrays are (n,) mono or (n, channels) at ``sample_rate``, converted like ``librosa.load``.
    """
    global _default_engine
    if n_threads:
        engine = Engine(n_threads)
    else:
        if _default_engine is None:
            _default_engine = Engine(0)
        engine = _default_engine
    if isinstance(audio, np.ndarray):
        mono_22k = audio.ndim == 1 and sample_rate == 22050
        output = engine.infer(audio) if mono_22k else engine.infer_pcm(audio, sample_rate)
    else:
        output = engine.infer_file(audio)
    return output_to_notes(output, options)
