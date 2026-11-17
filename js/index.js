// JavaScript API for the WebAssembly build of BasiCPP Pitch. Runs in browsers (main thread or Web
// Worker) and Node.js 18+. Results are bit-identical to the C++ library and to basic-pitch.

const NOTE_BINS = 88;
const CONTOUR_BINS = 264;

/**
 * Instantiates the WebAssembly module. `threads: true` loads the pthreads build
 * (basicpp_pitch_wasm_mt.mjs, from -DBPP_WASM_THREADS=ON), which needs SharedArrayBuffer: a
 * cross-origin isolated page, or Node.js. Other options go to Emscripten (e.g. `locateFile`).
 */
export async function loadBasicPitch({ threads = false, ...moduleOptions } = {}) {
  const { default: create } = threads ? await import("./basicpp_pitch_wasm_mt.mjs") : await import("./basicpp_pitch_wasm.mjs");
  const nThreads = threads ? Math.max(1, globalThis.navigator?.hardwareConcurrency ?? 4) : 1;
  return new BasicPitch(await create(moduleOptions), nThreads);
}

class Wasm {
  constructor(m) {
    this.m = m;
  }

  check(ok) {
    if (!ok) throw new Error(this.m.UTF8ToString(this.m._bpp_last_error()) || "basicpp-pitch error");
  }

  malloc(bytes) {
    const p = this.m._malloc(Math.max(bytes, 1));
    if (!p) throw new Error("basicpp-pitch: out of WebAssembly memory");
    return p;
  }

  // Runs fn(ptr) with a heap copy of a Float32Array, then frees it.
  withFloats(data, fn) {
    const p = this.malloc(data.length * 4);
    try {
      this.m.HEAPF32.set(data, p >> 2);
      return fn(p);
    } finally {
      this.m._free(p);
    }
  }

  // bpp_note_options; field offsets follow include/bpp/bpp_c.h (asserted in src/c_api.cpp).
  withOptions(o, fn) {
    const m = this.m;
    const p = this.malloc(64);
    try {
      m._bpp_note_options_default(p);
      const f64 = (offset, v) => { if (v !== undefined) m.HEAPF64[(p + offset) >> 3] = v; };
      const i32 = (offset, v) => { if (v !== undefined) m.HEAP32[(p + offset) >> 2] = v ? 1 : 0; };
      f64(0, o.onsetThreshold);
      f64(8, o.frameThreshold);
      f64(16, o.minimumNoteLengthMs);
      f64(24, o.minimumFrequency ?? 0);
      f64(32, o.maximumFrequency ?? 0);
      i32(40, o.inferOnsets);
      i32(44, o.melodiaTrick);
      i32(48, o.includePitchBends);
      i32(52, o.multiplePitchBends);
      f64(56, o.midiTempo);
      return fn(p);
    } finally {
      m._free(p);
    }
  }

  // Decodes nFrames of model output stored at note/onset/contour (heap pointers).
  decode(note, onset, contour, nFrames, options) {
    const m = this.m;
    return this.withOptions(options, (opt) => {
      const h = m._bpp_notes_decode(note, onset, contour, nFrames, opt);
      this.check(h);
      const scratch = this.malloc(40);
      try {
        const notes = [];
        for (let i = 0, n = m._bpp_notes_count(h); i < n; i++) {
          const nb = m._bpp_notes_get(h, i, scratch, scratch + 8, scratch + 16, scratch + 20, scratch + 24, scratch + 28);
          const ev = {
            startS: m.HEAPF64[scratch >> 3],
            endS: m.HEAPF64[(scratch + 8) >> 3],
            startFrame: m.HEAP32[(scratch + 16) >> 2],
            endFrame: m.HEAP32[(scratch + 20) >> 2],
            pitch: m.HEAP32[(scratch + 24) >> 2],
            amplitude: m.HEAPF32[(scratch + 28) >> 2],
          };
          if (nb > 0) {
            const b = this.malloc(nb * 4);
            m._bpp_notes_get_bends(h, i, b);
            ev.pitchBends = Array.from(m.HEAP32.subarray(b >> 2, (b >> 2) + nb));
            m._free(b);
          }
          notes.push(ev);
        }
        const size = m._bpp_notes_midi(h, opt, 0, 0);
        this.check(size > 0);
        const buf = this.malloc(size);
        m._bpp_notes_midi(h, opt, buf, size);
        const midi = m.HEAPU8.slice(buf, buf + size);
        m._free(buf);
        return { notes, midi, frames: nFrames };
      } finally {
        m._free(scratch);
        m._bpp_notes_free(h);
      }
    });
  }

  // Heap buffers for nFrames of model output; fn(note, onset, contour) fills and uses them.
  withFrames(nFrames, fn) {
    const note = this.malloc(nFrames * NOTE_BINS * 4);
    const onset = this.malloc(nFrames * NOTE_BINS * 4);
    const contour = this.malloc(nFrames * CONTOUR_BINS * 4);
    try {
      return fn(note, onset, contour);
    } finally {
      this.m._free(note);
      this.m._free(onset);
      this.m._free(contour);
    }
  }
}

function frameCount(audio, channels) {
  if (!(audio instanceof Float32Array)) throw new TypeError("audio must be a Float32Array");
  if (!Number.isInteger(channels) || channels < 1) throw new RangeError("channels must be a positive integer");
  if (audio.length % channels) throw new RangeError("audio length is not a multiple of the channel count");
  return audio.length / channels;
}

export class BasicPitch {
  #w;
  #engine;
  #threads;

  constructor(m, threads = 1) {
    this.#w = new Wasm(m);
    this.#threads = threads;
    this.#engine = m._bpp_engine_new(threads);
    this.#w.check(this.#engine);
  }

  /** Threads the engine runs on. */
  get threads() {
    return this.#threads;
  }

  /**
   * Transcribes interleaved float32 PCM at any sample rate; resampling matches librosa.load, so
   * the result equals basic-pitch on a WAV file holding the same samples. `onProgress(fraction)`
   * is called as analysis windows complete; returning false from it cancels (throws).
   */
  transcribe(audio, { sampleRate = 22050, channels = 1, onProgress, ...noteOptions } = {}) {
    const w = this.#w, m = w.m;
    const frames = frameCount(audio, channels);
    const pp = w.malloc(8);
    let samples = 0, n = 0;
    try {
      w.check(w.withFloats(audio, (p) => m._bpp_convert_pcm(p, frames, sampleRate, channels, pp, pp + 4)) === 0);
      samples = m.HEAP32[pp >> 2];
      n = m.HEAP32[(pp + 4) >> 2] >>> 0;
    } finally {
      m._free(pp);
    }
    let out = 0;
    try {
      out = m._bpp_output_new(n);
      w.check(out);
      // Windows are independent, so batching them changes nothing but the progress granularity.
      const total = m._bpp_window_count(n), batch = 2 * this.#threads;
      for (let first = 0; first < total; first += batch) {
        const count = Math.min(batch, total - first);
        w.check(m._bpp_engine_infer_windows(this.#engine, samples, n, first, count, out) === 0);
        if (onProgress?.((first + count) / total) === false) throw new Error("transcription cancelled");
      }
      if (total === 0) onProgress?.(1);
      const nFrames = m._bpp_output_frames(out);
      return w.withFrames(nFrames, (note, onset, contour) => {
        m._bpp_output_copy(out, note, onset, contour);
        return w.decode(note, onset, contour, nFrames, noteOptions);
      });
    } finally {
      if (out) m._bpp_output_free(out);
      m._bpp_free_audio(samples);
    }
  }

  /** Incremental transcription of live audio. */
  createStream({ sampleRate = 22050, channels = 1, threads = 1 } = {}) {
    return new Stream(this.#w, sampleRate, channels, threads);
  }

  dispose() {
    if (this.#engine) this.#w.m._bpp_engine_free(this.#engine);
    this.#engine = 0;
  }
}

export class Stream {
  #w;
  #h;
  #channels;
  #frames = 0;

  constructor(w, sampleRate, channels, threads) {
    this.#w = w;
    this.#channels = channels;
    this.#h = w.m._bpp_stream_new_pcm(threads, sampleRate, channels);
    w.check(this.#h);
  }

  /** Feeds interleaved samples; returns the number of new model frames (one window = 142). */
  push(audio) {
    const w = this.#w;
    const frames = frameCount(audio, this.#channels);
    const total = w.withFloats(audio, (p) => w.m._bpp_stream_push(this.#h, p, frames));
    w.check(total >= 0);
    const added = total - this.#frames;
    this.#frames = total;
    return added;
  }

  /** Flushes the stream; afterwards notes() covers all audio and equals BasicPitch.transcribe(). */
  finish() {
    const total = this.#w.m._bpp_stream_finish(this.#h);
    this.#w.check(total >= 0);
    this.#frames = total;
  }

  /** Model frames computed so far. */
  get frames() {
    return this.#frames;
  }

  /** Leading frames that are final no matter when the stream ends. */
  get finalFrames() {
    return this.#w.m._bpp_stream_final_frames(this.#h);
  }

  /**
   * Decodes notes from the frames so far (by default only the final ones). Notes near the end
   * may still change while audio keeps arriving.
   */
  notes(noteOptions = {}, { finalOnly = true } = {}) {
    const w = this.#w, m = w.m;
    const n = finalOnly ? this.finalFrames : this.#frames;
    return w.withFrames(n, (note, onset, contour) => {
      w.check(m._bpp_stream_read(this.#h, 0, n, note, onset, contour) === 0);
      return w.decode(note, onset, contour, n, noteOptions);
    });
  }

  dispose() {
    if (this.#h) this.#w.m._bpp_stream_free(this.#h);
    this.#h = 0;
  }
}
