# basicpp-pitch (WebAssembly)

[BasiCPP Pitch](https://github.com/yuanhenglee/basicpp-pitch) compiled to WebAssembly: Spotify's
basic-pitch transcription model in the browser or Node.js, with notes and MIDI bytes identical to
the Python library.

```js
import { loadBasicPitch } from "basicpp-pitch";

const bp = await loadBasicPitch();

// Whole recording: interleaved float32 PCM at its native rate (resampled like librosa.load)
const { notes, midi } = bp.transcribe(pcm, { sampleRate: 44100, channels: 2 });

// Live input
const stream = bp.createStream({ sampleRate: 48000 });
stream.push(chunk);                          // from an AudioWorklet, for example
const live = stream.notes();                 // notes over the frames that are already final
stream.finish();
const all = stream.notes();
```

Run it in a Web Worker for long files so the page stays responsive. The `.wasm` file is loaded
next to `basicpp_pitch_wasm.mjs`; pass `{ locateFile }` to `loadBasicPitch` to serve it elsewhere.

Build: see "WebAssembly" in the main README. This package includes libsoxr (LGPL-2.1); keep the
`.wasm` file separate and replaceable.
