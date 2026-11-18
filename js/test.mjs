// Checks the JavaScript API of a WebAssembly build against the golden fixture (MIDI bytes and note
// pitches/times from basic-pitch), offline and streaming, from raw 44.1 kHz stereo PCM.
//   node js/test.mjs <build>/js tests/fixtures/<name>
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [pkgDir, base] = process.argv.slice(2);
const { loadBasicPitch } = await import(pathToFileURL(join(resolve(pkgDir), "index.js")).href);

let failures = 0;
const check = (ok, what) => {
  console.log(`[${ok ? "PASS" : "FAIL"}] ${what}`);
  failures += !ok;
};

// 16-bit PCM WAV -> interleaved float32 scaled like libsndfile (x / 32768).
function readWav16(path) {
  const b = readFileSync(path);
  let channels = 0, sampleRate = 0, pcm = null;
  for (let pos = 12; pos + 8 <= b.length; ) {
    const id = b.toString("latin1", pos, pos + 4), size = b.readUInt32LE(pos + 4);
    if (id === "fmt ") {
      channels = b.readUInt16LE(pos + 10);
      sampleRate = b.readUInt32LE(pos + 12);
    } else if (id === "data") {
      const n = Math.min(size, b.length - pos - 8) >> 1;
      pcm = new Float32Array(n);
      for (let i = 0; i < n; i++) pcm[i] = b.readInt16LE(pos + 8 + 2 * i) / 32768;
    }
    pos += 8 + size + (size & 1);
  }
  return { pcm, channels, sampleRate };
}

// "<start hex> <end hex> <pitch> <amplitude hex> [bends...]" per line
function hexFloat(s) {
  const m = /^(-?)0x([01])\.([0-9a-f]*)p([+-]\d+)$/.exec(s);
  if (!m) return NaN;
  const frac = m[3] ? parseInt(m[3], 16) / 16 ** m[3].length : 0;
  return (m[1] ? -1 : 1) * (Number(m[2]) + frac) * 2 ** Number(m[4]);
}
const golden = readFileSync(base + ".notes", "utf8").trim().split(/\r?\n/).map((line) => {
  const f = line.trim().split(/\s+/);
  return { startS: hexFloat(f[0]), endS: hexFloat(f[1]), pitch: Number(f[2]), amplitude: Math.fround(hexFloat(f[3])),
           bends: f.slice(4).map(Number) };
});
const goldenMidi = new Uint8Array(readFileSync(base + ".mid"));

// The golden output comes from Windows, where libsoxr designs its filter with the Universal CRT's
// libm; WebAssembly's musl rounds a few of those calls differently, so resampled audio moves by about
// one ulp. Note times, pitches, bends and MIDI bytes stay identical; amplitudes move by ~1e-5.
const AMPLITUDE_TOL = 1e-4;
const sameNotes = (notes) =>
  notes.length === golden.length &&
  notes.every((n, i) => n.startS === golden[i].startS && n.endS === golden[i].endS && n.pitch === golden[i].pitch &&
    Math.abs(n.amplitude - golden[i].amplitude) <= AMPLITUDE_TOL && (n.pitchBends ?? []).join() === golden[i].bends.join());
const identical = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const sameBytes = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

const { pcm, channels, sampleRate } = readWav16(base + ".wav");
// A pthreads build directory only holds basicpp_pitch_wasm_mt.mjs.
const threads = !existsSync(join(resolve(pkgDir), "basicpp_pitch_wasm.mjs"));
const bp = await loadBasicPitch({ threads });
console.log(`       ${threads ? `pthreads build, ${bp.threads} threads` : "single-threaded build"}`);

const t0 = performance.now();
const progress = [];
const result = bp.transcribe(pcm, { sampleRate, channels, onProgress: (f) => void progress.push(f) });
const ms = performance.now() - t0;
check(progress.length > 0 && progress.every((f, i) => f > (progress[i - 1] ?? 0)) && progress.at(-1) === 1,
      `transcribe: progress ${progress.map((f) => f.toFixed(2)).join(", ")}`);
let cancelled = false;
try {
  bp.transcribe(pcm, { sampleRate, channels, onProgress: () => false });
} catch (e) {
  cancelled = /cancelled/.test(e.message);
}
check(cancelled, "transcribe: returning false from onProgress cancels");
check(sameNotes(result.notes), `transcribe: ${result.notes.length} note events identical (amplitudes within ${AMPLITUDE_TOL})`);
check(sameBytes(result.midi, goldenMidi), "transcribe: MIDI bytes identical");
const seconds = pcm.length / channels / sampleRate;
console.log(`       ${seconds.toFixed(1)} s of audio in ${ms.toFixed(0)} ms (${(seconds * 1000 / ms).toFixed(0)}x real time)`);

const stream = bp.createStream({ sampleRate, channels });
const chunk = 4410 * channels;  // 100 ms
let liveDecodes = 0;
for (let pos = 0; pos < pcm.length; pos += chunk) {
  if (stream.push(pcm.subarray(pos, Math.min(pos + chunk, pcm.length))) > 0) {
    stream.notes();  // live view over the final frames, as an app would draw it
    liveDecodes++;
  }
}
stream.finish();
const streamed = stream.notes();
check(liveDecodes > 0, `stream: ${liveDecodes} live decodes while pushing`);
check(identical(streamed.notes, result.notes) && sameBytes(streamed.midi, result.midi),
      "stream: notes and MIDI bit-identical to transcribe()");
stream.dispose();

let threw = false;
try {
  bp.createStream({ sampleRate: 10 });
} catch (e) {
  threw = /sample rate/.test(e.message);
}
check(threw, "invalid sample rate raises an Error with the engine's message");
bp.dispose();

console.log(failures ? "FAILED" : "ALL PASSED");
process.exit(failures ? 1 : 0);
