/** Note decoding parameters; omitted fields use basic_pitch.inference.predict()'s defaults. */
export interface NoteOptions {
  onsetThreshold?: number;
  frameThreshold?: number;
  minimumNoteLengthMs?: number;
  /** Hz; omit for none. */
  minimumFrequency?: number;
  /** Hz; omit for none. */
  maximumFrequency?: number;
  inferOnsets?: boolean;
  melodiaTrick?: boolean;
  includePitchBends?: boolean;
  multiplePitchBends?: boolean;
  midiTempo?: number;
}

export interface NoteEvent {
  startS: number;
  endS: number;
  startFrame: number;
  endFrame: number;
  /** MIDI note number. */
  pitch: number;
  /** Mean note activation in [0, 1]; MIDI velocity = round(127 * amplitude). */
  amplitude: number;
  /** One value per frame in 1/3 semitone; absent if none. */
  pitchBends?: number[];
}

export interface Transcription {
  notes: NoteEvent[];
  /** Standard MIDI File bytes, identical to basic-pitch's. */
  midi: Uint8Array;
  /** Model frames decoded (86.13 per second). */
  frames: number;
}

export interface PcmFormat {
  /** 1000 .. 1000000 Hz; default 22050. */
  sampleRate?: number;
  /** Interleaved channels; default 1. */
  channels?: number;
}

/** `threads: true` loads the pthreads build (needs SharedArrayBuffer). */
export declare function loadBasicPitch(options?: { threads?: boolean } & Record<string, unknown>): Promise<BasicPitch>;

export interface TranscribeOptions extends PcmFormat, NoteOptions {
  /** Called with the completed fraction as analysis windows finish; return false to cancel. */
  onProgress?: (fraction: number) => boolean | void;
}

export declare class BasicPitch {
  readonly threads: number;
  transcribe(audio: Float32Array, options?: TranscribeOptions): Transcription;
  createStream(options?: PcmFormat & { threads?: number }): Stream;
  dispose(): void;
}

export declare class Stream {
  push(audio: Float32Array): number;
  finish(): void;
  readonly frames: number;
  readonly finalFrames: number;
  notes(options?: NoteOptions, select?: { finalOnly?: boolean }): Transcription;
  dispose(): void;
}
