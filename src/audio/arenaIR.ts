/**
 * A synthetic arena impulse response.
 *
 * An indoor arena has a very particular sound and it is mostly not the direct
 * signal: a short pre-delay while the wavefront crosses to the first surface,
 * a scatter of discrete early reflections off the floor, the backboard and the
 * near stands, then a long diffuse tail off the bowl and the roof. Real rooms
 * of this size run an RT60 around two seconds, and the tail is not flat —
 * high frequencies are absorbed by twenty thousand people far faster than the
 * low end, so the decay gets darker as it goes.
 *
 * Building this rather than shipping a recorded IR keeps the no-external-assets
 * rule intact, and it costs one buffer allocation at startup.
 */

import { makeRng } from '../core/MathX';

export interface ArenaIROptions {
  /** Seconds until the reverb tail has fallen 60 dB. */
  rt60: number;
  /** Seconds before the first reflection arrives. */
  preDelay: number;
  /** How much of the energy is in discrete early reflections, 0..1. */
  earlyMix: number;
  /** Extra decay rate applied to the top end, as a multiple of the low end. */
  highDamping: number;
  seed: number;
}

export const COURT_IR: ArenaIROptions = {
  // Close, tight reflections — the sound of being on the floor.
  rt60: 0.85,
  preDelay: 0.008,
  earlyMix: 0.62,
  highDamping: 2.4,
  seed: 0x5011,
};

export const BOWL_IR: ArenaIROptions = {
  // The long one. This is what makes a horn sound like it is in an arena.
  rt60: 2.3,
  preDelay: 0.026,
  earlyMix: 0.22,
  highDamping: 3.6,
  seed: 0xb0c4,
};

/**
 * Renders a stereo impulse response into an AudioBuffer.
 *
 * The two channels are generated from independent noise so the tail decorrelates
 * and the reverb has width; the early reflections are placed at slightly
 * different times per channel for the same reason.
 */
export function makeArenaIR(ctx: BaseAudioContext, opts: ArenaIROptions): AudioBuffer {
  const rate = ctx.sampleRate;
  const length = Math.max(1, Math.floor(rate * (opts.rt60 + opts.preDelay + 0.05)));
  const buffer = ctx.createBuffer(2, length, rate);
  const rng = makeRng(opts.seed);

  // Exponential decay constant: amplitude falls by 60 dB (a factor of 1000)
  // over rt60 seconds.
  const decay = Math.log(1000) / opts.rt60;

  for (let ch = 0; ch < 2; ch++) {
    const data = buffer.getChannelData(ch);
    const preDelaySamples = Math.floor(opts.preDelay * rate * (ch === 0 ? 1 : 1.13));

    // Diffuse tail: noise under an exponential envelope, with the high end
    // rolled off progressively by a one-pole lowpass whose cutoff falls as the
    // tail decays. That falling cutoff is the audible signature of a big
    // absorptive room, and it is what a flat noise burst always gets wrong.
    let lp = 0;
    for (let i = preDelaySamples; i < length; i++) {
      const t = (i - preDelaySamples) / rate;
      const env = Math.exp(-decay * t);
      const noise = rng() * 2 - 1;
      // Coefficient walks from open to closed across the tail.
      const openness = Math.pow(env, 1 / opts.highDamping);
      const a = 0.06 + 0.9 * openness;
      lp += a * (noise - lp);
      data[i] = lp * env * (1 - opts.earlyMix);
    }

    // Early reflections: a sparse set of discrete taps, getting denser and
    // quieter with time the way real reflections do as the wavefront breaks up.
    let tap = preDelaySamples + Math.floor(rate * 0.004);
    let amp = opts.earlyMix;
    let gap = 0.006;
    while (tap < length && amp > 0.002) {
      const idx = Math.min(length - 1, tap);
      data[idx] += (rng() < 0.5 ? -1 : 1) * amp;
      amp *= 0.72 + rng() * 0.14;
      gap *= 1.24 + rng() * 0.2;
      tap += Math.max(1, Math.floor(gap * rate));
    }

    // Normalise per channel so the send level means the same thing regardless
    // of which IR is loaded.
    let peak = 0;
    for (let i = 0; i < length; i++) peak = Math.max(peak, Math.abs(data[i]));
    if (peak > 0) {
      const k = 0.85 / peak;
      for (let i = 0; i < length; i++) data[i] *= k;
    }
  }

  return buffer;
}
