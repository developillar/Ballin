/**
 * The crowd.
 *
 * A basketball crowd is not one sound, it is three layered ones, and getting
 * the layers wrong is what makes synthetic crowds sound like rain:
 *
 *   1. A broadband murmur — twenty thousand conversations averaged into noise
 *      with a strong low-mid weighting around 250–700 Hz. This never stops.
 *   2. A vowel-ish "roar" band around 500–1400 Hz that swells with excitement.
 *      This is what a crowd reacting sounds like, and it is almost absent at
 *      rest, which is why a bed that just gets louder reads as a volume knob
 *      rather than as a reaction.
 *   3. Transients — individual claps and shouts, sparse and stochastic. They
 *      are what stops the bed from sounding like a loop.
 *
 * Everything is generated from noise buffers built at startup. The murmur and
 * roar loop continuously; only their gains and filters are automated.
 */

import { clamp01, makeRng } from '../core/MathX';

/** Seconds of noise generated for each looping layer. Long enough not to pulse. */
const LOOP_SECONDS = 8;

function noiseBuffer(ctx: BaseAudioContext, seconds: number, seed: number): AudioBuffer {
  const rate = ctx.sampleRate;
  const length = Math.floor(rate * seconds);
  const buf = ctx.createBuffer(2, length, rate);
  const rng = makeRng(seed);
  for (let ch = 0; ch < 2; ch++) {
    const d = buf.getChannelData(ch);
    // Pink-ish noise via a small filter cascade. White noise on its own is far
    // too bright to read as a crowd; the -3 dB/octave tilt is most of the
    // difference between "people" and "static".
    let b0 = 0;
    let b1 = 0;
    let b2 = 0;
    for (let i = 0; i < length; i++) {
      const w = rng() * 2 - 1;
      b0 = 0.99765 * b0 + w * 0.099;
      b1 = 0.963 * b1 + w * 0.2965;
      b2 = 0.57 * b2 + w * 1.0526;
      d[i] = (b0 + b1 + b2 + w * 0.1848) * 0.16;
    }
    // Cross-fade the last 250 ms into the head so the loop point is inaudible.
    const fade = Math.floor(rate * 0.25);
    for (let i = 0; i < fade; i++) {
      const t = i / fade;
      d[i] = d[i] * t + d[length - fade + i] * (1 - t);
    }
  }
  return buf;
}

export interface CrowdBedOptions {
  /** How many stochastic transient voices to run. Scales with quality tier. */
  voices: number;
  seed: number;
}

export class CrowdBed {
  /** 0 = between plays, 1 = the building is coming down. */
  excitement = 0;

  private readonly ctx: AudioContext;
  private readonly out: GainNode;
  private readonly murmurGain: GainNode;
  private readonly roarGain: GainNode;
  private readonly roarFilter: BiquadFilterNode;
  private readonly transientGain: GainNode;
  private readonly sources: AudioBufferSourceNode[] = [];
  private readonly clapBuffer: AudioBuffer;
  private readonly rng: () => number;
  private readonly voices: number;

  /** Smoothed excitement, so a swell rises fast and settles slowly. */
  private level = 0;
  private nextTransient = 0;

  constructor(ctx: AudioContext, destination: AudioNode, opts: CrowdBedOptions) {
    this.ctx = ctx;
    this.rng = makeRng(opts.seed);
    this.voices = opts.voices;

    this.out = ctx.createGain();
    this.out.gain.value = 1;
    this.out.connect(destination);

    const noise = noiseBuffer(ctx, LOOP_SECONDS, opts.seed);

    // --- Murmur: the constant bed --------------------------------------
    this.murmurGain = ctx.createGain();
    this.murmurGain.gain.value = 0.16;
    const murmurBand = ctx.createBiquadFilter();
    murmurBand.type = 'bandpass';
    murmurBand.frequency.value = 420;
    murmurBand.Q.value = 0.55;
    const murmurShelf = ctx.createBiquadFilter();
    murmurShelf.type = 'highshelf';
    murmurShelf.frequency.value = 2200;
    murmurShelf.gain.value = -14;
    murmurBand.connect(murmurShelf).connect(this.murmurGain).connect(this.out);
    this.play(noise, murmurBand, 1);

    // --- Roar: the reactive layer --------------------------------------
    this.roarGain = ctx.createGain();
    this.roarGain.gain.value = 0;
    this.roarFilter = ctx.createBiquadFilter();
    this.roarFilter.type = 'bandpass';
    this.roarFilter.frequency.value = 620;
    this.roarFilter.Q.value = 1.35;
    this.roarFilter.connect(this.roarGain).connect(this.out);
    // Detuned against the murmur so the two layers do not phase-lock.
    this.play(noise, this.roarFilter, 0.87);

    this.transientGain = ctx.createGain();
    this.transientGain.gain.value = 0.5;
    this.transientGain.connect(this.out);

    this.clapBuffer = this.makeClap();
  }

  private play(buffer: AudioBuffer, target: AudioNode, rate: number): void {
    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    src.loop = true;
    src.playbackRate.value = rate;
    src.connect(target);
    src.start(this.ctx.currentTime + this.rng() * 0.5);
    this.sources.push(src);
  }

  /** A single clap: a very short, bright, filtered burst with a hard attack. */
  private makeClap(): AudioBuffer {
    const rate = this.ctx.sampleRate;
    const length = Math.floor(rate * 0.09);
    const buf = this.ctx.createBuffer(1, length, rate);
    const d = buf.getChannelData(0);
    const rng = makeRng(0xc1a9);
    let lp = 0;
    for (let i = 0; i < length; i++) {
      const t = i / length;
      // Two-stage decay: a 3 ms crack, then a short body.
      const env = Math.exp(-t * 34) * (1 - t);
      const w = rng() * 2 - 1;
      lp += 0.55 * (w - lp);
      d[i] = (w * 0.6 + lp * 0.4) * env;
    }
    return buf;
  }

  /** Fires one crowd transient at a random position in the stereo field. */
  private transient(intensity: number): void {
    const ctx = this.ctx;
    const src = ctx.createBufferSource();
    src.buffer = this.clapBuffer;
    // Wide pitch spread: a crowd is thousands of different hands.
    src.playbackRate.value = 0.7 + this.rng() * 0.9;

    const filter = ctx.createBiquadFilter();
    filter.type = 'bandpass';
    filter.frequency.value = 900 + this.rng() * 2600;
    filter.Q.value = 0.8;

    const pan = ctx.createStereoPanner();
    pan.pan.value = this.rng() * 1.7 - 0.85;

    const g = ctx.createGain();
    g.gain.value = (0.06 + this.rng() * 0.14) * intensity;

    src.connect(filter).connect(g).connect(pan).connect(this.transientGain);
    src.start();
  }

  /**
   * Drives the bed toward the current excitement. The asymmetry matters: a
   * crowd reacts in a few hundred milliseconds and takes several seconds to
   * settle, and a symmetric envelope sounds like an automated fader.
   */
  update(dt: number): void {
    const target = clamp01(this.excitement);
    const rate = target > this.level ? 6.5 : 0.55;
    this.level += (target - this.level) * (1 - Math.exp(-rate * dt));

    const t = this.ctx.currentTime;
    const l = this.level;

    // The murmur barely moves — most of the change is the roar arriving.
    this.murmurGain.gain.setTargetAtTime(0.16 + l * 0.1, t, 0.12);
    this.roarGain.gain.setTargetAtTime(l * l * 0.42, t, 0.08);
    // The roar also opens upward in pitch as it builds, which is what a crowd
    // rising to its feet actually does.
    this.roarFilter.frequency.setTargetAtTime(620 + l * 520, t, 0.15);
    this.roarFilter.Q.setTargetAtTime(1.35 - l * 0.5, t, 0.15);

    if (this.voices <= 0) return;
    this.nextTransient -= dt;
    if (this.nextTransient <= 0) {
      // Rate climbs steeply with excitement: idle chatter is a clap every
      // second or so, a made three is a wall of them.
      const perSecond = 1.2 + l * this.voices;
      this.nextTransient = -Math.log(1 - this.rng()) / perSecond;
      this.transient(0.35 + l * 0.9);
    }
  }

  /** A discrete reaction — a made basket, a block. Decays back on its own. */
  spike(amount: number): void {
    this.excitement = clamp01(Math.max(this.excitement, amount));
    const bursts = Math.round(2 + amount * 6);
    for (let i = 0; i < bursts; i++) this.transient(0.5 + amount * 0.8);
  }

  /** Bleeds excitement away. Called every frame; events push it back up. */
  settle(dt: number, floor = 0): void {
    this.excitement = Math.max(floor, this.excitement - dt * 0.22);
  }

  dispose(): void {
    for (const s of this.sources) {
      try {
        s.stop();
      } catch {
        // Already stopped; nothing to do.
      }
    }
    this.sources.length = 0;
    this.out.disconnect();
  }
}
