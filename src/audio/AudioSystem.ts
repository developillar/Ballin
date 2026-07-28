/**
 * Fully procedural audio — no sample downloads.
 *
 * Everything is synthesised through the Web Audio graph and placed in the room
 * against the camera. The signal path is:
 *
 *   voice ──► panner ──┬──────────────────────────────► dry ──┐
 *                      ├──► courtSend ──► court IR ──────────┤──► bus ──► comp ──► out
 *                      └──► bowlSend  ──► bowl IR  ──────────┘
 *
 * Two convolution reverbs rather than one, because an arena has two distinct
 * spaces in it: the tight slap off the floor and the boards, and the long tail
 * off the bowl. A ball bounce is mostly the first; a horn is mostly the second.
 * Using a single medium reverb for both makes everything sound like a gym.
 *
 * A compressor sits across the bus so that a rim clang landing on top of a
 * crowd swell ducks rather than clips — the same reason a broadcast mix has one.
 *
 * Nothing here starts until a user gesture unlocks the context, which is both
 * the browser's rule and the reason every method guards on `ctx`.
 */

import { Vector3 } from 'three';
import type { Engine, System } from '../core/Engine';
import { clamp, clamp01, makeRng } from '../core/MathX';
import { BOWL_IR, COURT_IR, makeArenaIR } from './arenaIR';
import { CrowdBed } from './crowdBed';

/** Crowd transient voices per tier — the only part of audio that scales. */
const VOICES = { low: 0, medium: 5, high: 11, ultra: 16 } as const;

export class AudioSystem implements System {
  readonly name = 'audio';
  readonly order = 80;

  ctx: AudioContext | null = null;
  master: GainNode | null = null;

  private bus: GainNode | null = null;
  private dry: GainNode | null = null;
  private courtSend: GainNode | null = null;
  private bowlSend: GainNode | null = null;
  private crowd: CrowdBed | null = null;

  private unlocked = false;
  private readonly rng = makeRng(0xa0d10);
  private readonly listener = new Vector3();
  private readonly camRight = new Vector3();
  private readonly relative = new Vector3();
  private engine: Engine | null = null;

  /** Rate-limits dribble hits so a fast crossover does not machine-gun. */
  private lastDribble = -1;

  init(engine: Engine): void {
    this.engine = engine;

    const unlock = () => {
      if (this.unlocked) return;
      this.unlocked = true;
      try {
        this.build(engine);
      } catch (err) {
        console.warn('[audio] failed to start', err);
        this.ctx = null;
      }
    };
    window.addEventListener('pointerdown', unlock, { once: true });
    window.addEventListener('keydown', unlock, { once: true });
    window.addEventListener('touchstart', unlock, { once: true });

    this.subscribe(engine);
  }

  private build(engine: Engine): void {
    const Ctx =
      window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = new Ctx();
    this.ctx = ctx;

    const master = ctx.createGain();
    master.gain.value = 0.75;

    // Gentle bus compression. Slow enough not to pump on the crowd bed, fast
    // enough to catch a rim clang's attack.
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -14;
    comp.knee.value = 22;
    comp.ratio.value = 3.2;
    comp.attack.value = 0.006;
    comp.release.value = 0.22;

    const bus = ctx.createGain();
    bus.gain.value = 1;
    bus.connect(comp).connect(master).connect(ctx.destination);

    this.master = master;
    this.bus = bus;

    this.dry = ctx.createGain();
    this.dry.gain.value = 1;
    this.dry.connect(bus);

    const court = ctx.createConvolver();
    court.buffer = makeArenaIR(ctx, COURT_IR);
    court.normalize = false;
    const courtWet = ctx.createGain();
    courtWet.gain.value = 0.5;
    court.connect(courtWet).connect(bus);
    this.courtSend = ctx.createGain();
    this.courtSend.gain.value = 1;
    this.courtSend.connect(court);

    const bowl = ctx.createConvolver();
    bowl.buffer = makeArenaIR(ctx, BOWL_IR);
    bowl.normalize = false;
    const bowlWet = ctx.createGain();
    bowlWet.gain.value = 0.34;
    // The tail is darkened further on the way out; a bright two-second reverb
    // sounds like a cathedral, not a sports arena full of soft bodies.
    const bowlTone = ctx.createBiquadFilter();
    bowlTone.type = 'lowpass';
    bowlTone.frequency.value = 2600;
    bowl.connect(bowlTone).connect(bowlWet).connect(bus);
    this.bowlSend = ctx.createGain();
    this.bowlSend.gain.value = 1;
    this.bowlSend.connect(bowl);

    const voices = VOICES[engine.quality.tier as keyof typeof VOICES] ?? VOICES.medium;
    // The crowd is already everywhere in the room, so it goes straight to the
    // bus with only a touch of the long tail rather than through a panner.
    this.crowd = new CrowdBed(ctx, bus, { voices, seed: 0xc0ffee });

    void ctx.resume();
  }

  // --- Graph helpers ------------------------------------------------------

  /**
   * Builds the per-voice output chain: a stereo position and distance
   * attenuation relative to the camera, plus the two reverb sends.
   *
   * `courtMix` and `bowlMix` are how much of this voice belongs to each space.
   * A dribble is nearly all court; a horn is nearly all bowl.
   */
  private voice(at: Vector3 | null, courtMix: number, bowlMix: number): GainNode | null {
    const ctx = this.ctx;
    if (!ctx || !this.dry || !this.courtSend || !this.bowlSend) return null;

    const head = ctx.createGain();
    head.gain.value = 1;

    let node: AudioNode = head;
    if (at && this.engine) {
      const cam = this.engine.camera;
      const dist = Math.max(0.6, at.distanceTo(this.listener.setFromMatrixPosition(cam.matrixWorld)));
      // Inverse-distance with a floor, then a mild air absorption roll-off so
      // the far end of the court sounds far rather than just quiet.
      head.gain.value = clamp(3.2 / dist, 0.05, 1.4);

      const pan = ctx.createStereoPanner();
      // Project onto the camera's right vector for a stable left/right image.
      this.camRight.set(1, 0, 0).applyQuaternion(cam.quaternion);
      this.relative.copy(at).sub(this.listener);
      pan.pan.value = clamp(this.relative.dot(this.camRight) / Math.max(dist, 1e-3), -0.85, 0.85);

      const air = ctx.createBiquadFilter();
      air.type = 'lowpass';
      air.frequency.value = clamp(19000 - dist * 480, 3200, 19000);

      head.connect(air).connect(pan);
      node = pan;
    }

    node.connect(this.dry);
    if (courtMix > 0) {
      const g = ctx.createGain();
      g.gain.value = courtMix;
      node.connect(g).connect(this.courtSend);
    }
    if (bowlMix > 0) {
      const g = ctx.createGain();
      g.gain.value = bowlMix;
      node.connect(g).connect(this.bowlSend);
    }
    return head;
  }

  /** A percussive amplitude envelope feeding a voice chain. */
  private env(target: GainNode, attack: number, decay: number, peak: number): GainNode | null {
    const ctx = this.ctx;
    if (!ctx) return null;
    const g = ctx.createGain();
    const t = ctx.currentTime;
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(peak, t + attack);
    g.gain.exponentialRampToValueAtTime(0.0001, t + attack + decay);
    g.connect(target);
    return g;
  }

  /**
   * A short burst of filtered noise — the basis of most contact sounds.
   * `curve` shapes the decay: 1 is linear, higher is snappier.
   */
  private noiseBurst(target: AudioNode, seconds: number, curve: number, when?: number): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const len = Math.max(1, Math.floor(ctx.sampleRate * seconds));
    const buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) {
      d[i] = (this.rng() * 2 - 1) * Math.pow(1 - i / len, curve);
    }
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(target);
    src.start(when ?? ctx.currentTime);
  }

  // --- Voices -------------------------------------------------------------

  /**
   * Ball on hardwood. Two components that have to arrive together: the leather
   * slap, which is a bright noise transient, and the ball's air cavity, which
   * is a low tone that drops in pitch as the ball deforms and recovers.
   */
  bounce(at: Vector3, speed: number, dribble: boolean): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const v = clamp01(speed / 12);
    const head = this.voice(at, dribble ? 0.85 : 0.7, dribble ? 0.12 : 0.3);
    if (!head) return;

    const body = this.env(head, 0.002, 0.16 + v * 0.1, 0.28 + v * 0.42);
    if (body) {
      const o = ctx.createOscillator();
      o.type = 'sine';
      const t = ctx.currentTime;
      // Harder bounces ring higher and drop further — the cavity is stiffer.
      o.frequency.setValueAtTime(155 + v * 90, t);
      o.frequency.exponentialRampToValueAtTime(52 + v * 18, t + 0.14);
      o.connect(body);
      o.start();
      o.stop(t + 0.3);
    }

    const slap = this.env(head, 0.001, 0.045, 0.12 + v * 0.3);
    if (slap) {
      const hp = ctx.createBiquadFilter();
      hp.type = 'bandpass';
      hp.frequency.value = 1500 + v * 2200;
      hp.Q.value = 0.7;
      hp.connect(slap);
      this.noiseBurst(hp, 0.05, 2.6);
    }
  }

  /**
   * Ball on iron. Modal synthesis: a ring's partials are inharmonic and, more
   * importantly, they decay at different rates — the high modes die in a
   * couple of hundred milliseconds while the fundamental hangs on for over a
   * second. Driving every partial from one shared envelope is what makes
   * synthetic metal sound like a bell-shaped buzzer instead of a rim.
   */
  clang(at: Vector3, speed: number): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const v = clamp01(speed / 9);
    const head = this.voice(at, 0.6, 0.55);
    if (!head) return;

    // Inharmonic ratios measured off a struck steel ring, roughly.
    const modes: [ratio: number, gain: number, decay: number][] = [
      [1.0, 1.0, 1.25],
      [1.593, 0.62, 0.72],
      [2.135, 0.44, 0.44],
      [2.296, 0.3, 0.36],
      [2.951, 0.22, 0.24],
      [4.06, 0.14, 0.15],
    ];
    const f0 = 392 * (0.97 + this.rng() * 0.06);
    const t = ctx.currentTime;

    for (const [ratio, gain, decay] of modes) {
      const o = ctx.createOscillator();
      o.type = 'sine';
      o.frequency.value = f0 * ratio;
      const g = ctx.createGain();
      const peak = gain * (0.06 + v * 0.24);
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(peak, t + 0.0015);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.0015 + decay);
      o.connect(g).connect(head);
      o.start();
      o.stop(t + decay + 0.05);
    }

    // The strike itself — a very short broadband tick that sells the contact.
    const tick = this.env(head, 0.0008, 0.02, 0.1 + v * 0.2);
    if (tick) {
      const bp = ctx.createBiquadFilter();
      bp.type = 'highpass';
      bp.frequency.value = 2400;
      bp.connect(tick);
      this.noiseBurst(bp, 0.02, 3);
    }
  }

  /**
   * Ball on backboard. Tempered glass in a steel frame: a low, dead thock with
   * very little sustain, plus the frame's own rattle. Far duller than the rim,
   * and getting that contrast right is most of what tells a listener which one
   * the ball hit.
   */
  board(at: Vector3, speed: number): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const v = clamp01(speed / 9);
    const head = this.voice(at, 0.75, 0.4);
    if (!head) return;
    const t = ctx.currentTime;

    for (const [f, gain, decay] of [
      [96, 1.0, 0.28],
      [174, 0.5, 0.18],
      [263, 0.28, 0.12],
    ] as const) {
      const o = ctx.createOscillator();
      o.type = 'sine';
      o.frequency.value = f * (0.98 + this.rng() * 0.04);
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(gain * (0.1 + v * 0.3), t + 0.003);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.003 + decay);
      o.connect(g).connect(head);
      o.start();
      o.stop(t + decay + 0.05);
    }

    const knock = this.env(head, 0.001, 0.07, 0.1 + v * 0.22);
    if (knock) {
      const lp = ctx.createBiquadFilter();
      lp.type = 'lowpass';
      lp.frequency.value = 1800;
      lp.Q.value = 0.6;
      lp.connect(knock);
      this.noiseBurst(lp, 0.08, 2.2);
    }
  }

  /**
   * The net. Not one sound — three or four nylon strands brushing the ball in
   * quick succession, each a short filtered noise chirp sweeping downward as
   * the ball drops through.
   */
  swish(at: Vector3): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const head = this.voice(at, 0.8, 0.25);
    if (!head) return;

    const strands = 3 + Math.floor(this.rng() * 3);
    for (let i = 0; i < strands; i++) {
      const delay = i * (0.012 + this.rng() * 0.02);
      const g = ctx.createGain();
      const t = ctx.currentTime + delay;
      const peak = 0.16 * (1 - i / (strands + 1));
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(peak, t + 0.004);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.1);
      g.connect(head);

      const bp = ctx.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.setValueAtTime(4200 + this.rng() * 1800, t);
      bp.frequency.exponentialRampToValueAtTime(1500, t + 0.1);
      bp.Q.value = 1.6;
      bp.connect(g);

      this.noiseBurst(bp, 0.12, 1.4, t);
    }
  }

  /**
   * Sneaker on hardwood. This is stick-slip friction: the shoe grips, releases,
   * and re-grips hundreds of times a second, which is why a squeak is a pitched
   * tone rather than a scrape. A sawtooth swept through a resonant bandpass is
   * the cheapest convincing model of it.
   */
  squeak(at: Vector3, intensity: number): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const i = clamp01(intensity);
    const head = this.voice(at, 0.9, 0.1);
    if (!head) return;

    const dur = 0.09 + this.rng() * 0.16;
    const t = ctx.currentTime;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.05 + i * 0.13, t + 0.012);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    g.connect(head);

    const o = ctx.createOscillator();
    o.type = 'sawtooth';
    const base = 620 + this.rng() * 520;
    o.frequency.setValueAtTime(base, t);
    // The rising-then-falling sweep is the characteristic shape.
    o.frequency.exponentialRampToValueAtTime(base * (1.5 + i * 0.6), t + dur * 0.35);
    o.frequency.exponentialRampToValueAtTime(base * 0.72, t + dur);

    const bp = ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.value = base * 2.1;
    bp.Q.value = 6.5;

    o.connect(bp).connect(g);
    o.start(t);
    o.stop(t + dur + 0.02);
  }

  /** A body landing. Mostly low thud plus the shoe slapping down. */
  land(at: Vector3, force: number): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const f = clamp01(force);
    const head = this.voice(at, 0.8, 0.3);
    if (!head) return;

    const g = this.env(head, 0.003, 0.16, 0.14 + f * 0.32);
    if (g) {
      const o = ctx.createOscillator();
      o.type = 'sine';
      const t = ctx.currentTime;
      o.frequency.setValueAtTime(112, t);
      o.frequency.exponentialRampToValueAtTime(44, t + 0.13);
      o.connect(g);
      o.start();
      o.stop(t + 0.25);
    }
    const slap = this.env(head, 0.001, 0.06, 0.06 + f * 0.16);
    if (slap) {
      const lp = ctx.createBiquadFilter();
      lp.type = 'lowpass';
      lp.frequency.value = 2600;
      lp.connect(slap);
      this.noiseBurst(lp, 0.07, 2.4);
    }
  }

  /** The official's whistle — two detuned tones plus the pea's warble. */
  whistle(): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const head = this.voice(null, 0.5, 0.7);
    if (!head) return;
    const t = ctx.currentTime;
    const dur = 0.55;

    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.2, t + 0.02);
    g.gain.setValueAtTime(0.2, t + dur - 0.08);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    g.connect(head);

    // The pea rattling inside is what makes a whistle a whistle: a fast
    // amplitude and pitch warble, not a clean tone.
    const lfo = ctx.createOscillator();
    lfo.type = 'sine';
    lfo.frequency.value = 34;
    const lfoGain = ctx.createGain();
    lfoGain.gain.value = 90;
    lfo.connect(lfoGain);

    for (const f of [3180, 3960]) {
      const o = ctx.createOscillator();
      o.type = 'sine';
      o.frequency.value = f;
      lfoGain.connect(o.frequency);
      const og = ctx.createGain();
      og.gain.value = 0.5;
      o.connect(og).connect(g);
      o.start(t);
      o.stop(t + dur + 0.05);
    }
    lfo.start(t);
    lfo.stop(t + dur + 0.05);
  }

  /** End-of-period horn. Heavy on the long reverb — that is the arena sound. */
  horn(seconds = 1.6): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const head = this.voice(null, 0.2, 1);
    if (!head) return;
    const t = ctx.currentTime;

    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.3, t + 0.04);
    g.gain.setValueAtTime(0.3, t + seconds - 0.12);
    g.gain.exponentialRampToValueAtTime(0.0001, t + seconds);
    g.connect(head);

    // A stack of harmonics with a slight beat between two nearly-tuned voices.
    for (const [f, gain] of [
      [138, 1],
      [139.6, 0.9],
      [276, 0.5],
      [414, 0.3],
      [552, 0.16],
    ] as const) {
      const o = ctx.createOscillator();
      o.type = 'sawtooth';
      o.frequency.value = f;
      const og = ctx.createGain();
      og.gain.value = gain * 0.22;
      const lp = ctx.createBiquadFilter();
      lp.type = 'lowpass';
      lp.frequency.value = 1400;
      o.connect(og).connect(lp).connect(g);
      o.start(t);
      o.stop(t + seconds + 0.05);
    }
  }

  /** Shot-clock and end-of-quarter beeps. */
  beep(frequency = 880, duration = 0.09): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const head = this.voice(null, 0.3, 0.4);
    if (!head) return;
    const g = this.env(head, 0.004, duration, 0.14);
    if (!g) return;
    const o = ctx.createOscillator();
    o.type = 'square';
    o.frequency.value = frequency;
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = 3200;
    o.connect(lp).connect(g);
    o.start();
    o.stop(ctx.currentTime + duration + 0.05);
  }

  // --- Wiring -------------------------------------------------------------

  private subscribe(engine: Engine): void {
    const bus = engine.bus;

    bus.on('floorBounce', ({ speed, position }) => this.bounce(position, speed, false));
    bus.on('dribble', ({ speed, position }) => {
      // Two dribbles cannot land within 90 ms of each other; anything closer is
      // the physics chattering, not a second bounce.
      const now = this.ctx?.currentTime ?? 0;
      if (now - this.lastDribble < 0.09) return;
      this.lastDribble = now;
      this.bounce(position, speed, true);
    });
    bus.on('rimContact', ({ speed, position }) => this.clang(position, speed));
    bus.on('boardContact', ({ speed, position }) => this.board(position, speed));
    bus.on('netSwish', ({ position }) => this.swish(position));
    bus.on('sneakerSqueak', ({ position, intensity }) => this.squeak(position, intensity));
    bus.on('jumpLand', ({ position, force }) => this.land(position, force));

    bus.on('scored', ({ points, swish }) => {
      this.crowd?.spike(points >= 3 ? (swish ? 0.95 : 0.8) : 0.55);
    });
    bus.on('block', () => this.crowd?.spike(0.85));
    bus.on('steal', () => this.crowd?.spike(0.6));
    bus.on('dunk', ({ power }) => this.crowd?.spike(clamp(0.7 + power * 0.3, 0, 1)));
    bus.on('crossover', ({ broke }) => {
      if (broke) this.crowd?.spike(0.7);
    });
    bus.on('missed', () => this.crowd?.spike(0.2));

    bus.on('clockTick', ({ shotClock }) => {
      if (shotClock <= 5 && shotClock > 0 && Number.isInteger(shotClock)) this.beep(720, 0.07);
    });
    bus.on('quarterEnd', () => this.horn());
    bus.on('gameEnd', () => {
      this.horn(2.4);
      this.crowd?.spike(1);
    });
  }

  update(dt: number, _alpha: number, engine: Engine): void {
    if (!this.crowd) return;
    // A crowd is never fully silent; the floor rises with the game clock later,
    // once GameSystem drives it.
    this.crowd.settle(dt, 0.08);
    this.crowd.update(dt);
    void engine;
  }

  /** Overall output level, 0..1. */
  setVolume(v: number): void {
    if (this.master) this.master.gain.value = clamp01(v) * 0.75;
  }

  dispose(): void {
    this.crowd?.dispose();
    this.crowd = null;
    void this.ctx?.close();
    this.ctx = null;
  }
}
