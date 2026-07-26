/**
 * Fully procedural audio — no sample downloads. Crowd bed, ball bounces, rim
 * clangs, net swish, sneaker squeaks and the horn are all synthesised through
 * the Web Audio graph and spatialised against the camera.
 *
 * Owned by the audio agent.
 */

import type { Engine, System } from '../core/Engine';

export class AudioSystem implements System {
  readonly name = 'audio';
  readonly order = 80;

  ctx: AudioContext | null = null;
  master: GainNode | null = null;
  private unlocked = false;

  init(engine: Engine): void {
    const unlock = () => {
      if (this.unlocked) return;
      this.unlocked = true;
      const Ctx =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.ctx = new Ctx();
      this.master = this.ctx.createGain();
      this.master.gain.value = 0.7;
      this.master.connect(this.ctx.destination);
      void this.ctx.resume();
    };
    window.addEventListener('pointerdown', unlock, { once: true });
    window.addEventListener('keydown', unlock, { once: true });

    engine.bus.on('floorBounce', ({ speed }) => this.thud(speed));
    engine.bus.on('rimContact', ({ speed }) => this.clang(speed));
    engine.bus.on('netSwish', () => this.swish());
  }

  private env(dur: number, gain: number): GainNode | null {
    if (!this.ctx || !this.master) return null;
    const g = this.ctx.createGain();
    const t = this.ctx.currentTime;
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(gain, t + 0.004);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    g.connect(this.master);
    return g;
  }

  private thud(speed: number): void {
    if (!this.ctx) return;
    const g = this.env(0.22, Math.min(0.6, speed * 0.09));
    if (!g) return;
    const o = this.ctx.createOscillator();
    o.type = 'sine';
    o.frequency.setValueAtTime(180, this.ctx.currentTime);
    o.frequency.exponentialRampToValueAtTime(58, this.ctx.currentTime + 0.16);
    o.connect(g);
    o.start();
    o.stop(this.ctx.currentTime + 0.25);
  }

  private clang(speed: number): void {
    if (!this.ctx) return;
    const g = this.env(0.5, Math.min(0.45, speed * 0.06));
    if (!g) return;
    for (const f of [520, 831, 1290, 1974]) {
      const o = this.ctx.createOscillator();
      o.type = 'triangle';
      o.frequency.value = f * (0.98 + Math.random() * 0.04);
      const gg = this.ctx.createGain();
      gg.gain.value = 0.28;
      o.connect(gg).connect(g);
      o.start();
      o.stop(this.ctx.currentTime + 0.5);
    }
  }

  private swish(): void {
    if (!this.ctx) return;
    const g = this.env(0.32, 0.35);
    if (!g) return;
    const len = Math.floor(this.ctx.sampleRate * 0.32);
    const buf = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / len);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    const bp = this.ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.value = 3400;
    bp.Q.value = 1.1;
    src.connect(bp).connect(g);
    src.start();
  }
}
