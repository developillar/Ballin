/**
 * Gameplay VFX.
 *
 * Two instanced pools — one alpha-blended for anything made of matter (floor
 * dust, sweat, confetti) and one additive for anything made of light (haze
 * motes, the glint off a droplet, the flash behind a swish). Two draw calls
 * total, which is the whole reason for the split.
 *
 * The restraint here is deliberate and it is the point. Real broadcast
 * basketball has no sparks off the rim, no shockwave rings, no speed lines. It
 * has dust hanging in the light, sweat catching a highlight, and confetti when
 * something is actually worth celebrating. Adding the arcade layer on top is
 * the single fastest way to make a frame stop reading as a broadcast, so the
 * effects below are all things a camera would really have caught.
 *
 * Other systems drive this through the event bus, or by calling the emitters
 * directly via `engine.get('vfx')` — nothing here reaches into another system.
 */

import { AdditiveBlending, Group, NormalBlending, Vector3, type DataTexture } from 'three';
import type { Engine, System } from '../core/Engine';
import { COURT } from '../core/Constants';
import { clamp, clamp01, makeRng } from '../core/MathX';
import { HazeField, ParticlePool, makeParticleAtlas } from './particles';

/** Pool capacities per tier. Motes are the standing cost; the rest is burst. */
const CAPACITY = {
  low: { matter: 0, light: 0, motes: 0 },
  medium: { matter: 260, light: 220, motes: 90 },
  high: { matter: 620, light: 520, motes: 260 },
  ultra: { matter: 1100, light: 900, motes: 420 },
} as const;

/** Maple dust, sampled off the court bake — never pure white. */
const DUST: [number, number, number] = [0.68, 0.6, 0.49];
const SWEAT: [number, number, number] = [0.82, 0.86, 0.9];
const HAZE: [number, number, number] = [0.85, 0.82, 0.72];
const CONFETTI: [number, number, number][] = [
  [0.85, 0.16, 0.2],
  [0.95, 0.75, 0.15],
  [0.9, 0.9, 0.92],
  [0.12, 0.28, 0.62],
];

export class VfxSystem implements System {
  readonly name = 'vfx';
  readonly order = 70;

  group = new Group();

  private atlas: DataTexture | null = null;
  private matter: ParticlePool | null = null;
  private light: ParticlePool | null = null;
  private haze: HazeField | null = null;

  private readonly rng = makeRng(0x1a2b3c4d);
  private readonly tmp = new Vector3();
  private readonly tmpVel = new Vector3();
  private elapsed = 0;

  init(engine: Engine): void {
    this.group.name = 'vfx';
    engine.scene.add(this.group);

    if (!engine.quality.particles) return;

    const cap = CAPACITY[engine.quality.tier as keyof typeof CAPACITY] ?? CAPACITY.medium;
    if (cap.matter === 0) return;

    this.atlas = makeParticleAtlas();
    this.matter = new ParticlePool(cap.matter, this.atlas, NormalBlending);
    this.light = new ParticlePool(cap.light, this.atlas, AdditiveBlending);
    this.group.add(this.matter.mesh, this.light.mesh);

    if (cap.motes > 0) {
      this.haze = new HazeField(this.atlas, {
        count: cap.motes,
        // Filling the volume the overhead banks actually light — court width
        // plus the apron, up to just under the rafters.
        extent: new Vector3(COURT.halfLength * 0.85, 7.5, COURT.halfWidth + 1.5),
        baseY: 1.2,
        color: HAZE,
        alpha: 0.16,
        size: 0.012,
      });
      this.group.add(this.haze.mesh);
    }

    this.subscribe(engine);
  }

  // --- Emitters -----------------------------------------------------------
  // Public so gameplay can call them for things that are not worth an event.

  /**
   * Dust kicked off the floor. `force` 0..1 — a dribble is about 0.15, a hard
   * landing 0.7, a dunk landing 1.
   */
  dust(at: Vector3, force: number): void {
    if (!this.matter) return;
    const f = clamp01(force);
    this.matter.emit({
      count: Math.round(2 + f * 16),
      position: this.tmp.set(at.x, Math.max(at.y, 0.008), at.z),
      // Dust goes out, not up: it is displaced air, not an explosion.
      velocity: this.tmpVel.set(0, 0.22 + f * 0.5, 0),
      speed: 0.5 + f * 1.9,
      spread: 1,
      life: [0.45, 0.35 + f * 1.1],
      size: [0.03 + f * 0.03, 0.09 + f * 0.16],
      color: DUST,
      colorJitter: 0.25,
      grow: 2.6 + f * 1.6,
      gravity: 0.35,
      drag: 0.02,
      alpha: 0.1 + f * 0.24,
      spin: 0.9,
      sprite: 'puff',
      floor: 0.004,
    });
  }

  /** Sweat thrown off a player. `dir` is the throw direction, usually the limb's velocity. */
  sweat(at: Vector3, dir: Vector3, amount: number): void {
    if (!this.matter || !this.light) return;
    const a = clamp01(amount);
    const count = Math.round(1 + a * 7);
    const vel = this.tmpVel.copy(dir).multiplyScalar(0.35);
    this.matter.emit({
      count,
      position: at,
      velocity: vel,
      speed: 0.8 + a * 1.6,
      spread: 0.6,
      life: [0.35, 0.75],
      size: [0.006, 0.014],
      color: SWEAT,
      colorJitter: 0.1,
      gravity: 7.2,
      drag: 0.25,
      alpha: 0.5,
      // Droplets elongate along travel; a round droplet at speed reads as a bead.
      stretch: 2.4,
      sprite: 'streak',
      floor: null,
    });
    // A matching additive glint, at a third the count, so a few of them catch
    // the overhead banks the way real sweat does.
    this.light.emit({
      count: Math.max(1, Math.round(count / 3)),
      position: at,
      velocity: vel,
      speed: 0.8 + a * 1.6,
      spread: 0.6,
      life: [0.3, 0.6],
      size: [0.008, 0.018],
      color: [0.9, 0.93, 1],
      gravity: 7.2,
      drag: 0.25,
      alpha: 0.32,
      sprite: 'dot',
    });
  }

  /** The brief flare of light through the net on a clean make. */
  swishFlash(at: Vector3): void {
    if (!this.light) return;
    this.light.emit({
      count: 14,
      position: at,
      speed: 0.55,
      spread: 1,
      life: [0.18, 0.34],
      size: [0.03, 0.08],
      color: [1, 0.94, 0.82],
      grow: 2.2,
      gravity: 1.4,
      drag: 0.1,
      alpha: 0.34,
      sprite: 'dot',
    });
  }

  /** Confetti. Reserved for moments that earn it. */
  confetti(at: Vector3, count: number): void {
    if (!this.matter) return;
    for (const color of CONFETTI) {
      this.matter.emit({
        count: Math.round(count / CONFETTI.length),
        position: at,
        velocity: this.tmpVel.set(0, 1.6, 0),
        speed: 2.4,
        spread: 1,
        life: [1.8, 3.6],
        size: [0.03, 0.055],
        color,
        colorJitter: 0.2,
        // Heavy drag and light gravity is what makes paper flutter instead of
        // falling like gravel.
        gravity: 1.5,
        drag: 0.35,
        alpha: 0.95,
        spin: 7,
        sprite: 'chip',
        floor: 0.01,
      });
    }
  }

  // --- Wiring -------------------------------------------------------------

  private subscribe(engine: Engine): void {
    const bus = engine.bus;

    bus.on('floorBounce', ({ speed, position }) => this.dust(position, clamp(speed / 9, 0.05, 0.5)));
    bus.on('dribble', ({ speed, position }) => this.dust(position, clamp(speed / 14, 0.04, 0.22)));
    bus.on('jumpLand', ({ position, force }) => this.dust(position, clamp(force, 0.25, 1)));
    bus.on('sneakerSqueak', ({ position, intensity }) => this.dust(position, intensity * 0.28));

    // `netSwish` carries the position the ball cleared the net at, which is the
    // only place the flash belongs. `scored` deliberately gets no handler: it
    // has no position on it, and guessing one puts light where nothing happened.
    bus.on('netSwish', ({ position }) => this.swishFlash(position));

    bus.on('gameEnd', () => this.celebrate());
  }

  /** Confetti drop over both halves, from the rafters. */
  private celebrate(): void {
    for (let i = 0; i < 6; i++) {
      this.confetti(
        this.tmp.set(
          (this.rng() * 2 - 1) * COURT.halfLength * 0.8,
          11,
          (this.rng() * 2 - 1) * COURT.halfWidth * 0.5,
        ),
        90,
      );
    }
  }

  update(dt: number, _alpha: number, _engine: Engine): void {
    this.elapsed += dt;
    this.matter?.update(dt);
    this.light?.update(dt);
    this.haze?.update(this.elapsed);
  }

  /** Live particle counts, for the debug overlay. */
  get counts(): { matter: number; light: number } {
    return { matter: this.matter?.liveCount ?? 0, light: this.light?.liveCount ?? 0 };
  }

  dispose(): void {
    this.matter?.dispose();
    this.light?.dispose();
    this.haze?.dispose();
    this.atlas?.dispose();
  }
}
