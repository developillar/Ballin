/**
 * Secondary motion.
 *
 * Rubric §9.2: *"A body with zero secondary motion is rigid even when the
 * primary animation is perfect."* Everything in here is small — most of it is
 * under two degrees — and collectively it is the difference between an athlete
 * and a rigged doll standing very still.
 *
 * Four layers, all additive on top of whatever the blend tree produced:
 *
 *   - **Breathing.** Ribcage expansion, a shoulder rise, a tiny head lift, on a
 *     rate that climbs with exertion and fatigue and stays audible-slow at rest.
 *   - **Fatigue.** A tired player rounds the shoulders, drops the hands, lets
 *     the head hang and shortens up. Steady-state, not oscillating.
 *   - **Micro-fidgets.** Seeded, low-amplitude, multi-octave noise on the head,
 *     hands and shoulders so no two players are ever in the same pose and no
 *     joint is ever *exactly* still.
 *   - **Impulses.** Spring-damped reactions to physical events: contact from a
 *     defender, and the shock that travels up the body every time a foot lands.
 *
 * All of it fades out under a running action so a jump shot is not fighting a
 * breath cycle at the release frame.
 */

import { Euler, Quaternion } from 'three';
import { BONE_INDEX, type BoneName } from '../entities/Skeleton';
import { identityPose, makePose, type Pose } from './Pose';
import { clamp, clamp01, makeRng } from '../core/MathX';

const _q = new Quaternion();
const _euler = new Euler();

/** Composes a small rotation onto a bone of an additive pose. */
export function addBone(pose: Pose, name: BoneName, x: number, y: number, z: number): void {
  if (x === 0 && y === 0 && z === 0) return;
  pose.rotations[BONE_INDEX[name]].multiply(_q.setFromEuler(_euler.set(x, y, z)));
}

const setBone = addBone;

/**
 * Critically-ish damped second-order spring. `freq` is in Hz, `damping` 1 is
 * critical — under 1 overshoots, which is exactly what a body does when it gets
 * hit.
 */
class Spring {
  value = 0;
  velocity = 0;

  step(dt: number, target: number, freq: number, damping: number): number {
    const w = 2 * Math.PI * freq;
    const a = -w * w * (this.value - target) - 2 * damping * w * this.velocity;
    this.velocity += a * dt;
    this.value += this.velocity * dt;
    return this.value;
  }

  kick(amount: number): void {
    this.velocity += amount;
  }
}

/** A contact knock, in the character's own frame. */
export interface Impulse {
  /** Forward/back, in body space. Positive pushes the chest back. */
  back: number;
  /** Lateral, in body space. Positive pushes toward the character's left. */
  side: number;
  /** Vertical — an undercut or a landing. Positive is upward. */
  lift: number;
}

export class SecondaryMotion {
  readonly pose: Pose = makePose();

  /** Extra root drop, metres. Landing shock and heavy breathing both move it. */
  rootDrop = 0;

  private t = 0;
  private breathPhase = 0;
  private readonly seed: number;
  private readonly fidget: number[] = [];

  private readonly leanBack = new Spring();
  private readonly leanSide = new Spring();
  private readonly shockDrop = new Spring();
  private readonly armFlail = new Spring();

  private exertion = 0;

  constructor(seed: number) {
    this.seed = seed;
    const rng = makeRng(seed);
    for (let i = 0; i < 10; i++) this.fidget.push(rng() * Math.PI * 2);
  }

  /** A physical knock. `strength` 0..1. */
  hit(impulse: Impulse, strength: number): void {
    const s = clamp01(strength);
    this.leanBack.kick(impulse.back * s * 9);
    this.leanSide.kick(impulse.side * s * 9);
    this.shockDrop.kick(-Math.abs(impulse.lift) * s * 1.1);
    this.armFlail.kick(s * 7 * (impulse.side >= 0 ? 1 : -1));
  }

  /** A foot struck the floor. Sends a small shock up the spine. */
  footStrike(force: number): void {
    const f = clamp01(force);
    this.shockDrop.kick(-0.16 - f * 0.5);
    this.leanBack.kick(f * 0.5);
  }

  /**
   * @param dt        seconds
   * @param speed     ground speed, m/s
   * @param fatigue   0..1
   * @param actionWeight how much a one-shot action currently owns the body
   */
  update(dt: number, speed: number, fatigue: number, actionWeight = 0): Pose {
    this.t += dt;
    const fat = clamp01(fatigue);
    const busy = clamp01(actionWeight);
    // Exertion trails speed — a player who just stopped sprinting is still
    // breathing hard.
    const want = clamp01(speed / 7) * 0.8 + fat * 0.5;
    this.exertion += (want - this.exertion) * clamp01(dt * (want > this.exertion ? 1.6 : 0.35));

    identityPose(this.pose);

    // --- Breathing --------------------------------------------------------
    const rate = 0.24 + this.exertion * 0.72; // Hz: ~14 → ~58 breaths/min
    this.breathPhase += dt * rate;
    const breath = Math.sin(this.breathPhase * Math.PI * 2);
    // Sharper inhale than exhale.
    const shaped = breath > 0 ? Math.pow(breath, 0.75) : -Math.pow(-breath, 1.35);
    const depth = (0.5 + this.exertion * 1.7) * (1 - busy * 0.65);
    setBone(this.pose, 'chest', -0.012 * shaped * depth, 0, 0);
    setBone(this.pose, 'upperChest', -0.016 * shaped * depth, 0, 0);
    setBone(this.pose, 'clavicleL', -0.02 * shaped * depth, 0, -0.014 * shaped * depth);
    setBone(this.pose, 'clavicleR', -0.02 * shaped * depth, 0, 0.014 * shaped * depth);
    setBone(this.pose, 'neck', 0.008 * shaped * depth, 0, 0);
    setBone(this.pose, 'spine', 0.006 * shaped * depth, 0, 0);

    // --- Fatigue ----------------------------------------------------------
    // Shoulders round forward, arms hang, chin drops, base softens. Scaled
    // down while sprinting: nobody runs with their arms dangling.
    const droop = fat * (1 - clamp01(speed / 6) * 0.55) * (1 - busy * 0.8);
    if (droop > 0.001) {
      setBone(this.pose, 'spine', 0.07 * droop, 0, 0);
      setBone(this.pose, 'chest', 0.05 * droop, 0, 0);
      setBone(this.pose, 'upperChest', 0.04 * droop, 0, 0);
      setBone(this.pose, 'neck', 0.1 * droop, 0, 0);
      setBone(this.pose, 'head', 0.11 * droop, 0, 0);
      setBone(this.pose, 'clavicleL', 0.07 * droop, 0, 0.09 * droop);
      setBone(this.pose, 'clavicleR', 0.07 * droop, 0, -0.09 * droop);
      setBone(this.pose, 'upperArmL', 0.12 * droop, 0, 0.1 * droop);
      setBone(this.pose, 'upperArmR', 0.12 * droop, 0, -0.1 * droop);
      setBone(this.pose, 'foreArmL', -0.16 * droop, 0, 0);
      setBone(this.pose, 'foreArmR', -0.16 * droop, 0, 0);
    }

    // --- Micro-fidgets ----------------------------------------------------
    // Two octaves, incommensurate rates, so it never reads as a loop.
    const idleness = (1 - clamp01(speed / 2.2)) * (1 - busy);
    const n = (i: number, f1: number, f2: number): number =>
      Math.sin(this.t * f1 + this.fidget[i]) * 0.68 + Math.sin(this.t * f2 + this.fidget[i] * 1.7) * 0.32;
    const micro = 1 - busy * 0.55;
    setBone(
      this.pose,
      'head',
      n(0, 0.37, 1.13) * 0.012 * micro,
      n(1, 0.29, 0.83) * 0.03 * (0.35 + idleness) * micro,
      n(2, 0.41, 1.31) * 0.009 * micro,
    );
    setBone(this.pose, 'neck', n(3, 0.31, 0.97) * 0.008 * micro, n(1, 0.29, 0.83) * 0.012 * micro, 0);
    setBone(this.pose, 'handL', n(4, 0.83, 2.11) * 0.05 * micro, 0, n(5, 0.61, 1.77) * 0.04 * micro);
    setBone(this.pose, 'handR', n(6, 0.79, 2.03) * 0.05 * micro, 0, n(7, 0.67, 1.83) * 0.04 * micro);
    setBone(this.pose, 'upperArmL', n(8, 0.23, 0.71) * 0.016 * idleness * micro, 0, 0);
    setBone(this.pose, 'upperArmR', n(9, 0.27, 0.67) * 0.016 * idleness * micro, 0, 0);
    // Weight shift: the free hip drifts while standing.
    const shift = n(0, 0.19, 0.53) * idleness;
    setBone(this.pose, 'hips', 0, shift * 0.02, shift * 0.035);

    // --- Impulses ---------------------------------------------------------
    const back = this.leanBack.step(dt, 0, 1.55, 0.42);
    const side = this.leanSide.step(dt, 0, 1.7, 0.4);
    const drop = this.shockDrop.step(dt, 0, 2.5, 0.55);
    const flail = this.armFlail.step(dt, 0, 1.35, 0.36);
    if (Math.abs(back) > 1e-4 || Math.abs(side) > 1e-4) {
      const b = clamp(back, -0.5, 0.5);
      const s = clamp(side, -0.5, 0.5);
      setBone(this.pose, 'hips', b * 0.12, s * 0.1, s * 0.14);
      setBone(this.pose, 'spine', b * 0.3, s * 0.16, s * 0.26);
      setBone(this.pose, 'chest', b * 0.22, s * 0.12, s * 0.2);
      setBone(this.pose, 'upperChest', b * 0.16, s * 0.1, s * 0.15);
      setBone(this.pose, 'neck', -b * 0.2, -s * 0.1, -s * 0.14);
      setBone(this.pose, 'head', -b * 0.26, -s * 0.16, -s * 0.2);
    }
    if (Math.abs(flail) > 1e-4) {
      const f = clamp(flail, -0.6, 0.6);
      setBone(this.pose, 'upperArmL', -f * 0.34, 0, -f * 0.3);
      setBone(this.pose, 'upperArmR', -f * 0.34, 0, -f * 0.3);
      setBone(this.pose, 'foreArmL', -Math.abs(f) * 0.2, 0, 0);
      setBone(this.pose, 'foreArmR', -Math.abs(f) * 0.2, 0, 0);
    }

    this.rootDrop = clamp(drop, -0.12, 0.05) * 0.16;
    this.pose.rootOffset.y += this.rootDrop;
    return this.pose;
  }
}
