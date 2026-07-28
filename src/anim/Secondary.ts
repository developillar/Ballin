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
 * hit, and what a hard stop has to do to read as weight (rubric §9.2).
 *
 * Sub-stepped. Semi-implicit Euler on a spring goes unstable once `ω·dt`
 * approaches 1, and a 2.5 Hz spring at a 100 ms frame (the animator's clamp) is
 * already there — one hitched frame would have thrown the whole body. Splitting
 * the step keeps it bounded no matter what frame time it is handed.
 */
export class Spring {
  value = 0;
  velocity = 0;

  step(dt: number, target: number, freq: number, damping: number): number {
    const w = 2 * Math.PI * freq;
    const n = Math.min(8, Math.max(1, Math.ceil((w * dt) / 0.25)));
    const h = dt / n;
    for (let i = 0; i < n; i++) {
      const a = -w * w * (this.value - target) - 2 * damping * w * this.velocity;
      this.velocity += a * h;
      this.value += this.velocity * h;
    }
    return this.value;
  }

  kick(amount: number): void {
    this.velocity += amount;
  }
}

/**
 * Real motion, in the character's own frame, sampled from the rig rather than
 * declared by gameplay. This is what the lag layer is driven from.
 */
export interface MotionDrive {
  /** Longitudinal acceleration, m/s². Positive is speeding up. */
  accelForward: number;
  /** Lateral (centripetal) acceleration, m/s². Positive pushes to the left. */
  accelSide: number;
  /** Yaw rate of the body, rad/s. Positive turns to the character's right. */
  turnRate: number;
  /** Vertical acceleration of the pelvis, m/s². */
  accelUp: number;
}

const NO_DRIVE: MotionDrive = { accelForward: 0, accelSide: 0, turnRate: 0, accelUp: 0 };

/**
 * The trailing mass, as a spring.
 *
 * Rubric §3.4/§3.5/§9.2 asks for 80–160 ms of lag *with a small overshoot*.
 * 1.7 Hz at ζ = 0.55 reaches half of a step in ~90 ms and overshoots it by
 * ~13% — inside the band, and it rings once on the way back rather than
 * decaying, which is what a body actually does when it stops dead.
 * Everything outboard of the sternum — neck, head, shoulder girdle, arms — is a
 * mass hung off a compliant spine and arrives late by about this much.
 */
const LAG_FREQ = 1.7;
const LAG_ZETA = 0.55;

const GRAVITY = 9.80665;

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

  /** How far the mass above the sternum has caught up with the drive. */
  private readonly lagFwd = new Spring();
  private readonly lagSide = new Spring();
  private readonly lagYaw = new Spring();
  private readonly lagUp = new Spring();
  /** Trail amounts actually applied last frame, for diagnostics. */
  trail = { pitch: 0, roll: 0, yaw: 0 };

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
   * @param drive     real motion in body space; omit for a static body
   */
  update(dt: number, speed: number, fatigue: number, actionWeight = 0, drive: MotionDrive = NO_DRIVE): Pose {
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

    // --- Trailing mass ----------------------------------------------------
    // The pelvis is what the ground drives. Everything above the sternum is a
    // mass on a compliant spine and arrives ~120 ms late; the arms, hung off
    // that, arrive later still. Track a lagged copy of the drive and rotate by
    // the *difference* — that difference is precisely the part of the motion the
    // upper body has not caught up with, so it is zero in steady state (no
    // double-counting of the balance lean the momentum layer already applies)
    // and largest exactly where the rubric says it must be visible: the frame a
    // sprint turns into a stop.
    //
    // The trunk-balance lean is the animator's job; this layer only ever touches
    // what hangs off the trunk.
    // Each channel is a second-order chase rather than an exponential one, so
    // the lag comes with the small overshoot the rubric asks for instead of a
    // dead-beat decay: `LAG_FREQ`/`LAG_ZETA` give a 50% rise at ~90 ms and a
    // ~13% overshoot when the drive steps back to zero at the end of a stop.
    const lagScale = 1 - busy * 0.55;
    const tFwd =
      -this.lagFwd.step(dt, clamp(drive.accelForward / GRAVITY, -0.9, 0.9), LAG_FREQ, LAG_ZETA) * lagScale;
    const tSide =
      -this.lagSide.step(dt, clamp(drive.accelSide / GRAVITY, -0.9, 0.9), LAG_FREQ, LAG_ZETA) * lagScale;
    const tYaw = -this.lagYaw.step(dt, clamp(drive.turnRate, -4.5, 4.5), LAG_FREQ * 0.9, LAG_ZETA) * 0.055 * lagScale;
    const tUp = -this.lagUp.step(dt, clamp(drive.accelUp / GRAVITY, -1.2, 1.2), LAG_FREQ * 1.2, LAG_ZETA) * lagScale;
    this.trail.pitch = tFwd;
    this.trail.roll = tSide;
    this.trail.yaw = tYaw;
    if (Math.abs(tFwd) > 1e-4 || Math.abs(tSide) > 1e-4 || Math.abs(tYaw) > 1e-4) {
      // Spine bones pitch forward on +X; the head counter-rotates against the
      // shoulders on Y, which §9.2 calls out by name.
      setBone(this.pose, 'neck', tFwd * 0.12, tYaw * 0.45, tSide * 0.1);
      setBone(this.pose, 'head', tFwd * 0.22, tYaw, tSide * 0.19);
      // The shoulder girdle is the hinge the arms swing off, so it takes a
      // little and passes the rest on.
      setBone(this.pose, 'clavicleL', tFwd * 0.07, tYaw * 0.3, tSide * 0.06);
      setBone(this.pose, 'clavicleR', tFwd * 0.07, tYaw * 0.3, tSide * 0.06);
      // Limb bones hang on −Y: a *forward* trail is negative X.
      setBone(this.pose, 'upperArmL', -tFwd * 0.34, 0, -tSide * 0.26);
      setBone(this.pose, 'upperArmR', -tFwd * 0.34, 0, -tSide * 0.26);
      setBone(this.pose, 'foreArmL', -Math.abs(tFwd) * 0.16, 0, 0);
      setBone(this.pose, 'foreArmR', -Math.abs(tFwd) * 0.16, 0, 0);
      setBone(this.pose, 'handL', -tFwd * 0.2, 0, -tSide * 0.16);
      setBone(this.pose, 'handR', -tFwd * 0.2, 0, -tSide * 0.16);
    }
    // A dropping pelvis leaves the head high and vice versa — this is the same
    // lag on the vertical channel, and it is what makes a landing read.
    if (Math.abs(tUp) > 1e-4) {
      setBone(this.pose, 'neck', -tUp * 0.05, 0, 0);
      setBone(this.pose, 'head', -tUp * 0.07, 0, 0);
    }

    // --- Impulses ---------------------------------------------------------
    // Discrete knocks only. The *continuous* balance lean belongs to the
    // animator's momentum layer, which derives it from θ ≈ atan(a/g); driving
    // these springs from acceleration as well would have the two fighting over
    // the same six bones.
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
