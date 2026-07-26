/**
 * Pose representation, clips and blending.
 *
 * There is no animation file format here — every clip is authored in code as a
 * set of keyframes over joint rotations. That sounds like a limitation and
 * mostly it is a feature: clips can be parameterised (shoot from anywhere on
 * the floor, dunk off either foot) and blended continuously rather than
 * snapping between baked takes.
 *
 * ## Rotation convention (verified against `entities/Skeleton.ts`)
 *
 * Every bone is built with an identity rest rotation, so **each bone's local
 * frame is the character frame**: +X is the character's *left*, +Y is up, +Z is
 * forward. What differs between bones is which way the bone body points:
 *
 *   - **Limb bones** (thigh, shin, foot, upperArm, foreArm, hand) hang along
 *     local **−Y**. A rotation about +X carries +Y toward +Z, therefore it
 *     carries −Y toward −Z: **positive X swings a limb backward, negative X
 *     swings it forward.**  Knee flexion (heel toward the seat) is therefore
 *     **positive X on the shin**; elbow flexion is **negative X on the
 *     foreArm** (the hand travels forward).
 *   - **Spine bones** (hips, spine, chest, upperChest, neck, head) stack along
 *     local **+Y**, so the sign flips: **positive X pitches the torso
 *     forward.**
 *   - Yaw is Y (positive = turning to the character's right, because +X is
 *     left), roll is Z (positive raises the +X / left side).
 *
 * Both statements come from the same right-handed rotation; only the direction
 * the bone body points differs. Getting this backwards inverts every knee in
 * the library, so it is spelled out here rather than left to memory.
 *
 * ## Interpolation
 *
 * Keys are interpolated with a **C1 cubic Hermite** over the Euler channels by
 * default (`interp: 'spline'`). The older behaviour — a smootherstep ease
 * between each adjacent pair — makes every joint decelerate to a dead stop on
 * every key, which is the single most "rigged doll" thing a clip player can do.
 * The spline keeps velocity continuous across keys and lets a well-placed key
 * overshoot slightly, which is what follow-through actually is.
 */

import { Quaternion, Vector3 } from 'three';
import { BONES, type BoneName, BONE_INDEX, type BuiltSkeleton } from '../entities/Skeleton';
import { clamp, clamp01, smootherstep } from '../core/MathX';

/** A full-body pose: one rotation per bone, plus a root offset. */
export interface Pose {
  rotations: Quaternion[];
  rootOffset: Vector3;
  /** Extra pelvis rotation applied on top of `rotations[hips]`. */
  rootYaw: number;
}

export function makePose(): Pose {
  return {
    rotations: BONES.map(() => new Quaternion()),
    rootOffset: new Vector3(),
    rootYaw: 0,
  };
}

export function copyPose(dst: Pose, src: Pose): Pose {
  for (let i = 0; i < dst.rotations.length; i++) dst.rotations[i].copy(src.rotations[i]);
  dst.rootOffset.copy(src.rootOffset);
  dst.rootYaw = src.rootYaw;
  return dst;
}

export function identityPose(dst: Pose): Pose {
  for (let i = 0; i < dst.rotations.length; i++) dst.rotations[i].identity();
  dst.rootOffset.set(0, 0, 0);
  dst.rootYaw = 0;
  return dst;
}

/** Spherical blend of two poses into `out`. */
export function blendPose(out: Pose, a: Pose, b: Pose, t: number): Pose {
  const k = clamp01(t);
  for (let i = 0; i < out.rotations.length; i++) {
    out.rotations[i].copy(a.rotations[i]).slerp(b.rotations[i], k);
  }
  out.rootOffset.lerpVectors(a.rootOffset, b.rootOffset, k);
  out.rootYaw = a.rootYaw + (b.rootYaw - a.rootYaw) * k;
  return out;
}

/**
 * Per-bone weighted blend — the *override* path. A jump shot does not ride on
 * top of a run cycle, it replaces it from the hips up (and, for the legs, from
 * the gather onward), so an override needs a masked slerp rather than the
 * additive multiply below.
 */
export function blendPoseMasked(
  out: Pose,
  base: Pose,
  target: Pose,
  weight: number,
  mask?: BoneMask,
  rootWeight = weight,
): Pose {
  const w = clamp01(weight);
  for (let i = 0; i < out.rotations.length; i++) {
    const k = w * (mask ? mask[i] : 1);
    out.rotations[i].copy(base.rotations[i]);
    if (k > 0.0001) out.rotations[i].slerp(target.rotations[i], k);
  }
  const rw = clamp01(rootWeight);
  out.rootOffset.lerpVectors(base.rootOffset, target.rootOffset, rw);
  out.rootYaw = base.rootYaw + (target.rootYaw - base.rootYaw) * rw;
  return out;
}

const _scratchQ = new Quaternion();
const IDENTITY = new Quaternion();

/**
 * Additive blend: layers `add` onto `base` at `weight`. Used for things that
 * ride on top of locomotion — breathing, a head turn, a shoulder shrug on
 * contact, the dribble pump, a lean into a screen.
 */
export function addPose(out: Pose, base: Pose, add: Pose, weight: number, mask?: BoneMask): Pose {
  const w = clamp01(weight);
  for (let i = 0; i < out.rotations.length; i++) {
    const m = mask ? mask[i] : 1;
    if (m <= 0 || w <= 0) {
      if (out !== base) out.rotations[i].copy(base.rotations[i]);
      continue;
    }
    _scratchQ.copy(add.rotations[i]);
    if (w * m < 1) _scratchQ.slerp(IDENTITY, 1 - w * m);
    if (out === base) out.rotations[i].multiply(_scratchQ);
    else out.rotations[i].copy(base.rotations[i]).multiply(_scratchQ);
  }
  if (out !== base) out.rootOffset.copy(base.rootOffset);
  out.rootOffset.addScaledVector(add.rootOffset, w);
  out.rootYaw = (out === base ? out.rootYaw : base.rootYaw) + add.rootYaw * w;
  return out;
}

/** Per-bone weights, for layering upper-body actions over lower-body motion. */
export type BoneMask = Float32Array;

export function makeMask(fill = 1): BoneMask {
  return new Float32Array(BONES.length).fill(fill);
}

/** Sets a mask over a named bone and everything beneath it in the hierarchy. */
export function maskSubtree(mask: BoneMask, rootName: BoneName, value: number): BoneMask {
  const stack: BoneName[] = [rootName];
  while (stack.length) {
    const name = stack.pop()!;
    mask[BONE_INDEX[name]] = value;
    for (const b of BONES) if (b.parent === name) stack.push(b.name);
  }
  return mask;
}

/** The standard upper-body mask: pass, steal and reach ride on top of running. */
export function upperBodyMask(): BoneMask {
  const m = makeMask(0);
  maskSubtree(m, 'spine', 1);
  // Feather the spine so the transition into the legs is not a hard seam.
  m[BONE_INDEX.spine] = 0.45;
  m[BONE_INDEX.chest] = 0.75;
  m[BONE_INDEX.upperChest] = 0.95;
  return m;
}

/** Everything. Shots, dunks, cuts and stops own the whole body. */
export function fullBodyMask(): BoneMask {
  return makeMask(1);
}

/** One arm plus its clavicle — the dribble pump, a one-handed reach. */
export function armMask(side: 'left' | 'right', shoulderWeight = 0.35): BoneMask {
  const m = makeMask(0);
  maskSubtree(m, side === 'left' ? 'clavicleL' : 'clavicleR', 1);
  m[BONE_INDEX[side === 'left' ? 'clavicleL' : 'clavicleR']] = shoulderWeight;
  return m;
}

/** Head and neck only — look-at pre-pose, a glance at the shot clock. */
export function headMask(): BoneMask {
  const m = makeMask(0);
  maskSubtree(m, 'neck', 1);
  m[BONE_INDEX.neck] = 0.6;
  return m;
}

// ---------------------------------------------------------------------------
// Clips
// ---------------------------------------------------------------------------

/** A single keyframe: a sparse set of joint rotations at a normalised time. */
export interface Keyframe {
  /** 0..1 through the clip. */
  t: number;
  /** Euler XYZ in radians, per named bone. Omitted bones hold the rest pose. */
  pose: Partial<Record<BoneName, [number, number, number]>>;
  /** Pelvis offset in metres, relative to the rest hips position. */
  root?: [number, number, number];
  rootYaw?: number;
  /** Legacy per-segment ease. Only consulted when the clip interpolates 'smooth'. */
  ease?: (t: number) => number;
}

export type ClipInterp = 'linear' | 'smooth' | 'spline';

export interface Clip {
  name: string;
  /** Duration in seconds at playback rate 1. */
  duration: number;
  loop: boolean;
  keys: Keyframe[];
  /**
   * Events fired as the clip passes a normalised time — the hook animation uses
   * to tell gameplay "the ball leaves the hand NOW".
   */
  events?: Array<{ t: number; name: string }>;
  /**
   * How keys are interpolated. Defaults to `'spline'`: a C1 cubic Hermite over
   * the Euler channels, so joints do not stall on every key.
   */
  interp?: ClipInterp;
  /**
   * How the clip composes when it is played as an action over locomotion.
   * `'override'` masks the gait out; `'additive'` rides on top of it.
   */
  layer?: 'additive' | 'override';
  /** Bone region an override action claims. */
  region?: 'upper' | 'full';
  /** Ground distance covered by one full cycle, as a fraction of standing height. */
  strideLength?: number;
  /** Fraction of the cycle each foot spends on the floor. */
  stanceFraction?: number;
  /** Normalised phase at which the left and right feet strike. */
  contactPhase?: readonly [number, number];
  /** Reference ground speed the cycle was authored at, m/s at 1.98 m tall. */
  refSpeed?: number;
}

// --- Sampling ---------------------------------------------------------------

interface Channel {
  bone: number;
  /** 3 values per key. */
  v: Float32Array;
  /** 3 tangents per key. */
  m: Float32Array;
}

interface PreparedClip {
  times: Float32Array;
  channels: Channel[];
  root: Float32Array;
  rootTangent: Float32Array;
  yaw: Float32Array;
  yawTangent: Float32Array;
  touched: Uint8Array;
  interp: ClipInterp;
}

const PREPARED = new WeakMap<Clip, PreparedClip>();

/**
 * Hermite tangents from time-aware central differences, with a magnitude limit
 * so a key surrounded by very unevenly spaced neighbours cannot fling the curve
 * out of range. Loops wrap; one-shots get flat ends, which reads as a natural
 * ease in and out.
 */
function buildTangents(times: Float32Array, values: Float32Array, stride: number, loop: boolean): Float32Array {
  const n = times.length;
  const out = new Float32Array(n * stride);
  if (n < 2) return out;
  for (let c = 0; c < stride; c++) {
    for (let i = 0; i < n; i++) {
      const iPrev = i > 0 ? i - 1 : loop ? n - 2 : 0;
      const iNext = i < n - 1 ? i + 1 : loop ? 1 : n - 1;
      const tPrev = i > 0 ? times[iPrev] : loop ? times[iPrev] - 1 : times[0];
      const tNext = i < n - 1 ? times[iNext] : loop ? times[iNext] + 1 : times[n - 1];
      const pPrev = values[iPrev * stride + c];
      const pNext = values[iNext * stride + c];
      const span = tNext - tPrev;
      if (span <= 1e-6) continue;
      let m = (pNext - pPrev) / span;
      const dl = times[i] - tPrev;
      const dr = tNext - times[i];
      const sL = dl > 1e-6 ? (values[i * stride + c] - pPrev) / dl : 0;
      const sR = dr > 1e-6 ? (pNext - values[i * stride + c]) / dr : 0;
      const lim = 3 * Math.max(Math.abs(sL), Math.abs(sR));
      m = clamp(m, -lim, lim);
      out[i * stride + c] = m;
    }
  }
  return out;
}

function prepare(clip: Clip): PreparedClip {
  const cached = PREPARED.get(clip);
  if (cached) return cached;

  const keys = clip.keys.slice().sort((a, b) => a.t - b.t);
  const n = Math.max(1, keys.length);
  const times = new Float32Array(n);
  for (let i = 0; i < keys.length; i++) times[i] = keys[i].t;

  const touched = new Uint8Array(BONES.length);
  for (const k of keys) for (const name of Object.keys(k.pose)) touched[BONE_INDEX[name as BoneName]] = 1;

  const channels: Channel[] = [];
  const loop = clip.loop;
  for (let bi = 0; bi < BONES.length; bi++) {
    if (!touched[bi]) continue;
    const name = BONES[bi].name;
    const v = new Float32Array(n * 3);
    for (let i = 0; i < keys.length; i++) {
      const p = keys[i].pose[name];
      if (p) {
        v[i * 3] = p[0];
        v[i * 3 + 1] = p[1];
        v[i * 3 + 2] = p[2];
      }
    }
    channels.push({ bone: bi, v, m: buildTangents(times, v, 3, loop) });
  }

  const root = new Float32Array(n * 3);
  const yaw = new Float32Array(n);
  for (let i = 0; i < keys.length; i++) {
    const r = keys[i].root;
    if (r) {
      root[i * 3] = r[0];
      root[i * 3 + 1] = r[1];
      root[i * 3 + 2] = r[2];
    }
    yaw[i] = keys[i].rootYaw ?? 0;
  }

  const prepared: PreparedClip = {
    times,
    channels,
    root,
    rootTangent: buildTangents(times, root, 3, loop),
    yaw,
    yawTangent: buildTangents(times, yaw, 1, loop),
    touched,
    interp: clip.interp ?? 'spline',
  };
  PREPARED.set(clip, prepared);
  return prepared;
}

/** Discards the sampling cache for a rebuilt clip. */
export function invalidateClip(clip: Clip): void {
  PREPARED.delete(clip);
}

function hermite(p0: number, m0: number, p1: number, m1: number, s: number, h: number): number {
  const s2 = s * s;
  const s3 = s2 * s;
  return (
    (2 * s3 - 3 * s2 + 1) * p0 +
    (s3 - 2 * s2 + s) * h * m0 +
    (-2 * s3 + 3 * s2) * p1 +
    (s3 - s2) * h * m1
  );
}

const _e = new Quaternion();

function quatFromEuler(out: Quaternion, x: number, y: number, z: number): Quaternion {
  // XYZ intrinsic, matching Three's default Euler order.
  const c1 = Math.cos(x / 2);
  const c2 = Math.cos(y / 2);
  const c3 = Math.cos(z / 2);
  const s1 = Math.sin(x / 2);
  const s2 = Math.sin(y / 2);
  const s3 = Math.sin(z / 2);
  out.set(
    s1 * c2 * c3 + c1 * s2 * s3,
    c1 * s2 * c3 - s1 * c2 * s3,
    c1 * c2 * s3 + s1 * s2 * c3,
    c1 * c2 * c3 - s1 * s2 * s3,
  );
  return out;
}

/** Samples a clip at normalised time `u` into `out`. */
export function sampleClip(out: Pose, clip: Clip, u: number): Pose {
  const p = prepare(clip);
  const n = p.times.length;
  if (n === 0) return identityPose(out);
  const t = clip.loop ? ((u % 1) + 1) % 1 : clamp01(u);

  // Locate the segment.
  let i0 = 0;
  while (i0 + 1 < n && p.times[i0 + 1] <= t) i0++;
  let i1 = i0 + 1;
  let t0 = p.times[i0];
  let t1 = i1 < n ? p.times[i1] : p.times[n - 1];
  if (i1 >= n) {
    if (clip.loop) {
      i1 = 0;
      t1 = p.times[0] + 1;
    } else {
      i1 = n - 1;
      t0 = p.times[n - 1];
      t1 = t0 + 1;
    }
  }
  if (clip.loop && t < p.times[0]) {
    // Before the first key: the wrapped segment from the last key.
    i0 = n - 1;
    i1 = 0;
    t0 = p.times[n - 1] - 1;
    t1 = p.times[0];
  }
  const h = Math.max(1e-6, t1 - t0);
  let s = clamp01((t - t0) / h);

  const mode = p.interp;
  if (mode === 'smooth') {
    const key = clip.keys[Math.min(clip.keys.length - 1, i1)];
    s = (key?.ease ?? smootherstep)(s);
  }
  const spline = mode === 'spline';

  const a0 = i0 * 3;
  const a1 = i1 * 3;

  for (let bi = 0, ci = 0; bi < BONES.length; bi++) {
    if (!p.touched[bi]) {
      out.rotations[bi].identity();
      continue;
    }
    const ch = p.channels[ci++];
    let x: number;
    let y: number;
    let z: number;
    if (spline) {
      x = hermite(ch.v[a0], ch.m[a0], ch.v[a1], ch.m[a1], s, h);
      y = hermite(ch.v[a0 + 1], ch.m[a0 + 1], ch.v[a1 + 1], ch.m[a1 + 1], s, h);
      z = hermite(ch.v[a0 + 2], ch.m[a0 + 2], ch.v[a1 + 2], ch.m[a1 + 2], s, h);
    } else {
      x = ch.v[a0] + (ch.v[a1] - ch.v[a0]) * s;
      y = ch.v[a0 + 1] + (ch.v[a1 + 1] - ch.v[a0 + 1]) * s;
      z = ch.v[a0 + 2] + (ch.v[a1 + 2] - ch.v[a0 + 2]) * s;
    }
    quatFromEuler(out.rotations[bi], x, y, z);
  }

  if (spline) {
    out.rootOffset.set(
      hermite(p.root[a0], p.rootTangent[a0], p.root[a1], p.rootTangent[a1], s, h),
      hermite(p.root[a0 + 1], p.rootTangent[a0 + 1], p.root[a1 + 1], p.rootTangent[a1 + 1], s, h),
      hermite(p.root[a0 + 2], p.rootTangent[a0 + 2], p.root[a1 + 2], p.rootTangent[a1 + 2], s, h),
    );
    out.rootYaw = hermite(p.yaw[i0], p.yawTangent[i0], p.yaw[i1], p.yawTangent[i1], s, h);
  } else {
    out.rootOffset.set(
      p.root[a0] + (p.root[a1] - p.root[a0]) * s,
      p.root[a0 + 1] + (p.root[a1 + 1] - p.root[a0 + 1]) * s,
      p.root[a0 + 2] + (p.root[a1 + 2] - p.root[a0 + 2]) * s,
    );
    out.rootYaw = p.yaw[i0] + (p.yaw[i1] - p.yaw[i0]) * s;
  }
  void _e;
  return out;
}

/** Writes a pose onto a skeleton's bones. */
export function applyPose(sk: BuiltSkeleton, pose: Pose): void {
  for (let i = 0; i < sk.bones.length; i++) {
    sk.bones[i].quaternion.copy(pose.rotations[i]);
  }
  const hips = sk.byName.hips;
  const rest = BONES[BONE_INDEX.hips].offset;
  hips.position.set(
    rest[0] * sk.height + pose.rootOffset.x,
    rest[1] * sk.height + pose.rootOffset.y,
    rest[2] * sk.height + pose.rootOffset.z,
  );
  sk.byName.root.rotation.set(0, pose.rootYaw, 0);
  sk.byName.root.updateMatrixWorld(true);
}

/**
 * A cross-fading clip player. Gameplay asks for a state; the player handles the
 * blend so transitions never pop.
 */
export class ClipPlayer {
  private current: Clip | null = null;
  private previous: Clip | null = null;
  private time = 0;
  private prevTime = 0;
  private prevRate = 1;
  private fade = 1;
  private fadeRate = 1;
  private rate = 1;
  private firedEvents = new Set<string>();

  private poseA = makePose();
  private poseB = makePose();
  readonly output = makePose();

  onEvent: ((name: string) => void) | null = null;

  get clipName(): string | null {
    return this.current?.name ?? null;
  }

  get clip(): Clip | null {
    return this.current;
  }

  get normalisedTime(): number {
    return this.current ? this.time / this.current.duration : 0;
  }

  /** 0 while a cross-fade is still leaving the previous clip, 1 when settled. */
  get fadeWeight(): number {
    return this.fade;
  }

  /** Switches clips, cross-fading over `fadeSeconds`. */
  play(clip: Clip, fadeSeconds = 0.18, rate = 1, startAt = 0): void {
    if (this.current === clip) {
      this.rate = rate;
      return;
    }
    this.previous = this.current;
    this.prevTime = this.time;
    this.prevRate = this.rate;
    this.current = clip;
    this.time = startAt * clip.duration;
    this.rate = rate;
    this.fade = fadeSeconds > 0 && this.previous ? 0 : 1;
    this.fadeRate = fadeSeconds > 0 ? 1 / fadeSeconds : Infinity;
    this.firedEvents.clear();
  }

  stop(): void {
    this.current = null;
    this.previous = null;
    this.firedEvents.clear();
  }

  update(dt: number): Pose {
    const clip = this.current;
    if (!clip) return this.output;

    this.time += dt * this.rate;
    if (clip.loop) this.time = ((this.time % clip.duration) + clip.duration) % clip.duration;
    else this.time = Math.min(this.time, clip.duration);

    const u = this.time / clip.duration;
    sampleClip(this.poseA, clip, u);

    if (this.fade < 1 && this.previous) {
      this.prevTime += dt * this.prevRate;
      const pu = this.prevTime / this.previous.duration;
      sampleClip(this.poseB, this.previous, pu);
      this.fade = Math.min(1, this.fade + this.fadeRate * dt);
      blendPose(this.output, this.poseB, this.poseA, smootherstep(this.fade));
    } else {
      copyPose(this.output, this.poseA);
      this.previous = null;
    }

    if (clip.events) {
      for (const ev of clip.events) {
        const key = `${ev.name}@${ev.t}`;
        if (u >= ev.t && !this.firedEvents.has(key)) {
          this.firedEvents.add(key);
          this.onEvent?.(ev.name);
        }
      }
      if (clip.loop && u < 0.02) this.firedEvents.clear();
    }

    return this.output;
  }
}
