/**
 * Pose representation, clips and blending.
 *
 * There is no animation file format here — every clip is authored in code as a
 * small set of keyframes over joint rotations. That sounds like a limitation
 * and mostly it is a feature: clips can be parameterised (shoot from anywhere
 * on the floor, dunk at any rim approach) and blended continuously rather than
 * snapping between baked takes.
 */

import { Quaternion, Vector3 } from 'three';
import { BONES, type BoneName, BONE_INDEX, type BuiltSkeleton } from '../entities/Skeleton';
import { clamp01, smootherstep } from '../core/MathX';

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
 * Additive blend: layers `add` onto `base` at `weight`. Used for things that
 * should ride on top of locomotion — a head turn, a shoulder shrug on contact,
 * a lean into a screen.
 */
export function addPose(out: Pose, base: Pose, add: Pose, weight: number, mask?: BoneMask): Pose {
  const w = clamp01(weight);
  const scratch = new Quaternion();
  for (let i = 0; i < out.rotations.length; i++) {
    const m = mask ? mask[i] : 1;
    if (m <= 0) {
      out.rotations[i].copy(base.rotations[i]);
      continue;
    }
    scratch.copy(add.rotations[i]);
    if (w * m < 1) scratch.slerp(IDENTITY, 1 - w * m);
    out.rotations[i].copy(base.rotations[i]).multiply(scratch);
  }
  out.rootOffset.copy(base.rootOffset).addScaledVector(add.rootOffset, w);
  out.rootYaw = base.rootYaw + add.rootYaw * w;
  return out;
}

const IDENTITY = new Quaternion();

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

/** The standard upper-body mask: shoot, pass and reach ride on top of running. */
export function upperBodyMask(): BoneMask {
  const m = makeMask(0);
  maskSubtree(m, 'spine', 1);
  // Feather the spine so the transition into the legs is not a hard seam.
  m[BONE_INDEX.spine] = 0.45;
  m[BONE_INDEX.chest] = 0.75;
  m[BONE_INDEX.upperChest] = 0.95;
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
  /** Eases into this key. Defaults to smootherstep. */
  ease?: (t: number) => number;
}

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
}

const _e = new Quaternion();
const _euler = { x: 0, y: 0, z: 0 };

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
  const t = clip.loop ? ((u % 1) + 1) % 1 : clamp01(u);
  const keys = clip.keys;
  if (keys.length === 0) return out;

  let i1 = 0;
  while (i1 < keys.length && keys[i1].t <= t) i1++;
  const prev = keys[Math.max(0, i1 - 1)];
  const next = clip.loop && i1 >= keys.length ? keys[0] : keys[Math.min(keys.length - 1, i1)];

  let span = next.t - prev.t;
  if (span <= 0) span = clip.loop ? 1 - prev.t + next.t : 1;
  let local = span > 0 ? (t - prev.t) / span : 0;
  if (local < 0) local += 1 / span;
  local = clamp01(local);
  const k = (next.ease ?? smootherstep)(local);

  for (let bi = 0; bi < BONES.length; bi++) {
    const name = BONES[bi].name;
    const a = prev.pose[name];
    const b = next.pose[name];
    if (!a && !b) {
      out.rotations[bi].identity();
      continue;
    }
    const av = a ?? [0, 0, 0];
    const bv = b ?? [0, 0, 0];
    quatFromEuler(out.rotations[bi], av[0], av[1], av[2]);
    quatFromEuler(_e, bv[0], bv[1], bv[2]);
    out.rotations[bi].slerp(_e, k);
  }

  const ar = prev.root ?? [0, 0, 0];
  const br = next.root ?? [0, 0, 0];
  out.rootOffset.set(
    ar[0] + (br[0] - ar[0]) * k,
    ar[1] + (br[1] - ar[1]) * k,
    ar[2] + (br[2] - ar[2]) * k,
  );
  out.rootYaw = (prev.rootYaw ?? 0) + ((next.rootYaw ?? 0) - (prev.rootYaw ?? 0)) * k;
  void _euler;
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
  sk.byName.root.rotation.y = pose.rootYaw;
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

  get normalisedTime(): number {
    return this.current ? this.time / this.current.duration : 0;
  }

  /** Switches clips, cross-fading over `fadeSeconds`. */
  play(clip: Clip, fadeSeconds = 0.18, rate = 1): void {
    if (this.current === clip) {
      this.rate = rate;
      return;
    }
    this.previous = this.current;
    this.prevTime = this.time;
    this.current = clip;
    this.time = 0;
    this.rate = rate;
    this.fade = fadeSeconds > 0 && this.previous ? 0 : 1;
    this.fadeRate = fadeSeconds > 0 ? 1 / fadeSeconds : Infinity;
    this.firedEvents.clear();
  }

  update(dt: number): Pose {
    const clip = this.current;
    if (!clip) return this.output;

    this.time += dt * this.rate;
    if (clip.loop) this.time %= clip.duration;
    else this.time = Math.min(this.time, clip.duration);

    const u = this.time / clip.duration;
    sampleClip(this.poseA, clip, u);

    if (this.fade < 1 && this.previous) {
      this.prevTime += dt;
      const pu = this.prevTime / this.previous.duration;
      sampleClip(this.poseB, this.previous, pu);
      this.fade = Math.min(1, this.fade + this.fadeRate * dt);
      blendPose(this.output, this.poseB, this.poseA, this.fade);
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
