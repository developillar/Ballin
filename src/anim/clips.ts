/**
 * Code-authored animation clips.
 *
 * There is no animation file format in this project, so every clip is built
 * here from analytic curves and a handful of authored keys. The upside is that
 * clips are *parametric*: a jump shot can be rebuilt for a different release
 * quality, a dunk for a different approach, without an artist round-trip.
 *
 * ## Sign convention — verified against `entities/Skeleton.ts`
 *
 * Bones are built with identity rest rotations, so a bone's local axes are the
 * character's axes: **+X is the character's LEFT, +Y up, +Z forward.** What
 * changes between bones is the direction the bone body points:
 *
 *   | bone family | body points along | +X rotation does |
 *   |---|---|---|
 *   | thigh, shin, foot, upperArm, foreArm, hand | local −Y | swings it **backward** |
 *   | hips, spine, chest, upperChest, neck, head | local +Y | pitches it **forward** |
 *
 * Therefore, and this is the part that inverts an entire library if you get it
 * wrong:
 *
 *   - **thigh / upperArm negative X swings the limb forward.**
 *   - **shin positive X bends the knee** (heel toward the seat). A shin can
 *     never be negative — that is a knee bending backward.
 *   - **foreArm negative X bends the elbow** (hand travels forward). A foreArm
 *     can never be positive.
 *   - **spine positive X leans the torso forward.**
 *   - Y is yaw, positive turns toward the character's right (because +X is
 *     left). Z is roll, positive lifts the +X / left side.
 *
 * `tools`-free verification: `footZ()` below is the closed-form forward
 * kinematics of the leg under exactly this convention, and it is what the
 * stride length — and therefore the absence of foot-skating — is derived from.
 * If the convention were inverted, stride length would come out negative.
 *
 * Owned by the animation agent.
 */

import { LANDMARK, type BoneName } from '../entities/Skeleton';
import type { Clip, Keyframe } from './Pose';
import { invalidateClip } from './Pose';
import type { ActionKind, ActionParams, AnimatorOptions } from './AnimatorTypes';
import { clamp, clamp01 } from '../core/MathX';

type PoseMap = Partial<Record<BoneName, [number, number, number]>>;

function key(t: number, pose: PoseMap, extra: Partial<Keyframe> = {}): Keyframe {
  return { t, pose, ...extra };
}

function clip(
  name: string,
  duration: number,
  loop: boolean,
  keys: Keyframe[],
  a?: Clip['events'] | Partial<Clip>,
  b: Partial<Clip> = {},
): Clip {
  const events = Array.isArray(a) ? a : undefined;
  const extra = (Array.isArray(a) ? b : a) ?? {};
  return { name, duration, loop, keys, interp: 'spline', events, ...extra };
}

/** Mirrors a pose's left/right limbs — halves the authoring for symmetric moves. */
export function mirror(p: PoseMap): PoseMap {
  const out: PoseMap = {};
  const swap: Record<string, string> = {
    thighL: 'thighR', thighR: 'thighL',
    shinL: 'shinR', shinR: 'shinL',
    footL: 'footR', footR: 'footL',
    toeL: 'toeR', toeR: 'toeL',
    upperArmL: 'upperArmR', upperArmR: 'upperArmL',
    foreArmL: 'foreArmR', foreArmR: 'foreArmL',
    handL: 'handR', handR: 'handL',
    clavicleL: 'clavicleR', clavicleR: 'clavicleL',
  };
  for (const [k, v] of Object.entries(p) as Array<[BoneName, [number, number, number]]>) {
    const name = (swap[k] ?? k) as BoneName;
    // Reflection across the sagittal plane: pitch survives, yaw and roll flip.
    out[name] = [v[0], -v[1], -v[2]];
  }
  return out;
}

// ---------------------------------------------------------------------------
// Curve helper
// ---------------------------------------------------------------------------

type Table = ReadonlyArray<readonly [number, number]>;

/**
 * Catmull-Rom over a sparse control table, evaluated at `p ∈ [0,1)`. Gait
 * channels are authored as a handful of anatomical landmarks (heel strike,
 * loading, mid-stance, toe-off, peak swing) and this turns them into the smooth
 * continuous curve the body actually follows.
 */
function curve(table: Table, p: number, loop = true): number {
  const n = table.length;
  if (n === 0) return 0;
  if (n === 1) return table[0][1];
  let t = loop ? ((p % 1) + 1) % 1 : clamp01(p);
  let i0 = 0;
  while (i0 + 1 < n && table[i0 + 1][0] <= t) i0++;
  const wrap = i0 === n - 1;
  const iA = wrap ? (loop ? n - 1 : n - 2) : i0;
  const iB = wrap ? (loop ? 0 : n - 1) : i0 + 1;
  const tA = table[iA][0];
  const tB = wrap && loop ? table[iB][0] + 1 : table[iB][0];
  if (t < table[0][0] && loop) {
    // Before the first control point: the wrapped segment.
    return curveSeg(table, n - 1, 0, table[n - 1][0] - 1, table[0][0], t, loop);
  }
  return curveSeg(table, iA, iB, tA, tB, t, loop);
}

function curveSeg(
  table: Table,
  iA: number,
  iB: number,
  tA: number,
  tB: number,
  t: number,
  loop: boolean,
): number {
  const n = table.length;
  const idx = (i: number) => (loop ? ((i % n) + n) % n : clamp(i, 0, n - 1));
  const pPrev = table[idx(iA - 1)][1];
  const pA = table[iA][1];
  const pB = table[iB][1];
  const pNext = table[idx(iB + 1)][1];
  const h = Math.max(1e-6, tB - tA);
  const s = clamp01((t - tA) / h);
  const mA = (pB - pPrev) * 0.5;
  const mB = (pNext - pA) * 0.5;
  const s2 = s * s;
  const s3 = s2 * s;
  return (
    (2 * s3 - 3 * s2 + 1) * pA +
    (s3 - 2 * s2 + s) * mA +
    (-2 * s3 + 3 * s2) * pB +
    (s3 - s2) * mB
  );
}

// ---------------------------------------------------------------------------
// Locomotion — one parametric gait generator
// ---------------------------------------------------------------------------

/**
 * The leg cycle is authored in *stance-normalised* time `q`, where `q ∈ [0,0.5]`
 * is the contact phase and `q ∈ [0.5,1]` is the swing. That decouples the shape
 * of the cycle from the stance fraction, which changes a lot between a walk
 * (62% contact) and a sprint (26%).
 */
const HIP_Q: Table = [
  // Contact is deliberately only half of peak swing flexion. Reaching the foot
  // all the way out to the peak-swing angle is *over-striding* — it puts the
  // ankle so far in front of the hip that the leg is 99% extended at touchdown
  // and the pelvis has to sink to reach the floor. It is the most common error
  // in a hand-authored run and it makes foot planting geometrically impossible.
  [0.00, -0.50], // heel strike
  [0.10, -0.36], // loading
  [0.22, -0.06], // mid-stance, hip passing under the body
  [0.36, 0.46], // late stance
  [0.50, 1.00], // toe-off: hip extended behind (scaled by `extend`)
  [0.60, 0.52],
  [0.72, -0.36], // swinging through
  [0.84, -0.86],
  [0.92, -1.00], // peak swing flexion
  [1.00, -0.50], // the paw-back: the foot is already moving backward at contact
];

const KNEE_Q: Table = [
  [0.00, 0.28], // strike: knee softly flexed, never locked
  [0.10, 0.44], // loading response — this is the shock absorber
  [0.22, 0.40],
  [0.36, 0.24],
  [0.50, 0.15], // near extension driving off the toe
  [0.58, 0.62],
  [0.68, 1.00], // heel to the seat
  [0.80, 0.74],
  [0.90, 0.38],
  [1.00, 0.28],
];

const ANKLE_Q: Table = [
  [0.00, -0.34], // dorsiflexed on approach
  [0.10, 0.04], // foot flat
  [0.25, 0.18],
  [0.40, 0.62],
  [0.50, 1.00], // plantarflexed, driving off the toe
  [0.60, 0.44],
  [0.72, -0.26], // toes pulled up to clear the floor
  [0.86, -0.44],
  [1.00, -0.34],
];

export interface GaitSpec {
  name: string;
  /** Reference ground speed the cycle reads correctly at, m/s for a 1.98 m player. */
  refSpeed: number;
  /** Peak hip flexion, radians. */
  swing: number;
  /** Hip extension at toe-off, as a multiple of `swing`. */
  extend: number;
  /** Peak swing-phase knee flexion, radians. */
  knee: number;
  /** Peak plantarflexion at toe-off, radians. */
  ankle: number;
  /** Fraction of the cycle each foot spends in contact. */
  stance: number;
  /** Shoulder counter-swing amplitude, radians. */
  arm: number;
  /** Base elbow flexion. */
  elbow: number;
  /** Net forward pitch of the trunk, radians. */
  lean: number;
  /**
   * Pelvis vertical oscillation amplitude, **metres at the reference height**
   * (1.98 m). `Pose.rootOffset` is in metres, so this is too; the blend tree
   * scales it for shorter or taller players.
   */
  bounce: number;
  /**
   * Standing crouch, metres at the reference height (negative lowers the
   * pelvis). Set so the authored leg puts the ankle a few millimetres *below*
   * the floor at contact: a foot that has to be lifted is always solvable, a
   * foot that has to be stretched to reach is not, and the plant IK releases.
   */
  crouch: number;
  /** Pelvic obliquity — the swing-side hip drop. */
  list: number;
  /** Pelvic transverse rotation. */
  pelvisYaw: number;
  /** Net shoulder-girdle counter-rotation. */
  shoulderYaw: number;
  /**
   * Mirrors the sagittal leg motion: the legs reach behind and push forward.
   * That is a backpedal — the knee still only bends one way, which is why this
   * is a flag rather than a negative `swing`.
   */
  reverse?: boolean;
}

/** Ankle position, forward of the hip, from the two leg angles. Fractions of height. */
function footZ(hip: number, knee: number): number {
  return -LANDMARK.thigh * Math.sin(hip) - LANDMARK.shin * Math.sin(hip + knee);
}

function hipAngle(spec: GaitSpec, q: number): number {
  const v = curve(HIP_Q, q);
  const a = v >= 0 ? v * spec.swing * spec.extend : v * spec.swing;
  return spec.reverse ? -a : a;
}

const kneeAngle = (spec: GaitSpec, q: number): number => curve(KNEE_Q, q) * spec.knee;
const ankleAngle = (spec: GaitSpec, q: number): number =>
  curve(ANKLE_Q, q) * spec.ankle * (spec.reverse ? -1 : 1);

/** Cycle phase → stance-normalised time for one leg. */
function toQ(spec: GaitSpec, p: number): number {
  const x = ((p % 1) + 1) % 1;
  return x < spec.stance ? (x / spec.stance) * 0.5 : 0.5 + ((x - spec.stance) / (1 - spec.stance)) * 0.5;
}

/**
 * Ground distance covered by one full cycle, as a fraction of standing height.
 *
 * This is the number that makes feet stop skating. During contact the foot is
 * pinned to the floor, so the pelvis travels forward by exactly the ankle's
 * backward excursion in body space; dividing by the stance fraction converts
 * that into distance per cycle. Play the clip at `speed / strideLength` cycles
 * per second and the stride rate matches the ground speed by construction.
 */
function measureStride(spec: GaitSpec): number {
  // Sample rather than trusting the two endpoints — the ankle path is not linear.
  const z0 = footZ(hipAngle(spec, 0), kneeAngle(spec, 0));
  const z1 = footZ(hipAngle(spec, 0.5), kneeAngle(spec, 0.5));
  return Math.max(0.08, Math.abs(z0 - z1) / spec.stance);
}

const TRUNK_SHARE = { hips: 0.12, spine: 0.3, chest: 0.31, upperChest: 0.27 };
const TWIST_SHARE = { spine: 0.16, chest: 0.34, upperChest: 0.5 };

/** How many phase samples a gait cycle is baked at. */
const GAIT_SAMPLES = 20;

export function buildGait(spec: GaitSpec): Clip {
  const stride = measureStride(spec);
  const keys: Keyframe[] = [];

  // Mid-stance is where the pelvis bottoms out; everything else is phased off it.
  const midStance = spec.stance * 0.24;

  for (let i = 0; i <= GAIT_SAMPLES; i++) {
    const p = i / GAIT_SAMPLES;
    const qL = toQ(spec, p);
    const qR = toQ(spec, p + 0.5);

    // --- Pelvis ----------------------------------------------------------
    const bobPhase = Math.cos(4 * Math.PI * (p - midStance));
    const listPhase = Math.cos(2 * Math.PI * (p - midStance));
    const twistPhase = Math.cos(2 * Math.PI * (p - 0.95));

    const hipsX = spec.lean * TRUNK_SHARE.hips;
    const hipsY = -spec.pelvisYaw * twistPhase;
    const hipsZ = spec.list * listPhase;

    // Shoulders must end up counter-rotated to the pelvis, and the pelvis yaw
    // is inherited by the spine, so the spine has to undo it first.
    const twist = (spec.shoulderYaw + spec.pelvisYaw) * twistPhase;
    // The trunk counters the pelvic list too — that is what keeps the head level.
    const roll = -spec.list * 1.35 * listPhase;

    const spineX = spec.lean * TRUNK_SHARE.spine;
    const chestX = spec.lean * TRUNK_SHARE.chest;
    const upperX = spec.lean * TRUNK_SHARE.upperChest;

    // --- Legs. Thigh angles are absolute, so undo the pelvis tilt. --------
    const thighLx = hipAngle(spec, qL) - hipsX;
    const thighRx = hipAngle(spec, qR) - hipsX;
    const shinLx = kneeAngle(spec, qL);
    const shinRx = kneeAngle(spec, qR);
    const footLx = ankleAngle(spec, qL);
    const footRx = ankleAngle(spec, qR);
    // Feet track slightly outboard, and the swing leg crosses toward the midline.
    const legSplay = 0.045 + spec.list * 0.4;

    // --- Arms ------------------------------------------------------------
    // Contralateral: left arm is back when the left leg is forward (p = 0).
    const armPhaseL = Math.cos(2 * Math.PI * (p - 0.03));
    const armPhaseR = -armPhaseL;
    const armLx = spec.arm * armPhaseL;
    const armRx = spec.arm * armPhaseR;
    // Elbows close on the forward swing and open behind — that asymmetry is
    // most of what separates a runner from a marching toy.
    const elbowL = -(spec.elbow + spec.elbow * 0.42 * (1 - armPhaseL) * 0.5);
    const elbowR = -(spec.elbow + spec.elbow * 0.42 * (1 - armPhaseR) * 0.5);
    const armAbduct = 0.11 + spec.arm * 0.18;
    // The hand crosses slightly toward the midline at the front of the swing.
    const armLy = -0.1 * (1 - armPhaseL) * 0.5;
    const armRy = 0.1 * (1 - armPhaseR) * 0.5;

    // --- Head stays level ------------------------------------------------
    const trunkX = hipsX + spineX + chestX + upperX;
    const neckX = -trunkX * 0.28;
    const headX = -trunkX * 0.42 + 0.03;
    const headY = -twist * 0.55 - hipsY * 0.2;
    const headZ = -roll * 0.5;

    keys.push(
      key(
        p,
        {
          hips: [hipsX, hipsY, hipsZ],
          spine: [spineX, twist * TWIST_SHARE.spine, roll * TWIST_SHARE.spine],
          chest: [chestX, twist * TWIST_SHARE.chest, roll * TWIST_SHARE.chest],
          upperChest: [upperX, twist * TWIST_SHARE.upperChest, roll * TWIST_SHARE.upperChest],
          neck: [neckX, headY * 0.35, headZ * 0.35],
          head: [headX, headY, headZ],

          clavicleL: [armLx * 0.12, 0, -0.02 - armLx * 0.06],
          upperArmL: [armLx, armLy, -armAbduct],
          foreArmL: [elbowL, 0, 0],
          handL: [-0.12 + armLx * 0.1, 0, -0.1],
          clavicleR: [armRx * 0.12, 0, 0.02 + armRx * 0.06],
          upperArmR: [armRx, armRy, armAbduct],
          foreArmR: [elbowR, 0, 0],
          handR: [-0.12 + armRx * 0.1, 0, 0.1],

          thighL: [thighLx, legSplay * 0.5, legSplay],
          shinL: [shinLx, 0, 0],
          footL: [footLx, -legSplay * 0.6, 0],
          toeL: [Math.max(0, -footLx) * 0.5, 0, 0],
          thighR: [thighRx, -legSplay * 0.5, -legSplay],
          shinR: [shinRx, 0, 0],
          footR: [footRx, legSplay * 0.6, 0],
          toeR: [Math.max(0, -footRx) * 0.5, 0, 0],
        },
        { root: [0, spec.crouch - spec.bounce * bobPhase, 0] },
      ),
    );
  }

  return clip(spec.name, 1, true, keys, {
    strideLength: stride,
    stanceFraction: spec.stance,
    contactPhase: [0, 0.5],
    refSpeed: spec.refSpeed,
  });
}

/**
 * The speed ladder. Each rung is a real gait, not a scaled version of the one
 * below it: stance fraction shortens, the trunk pitches further forward, the
 * knee folds higher and the arms drive harder as speed climbs.
 */
export const GAIT_SPECS: readonly GaitSpec[] = [
  {
    name: 'walk',
    refSpeed: 1.35,
    swing: 0.52, extend: 0.9, knee: 0.7, ankle: 0.34, stance: 0.55,
    arm: 0.16, elbow: 0.34, lean: 0.045, bounce: 0.01, crouch: -0.018,
    list: 0.05, pelvisYaw: 0.055, shoulderYaw: 0.05,
  },
  {
    name: 'jog',
    refSpeed: 2.9,
    swing: 0.62, extend: 0.82, knee: 1.1, ankle: 0.44, stance: 0.33,
    arm: 0.36, elbow: 0.72, lean: 0.11, bounce: 0.02, crouch: -0.02,
    list: 0.07, pelvisYaw: 0.085, shoulderYaw: 0.085,
  },
  {
    name: 'run',
    refSpeed: 5.0,
    swing: 0.78, extend: 0.82, knee: 1.5, ankle: 0.5, stance: 0.25,
    arm: 0.56, elbow: 0.95, lean: 0.2, bounce: 0.028, crouch: -0.025,
    list: 0.085, pelvisYaw: 0.115, shoulderYaw: 0.115,
  },
  {
    name: 'sprint',
    refSpeed: 7.9,
    swing: 0.95, extend: 0.8, knee: 1.95, ankle: 0.56, stance: 0.2,
    arm: 0.8, elbow: 1.22, lean: 0.31, bounce: 0.036, crouch: -0.032,
    list: 0.095, pelvisYaw: 0.145, shoulderYaw: 0.15,
  },
];

/** Backpedal: short choppy steps, chest up, hips under, weight on the balls. */
const BACKPEDAL_SPEC: GaitSpec = {
  name: 'backpedal',
  refSpeed: 3.4,
  swing: 0.44, extend: 0.9, knee: 1.05, ankle: 0.5, stance: 0.38,
  arm: 0.3, elbow: 0.85, lean: -0.05, bounce: 0.018, crouch: -0.06,
  list: 0.05, pelvisYaw: 0.06, shoulderYaw: 0.06,
  reverse: true,
};

export const GAITS = GAIT_SPECS.map(buildGait);
export const BACKPEDAL = buildGait(BACKPEDAL_SPEC);

// --- Non-cyclic-ground states ----------------------------------------------

const idle = clip('idle', 4.6, true, [
  key(0, {
    hips: [0.015, 0.03, 0.035],
    thighL: [-0.09, 0.07, 0.045], shinL: [0.15, 0, 0], footL: [-0.05, -0.04, 0],
    thighR: [-0.02, -0.06, -0.055], shinR: [0.07, 0, 0], footR: [-0.03, 0.03, 0],
    spine: [0.045, -0.02, -0.03], chest: [0.03, -0.015, -0.02], upperChest: [0.025, -0.01, -0.015],
    clavicleL: [0, 0, -0.03], upperArmL: [0.1, -0.03, -0.16], foreArmL: [-0.42, 0, 0], handL: [-0.15, 0, -0.12],
    clavicleR: [0, 0, 0.03], upperArmR: [0.08, 0.03, 0.15], foreArmR: [-0.38, 0, 0], handR: [-0.15, 0, 0.12],
    neck: [-0.02, 0.03, 0], head: [0.03, 0.06, -0.015],
  }, { root: [0, -0.012, 0] }),
  key(0.34, {
    hips: [0.02, 0.02, 0.03],
    thighL: [-0.075, 0.07, 0.045], shinL: [0.13, 0, 0], footL: [-0.045, -0.04, 0],
    thighR: [-0.035, -0.06, -0.055], shinR: [0.09, 0, 0], footR: [-0.04, 0.03, 0],
    spine: [0.05, -0.01, -0.025], chest: [0.035, -0.01, -0.018], upperChest: [0.03, 0, -0.012],
    clavicleL: [0.01, 0, -0.04], upperArmL: [0.13, -0.03, -0.175], foreArmL: [-0.47, 0, 0], handL: [-0.16, 0, -0.12],
    clavicleR: [0.01, 0, 0.04], upperArmR: [0.11, 0.03, 0.165], foreArmR: [-0.43, 0, 0], handR: [-0.16, 0, 0.12],
    neck: [-0.025, 0, 0], head: [0.02, -0.01, -0.01],
  }, { root: [0, -0.021, 0] }),
  key(0.62, {
    hips: [0.015, -0.03, -0.03],
    thighL: [-0.03, 0.06, 0.05], shinL: [0.08, 0, 0], footL: [-0.03, -0.03, 0],
    thighR: [-0.085, -0.07, -0.05], shinR: [0.14, 0, 0], footR: [-0.05, 0.04, 0],
    spine: [0.04, 0.025, 0.03], chest: [0.028, 0.018, 0.022], upperChest: [0.022, 0.012, 0.016],
    clavicleL: [0, 0, -0.028], upperArmL: [0.07, -0.02, -0.15], foreArmL: [-0.36, 0, 0], handL: [-0.14, 0, -0.12],
    clavicleR: [0, 0, 0.028], upperArmR: [0.1, 0.02, 0.16], foreArmR: [-0.4, 0, 0], handR: [-0.14, 0, 0.12],
    neck: [-0.015, -0.03, 0], head: [0.035, -0.06, 0.015],
  }, { root: [0, -0.01, 0] }),
  key(1, {
    hips: [0.015, 0.03, 0.035],
    thighL: [-0.09, 0.07, 0.045], shinL: [0.15, 0, 0], footL: [-0.05, -0.04, 0],
    thighR: [-0.02, -0.06, -0.055], shinR: [0.07, 0, 0], footR: [-0.03, 0.03, 0],
    spine: [0.045, -0.02, -0.03], chest: [0.03, -0.015, -0.02], upperChest: [0.025, -0.01, -0.015],
    clavicleL: [0, 0, -0.03], upperArmL: [0.1, -0.03, -0.16], foreArmL: [-0.42, 0, 0], handL: [-0.15, 0, -0.12],
    clavicleR: [0, 0, 0.03], upperArmR: [0.08, 0.03, 0.15], foreArmR: [-0.38, 0, 0], handR: [-0.15, 0, 0.12],
    neck: [-0.02, 0.03, 0], head: [0.03, 0.06, -0.015],
  }, { root: [0, -0.012, 0] }),
], { contactPhase: [0, 0], stanceFraction: 1 });

/** Defensive stance: wide base, hips low, hands active, weight rocking. */
const stance = clip('stance', 2.3, true, [
  key(0, {
    hips: [0.06, 0.04, 0.02],
    thighL: [-0.5, 0.34, 0.16], shinL: [0.92, 0, 0], footL: [-0.28, 0.26, 0],
    thighR: [-0.44, -0.32, -0.14], shinR: [0.86, 0, 0], footR: [-0.24, -0.26, 0],
    spine: [0.2, -0.02, 0], chest: [0.13, -0.01, 0], upperChest: [0.08, 0, 0],
    clavicleL: [0, 0, -0.06], upperArmL: [0.16, -0.18, -0.66], foreArmL: [-0.72, 0, 0], handL: [-0.26, 0, -0.34],
    clavicleR: [0, 0, 0.06], upperArmR: [0.16, 0.18, 0.66], foreArmR: [-0.72, 0, 0], handR: [-0.26, 0, 0.34],
    neck: [-0.12, 0, 0], head: [-0.12, 0.03, 0],
  }, { root: [0, -0.148, 0] }),
  key(0.38, {
    hips: [0.07, -0.05, -0.02],
    thighL: [-0.46, 0.32, 0.15], shinL: [0.88, 0, 0], footL: [-0.26, 0.25, 0],
    thighR: [-0.53, -0.34, -0.16], shinR: [0.96, 0, 0], footR: [-0.3, -0.27, 0],
    spine: [0.23, 0.02, 0], chest: [0.15, 0.01, 0], upperChest: [0.09, 0, 0],
    clavicleL: [0.02, 0, -0.09], upperArmL: [0.1, -0.24, -0.78], foreArmL: [-0.62, 0, 0], handL: [-0.3, 0, -0.36],
    clavicleR: [0.02, 0, 0.09], upperArmR: [0.1, 0.24, 0.78], foreArmR: [-0.62, 0, 0], handR: [-0.3, 0, 0.36],
    neck: [-0.13, 0, 0], head: [-0.13, -0.04, 0],
  }, { root: [0, -0.176, 0] }),
  key(0.72, {
    hips: [0.055, 0.02, 0.01],
    thighL: [-0.52, 0.35, 0.16], shinL: [0.95, 0, 0], footL: [-0.29, 0.27, 0],
    thighR: [-0.42, -0.31, -0.13], shinR: [0.83, 0, 0], footR: [-0.23, -0.25, 0],
    spine: [0.19, -0.01, 0], chest: [0.12, 0, 0], upperChest: [0.075, 0, 0],
    clavicleL: [0, 0, -0.05], upperArmL: [0.2, -0.14, -0.6], foreArmL: [-0.8, 0, 0], handL: [-0.22, 0, -0.3],
    clavicleR: [0, 0, 0.05], upperArmR: [0.2, 0.14, 0.6], foreArmR: [-0.8, 0, 0], handR: [-0.22, 0, 0.3],
    neck: [-0.11, 0, 0], head: [-0.11, 0.05, 0],
  }, { root: [0, -0.142, 0] }),
  key(1, {
    hips: [0.06, 0.04, 0.02],
    thighL: [-0.5, 0.34, 0.16], shinL: [0.92, 0, 0], footL: [-0.28, 0.26, 0],
    thighR: [-0.44, -0.32, -0.14], shinR: [0.86, 0, 0], footR: [-0.24, -0.26, 0],
    spine: [0.2, -0.02, 0], chest: [0.13, -0.01, 0], upperChest: [0.08, 0, 0],
    clavicleL: [0, 0, -0.06], upperArmL: [0.16, -0.18, -0.66], foreArmL: [-0.72, 0, 0], handL: [-0.26, 0, -0.34],
    clavicleR: [0, 0, 0.06], upperArmR: [0.16, 0.18, 0.66], foreArmR: [-0.72, 0, 0], handR: [-0.26, 0, 0.34],
    neck: [-0.12, 0, 0], head: [-0.12, 0.03, 0],
  }, { root: [0, -0.148, 0] }),
], { contactPhase: [0, 0], stanceFraction: 1 });

/**
 * Defensive slide. The feet never cross: the lead foot steps out, the trail
 * foot recovers, and the pelvis stays square to the ball-handler.
 */
function slideCycle(name: string, dir: 1 | -1): Clip {
  const push = (p: PoseMap): PoseMap => (dir > 0 ? p : mirror(p));
  const step: PoseMap = {
    hips: [0.05, 0, 0.03],
    thighL: [-0.42, 0.58, 0.2], shinL: [0.8, 0, 0], footL: [-0.24, 0.4, 0],
    thighR: [-0.5, -0.16, -0.06], shinR: [0.96, 0, 0], footR: [-0.3, -0.14, 0],
    spine: [0.19, 0, 0.05], chest: [0.12, 0, 0.03], upperChest: [0.08, 0, 0.02],
    clavicleL: [0, 0, -0.08], upperArmL: [0.08, -0.26, -0.86], foreArmL: [-0.6, 0, 0], handL: [-0.26, 0, -0.38],
    clavicleR: [0, 0, 0.05], upperArmR: [0.18, 0.14, 0.58], foreArmR: [-0.82, 0, 0], handR: [-0.2, 0, 0.28],
    head: [-0.1, 0, 0],
  };
  const recover: PoseMap = {
    hips: [0.06, 0, 0.01],
    thighL: [-0.5, 0.3, 0.13], shinL: [0.95, 0, 0], footL: [-0.28, 0.24, 0],
    thighR: [-0.44, -0.34, -0.15], shinR: [0.86, 0, 0], footR: [-0.24, -0.26, 0],
    spine: [0.22, 0, 0.02], chest: [0.14, 0, 0.01], upperChest: [0.09, 0, 0],
    clavicleL: [0, 0, -0.06], upperArmL: [0.14, -0.2, -0.7], foreArmL: [-0.7, 0, 0], handL: [-0.24, 0, -0.34],
    clavicleR: [0, 0, 0.06], upperArmR: [0.14, 0.2, 0.66], foreArmR: [-0.72, 0, 0], handR: [-0.24, 0, 0.32],
    head: [-0.11, 0, 0],
  };
  return clip(name, 1, true, [
    key(0, push(recover), { root: [0, -0.155, 0] }),
    key(0.26, push(step), { root: [0, -0.178, 0] }),
    key(0.54, push(recover), { root: [0, -0.146, 0] }),
    key(0.78, push(step), { root: [0, -0.17, 0] }),
    key(1, push(recover), { root: [0, -0.155, 0] }),
  ], { strideLength: 0.62, stanceFraction: 0.55, contactPhase: [0.04, 0.54], refSpeed: 3.2 });
}

/** Airborne hang. Legs trail and split slightly; arms stay up and wide. */
const air = clip('air', 1.1, true, [
  key(0, {
    hips: [-0.04, 0, 0],
    thighL: [-0.46, 0.06, 0.04], shinL: [0.62, 0, 0], footL: [0.3, 0, 0],
    thighR: [-0.14, -0.06, -0.04], shinR: [0.34, 0, 0], footR: [0.26, 0, 0],
    spine: [-0.05, 0, 0], chest: [-0.04, 0, 0], upperChest: [-0.03, 0, 0],
    upperArmL: [-0.6, -0.05, -0.34], foreArmL: [-0.62, 0, 0], handL: [-0.1, 0, -0.12],
    upperArmR: [-0.78, 0.05, 0.34], foreArmR: [-0.5, 0, 0], handR: [-0.1, 0, 0.12],
    head: [-0.08, 0, 0],
  }),
  key(0.5, {
    hips: [-0.05, 0, 0],
    thighL: [-0.4, 0.06, 0.04], shinL: [0.7, 0, 0], footL: [0.34, 0, 0],
    thighR: [-0.2, -0.06, -0.04], shinR: [0.28, 0, 0], footR: [0.22, 0, 0],
    spine: [-0.06, 0, 0], chest: [-0.045, 0, 0], upperChest: [-0.035, 0, 0],
    upperArmL: [-0.66, -0.05, -0.32], foreArmL: [-0.58, 0, 0], handL: [-0.1, 0, -0.12],
    upperArmR: [-0.84, 0.05, 0.32], foreArmR: [-0.46, 0, 0], handR: [-0.1, 0, 0.12],
    head: [-0.09, 0, 0],
  }),
  key(1, {
    hips: [-0.04, 0, 0],
    thighL: [-0.46, 0.06, 0.04], shinL: [0.62, 0, 0], footL: [0.3, 0, 0],
    thighR: [-0.14, -0.06, -0.04], shinR: [0.34, 0, 0], footR: [0.26, 0, 0],
    spine: [-0.05, 0, 0], chest: [-0.04, 0, 0], upperChest: [-0.03, 0, 0],
    upperArmL: [-0.6, -0.05, -0.34], foreArmL: [-0.62, 0, 0], handL: [-0.1, 0, -0.12],
    upperArmR: [-0.78, 0.05, 0.34], foreArmR: [-0.5, 0, 0], handR: [-0.1, 0, 0.12],
    head: [-0.08, 0, 0],
  }),
], { contactPhase: [0, 0], stanceFraction: 0 });

/**
 * The dribble pump, as an *additive* one-arm layer. Two bounces per stride so
 * the ball stays under the hip, with the hand riding the ball up and pushing it
 * back down rather than slapping at it.
 */
function dribblePump(hand: 'left' | 'right', beatsPerCycle: number, height: number): Clip {
  const arm: BoneName = hand === 'right' ? 'upperArmR' : 'upperArmL';
  const fore: BoneName = hand === 'right' ? 'foreArmR' : 'foreArmL';
  const wrist: BoneName = hand === 'right' ? 'handR' : 'handL';
  const clav: BoneName = hand === 'right' ? 'clavicleR' : 'clavicleL';
  const s = hand === 'right' ? 1 : -1;
  const n = 12;
  const keys: Keyframe[] = [];
  for (let i = 0; i <= n; i++) {
    const p = i / n;
    const beat = (p * beatsPerCycle) % 1;
    // Asymmetric: a fast push down, a slower ride up with the ball.
    const push = beat < 0.34 ? -Math.sin((beat / 0.34) * Math.PI) : 0;
    const ride = beat >= 0.34 ? Math.sin(((beat - 0.34) / 0.66) * Math.PI) : 0;
    const drop = (push * 0.9 + ride * 0.55) * height;
    keys.push(
      key(p, {
        [clav]: [drop * 0.1, 0, 0],
        [arm]: [0.3 + drop * 0.55, -0.06 * s, 0.22 * s],
        [fore]: [-0.62 - drop * 0.5, 0, 0],
        [wrist]: [-0.25 - drop * 0.7, 0, 0.16 * s],
      }),
    );
  }
  return clip(`dribble-${hand}`, 1, true, keys, { layer: 'additive' });
}

export const LOCOMOTION = {
  idle,
  gaits: GAITS,
  backpedal: BACKPEDAL,
  stance,
  slideL: slideCycle('slideL', 1),
  slideR: slideCycle('slideR', -1),
  air,
  dribbleL: dribblePump('left', 2, 1),
  dribbleR: dribblePump('right', 2, 1),
} as const;

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/** A ballistic root arc: `h` metres of lift between `t0` and `t1`. */
function arcY(h: number, t0: number, t1: number, t: number): number {
  if (t <= t0 || t >= t1) return 0;
  const s = (t - t0) / (t1 - t0);
  return 4 * h * s * (1 - s);
}

/**
 * Jump shot — the signature animation of the sport, so it gets the most keys.
 *
 * The sequence, and the timings, are the real ones:
 *   gather (0.00–0.10) → dip (0.10–0.24) → drive through the legs (0.24–0.36)
 *   → ball to the set point with the elbow tucking under (0.36–0.44)
 *   → off-hand leaves (0.46) → release near the top of the jump (0.50)
 *   → wrist snap and a held follow-through (0.52–0.74) → land balanced (0.94).
 *
 * `quality` is not cosmetic. A good release has the elbow under the ball, a
 * vertical rise, a long hold and a square landing; a bad one flares the elbow,
 * drifts sideways, snaps off early and lands off-balance.
 */
function jumpShot(quality: number, hand: 'left' | 'right', variant: 'set' | 'fade' | 'quick' = 'set'): Clip {
  const q = clamp01(quality);
  const bad = 1 - q;
  const L = hand === 'left';
  const s = L ? -1 : 1; // +1 for a right-hander: their shooting side is −X.
  const shootArm: BoneName = L ? 'upperArmL' : 'upperArmR';
  const shootFore: BoneName = L ? 'foreArmL' : 'foreArmR';
  const shootHand: BoneName = L ? 'handL' : 'handR';
  const shootClav: BoneName = L ? 'clavicleL' : 'clavicleR';
  const guideArm: BoneName = L ? 'upperArmR' : 'upperArmL';
  const guideFore: BoneName = L ? 'foreArmR' : 'foreArmL';
  const guideHand: BoneName = L ? 'handR' : 'handL';

  const fade = variant === 'fade';
  const quick = variant === 'quick';
  const dur = quick ? 0.82 : fade ? 1.22 : 1.08;
  const lift = quick ? 0.24 : fade ? 0.3 : 0.34;
  // Elbow flare is the classic bad-form tell: the upper arm rolls outward and
  // the ball comes off the side of the hand instead of the fingertips.
  const flare = bad * 0.62;
  // A rushed shot drifts off the line of the shot and lands off-balance.
  const drift = bad * (fade ? 0.16 : 0.11);
  // …and the shoulders turn off the target instead of staying square.
  const skew = bad * 0.22 * s;

  const stanceWidth = 0.05 + q * 0.03;

  const gather: PoseMap = {
    hips: [0.06, -0.05 * s, 0],
    thighL: [-0.24, stanceWidth, 0.05], shinL: [0.42, 0, 0], footL: [-0.16, -0.03, 0],
    thighR: [-0.22, -stanceWidth, -0.05], shinR: [0.4, 0, 0], footR: [-0.15, 0.03, 0],
    spine: [0.12, -0.05 * s, 0], chest: [0.08, -0.04 * s, 0], upperChest: [0.05, -0.03 * s, 0],
    [shootClav]: [0.02, 0, 0.04 * s],
    [shootArm]: [0.34, -0.12 * s, 0.34 * s], [shootFore]: [-1.26, 0, 0], [shootHand]: [-0.24, 0, 0.2 * s],
    [guideArm]: [0.32, 0.12 * s, -0.44 * s], [guideFore]: [-1.34, 0, 0], [guideHand]: [-0.2, 0, -0.24 * s],
    neck: [-0.06, 0, 0], head: [-0.14, -0.03 * s, 0],
  };
  // The dip: hips sink, shins pitch forward over the toes, ball drops to the
  // waist. Depth is where the shot gets its power.
  const dipDepth = quick ? 0.13 : 0.2;
  const dip: PoseMap = {
    hips: [0.2, -0.06 * s, 0],
    thighL: [-0.72, stanceWidth, 0.05], shinL: [1.14, 0, 0], footL: [-0.44, -0.03, 0],
    thighR: [-0.7, -stanceWidth, -0.05], shinR: [1.12, 0, 0], footR: [-0.43, 0.03, 0],
    spine: [0.22, -0.06 * s, 0], chest: [0.14, -0.05 * s, 0], upperChest: [0.09, -0.03 * s, 0],
    [shootClav]: [0.03, 0, 0.05 * s],
    [shootArm]: [0.46, -0.14 * s, 0.3 * s], [shootFore]: [-1.5, 0, 0], [shootHand]: [-0.34, 0, 0.22 * s],
    [guideArm]: [0.44, 0.14 * s, -0.4 * s], [guideFore]: [-1.52, 0, 0], [guideHand]: [-0.26, 0, -0.26 * s],
    neck: [-0.1, 0, 0], head: [-0.2, -0.03 * s, 0],
  };
  // Triple extension. Ankles, knees and hips fire together and the ball starts
  // up the centre line.
  const drive: PoseMap = {
    hips: [0.08, -0.03 * s, 0],
    thighL: [-0.3, stanceWidth, 0.04], shinL: [0.46, 0, 0], footL: [0.26, -0.02, 0],
    thighR: [-0.28, -stanceWidth, -0.04], shinR: [0.44, 0, 0], footR: [0.26, 0.02, 0],
    spine: [0.06, -0.03 * s, 0], chest: [0.03, -0.02 * s, 0], upperChest: [0.01, -0.02 * s, 0],
    [shootClav]: [-0.05, 0, 0.06 * s],
    [shootArm]: [-0.62, -0.1 * s, 0.22 * s + flare], [shootFore]: [-1.86, 0, 0], [shootHand]: [-0.5, 0, 0.14 * s],
    [guideArm]: [-0.5, 0.1 * s, -0.38 * s], [guideFore]: [-1.74, 0, 0], [guideHand]: [-0.3, 0, -0.2 * s],
    neck: [-0.1, 0, 0], head: [-0.2, -0.02 * s, 0],
  };
  // Set point: ball above the brow on the shooting side, elbow under the ball,
  // forearm vertical, guide hand on the side with the palm turned in.
  const setPoint: PoseMap = {
    hips: [-0.02, -0.02 * s, 0],
    thighL: [-0.2 - (fade ? 0.18 : 0), stanceWidth * 0.8, 0.03], shinL: [0.4, 0, 0], footL: [0.34, 0, 0],
    thighR: [-0.18 - (fade ? 0.18 : 0), -stanceWidth * 0.8, -0.03], shinR: [0.38, 0, 0], footR: [0.34, 0, 0],
    spine: [fade ? -0.14 : -0.02, -0.02 * s + skew, drift], chest: [fade ? -0.1 : -0.02, -0.02 * s + skew * 0.7, drift * 0.6],
    upperChest: [fade ? -0.08 : -0.01, -0.02 * s + skew * 0.5, drift * 0.4],
    [shootClav]: [-0.1, 0, 0.06 * s],
    // A good set point puts the elbow directly under the ball: forearm near
    // vertical, upper arm rolled *in*. A bad one flares it out to the side.
    [shootArm]: [-1.5 + bad * 0.34, -0.06 * s, 0.16 * s + flare],
    [shootFore]: [-1.92 + bad * 0.3, 0, 0], [shootHand]: [-0.62, 0, 0.1 * s],
    [guideArm]: [-1.24, 0.08 * s, -0.42 * s], [guideFore]: [-1.62, 0, 0], [guideHand]: [-0.34, 0, -0.36 * s],
    neck: [-0.12, 0, 0], head: [-0.22, -0.02 * s, 0],
  };
  // Release: elbow extends, wrist begins to snap, off-hand already peeling away.
  const release: PoseMap = {
    ...setPoint,
    spine: [fade ? -0.2 : -0.05, -0.02 * s, drift], chest: [fade ? -0.14 : -0.04, -0.02 * s, drift * 0.6],
    [shootArm]: [-1.86 - q * 0.36, -0.04 * s, 0.1 * s + flare * 0.85],
    [shootFore]: [-0.96 + q * 0.62, 0, 0],
    [shootHand]: [0.16 + q * 0.56, 0, 0.06 * s + flare * 0.4],
    [guideArm]: [-1.34, 0.12 * s, -0.56 * s], [guideFore]: [-1.28, 0, 0], [guideHand]: [-0.2, 0, -0.5 * s],
    head: [-0.26, -0.02 * s, 0],
  };
  // Follow-through: fingers hang over the front of the rim, wrist flexed. Held.
  const follow: PoseMap = {
    ...release,
    // The follow-through: a confident one holds a long, straight arm with the
    // wrist snapped over. A poor one collapses back toward the body early.
    [shootArm]: [-2.02 - q * 0.42, -0.02 * s, 0.07 * s + flare * 0.55],
    [shootFore]: [-0.62 + q * 0.56, 0, 0],
    [shootHand]: [0.34 + q * 0.62, 0, 0.04 * s + flare * 0.3],
    [guideArm]: [-1.0, 0.16 * s, -0.66 * s], [guideFore]: [-1.0, 0, 0],
    thighL: [-0.36 - (fade ? 0.22 : 0), stanceWidth * 0.7, 0.03], shinL: [0.62, 0, 0], footL: [0.28, 0, 0],
    thighR: [-0.34 - (fade ? 0.22 : 0), -stanceWidth * 0.7, -0.03], shinR: [0.6, 0, 0], footR: [0.28, 0, 0],
  };
  // Reaching for the floor.
  const reach: PoseMap = {
    ...follow,
    [shootArm]: [-1.9, -0.02 * s, 0.12 * s], [shootFore]: [-0.5, 0, 0], [shootHand]: [0.4, 0, 0],
    [guideArm]: [-0.7, 0.14 * s, -0.6 * s], [guideFore]: [-0.9, 0, 0],
    thighL: [-0.2, stanceWidth, 0.04], shinL: [0.3, 0, 0], footL: [-0.22, 0, 0],
    thighR: [-0.18, -stanceWidth, -0.04], shinR: [0.28, 0, 0], footR: [-0.22, 0, 0],
    spine: [0.06, 0, drift * 0.5],
  };
  // Absorb: a real landing has a give in it.
  const land: PoseMap = {
    hips: [0.14, -0.02 * s, bad * 0.06],
    thighL: [-0.52, stanceWidth, 0.05], shinL: [0.86, 0, 0], footL: [-0.26, 0, 0],
    thighR: [-0.5, -stanceWidth, -0.05], shinR: [0.84, 0, 0], footR: [-0.25, 0, 0],
    spine: [0.16, -0.02 * s, drift * 0.8], chest: [0.1, 0, 0], upperChest: [0.06, 0, 0],
    [shootArm]: [-0.4, 0, 0.2 * s], [shootFore]: [-0.7, 0, 0], [shootHand]: [0.05, 0, 0],
    [guideArm]: [-0.3, 0, -0.32 * s], [guideFore]: [-0.7, 0, 0],
    head: [-0.06, 0, 0],
  };

  const t = quick
    ? { gather: 0.06, dip: 0.16, drive: 0.28, set: 0.38, rel: 0.46, fol: 0.62, reach: 0.82, land: 0.92 }
    // The gather→dip window is the anticipation §9.2 measures, and it wants
    // 60–140 ms. At 1.08 s the old 0.09→0.23 spacing put it at 151 ms, which
    // reads as a squat rather than a dip; 0.115→0.205 is 97 ms.
    : { gather: 0.115, dip: 0.205, drive: 0.35, set: 0.44, rel: 0.51, fol: 0.68, reach: 0.86, land: 0.94 };
  // Feet leave the floor a touch after the drive and land just before `land`.
  const t0 = t.drive + 0.015;
  const t1 = t.reach + 0.04;
  // A rushed shot drifts laterally and does not land where it took off.
  const rootAt = (u: number, extra = 0): [number, number, number] => [
    (fade ? -0.14 : 0) * (u > t0 ? 1 : u / Math.max(1e-3, t0)) - drift * 0.55 * clamp01((u - t.dip) / 0.5),
    arcY(lift * (0.82 + q * 0.22), t0, t1, u) + extra,
    0,
  ];

  return clip(
    fade ? 'fadeaway' : quick ? 'quickShot' : 'jumpShot',
    dur,
    false,
    [
      key(0, {}, { root: [0, 0, 0] }),
      key(t.gather, gather, { root: rootAt(t.gather, -0.045) }),
      key(t.dip, dip, { root: rootAt(t.dip, -dipDepth) }),
      key(t.drive, drive, { root: rootAt(t.drive, -0.035) }),
      key(t.set, setPoint, { root: rootAt(t.set) }),
      key(t.rel, release, { root: rootAt(t.rel) }),
      key(t.fol, follow, { root: rootAt(t.fol) }),
      key(t.reach, reach, { root: rootAt(t.reach) }),
      key(t.land, land, { root: rootAt(t.land, -0.11) }),
      key(1, {}, { root: [0, 0, 0] }),
    ],
    [
      { t: t.set + 0.02, name: 'apex' },
      // A poor release leaves the hand early, on the way up rather than at the
      // top of the jump — which is exactly why it misses.
      { t: t.rel - bad * 0.05, name: 'release' },
      { t: t.land, name: 'land' },
    ],
    { layer: 'override', region: 'full' },
  );
}

/** Layup finishes. All three drive off the opposite knee and rise off one foot. */
function layup(hand: 'left' | 'right', finish: 'overhand' | 'fingerRoll' | 'reverse'): Clip {
  const L = hand === 'left';
  const s = L ? -1 : 1;
  const arm: BoneName = L ? 'upperArmL' : 'upperArmR';
  const fore: BoneName = L ? 'foreArmL' : 'foreArmR';
  const wrist: BoneName = L ? 'handL' : 'handR';
  const off: BoneName = L ? 'upperArmR' : 'upperArmL';
  const offFore: BoneName = L ? 'foreArmR' : 'foreArmL';
  // Right-handed layup drives off the LEFT knee.
  const driveThigh: BoneName = L ? 'thighR' : 'thighL';
  const driveShin: BoneName = L ? 'shinR' : 'shinL';
  const plantThigh: BoneName = L ? 'thighL' : 'thighR';
  const plantShin: BoneName = L ? 'shinL' : 'shinR';
  const reverse = finish === 'reverse';
  const roll = finish === 'fingerRoll';
  const twist = reverse ? 0.55 * s : 0.16 * s;

  const gather: PoseMap = {
    hips: [0.14, -0.1 * s, 0],
    [plantThigh]: [-0.36, 0.05 * s, 0], [plantShin]: [0.72, 0, 0],
    [driveThigh]: [-0.5, -0.05 * s, 0], [driveShin]: [0.9, 0, 0],
    spine: [0.2, -0.12 * s, 0], chest: [0.12, -0.08 * s, 0], upperChest: [0.08, -0.06 * s, 0],
    [arm]: [0.3, -0.1 * s, 0.3 * s], [fore]: [-1.4, 0, 0], [wrist]: [-0.3, 0, 0.16 * s],
    [off]: [0.3, 0.1 * s, -0.4 * s], [offFore]: [-1.4, 0, 0],
    head: [-0.18, -0.1 * s, 0],
  };
  const plant: PoseMap = {
    hips: [0.22, -0.06 * s, -0.05 * s],
    [plantThigh]: [-0.62, 0.05 * s, 0], [plantShin]: [1.1, 0, 0],
    [driveThigh]: [-0.34, -0.05 * s, 0], [driveShin]: [0.7, 0, 0],
    spine: [0.26, -0.08 * s, 0], chest: [0.16, -0.06 * s, 0], upperChest: [0.1, -0.04 * s, 0],
    [arm]: [0.16, -0.1 * s, 0.28 * s], [fore]: [-1.5, 0, 0],
    [off]: [0.2, 0.1 * s, -0.4 * s], [offFore]: [-1.5, 0, 0],
    head: [-0.2, -0.06 * s, 0],
  };
  const rise: PoseMap = {
    hips: [-0.02, twist * 0.5, -0.04 * s],
    [plantThigh]: [-0.14, 0.04 * s, 0], [plantShin]: [0.3, 0, 0],
    [driveThigh]: [-1.5, -0.06 * s, 0], [driveShin]: [1.32, 0, 0],
    spine: [-0.04, twist * 0.6, 0], chest: [-0.04, twist * 0.5, 0], upperChest: [-0.03, twist * 0.4, 0],
    [arm]: [-2.14, -0.12 * s, 0.18 * s], [fore]: [-0.66, 0, 0], [wrist]: [reverse ? -0.5 : -0.34, 0, 0.1 * s],
    [off]: [-1.0, 0.12 * s, -0.5 * s], [offFore]: [-1.0, 0, 0],
    neck: [-0.14, 0, 0], head: [-0.34, twist * 0.5, 0],
  };
  const finishPose: PoseMap = {
    ...rise,
    [arm]: [-2.5 - (roll ? 0.06 : 0), -0.1 * s, 0.14 * s],
    // Overhand snaps the wrist down; the finger roll lays the palm up and lets
    // the ball run off the fingertips.
    [fore]: [roll ? -0.28 : -0.34, 0, 0],
    [wrist]: [roll ? -0.62 : 0.5, 0, roll ? 0.5 * s : 0.06 * s],
    [driveThigh]: [-1.34, -0.06 * s, 0], [driveShin]: [1.1, 0, 0],
    [off]: [-0.8, 0.14 * s, -0.56 * s], [offFore]: [-1.2, 0, 0],
    head: [-0.36, twist * 0.6, 0],
  };
  const landPose: PoseMap = {
    hips: [0.16, twist * 0.2, 0],
    [plantThigh]: [-0.48, 0.05 * s, 0], [plantShin]: [0.84, 0, 0],
    [driveThigh]: [-0.44, -0.05 * s, 0], [driveShin]: [0.8, 0, 0],
    spine: [0.18, twist * 0.2, 0], chest: [0.1, 0, 0],
    [arm]: [-0.4, 0, 0.2 * s], [fore]: [-0.8, 0, 0],
    [off]: [-0.3, 0, -0.3 * s], [offFore]: [-0.8, 0, 0],
  };

  const t0 = 0.44;
  const t1 = 0.86;
  const lift = 0.46;
  const rootAt = (u: number, extra = 0): [number, number, number] => [0, arcY(lift, t0, t1, u) + extra, 0];
  // A reverse finish carries the body under the rim and turns the shoulders
  // through; the pelvis has to follow or the spine does impossible things.
  const spin = reverse ? Math.PI * 0.5 * s : 0;

  return clip(
    reverse ? 'reverseLayup' : roll ? 'fingerRoll' : 'layup',
    reverse ? 1.08 : 1.0,
    false,
    [
      key(0, {}, { root: [0, 0, 0], rootYaw: 0 }),
      key(0.16, gather, { root: rootAt(0.16, -0.07), rootYaw: spin * 0.05 }),
      key(0.34, plant, { root: rootAt(0.34, -0.15), rootYaw: spin * 0.16 }),
      key(0.56, rise, { root: rootAt(0.56), rootYaw: spin * 0.6 }),
      key(0.66, finishPose, { root: rootAt(0.66), rootYaw: spin * 0.85 }),
      key(0.86, { ...finishPose, [arm]: [-2.1, -0.06 * s, 0.2 * s] }, { root: rootAt(0.86), rootYaw: spin }),
      key(0.94, landPose, { root: [0, -0.1, 0], rootYaw: spin }),
      key(1, {}, { root: [0, 0, 0], rootYaw: spin }),
    ],
    [
      { t: 0.6, name: 'apex' },
      { t: 0.64, name: 'release' },
      { t: 0.94, name: 'land' },
    ],
    { layer: 'override', region: 'full' },
  );
}

/** Floater — one-foot push shot off the drive, quick and very high-arcing. */
function floater(hand: 'left' | 'right'): Clip {
  const L = hand === 'left';
  const s = L ? -1 : 1;
  const arm: BoneName = L ? 'upperArmL' : 'upperArmR';
  const fore: BoneName = L ? 'foreArmL' : 'foreArmR';
  const wrist: BoneName = L ? 'handL' : 'handR';
  const off: BoneName = L ? 'upperArmR' : 'upperArmL';
  const offFore: BoneName = L ? 'foreArmR' : 'foreArmL';
  const driveThigh: BoneName = L ? 'thighR' : 'thighL';
  const driveShin: BoneName = L ? 'shinR' : 'shinL';
  const plantThigh: BoneName = L ? 'thighL' : 'thighR';
  const plantShin: BoneName = L ? 'shinL' : 'shinR';

  const t0 = 0.4;
  const t1 = 0.82;
  const rootAt = (u: number, extra = 0): [number, number, number] => [0, arcY(0.3, t0, t1, u) + extra, 0];

  return clip('floater', 0.86, false, [
    key(0, {}, { root: [0, 0, 0] }),
    key(0.18, {
      hips: [0.16, -0.08 * s, 0],
      [plantThigh]: [-0.52, 0.05 * s, 0], [plantShin]: [0.96, 0, 0],
      [driveThigh]: [-0.4, -0.05 * s, 0], [driveShin]: [0.78, 0, 0],
      spine: [0.2, -0.1 * s, 0], chest: [0.12, -0.06 * s, 0],
      [arm]: [0.2, -0.1 * s, 0.3 * s], [fore]: [-1.44, 0, 0],
      [off]: [0.22, 0.1 * s, -0.4 * s], [offFore]: [-1.44, 0, 0],
      head: [-0.18, -0.06 * s, 0],
    }, { root: rootAt(0.18, -0.15) }),
    key(0.46, {
      hips: [-0.02, -0.02 * s, 0],
      [plantThigh]: [-0.16, 0.04 * s, 0], [plantShin]: [0.32, 0, 0],
      [driveThigh]: [-1.28, -0.06 * s, 0], [driveShin]: [1.2, 0, 0],
      spine: [-0.06, -0.02 * s, 0], chest: [-0.05, 0, 0], upperChest: [-0.04, 0, 0],
      [arm]: [-1.86, -0.08 * s, 0.16 * s], [fore]: [-1.5, 0, 0], [wrist]: [-0.5, 0, 0.1 * s],
      [off]: [-1.1, 0.1 * s, -0.5 * s], [offFore]: [-1.2, 0, 0],
      head: [-0.3, 0, 0],
    }, { root: rootAt(0.46) }),
    key(0.56, {
      hips: [-0.04, 0, 0],
      [plantThigh]: [-0.2, 0.04 * s, 0], [plantShin]: [0.36, 0, 0],
      [driveThigh]: [-1.2, -0.06 * s, 0], [driveShin]: [1.1, 0, 0],
      spine: [-0.1, 0, 0], chest: [-0.07, 0, 0], upperChest: [-0.05, 0, 0],
      // Soft high release: the elbow barely extends, the wrist does the work.
      [arm]: [-2.42, -0.04 * s, 0.1 * s], [fore]: [-0.86, 0, 0], [wrist]: [0.72, 0, 0.04 * s],
      [off]: [-0.86, 0.14 * s, -0.6 * s], [offFore]: [-1.1, 0, 0],
      head: [-0.34, 0, 0],
    }, { root: rootAt(0.56) }),
    key(0.78, {
      [arm]: [-2.5, -0.02 * s, 0.08 * s], [fore]: [-0.66, 0, 0], [wrist]: [0.78, 0, 0],
      [driveThigh]: [-0.7, -0.05 * s, 0], [driveShin]: [0.8, 0, 0],
      [plantThigh]: [-0.3, 0.05 * s, 0], [plantShin]: [0.5, 0, 0],
      spine: [0.02, 0, 0],
    }, { root: rootAt(0.78) }),
    key(0.92, {
      hips: [0.14, 0, 0],
      thighL: [-0.44, 0.05, 0], shinL: [0.8, 0, 0],
      thighR: [-0.42, -0.05, 0], shinR: [0.78, 0, 0],
      spine: [0.16, 0, 0],
      [arm]: [-0.5, 0, 0.2 * s], [fore]: [-0.8, 0, 0],
    }, { root: [0, -0.1, 0] }),
    key(1, {}, { root: [0, 0, 0] }),
  ], [{ t: 0.5, name: 'apex' }, { t: 0.55, name: 'release' }, { t: 0.92, name: 'land' }],
    { layer: 'override', region: 'full' });
}

/** Hook shot — shoulder turned, off-arm sealing, ball swept over the top. */
function hookShot(hand: 'left' | 'right'): Clip {
  const L = hand === 'left';
  const s = L ? -1 : 1;
  const arm: BoneName = L ? 'upperArmL' : 'upperArmR';
  const fore: BoneName = L ? 'foreArmL' : 'foreArmR';
  const wrist: BoneName = L ? 'handL' : 'handR';
  const off: BoneName = L ? 'upperArmR' : 'upperArmL';
  const offFore: BoneName = L ? 'foreArmR' : 'foreArmL';
  const driveThigh: BoneName = L ? 'thighR' : 'thighL';
  const driveShin: BoneName = L ? 'shinR' : 'shinL';
  const rootAt = (u: number, extra = 0): [number, number, number] => [0, arcY(0.24, 0.34, 0.8, u) + extra, 0];

  return clip('hookShot', 1.05, false, [
    key(0, {}, { root: [0, 0, 0] }),
    key(0.2, {
      hips: [0.08, 0.3 * s, 0],
      thighL: [-0.4, 0.1, 0.06], shinL: [0.74, 0, 0],
      thighR: [-0.38, -0.1, -0.06], shinR: [0.72, 0, 0],
      spine: [0.12, 0.24 * s, 0], chest: [0.08, 0.2 * s, 0], upperChest: [0.05, 0.16 * s, 0],
      [arm]: [0.4, -0.2 * s, 0.62 * s], [fore]: [-1.2, 0, 0],
      [off]: [0.1, 0.2 * s, -0.9 * s], [offFore]: [-1.5, 0, 0],
      head: [-0.1, -0.2 * s, 0],
    }, { root: rootAt(0.2, -0.14) }),
    key(0.46, {
      hips: [0.0, 0.22 * s, -0.08 * s],
      [driveThigh]: [-1.24, -0.06 * s, 0], [driveShin]: [1.06, 0, 0],
      spine: [-0.04, 0.16 * s, -0.1 * s], chest: [-0.04, 0.12 * s, -0.08 * s], upperChest: [-0.03, 0.1 * s, -0.06 * s],
      // Sweeping arc: the arm goes out sideways then over the top.
      [arm]: [-0.5, -0.5 * s, 1.5 * s], [fore]: [-0.5, 0, 0], [wrist]: [-0.3, 0, 0.2 * s],
      [off]: [-0.9, 0.3 * s, -1.1 * s], [offFore]: [-0.7, 0, 0],
      head: [-0.24, -0.24 * s, 0],
    }, { root: rootAt(0.46) }),
    key(0.6, {
      hips: [-0.02, 0.16 * s, -0.12 * s],
      [driveThigh]: [-1.1, -0.06 * s, 0], [driveShin]: [0.96, 0, 0],
      spine: [-0.08, 0.1 * s, -0.14 * s], chest: [-0.06, 0.08 * s, -0.1 * s],
      [arm]: [-2.2, -0.24 * s, 0.5 * s], [fore]: [-0.28, 0, 0], [wrist]: [0.6, 0, 0.1 * s],
      [off]: [-0.7, 0.3 * s, -1.0 * s], [offFore]: [-0.9, 0, 0],
      head: [-0.32, -0.16 * s, 0],
    }, { root: rootAt(0.6) }),
    key(0.8, {
      [arm]: [-2.34, -0.14 * s, 0.34 * s], [fore]: [-0.2, 0, 0], [wrist]: [0.66, 0, 0],
      spine: [0.0, 0.06 * s, -0.08 * s],
      [driveThigh]: [-0.6, -0.05 * s, 0], [driveShin]: [0.7, 0, 0],
    }, { root: rootAt(0.8) }),
    key(0.93, {
      hips: [0.14, 0, 0],
      thighL: [-0.46, 0.06, 0], shinL: [0.82, 0, 0],
      thighR: [-0.44, -0.06, 0], shinR: [0.8, 0, 0],
      spine: [0.16, 0, 0],
      [arm]: [-0.5, 0, 0.24 * s], [fore]: [-0.8, 0, 0],
    }, { root: [0, -0.1, 0] }),
    key(1, {}, { root: [0, 0, 0] }),
  ], [{ t: 0.52, name: 'apex' }, { t: 0.58, name: 'release' }, { t: 0.93, name: 'land' }],
    { layer: 'override', region: 'full' });
}

/**
 * Dunks. Style and takeoff both change the read: a two-foot gather is a heavier
 * squat with both arms cocked, a one-foot approach is a long last stride and a
 * knee drive; a tomahawk cocks the ball behind the head.
 */
function dunk(
  power: number,
  hand: 'left' | 'right',
  style: 'oneHand' | 'twoHand' | 'tomahawk',
  twoFoot: boolean,
): Clip {
  const p = clamp01(power);
  const L = hand === 'left';
  const s = L ? -1 : 1;
  const arm: BoneName = L ? 'upperArmL' : 'upperArmR';
  const fore: BoneName = L ? 'foreArmL' : 'foreArmR';
  const wrist: BoneName = L ? 'handL' : 'handR';
  const off: BoneName = L ? 'upperArmR' : 'upperArmL';
  const offFore: BoneName = L ? 'foreArmR' : 'foreArmL';
  const offWrist: BoneName = L ? 'handR' : 'handL';
  const driveThigh: BoneName = L ? 'thighR' : 'thighL';
  const driveShin: BoneName = L ? 'shinR' : 'shinL';
  const two = style === 'twoHand';
  const tom = style === 'tomahawk';
  const lift = 0.62 + p * 0.22;
  const t0 = 0.32;
  const t1 = 0.84;
  const rootAt = (u: number, extra = 0): [number, number, number] => [0, arcY(lift, t0, t1, u) + extra, 0];

  const gather: PoseMap = twoFoot
    ? {
        hips: [0.34, 0, 0],
        thighL: [-0.86, 0.07, 0.05], shinL: [1.42, 0, 0], footL: [-0.5, 0, 0],
        thighR: [-0.86, -0.07, -0.05], shinR: [1.42, 0, 0], footR: [-0.5, 0, 0],
        spine: [0.34, 0, 0], chest: [0.2, 0, 0], upperChest: [0.12, 0, 0],
        upperArmL: [0.9, 0, -0.24], foreArmL: [-0.5, 0, 0],
        upperArmR: [0.9, 0, 0.24], foreArmR: [-0.5, 0, 0],
        head: [-0.2, 0, 0],
      }
    : {
        hips: [0.24, -0.06 * s, 0],
        thighL: [L ? -0.42 : -0.78, 0.06, 0.04], shinL: [L ? 0.76 : 1.26, 0, 0], footL: [-0.4, 0, 0],
        thighR: [L ? -0.78 : -0.42, -0.06, -0.04], shinR: [L ? 1.26 : 0.76, 0, 0], footR: [-0.4, 0, 0],
        spine: [0.28, -0.08 * s, 0], chest: [0.17, -0.05 * s, 0], upperChest: [0.1, -0.03 * s, 0],
        [arm]: [0.55, -0.1 * s, 0.3 * s], [fore]: [-1.1, 0, 0],
        [off]: [0.5, 0.1 * s, -0.36 * s], [offFore]: [-1.2, 0, 0],
        head: [-0.22, -0.05 * s, 0],
      };

  const cock: PoseMap = tom
    ? {
        // Ball taken back behind the head.
        hips: [-0.06, -0.14 * s, 0],
        [driveThigh]: [-1.15, -0.06 * s, 0], [driveShin]: [1.0, 0, 0],
        thighL: twoFoot ? [-0.5, 0.06, 0.04] : [L ? -0.5 : -0.24, 0.06, 0.04],
        shinL: twoFoot ? [0.66, 0, 0] : [L ? 0.66 : 0.42, 0, 0],
        thighR: twoFoot ? [-0.28, -0.06, -0.04] : [L ? -0.24 : -0.5, -0.06, -0.04],
        shinR: twoFoot ? [0.42, 0, 0] : [L ? 0.42 : 0.66, 0, 0],
        spine: [-0.22, -0.16 * s, 0], chest: [-0.14, -0.12 * s, 0], upperChest: [-0.1, -0.1 * s, 0],
        [arm]: [-2.66, -0.3 * s, 0.24 * s], [fore]: [-1.5, 0, 0], [wrist]: [-0.4, 0, 0.1 * s],
        [off]: [-1.5, 0.24 * s, -0.6 * s], [offFore]: [-0.9, 0, 0],
        neck: [-0.14, 0, 0], head: [-0.42, -0.06 * s, 0],
      }
    : two
      ? {
          hips: [-0.08, 0, 0],
          thighL: [-0.44, 0.06, 0.04], shinL: [0.62, 0, 0], footL: [0.3, 0, 0],
          thighR: [-0.44, -0.06, -0.04], shinR: [0.62, 0, 0], footR: [0.3, 0, 0],
          spine: [-0.2, 0, 0], chest: [-0.14, 0, 0], upperChest: [-0.1, 0, 0],
          upperArmL: [-2.5 - p * 0.16, -0.06, -0.3], foreArmL: [-0.5, 0, 0], handL: [-0.3, 0, -0.1],
          upperArmR: [-2.5 - p * 0.16, 0.06, 0.3], foreArmR: [-0.5, 0, 0], handR: [-0.3, 0, 0.1],
          neck: [-0.16, 0, 0], head: [-0.44, 0, 0],
        }
      : {
          hips: [-0.05, -0.1 * s, 0],
          [driveThigh]: [-1.2, -0.06 * s, 0], [driveShin]: [1.05, 0, 0],
          thighL: twoFoot ? [-0.48, 0.06, 0.04] : [L ? -0.48 : -0.22, 0.06, 0.04],
          shinL: twoFoot ? [0.64, 0, 0] : [L ? 0.64 : 0.4, 0, 0],
          thighR: twoFoot ? [-0.26, -0.06, -0.04] : [L ? -0.22 : -0.48, -0.06, -0.04],
          shinR: twoFoot ? [0.4, 0, 0] : [L ? 0.4 : 0.64, 0, 0],
          spine: [-0.18, -0.12 * s, 0], chest: [-0.12, -0.09 * s, 0], upperChest: [-0.08, -0.07 * s, 0],
          [arm]: [-2.44 - p * 0.2, -0.14 * s, 0.14 * s], [fore]: [-0.42, 0, 0], [wrist]: [-0.24, 0, 0.08 * s],
          [off]: [-1.4, 0.2 * s, -0.56 * s], [offFore]: [-0.8, 0, 0],
          neck: [-0.16, 0, 0], head: [-0.44, -0.04 * s, 0],
        };

  const jam: PoseMap = tom
    ? {
        ...cock,
        spine: [0.05, -0.06 * s, 0], chest: [0.04, -0.04 * s, 0], upperChest: [0.03, -0.03 * s, 0],
        [arm]: [-2.5, -0.06 * s, 0.1 * s], [fore]: [-0.14, 0, 0], [wrist]: [0.5, 0, 0],
        [off]: [-1.1, 0.16 * s, -0.5 * s], [offFore]: [-0.7, 0, 0],
      }
    : two
      ? {
          ...cock,
          spine: [0.06, 0, 0], chest: [0.04, 0, 0], upperChest: [0.03, 0, 0],
          upperArmL: [-2.18, -0.04, -0.24], foreArmL: [-0.24, 0, 0], handL: [0.4, 0, 0],
          upperArmR: [-2.18, 0.04, 0.24], foreArmR: [-0.24, 0, 0], handR: [0.4, 0, 0],
        }
      : {
          ...cock,
          spine: [0.04, -0.06 * s, 0], chest: [0.03, -0.04 * s, 0],
          [arm]: [-2.14, -0.06 * s, 0.1 * s], [fore]: [-0.14, 0, 0], [wrist]: [0.44, 0, 0],
          [offWrist]: [0.1, 0, 0],
        };

  const hang: PoseMap = {
    hips: [0.04, 0, 0],
    thighL: [-0.62, 0.06, 0.04], shinL: [0.9, 0, 0], footL: [0.2, 0, 0],
    thighR: [-0.62, -0.06, -0.04], shinR: [0.9, 0, 0], footR: [0.2, 0, 0],
    spine: [0.1, 0, 0], chest: [0.06, 0, 0],
    [arm]: [-2.0, 0, 0.16 * s], [fore]: [-0.4, 0, 0],
    [off]: [-1.0, 0, -0.4 * s], [offFore]: [-0.8, 0, 0],
    head: [-0.1, 0, 0],
  };
  const landPose: PoseMap = {
    hips: [0.24, 0, 0],
    thighL: [-0.74, 0.07, 0.05], shinL: [1.2, 0, 0], footL: [-0.36, 0, 0],
    thighR: [-0.74, -0.07, -0.05], shinR: [1.2, 0, 0], footR: [-0.36, 0, 0],
    spine: [0.26, 0, 0], chest: [0.16, 0, 0], upperChest: [0.1, 0, 0],
    upperArmL: [0.34, 0, -0.4], foreArmL: [-0.5, 0, 0],
    upperArmR: [0.34, 0, 0.4], foreArmR: [-0.5, 0, 0],
    head: [0.06, 0, 0],
  };

  return clip(
    tom ? 'dunkTomahawk' : two ? 'dunkTwoHand' : 'dunk',
    1.36,
    false,
    [
      key(0, {}, { root: [0, 0, 0] }),
      key(0.2, gather, { root: rootAt(0.2, twoFoot ? -0.3 : -0.22) }),
      key(0.28, gather, { root: rootAt(0.28, -0.1) }),
      key(0.5, cock, { root: rootAt(0.5) }),
      key(0.58, jam, { root: rootAt(0.58) }),
      key(0.72, hang, { root: rootAt(0.72) }),
      key(0.86, landPose, { root: [0, -0.2, 0] }),
      key(0.94, landPose, { root: [0, -0.14, 0] }),
      key(1, {}, { root: [0, 0, 0] }),
    ],
    [
      { t: 0.5, name: 'apex' },
      { t: 0.56, name: 'release' },
      { t: 0.86, name: 'land' },
    ],
    { layer: 'override', region: 'full' },
  );
}

/** Euro-step: a long lateral stride one way, then the other, ball swung across. */
function euroStep(hand: 'left' | 'right', firstDir: 1 | -1): Clip {
  const s = hand === 'left' ? -1 : 1;
  const d = firstDir;
  const arm: BoneName = hand === 'left' ? 'upperArmL' : 'upperArmR';
  const fore: BoneName = hand === 'left' ? 'foreArmL' : 'foreArmR';
  const off: BoneName = hand === 'left' ? 'upperArmR' : 'upperArmL';
  const offFore: BoneName = hand === 'left' ? 'foreArmR' : 'foreArmL';

  const stepOut: PoseMap = {
    hips: [0.14, 0.16 * d, 0.1 * d],
    thighL: [d > 0 ? -0.9 : -0.3, 0.34 * d, 0.16 * d], shinL: [d > 0 ? 0.72 : 0.6, 0, 0],
    thighR: [d > 0 ? -0.3 : -0.9, 0.34 * d, 0.16 * d], shinR: [d > 0 ? 0.6 : 0.72, 0, 0],
    spine: [0.16, 0.1 * d, -0.16 * d], chest: [0.1, 0.08 * d, -0.12 * d], upperChest: [0.06, 0.06 * d, -0.08 * d],
    // Ball swung away from the defender.
    [arm]: [0.36, -0.3 * d, 0.5 * s], [fore]: [-1.2, 0, 0],
    [off]: [0.34, 0.3 * d, -0.6 * s], [offFore]: [-1.3, 0, 0],
    head: [-0.12, 0.2 * d, 0],
  };
  const stepBack = mirror(stepOut);

  return clip('euroStep', 0.94, false, [
    key(0, {}, { root: [0, 0, 0] }),
    key(0.22, stepOut, { root: [0.12 * d, -0.13, 0] }),
    key(0.4, stepOut, { root: [0.2 * d, -0.16, 0] }),
    key(0.66, stepBack, { root: [-0.16 * d, -0.15, 0] }),
    key(0.82, stepBack, { root: [-0.2 * d, -0.12, 0] }),
    key(1, {}, { root: [0, 0, 0] }),
  ], [{ t: 0.4, name: 'plantA' }, { t: 0.8, name: 'plantB' }],
    { layer: 'override', region: 'full' });
}

/** Post-up: back to the basket, hips low, seal arm out, ball high and away. */
function postUp(hand: 'left' | 'right'): Clip {
  const s = hand === 'left' ? -1 : 1;
  const sealArm: BoneName = hand === 'left' ? 'upperArmR' : 'upperArmL';
  const sealFore: BoneName = hand === 'left' ? 'foreArmR' : 'foreArmL';
  const ballArm: BoneName = hand === 'left' ? 'upperArmL' : 'upperArmR';
  const ballFore: BoneName = hand === 'left' ? 'foreArmL' : 'foreArmR';
  const base: PoseMap = {
    hips: [0.16, -0.08 * s, 0.04 * s],
    thighL: [-0.6, 0.3, 0.16], shinL: [1.0, 0, 0], footL: [-0.3, 0.22, 0],
    thighR: [-0.58, -0.3, -0.16], shinR: [0.98, 0, 0], footR: [-0.29, -0.22, 0],
    spine: [0.2, -0.1 * s, 0], chest: [0.13, -0.08 * s, 0], upperChest: [0.08, -0.06 * s, 0],
    [sealArm]: [0.3, 0.2 * s, -1.24 * s], [sealFore]: [-0.5, 0, 0],
    [ballArm]: [0.3, -0.2 * s, 0.72 * s], [ballFore]: [-1.36, 0, 0],
    head: [-0.06, -0.4 * s, 0],
  };
  const push: PoseMap = {
    ...base,
    hips: [0.2, -0.1 * s, 0.05 * s],
    spine: [0.25, -0.12 * s, 0], chest: [0.16, -0.1 * s, 0],
    [sealArm]: [0.24, 0.24 * s, -1.36 * s], [sealFore]: [-0.4, 0, 0],
    thighL: [-0.66, 0.3, 0.16], shinL: [1.08, 0, 0],
    thighR: [-0.64, -0.3, -0.16], shinR: [1.06, 0, 0],
  };
  return clip('postUp', 1.6, true, [
    key(0, base, { root: [0, -0.14, 0] }),
    key(0.38, push, { root: [0, -0.175, 0] }),
    key(0.72, base, { root: [0, -0.135, 0] }),
    key(1, base, { root: [0, -0.14, 0] }),
  ], { layer: 'override', region: 'full' });
}

/** Drop step: pivot off the front foot and open to the rim. */
function dropStep(hand: 'left' | 'right'): Clip {
  const s = hand === 'left' ? -1 : 1;
  return clip('dropStep', 0.82, false, [
    key(0, {}, { root: [0, 0, 0], rootYaw: 0 }),
    key(0.24, {
      hips: [0.2, -0.16 * s, 0.06 * s],
      thighL: [-0.66, 0.3, 0.16], shinL: [1.08, 0, 0],
      thighR: [-0.6, -0.3, -0.16], shinR: [1.02, 0, 0],
      spine: [0.24, -0.12 * s, 0], chest: [0.15, -0.1 * s, 0],
      upperArmL: [0.28, 0, -0.7], foreArmL: [-1.3, 0, 0],
      upperArmR: [0.28, 0, 0.7], foreArmR: [-1.3, 0, 0],
      head: [-0.08, -0.3 * s, 0],
    }, { root: [0, -0.18, 0], rootYaw: -0.35 * s }),
    key(0.56, {
      hips: [0.14, 0.1 * s, -0.06 * s],
      thighL: [-0.9, 0.22, 0.1], shinL: [0.9, 0, 0],
      thighR: [-0.4, -0.22, -0.1], shinR: [0.86, 0, 0],
      spine: [0.16, 0.08 * s, 0], chest: [0.1, 0.06 * s, 0],
      upperArmL: [0.2, 0, -0.5], foreArmL: [-1.2, 0, 0],
      upperArmR: [0.2, 0, 0.5], foreArmR: [-1.2, 0, 0],
      head: [-0.14, 0.2 * s, 0],
    }, { root: [0, -0.15, 0], rootYaw: -1.15 * s }),
    key(1, {}, { root: [0, -0.03, 0], rootYaw: -1.45 * s }),
  ], [{ t: 0.5, name: 'plantA' }], { layer: 'override', region: 'full' });
}

/** Defensive close-out: choppy short steps, hands high, chest up, hips sinking. */
function closeout(): Clip {
  const chop = (lead: 1 | -1): PoseMap => ({
    hips: [0.1, 0, 0.03 * lead],
    thighL: [lead > 0 ? -0.62 : -0.4, 0.24, 0.12], shinL: [lead > 0 ? 0.9 : 0.72, 0, 0], footL: [-0.24, 0.18, 0],
    thighR: [lead > 0 ? -0.4 : -0.62, -0.24, -0.12], shinR: [lead > 0 ? 0.72 : 0.9, 0, 0], footR: [-0.24, -0.18, 0],
    spine: [0.14, 0, 0], chest: [0.09, 0, 0], upperChest: [0.06, 0, 0],
    clavicleL: [-0.06, 0, -0.06], upperArmL: [-1.9, -0.1, -0.42], foreArmL: [-0.5, 0, 0], handL: [-0.2, 0, -0.1],
    clavicleR: [-0.06, 0, 0.06], upperArmR: [-1.9, 0.1, 0.42], foreArmR: [-0.5, 0, 0], handR: [-0.2, 0, 0.1],
    neck: [-0.06, 0, 0], head: [-0.1, 0, 0],
  });
  return clip('closeout', 0.92, false, [
    key(0, {}, { root: [0, 0, 0] }),
    key(0.18, chop(1), { root: [0, -0.09, 0] }),
    key(0.36, chop(-1), { root: [0, -0.12, 0] }),
    key(0.54, chop(1), { root: [0, -0.145, 0] }),
    key(0.72, chop(-1), { root: [0, -0.165, 0] }),
    key(1, chop(1), { root: [0, -0.16, 0] }),
  ], [{ t: 0.2, name: 'plantA' }, { t: 0.56, name: 'plantB' }],
    { layer: 'override', region: 'full' });
}

/** Contest: straight up, hand high, no body lean — verticality. */
function contest(hand: 'left' | 'right', jump: boolean): Clip {
  const s = hand === 'left' ? -1 : 1;
  const arm: BoneName = hand === 'left' ? 'upperArmL' : 'upperArmR';
  const fore: BoneName = hand === 'left' ? 'foreArmL' : 'foreArmR';
  const off: BoneName = hand === 'left' ? 'upperArmR' : 'upperArmL';
  const offFore: BoneName = hand === 'left' ? 'foreArmR' : 'foreArmL';
  const lift = jump ? 0.44 : 0;
  const rootAt = (u: number, extra = 0): [number, number, number] => [0, arcY(lift, 0.28, 0.74, u) + extra, 0];
  const up: PoseMap = {
    hips: [-0.02, 0, 0],
    thighL: [-0.16, 0.06, 0.04], shinL: [0.26, 0, 0], footL: [jump ? 0.24 : -0.06, 0, 0],
    thighR: [-0.14, -0.06, -0.04], shinR: [0.24, 0, 0], footR: [jump ? 0.24 : -0.06, 0, 0],
    spine: [-0.04, 0, 0], chest: [-0.03, 0, 0], upperChest: [-0.02, 0, 0],
    [arm]: [-2.86, -0.04 * s, 0.06 * s], [fore]: [-0.1, 0, 0],
    [off]: [-1.7, 0.06 * s, -0.5 * s], [offFore]: [-0.4, 0, 0],
    neck: [-0.16, 0, 0], head: [-0.4, 0, 0],
  };
  return clip(jump ? 'contestJump' : 'contest', jump ? 0.95 : 0.8, false, [
    key(0, {}, { root: [0, 0, 0] }),
    key(0.16, {
      hips: [0.16, 0, 0],
      thighL: [-0.5, 0.08, 0.05], shinL: [0.92, 0, 0],
      thighR: [-0.48, -0.08, -0.05], shinR: [0.9, 0, 0],
      spine: [0.18, 0, 0], chest: [0.11, 0, 0],
      [arm]: [-1.0, 0, 0.2 * s], [fore]: [-0.5, 0, 0],
      [off]: [-0.4, 0, -0.4 * s], [offFore]: [-0.6, 0, 0],
    }, { root: rootAt(0.16, -0.15) }),
    key(0.4, up, { root: rootAt(0.4) }),
    key(0.62, up, { root: rootAt(0.62) }),
    key(0.84, {
      hips: [0.14, 0, 0],
      thighL: [-0.44, 0.07, 0.05], shinL: [0.8, 0, 0],
      thighR: [-0.42, -0.07, -0.05], shinR: [0.78, 0, 0],
      spine: [0.14, 0, 0],
      [arm]: [-1.2, 0, 0.24 * s], [fore]: [-0.5, 0, 0],
    }, { root: [0, -0.1, 0] }),
    key(1, {}, { root: [0, 0, 0] }),
  ], jump ? [{ t: 0.44, name: 'apex' }, { t: 0.84, name: 'land' }] : [],
    { layer: 'override', region: 'full' });
}

/** Box-out: hips back into the defender, wide base, arms out and level. */
function boxOut(): Clip {
  const wide: PoseMap = {
    hips: [0.26, 0, 0],
    thighL: [-0.72, 0.38, 0.2], shinL: [1.06, 0, 0], footL: [-0.3, 0.3, 0],
    thighR: [-0.7, -0.38, -0.2], shinR: [1.04, 0, 0], footR: [-0.29, -0.3, 0],
    spine: [0.26, 0, 0], chest: [0.16, 0, 0], upperChest: [0.1, 0, 0],
    clavicleL: [0, 0, -0.14], upperArmL: [0.06, -0.2, -1.42], foreArmL: [-0.7, 0, 0], handL: [-0.2, 0, -0.2],
    clavicleR: [0, 0, 0.14], upperArmR: [0.06, 0.2, 1.42], foreArmR: [-0.7, 0, 0], handR: [-0.2, 0, 0.2],
    neck: [-0.14, 0, 0], head: [-0.24, 0, 0],
  };
  const drive: PoseMap = {
    ...wide,
    hips: [0.32, 0, 0],
    thighL: [-0.8, 0.4, 0.21], shinL: [1.16, 0, 0],
    thighR: [-0.78, -0.4, -0.21], shinR: [1.14, 0, 0],
    spine: [0.3, 0, 0],
    upperArmL: [0.02, -0.24, -1.5], upperArmR: [0.02, 0.24, 1.5],
  };
  return clip('boxOut', 1.3, true, [
    key(0, wide, { root: [0, -0.2, -0.03] }),
    key(0.34, drive, { root: [0, -0.23, -0.06] }),
    key(0.7, wide, { root: [0, -0.195, -0.028] }),
    key(1, wide, { root: [0, -0.2, -0.03] }),
  ], { layer: 'override', region: 'full' });
}

/** Rebound: two-foot gather, both hands high, snatch and chin the ball. */
function rebound(): Clip {
  const rootAt = (u: number, extra = 0): [number, number, number] => [0, arcY(0.5, 0.28, 0.76, u) + extra, 0];
  return clip('rebound', 1.02, false, [
    key(0, {}, { root: [0, 0, 0] }),
    key(0.18, {
      hips: [0.3, 0, 0],
      thighL: [-0.78, 0.07, 0.05], shinL: [1.3, 0, 0], footL: [-0.46, 0, 0],
      thighR: [-0.78, -0.07, -0.05], shinR: [1.3, 0, 0], footR: [-0.46, 0, 0],
      spine: [0.3, 0, 0], chest: [0.18, 0, 0], upperChest: [0.11, 0, 0],
      upperArmL: [0.8, 0, -0.3], foreArmL: [-0.6, 0, 0],
      upperArmR: [0.8, 0, 0.3], foreArmR: [-0.6, 0, 0],
      head: [-0.16, 0, 0],
    }, { root: rootAt(0.18, -0.26) }),
    key(0.44, {
      hips: [-0.05, 0, 0],
      thighL: [-0.34, 0.06, 0.04], shinL: [0.5, 0, 0], footL: [0.28, 0, 0],
      thighR: [-0.3, -0.06, -0.04], shinR: [0.46, 0, 0], footR: [0.28, 0, 0],
      spine: [-0.1, 0, 0], chest: [-0.07, 0, 0], upperChest: [-0.05, 0, 0],
      upperArmL: [-2.72, -0.08, -0.24], foreArmL: [-0.16, 0, 0], handL: [-0.2, 0, -0.1],
      upperArmR: [-2.72, 0.08, 0.24], foreArmR: [-0.16, 0, 0], handR: [-0.2, 0, 0.1],
      neck: [-0.18, 0, 0], head: [-0.46, 0, 0],
    }, { root: rootAt(0.44) }),
    key(0.58, {
      // Snatch: elbows fold in, ball to the chin.
      hips: [0.02, 0, 0],
      thighL: [-0.46, 0.06, 0.04], shinL: [0.66, 0, 0], footL: [0.22, 0, 0],
      thighR: [-0.44, -0.06, -0.04], shinR: [0.64, 0, 0], footR: [0.22, 0, 0],
      spine: [0.04, 0, 0], chest: [0.03, 0, 0],
      upperArmL: [-1.5, -0.16, -0.66], foreArmL: [-1.6, 0, 0], handL: [-0.3, 0, -0.2],
      upperArmR: [-1.5, 0.16, 0.66], foreArmR: [-1.6, 0, 0], handR: [-0.3, 0, 0.2],
      head: [-0.16, 0, 0],
    }, { root: rootAt(0.58) }),
    key(0.8, {
      hips: [0.14, 0, 0],
      thighL: [-0.6, 0.1, 0.06], shinL: [0.98, 0, 0], footL: [-0.28, 0, 0],
      thighR: [-0.58, -0.1, -0.06], shinR: [0.96, 0, 0], footR: [-0.28, 0, 0],
      spine: [0.18, 0, 0], chest: [0.11, 0, 0],
      upperArmL: [-0.7, -0.2, -0.8], foreArmL: [-1.8, 0, 0],
      upperArmR: [-0.7, 0.2, 0.8], foreArmR: [-1.8, 0, 0],
    }, { root: [0, -0.14, 0] }),
    key(1, {}, { root: [0, 0, 0] }),
  ], [{ t: 0.46, name: 'apex' }, { t: 0.52, name: 'catch' }, { t: 0.8, name: 'land' }],
    { layer: 'override', region: 'full' });
}

/** Passes ride on the upper body so a player can pass on the move. */
function pass(kind: 'chest' | 'bounce' | 'overhead', hand: 'left' | 'right'): Clip {
  const s = hand === 'left' ? -1 : 1;
  const over = kind === 'overhead';
  const bounce = kind === 'bounce';
  const wind: PoseMap = over
    ? {
        upperArmL: [-2.3, -0.08, -0.34], foreArmL: [-1.1, 0, 0], handL: [-0.3, 0, -0.14],
        upperArmR: [-2.3, 0.08, 0.34], foreArmR: [-1.1, 0, 0], handR: [-0.3, 0, 0.14],
        spine: [-0.1, 0, 0], chest: [-0.07, 0, 0], upperChest: [-0.05, 0, 0],
      }
    : {
        upperArmL: [0.46, -0.1, -0.44], foreArmL: [-1.78, 0, 0], handL: [-0.3, 0, -0.16],
        upperArmR: [0.46, 0.1, 0.44], foreArmR: [-1.78, 0, 0], handR: [-0.3, 0, 0.16],
        spine: [0.08, -0.16 * s, 0], chest: [0.05, -0.13 * s, 0], upperChest: [0.03, -0.1 * s, 0],
      };
  const armX = over ? -1.7 : bounce ? 0.24 : -0.42;
  const release: PoseMap = over
    ? {
        upperArmL: [armX, -0.04, -0.24], foreArmL: [-0.2, 0, 0], handL: [0.4, 0, -0.06],
        upperArmR: [armX, 0.04, 0.24], foreArmR: [-0.2, 0, 0], handR: [0.4, 0, 0.06],
        spine: [0.1, 0, 0], chest: [0.07, 0, 0], upperChest: [0.05, 0, 0],
      }
    : {
        upperArmL: [armX, -0.04, -0.16], foreArmL: [-0.2, 0, 0], handL: [bounce ? 0.4 : 0.2, 0, 0],
        upperArmR: [armX, 0.04, 0.16], foreArmR: [-0.2, 0, 0], handR: [bounce ? 0.4 : 0.2, 0, 0],
        spine: [bounce ? 0.16 : -0.03, 0.18 * s, 0], chest: [bounce ? 0.1 : -0.02, 0.15 * s, 0],
        upperChest: [0.02, 0.12 * s, 0],
      };
  // Overshoot past the release: the arms keep going and the wrists finish over.
  const follow: PoseMap = {
    ...release,
    upperArmL: [armX - 0.12, -0.02, -0.12],
    upperArmR: [armX - 0.12, 0.02, 0.12],
    foreArmL: [-0.12, 0, 0], foreArmR: [-0.12, 0, 0],
    handL: [0.5, 0, 0], handR: [0.5, 0, 0],
  };
  return clip(
    over ? 'overheadPass' : bounce ? 'bouncePass' : 'pass',
    0.5,
    false,
    [
      key(0, {}),
      key(0.3, wind),
      key(0.5, release),
      key(0.66, follow),
      key(1, {}),
    ],
    [{ t: 0.5, name: 'release' }],
    { layer: 'additive', region: 'upper' },
  );
}

/** Shot block: gather, explode, and a long arm at the top with a late swipe. */
function block(hand: 'left' | 'right'): Clip {
  const s = hand === 'left' ? -1 : 1;
  const arm: BoneName = hand === 'left' ? 'upperArmL' : 'upperArmR';
  const fore: BoneName = hand === 'left' ? 'foreArmL' : 'foreArmR';
  const wrist: BoneName = hand === 'left' ? 'handL' : 'handR';
  const off: BoneName = hand === 'left' ? 'upperArmR' : 'upperArmL';
  const offFore: BoneName = hand === 'left' ? 'foreArmR' : 'foreArmL';
  const rootAt = (u: number, extra = 0): [number, number, number] => [0, arcY(0.56, 0.26, 0.76, u) + extra, 0];
  return clip('block', 1.0, false, [
    key(0, {}, { root: [0, 0, 0] }),
    key(0.16, {
      hips: [0.28, 0, 0],
      thighL: [-0.74, 0.07, 0.05], shinL: [1.26, 0, 0], footL: [-0.44, 0, 0],
      thighR: [-0.74, -0.07, -0.05], shinR: [1.26, 0, 0], footR: [-0.44, 0, 0],
      spine: [0.28, 0, 0], chest: [0.17, 0, 0],
      upperArmL: [0.7, 0, -0.28], foreArmL: [-0.5, 0, 0],
      upperArmR: [0.7, 0, 0.28], foreArmR: [-0.5, 0, 0],
      head: [-0.14, 0, 0],
    }, { root: rootAt(0.16, -0.24) }),
    key(0.42, {
      hips: [-0.06, -0.06 * s, 0],
      thighL: [-0.24, 0.06, 0.04], shinL: [0.36, 0, 0], footL: [0.3, 0, 0],
      thighR: [-0.22, -0.06, -0.04], shinR: [0.34, 0, 0], footR: [0.3, 0, 0],
      spine: [-0.12, -0.06 * s, 0], chest: [-0.08, -0.05 * s, 0], upperChest: [-0.06, -0.04 * s, 0],
      [arm]: [-2.92, -0.06 * s, 0.06 * s], [fore]: [-0.08, 0, 0], [wrist]: [-0.3, 0, 0],
      [off]: [-1.8, 0.08 * s, -0.44 * s], [offFore]: [-0.4, 0, 0],
      neck: [-0.2, 0, 0], head: [-0.48, -0.04 * s, 0],
    }, { root: rootAt(0.42) }),
    key(0.56, {
      hips: [-0.06, -0.1 * s, 0],
      thighL: [-0.3, 0.06, 0.04], shinL: [0.42, 0, 0], footL: [0.3, 0, 0],
      thighR: [-0.28, -0.06, -0.04], shinR: [0.4, 0, 0], footR: [0.3, 0, 0],
      spine: [-0.1, -0.1 * s, 0], chest: [-0.07, -0.08 * s, 0],
      // The swipe.
      [arm]: [-2.8, -0.24 * s, 0.34 * s], [fore]: [-0.16, 0, 0], [wrist]: [0.2, 0, 0.2 * s],
      [off]: [-1.6, 0.1 * s, -0.5 * s], [offFore]: [-0.5, 0, 0],
      head: [-0.44, -0.08 * s, 0],
    }, { root: rootAt(0.56) }),
    key(0.82, {
      hips: [0.2, 0, 0],
      thighL: [-0.66, 0.07, 0.05], shinL: [1.08, 0, 0], footL: [-0.3, 0, 0],
      thighR: [-0.64, -0.07, -0.05], shinR: [1.06, 0, 0], footR: [-0.3, 0, 0],
      spine: [0.22, 0, 0], chest: [0.13, 0, 0],
      [arm]: [-1.2, 0, 0.24 * s], [fore]: [-0.6, 0, 0],
      [off]: [-0.6, 0, -0.4 * s], [offFore]: [-0.7, 0, 0],
    }, { root: [0, -0.16, 0] }),
    key(1, {}, { root: [0, 0, 0] }),
  ], [{ t: 0.44, name: 'apex' }, { t: 0.82, name: 'land' }],
    { layer: 'override', region: 'full' });
}

// --- Transitions ------------------------------------------------------------

/**
 * A hard stop. Two-foot chop, hips drop and go *behind* the feet, torso leans
 * back against the direction of travel, then settles. This is the difference
 * between a player who stops and one who slides to a halt.
 */
function hardStop(): Clip {
  return clip('hardStop', 0.62, false, [
    key(0, {}, { root: [0, 0, 0] }),
    key(0.2, {
      hips: [-0.16, 0, 0],
      // Lead leg spears forward to catch the weight.
      thighL: [-0.95, 0.16, 0.09], shinL: [0.5, 0, 0], footL: [-0.5, 0, 0],
      thighR: [-0.2, -0.16, -0.09], shinR: [0.9, 0, 0], footR: [0.24, 0, 0],
      spine: [-0.2, 0, 0], chest: [-0.14, 0, 0], upperChest: [-0.1, 0, 0],
      upperArmL: [-0.5, -0.2, -0.7], foreArmL: [-0.7, 0, 0],
      upperArmR: [-0.5, 0.2, 0.7], foreArmR: [-0.7, 0, 0],
      neck: [0.08, 0, 0], head: [0.14, 0, 0],
    }, { root: [0, -0.13, -0.05] }),
    key(0.44, {
      hips: [0.06, 0, 0],
      thighL: [-0.86, 0.2, 0.11], shinL: [1.12, 0, 0], footL: [-0.36, 0, 0],
      thighR: [-0.44, -0.2, -0.11], shinR: [1.06, 0, 0], footR: [-0.1, 0, 0],
      spine: [0.06, 0, 0], chest: [0.04, 0, 0], upperChest: [0.03, 0, 0],
      upperArmL: [-0.2, -0.16, -0.6], foreArmL: [-0.9, 0, 0],
      upperArmR: [-0.2, 0.16, 0.6], foreArmR: [-0.9, 0, 0],
      head: [-0.04, 0, 0],
    }, { root: [0, -0.2, -0.03] }),
    key(0.72, {
      hips: [0.12, 0, 0],
      thighL: [-0.56, 0.2, 0.11], shinL: [0.98, 0, 0],
      thighR: [-0.5, -0.2, -0.11], shinR: [0.94, 0, 0],
      spine: [0.14, 0, 0], chest: [0.09, 0, 0],
      upperArmL: [0.02, -0.1, -0.44], foreArmL: [-0.7, 0, 0],
      upperArmR: [0.02, 0.1, 0.44], foreArmR: [-0.7, 0, 0],
    }, { root: [0, -0.15, 0] }),
    key(1, {}, { root: [0, 0, 0] }),
  ], [{ t: 0.18, name: 'plantA' }, { t: 0.36, name: 'plantB' }],
    { layer: 'override', region: 'full' });
}

/** Jump stop: both feet land together, hips sink, base widens, weight settles. */
function jumpStop(): Clip {
  return clip('jumpStop', 0.66, false, [
    key(0, {}, { root: [0, 0, 0] }),
    key(0.16, {
      hips: [0.04, 0, 0],
      thighL: [-0.62, 0.1, 0.06], shinL: [0.9, 0, 0], footL: [0.12, 0, 0],
      thighR: [-0.6, -0.1, -0.06], shinR: [0.88, 0, 0], footR: [0.12, 0, 0],
      spine: [0.02, 0, 0],
      upperArmL: [-0.4, -0.1, -0.5], foreArmL: [-1.0, 0, 0],
      upperArmR: [-0.4, 0.1, 0.5], foreArmR: [-1.0, 0, 0],
    }, { root: [0, 0.075, 0] }),
    key(0.34, {
      hips: [0.16, 0, 0],
      thighL: [-0.68, 0.3, 0.17], shinL: [1.16, 0, 0], footL: [-0.28, 0.2, 0],
      thighR: [-0.66, -0.3, -0.17], shinR: [1.14, 0, 0], footR: [-0.27, -0.2, 0],
      spine: [0.2, 0, 0], chest: [0.12, 0, 0], upperChest: [0.08, 0, 0],
      upperArmL: [0.1, -0.16, -0.78], foreArmL: [-1.1, 0, 0],
      upperArmR: [0.1, 0.16, 0.78], foreArmR: [-1.1, 0, 0],
      head: [-0.1, 0, 0],
    }, { root: [0, -0.2, -0.02] }),
    key(0.6, {
      hips: [0.12, 0, 0],
      thighL: [-0.56, 0.3, 0.17], shinL: [0.98, 0, 0], footL: [-0.24, 0.2, 0],
      thighR: [-0.54, -0.3, -0.17], shinR: [0.96, 0, 0], footR: [-0.23, -0.2, 0],
      spine: [0.16, 0, 0], chest: [0.1, 0, 0],
      upperArmL: [0.12, -0.12, -0.66], foreArmL: [-1.05, 0, 0],
      upperArmR: [0.12, 0.12, 0.66], foreArmR: [-1.05, 0, 0],
    }, { root: [0, -0.15, 0] }),
    key(1, {}, { root: [0, -0.02, 0] }),
  ], [{ t: 0.3, name: 'plantA' }],
    { layer: 'override', region: 'full' });
}

/**
 * Plant and cut. The outside foot lands *wide*, the knee and ankle take the
 * load, the whole body leans into the new heading and then pushes off it. A
 * player who changes direction without this is sliding, not cutting.
 */
function cut(dir: 1 | -1): Clip {
  const d = dir;
  const plant: PoseMap = {
    hips: [0.14, 0.1 * d, 0.16 * d],
    thighL: [d > 0 ? -0.42 : -0.66, 0.5 * d, 0.3 * d], shinL: [d > 0 ? 0.86 : 1.0, 0, 0],
    footL: [-0.24, 0.32 * d, 0.22 * d],
    thighR: [d > 0 ? -0.66 : -0.42, 0.5 * d, 0.3 * d], shinR: [d > 0 ? 1.0 : 0.86, 0, 0],
    footR: [-0.24, 0.32 * d, 0.22 * d],
    spine: [0.16, 0.14 * d, -0.24 * d], chest: [0.1, 0.12 * d, -0.18 * d], upperChest: [0.06, 0.1 * d, -0.12 * d],
    clavicleL: [0, 0, -0.06], upperArmL: [0.1, -0.24 * d, -0.66], foreArmL: [-1.0, 0, 0],
    clavicleR: [0, 0, 0.06], upperArmR: [0.1, -0.24 * d, 0.66], foreArmR: [-1.0, 0, 0],
    neck: [-0.06, 0.1 * d, 0], head: [-0.1, 0.34 * d, -0.1 * d],
  };
  const drive: PoseMap = {
    hips: [0.08, 0.16 * d, 0.06 * d],
    thighL: [d > 0 ? -0.86 : -0.1, 0.3 * d, 0.14 * d], shinL: [d > 0 ? 0.9 : 0.5, 0, 0],
    thighR: [d > 0 ? -0.1 : -0.86, 0.3 * d, 0.14 * d], shinR: [d > 0 ? 0.5 : 0.9, 0, 0],
    spine: [0.2, 0.16 * d, -0.1 * d], chest: [0.13, 0.14 * d, -0.07 * d], upperChest: [0.08, 0.12 * d, -0.05 * d],
    upperArmL: [d > 0 ? -0.4 : 0.5, -0.1 * d, -0.28], foreArmL: [-0.9, 0, 0],
    upperArmR: [d > 0 ? 0.5 : -0.4, -0.1 * d, 0.28], foreArmR: [-0.9, 0, 0],
    head: [-0.08, 0.3 * d, 0],
  };
  return clip(`cut${d > 0 ? 'L' : 'R'}`, 0.6, false, [
    key(0, {}, { root: [0, 0, 0], rootYaw: 0 }),
    key(0.24, plant, { root: [0.06 * d, -0.155, 0], rootYaw: 0.1 * d }),
    key(0.42, plant, { root: [0.09 * d, -0.185, 0], rootYaw: 0.24 * d }),
    key(0.68, drive, { root: [-0.02 * d, -0.11, 0], rootYaw: 0.4 * d }),
    key(1, {}, { root: [0, -0.02, 0], rootYaw: 0.46 * d }),
  ], [{ t: 0.26, name: 'plantA' }, { t: 0.6, name: 'plantB' }],
    { layer: 'override', region: 'full' });
}

/** Pivot on the ball of the front foot, ball swung through, shoulders leading. */
function pivot(dir: 1 | -1, turn = Math.PI * 0.65): Clip {
  const d = dir;
  const mid: PoseMap = {
    hips: [0.12, 0.14 * d, 0.04 * d],
    thighL: [d > 0 ? -0.7 : -0.34, 0.22, 0.12], shinL: [d > 0 ? 0.94 : 0.7, 0, 0],
    thighR: [d > 0 ? -0.34 : -0.7, -0.22, -0.12], shinR: [d > 0 ? 0.7 : 0.94, 0, 0],
    spine: [0.14, 0.16 * d, 0], chest: [0.09, 0.14 * d, 0], upperChest: [0.06, 0.12 * d, 0],
    upperArmL: [0.24, -0.14 * d, -0.62], foreArmL: [-1.24, 0, 0],
    upperArmR: [0.24, -0.14 * d, 0.62], foreArmR: [-1.24, 0, 0],
    head: [-0.08, 0.34 * d, 0],
  };
  return clip('pivot', 0.7, false, [
    key(0, {}, { root: [0, 0, 0], rootYaw: 0 }),
    key(0.3, mid, { root: [0, -0.11, 0], rootYaw: turn * 0.32 * d }),
    key(0.66, mid, { root: [0, -0.13, 0], rootYaw: turn * 0.86 * d }),
    key(1, {}, { root: [0, -0.03, 0], rootYaw: turn * d }),
  ], [{ t: 0.34, name: 'plantA' }],
    { layer: 'override', region: 'full' });
}

/** Explosive first step out of a standstill. */
function accelBurst(dir: 1 | -1): Clip {
  const d = dir;
  return clip('accelBurst', 0.56, false, [
    key(0, {}, { root: [0, 0, 0] }),
    key(0.14, {
      hips: [0.3, 0.06 * d, 0],
      thighL: [d > 0 ? -0.34 : -0.9, 0.12, 0.07], shinL: [d > 0 ? 0.96 : 0.8, 0, 0], footL: [-0.32, 0, 0],
      thighR: [d > 0 ? -0.9 : -0.34, -0.12, -0.07], shinR: [d > 0 ? 0.8 : 0.96, 0, 0], footR: [-0.32, 0, 0],
      spine: [0.34, 0.06 * d, 0], chest: [0.22, 0.05 * d, 0], upperChest: [0.14, 0.04 * d, 0],
      upperArmL: [d > 0 ? 0.9 : -0.8, 0, -0.16], foreArmL: [-1.25, 0, 0],
      upperArmR: [d > 0 ? -0.8 : 0.9, 0, 0.16], foreArmR: [-1.25, 0, 0],
      neck: [-0.2, 0, 0], head: [-0.28, 0, 0],
    }, { root: [0, -0.12, 0] }),
    key(0.44, {
      hips: [0.26, -0.06 * d, 0],
      thighL: [d > 0 ? -1.15 : -0.1, 0.1, 0.06], shinL: [d > 0 ? 1.15 : 0.4, 0, 0], footL: [0.26, 0, 0],
      thighR: [d > 0 ? -0.1 : -1.15, -0.1, -0.06], shinR: [d > 0 ? 0.4 : 1.15, 0, 0], footR: [0.26, 0, 0],
      spine: [0.3, -0.05 * d, 0], chest: [0.2, -0.04 * d, 0], upperChest: [0.13, -0.03 * d, 0],
      upperArmL: [d > 0 ? -0.85 : 0.95, 0, -0.14], foreArmL: [-1.35, 0, 0],
      upperArmR: [d > 0 ? 0.95 : -0.85, 0, 0.14], foreArmR: [-1.35, 0, 0],
      neck: [-0.18, 0, 0], head: [-0.26, 0, 0],
    }, { root: [0, -0.05, 0] }),
    key(1, {}, { root: [0, 0, 0] }),
  ], [{ t: 0.16, name: 'plantA' }, { t: 0.46, name: 'plantB' }],
    { layer: 'override', region: 'full' });
}

// --- Handle work ------------------------------------------------------------

function crossover(hand: 'left' | 'right', between: boolean): Clip {
  const s = hand === 'left' ? -1 : 1;
  const low: PoseMap = {
    hips: [0.2, 0.16 * s, 0.06 * s],
    thighL: [-0.62, 0.4, 0.2], shinL: [1.02, 0, 0], footL: [-0.28, 0.3, 0],
    thighR: [-0.44, -0.28, -0.14], shinR: [0.84, 0, 0], footR: [-0.22, -0.22, 0],
    spine: [0.26, 0.18 * s, -0.08 * s], chest: [0.16, 0.15 * s, -0.05 * s], upperChest: [0.1, 0.12 * s, -0.03 * s],
    upperArmL: [0.44, -0.1, between ? -0.36 : -0.62], foreArmL: [-1.34, 0, 0], handL: [-0.4, 0, -0.2],
    upperArmR: [0.44, 0.1, between ? 0.36 : 0.62], foreArmR: [-1.34, 0, 0], handR: [-0.4, 0, 0.2],
    neck: [-0.1, 0, 0], head: [-0.14, 0.2 * s, 0],
  };
  return clip(between ? 'betweenLegs' : 'crossover', 0.5, false, [
    key(0, {}, { root: [0, 0, 0] }),
    key(0.3, low, { root: [0.03 * s, -0.15, 0] }),
    key(0.56, mirror(low), { root: [-0.03 * s, -0.16, 0] }),
    key(1, {}, { root: [0, -0.02, 0] }),
  ], { layer: 'override', region: 'full' });
}

function stepback(hand: 'left' | 'right'): Clip {
  const s = hand === 'left' ? -1 : 1;
  return clip('stepback', 0.66, false, [
    key(0, {}, { root: [0, 0, 0] }),
    key(0.2, {
      hips: [0.24, 0.1 * s, 0],
      thighL: [-0.74, 0.24, 0.13], shinL: [1.14, 0, 0], footL: [-0.34, 0.18, 0],
      thighR: [-0.68, -0.24, -0.13], shinR: [1.08, 0, 0], footR: [-0.32, -0.18, 0],
      spine: [0.28, 0.12 * s, 0], chest: [0.18, 0.1 * s, 0],
      upperArmL: [0.4, 0, -0.5], foreArmL: [-1.4, 0, 0],
      upperArmR: [0.4, 0, 0.5], foreArmR: [-1.4, 0, 0],
      head: [-0.14, 0.16 * s, 0],
    }, { root: [0, -0.18, 0.03] }),
    key(0.46, {
      hips: [-0.1, 0, 0],
      // Push off the front foot and ride back on the trail leg.
      thighL: [0.34, 0.14, 0.08], shinL: [0.72, 0, 0], footL: [0.28, 0, 0],
      thighR: [-0.42, -0.14, -0.08], shinR: [0.86, 0, 0], footR: [-0.24, 0, 0],
      spine: [-0.14, 0, 0], chest: [-0.1, 0, 0], upperChest: [-0.07, 0, 0],
      upperArmL: [0.16, 0, -0.42], foreArmL: [-1.5, 0, 0],
      upperArmR: [0.16, 0, 0.42], foreArmR: [-1.5, 0, 0],
      head: [0.02, 0, 0],
    }, { root: [0, -0.06, -0.16] }),
    key(0.74, {
      hips: [0.14, 0, 0],
      thighL: [-0.3, 0.2, 0.11], shinL: [0.72, 0, 0], footL: [-0.2, 0.14, 0],
      thighR: [-0.64, -0.2, -0.11], shinR: [1.04, 0, 0], footR: [-0.3, -0.14, 0],
      spine: [0.14, 0, 0], chest: [0.09, 0, 0],
      upperArmL: [0.3, 0, -0.44], foreArmL: [-1.42, 0, 0],
      upperArmR: [0.3, 0, 0.44], foreArmR: [-1.42, 0, 0],
    }, { root: [0, -0.16, -0.2] }),
    key(1, {}, { root: [0, -0.03, -0.2] }),
  ], [{ t: 0.22, name: 'plantA' }, { t: 0.7, name: 'plantB' }],
    { layer: 'override', region: 'full' });
}

function spinMove(dir: 1 | -1): Clip {
  const d = dir;
  return clip('spin', 0.72, false, [
    key(0, {}, { root: [0, 0, 0], rootYaw: 0 }),
    key(0.26, {
      hips: [0.2, 0.24 * d, 0.06 * d],
      thighL: [-0.68, 0.3, 0.16], shinL: [1.04, 0, 0],
      thighR: [-0.36, -0.24, -0.12], shinR: [0.76, 0, 0],
      spine: [0.24, 0.2 * d, -0.06 * d], chest: [0.15, 0.17 * d, 0], upperChest: [0.1, 0.14 * d, 0],
      upperArmL: [0.36, 0, -0.66], foreArmL: [-1.4, 0, 0],
      upperArmR: [0.36, 0, 0.66], foreArmR: [-1.4, 0, 0],
      head: [-0.1, 0.5 * d, 0],
    }, { root: [0, -0.16, 0], rootYaw: 0.9 * d }),
    key(0.58, {
      hips: [0.16, -0.1 * d, -0.05 * d],
      thighL: [-0.34, 0.26, 0.14], shinL: [0.78, 0, 0],
      thighR: [-0.72, -0.26, -0.14], shinR: [1.08, 0, 0],
      spine: [0.2, -0.08 * d, 0.04 * d], chest: [0.13, -0.06 * d, 0],
      upperArmL: [0.3, 0, -0.56], foreArmL: [-1.34, 0, 0],
      upperArmR: [0.3, 0, 0.56], foreArmR: [-1.34, 0, 0],
      head: [-0.12, 0.2 * d, 0],
    }, { root: [0, -0.14, 0], rootYaw: 2.2 * d }),
    key(1, {}, { root: [0, -0.02, 0], rootYaw: Math.PI * d }),
  ], [{ t: 0.28, name: 'plantA' }, { t: 0.62, name: 'plantB' }],
    { layer: 'override', region: 'full' });
}

function steal(hand: 'left' | 'right'): Clip {
  const s = hand === 'left' ? -1 : 1;
  const arm: BoneName = hand === 'left' ? 'upperArmL' : 'upperArmR';
  const fore: BoneName = hand === 'left' ? 'foreArmL' : 'foreArmR';
  const wrist: BoneName = hand === 'left' ? 'handL' : 'handR';
  return clip('steal', 0.46, false, [
    key(0, {}),
    key(0.24, {
      [arm]: [0.3, 0.2 * s, 0.5 * s], [fore]: [-1.2, 0, 0],
      spine: [0.1, -0.16 * s, 0], upperChest: [0.04, -0.12 * s, 0],
    }),
    key(0.46, {
      [arm]: [-1.15, -0.28 * s, 0.48 * s], [fore]: [-0.24, 0, 0], [wrist]: [-0.3, 0, 0.3 * s],
      spine: [0.16, 0.28 * s, 0], chest: [0.1, 0.22 * s, 0], upperChest: [0.06, 0.18 * s, 0],
      head: [-0.1, 0.3 * s, 0],
    }),
    key(0.68, {
      [arm]: [-0.7, -0.1 * s, 0.4 * s], [fore]: [-0.6, 0, 0],
      spine: [0.1, 0.14 * s, 0],
    }),
    key(1, {}),
  ], [{ t: 0.44, name: 'swipe' }], { layer: 'additive', region: 'upper' });
}

function celebrate(seedFlavour: number): Clip {
  const flavour = seedFlavour % 2;
  return flavour === 0
    ? clip('celebrate', 1.5, false, [
        key(0, {}, { root: [0, 0, 0] }),
        key(0.22, {
          upperArmL: [-2.4, -0.1, -0.46], foreArmL: [-0.4, 0, 0], handL: [-0.3, 0, 0],
          upperArmR: [-2.4, 0.1, 0.46], foreArmR: [-0.4, 0, 0], handR: [-0.3, 0, 0],
          spine: [-0.2, 0, 0], chest: [-0.13, 0, 0], upperChest: [-0.09, 0, 0],
          thighL: [-0.2, 0.1, 0.06], shinL: [0.4, 0, 0],
          thighR: [-0.18, -0.1, -0.06], shinR: [0.38, 0, 0],
          neck: [-0.14, 0, 0], head: [-0.34, 0, 0],
        }, { root: [0, 0.09, 0] }),
        key(0.5, {
          upperArmL: [-1.85, -0.2, -0.8], foreArmL: [-1.0, 0, 0],
          upperArmR: [-1.85, 0.2, 0.8], foreArmR: [-1.0, 0, 0],
          spine: [-0.1, 0.16, 0], chest: [-0.07, 0.13, 0], upperChest: [-0.05, 0.1, 0],
          head: [-0.2, 0.2, 0],
        }, { root: [0, -0.03, 0] }),
        key(0.78, {
          upperArmL: [-2.1, -0.14, -0.6], foreArmL: [-0.7, 0, 0],
          upperArmR: [-2.1, 0.14, 0.6], foreArmR: [-0.7, 0, 0],
          spine: [-0.14, -0.14, 0], chest: [-0.09, -0.11, 0],
          head: [-0.26, -0.18, 0],
        }, { root: [0, 0.03, 0] }),
        key(1, {}, { root: [0, 0, 0] }),
      ], { layer: 'override', region: 'full' })
    : clip('celebrate', 1.4, false, [
        key(0, {}, { root: [0, 0, 0] }),
        key(0.26, {
          // Three-goggles / shimmy flavour.
          upperArmL: [-1.5, -0.4, -0.5], foreArmL: [-1.9, 0, 0], handL: [-0.5, 0, -0.3],
          upperArmR: [-1.5, 0.4, 0.5], foreArmR: [-1.9, 0, 0], handR: [-0.5, 0, 0.3],
          spine: [-0.08, 0.18, 0.1], chest: [-0.05, 0.14, 0.07], upperChest: [-0.04, 0.11, 0.05],
          head: [-0.12, 0.14, 0.05],
        }, { root: [0, -0.02, 0] }),
        key(0.56, {
          upperArmL: [-1.5, -0.4, -0.5], foreArmL: [-1.9, 0, 0], handL: [-0.5, 0, -0.3],
          upperArmR: [-1.5, 0.4, 0.5], foreArmR: [-1.9, 0, 0], handR: [-0.5, 0, 0.3],
          spine: [-0.08, -0.18, -0.1], chest: [-0.05, -0.14, -0.07], upperChest: [-0.04, -0.11, -0.05],
          head: [-0.12, -0.14, -0.05],
        }, { root: [0, -0.04, 0] }),
        key(0.82, {
          upperArmL: [-1.2, -0.3, -0.5], foreArmL: [-1.4, 0, 0],
          upperArmR: [-1.2, 0.3, 0.5], foreArmR: [-1.4, 0, 0],
          spine: [-0.04, 0.1, 0.05], head: [-0.08, 0.08, 0],
        }, { root: [0, -0.01, 0] }),
        key(1, {}, { root: [0, 0, 0] }),
      ], { layer: 'override', region: 'full' });
}

function dejected(): Clip {
  return clip('dejected', 1.5, false, [
    key(0, {}, { root: [0, 0, 0] }),
    key(0.3, {
      spine: [0.24, 0.06, 0], chest: [0.15, 0.04, 0], upperChest: [0.1, 0.03, 0],
      neck: [0.2, 0, 0], head: [0.28, 0.08, 0],
      clavicleL: [0.06, 0, 0.06], upperArmL: [0.24, 0, -0.04], foreArmL: [-0.24, 0, 0],
      clavicleR: [0.06, 0, -0.06], upperArmR: [0.24, 0, 0.04], foreArmR: [-0.24, 0, 0],
      thighL: [-0.1, 0.06, 0.04], shinL: [0.2, 0, 0],
      thighR: [-0.08, -0.06, -0.04], shinR: [0.18, 0, 0],
    }, { root: [0, -0.045, 0] }),
    key(0.66, {
      spine: [0.26, -0.05, 0], chest: [0.16, -0.03, 0], upperChest: [0.11, -0.02, 0],
      neck: [0.22, 0, 0], head: [0.3, -0.06, 0],
      upperArmL: [0.3, 0, -0.1], foreArmL: [-0.5, 0, 0],
      upperArmR: [0.3, 0, 0.1], foreArmR: [-0.5, 0, 0],
      thighL: [-0.12, 0.06, 0.04], shinL: [0.22, 0, 0],
      thighR: [-0.1, -0.06, -0.04], shinR: [0.2, 0, 0],
    }, { root: [0, -0.05, 0] }),
    key(1, {}, { root: [0, 0, 0] }),
  ], { layer: 'override', region: 'full' });
}

// ---------------------------------------------------------------------------

/**
 * Cheap memo: clips are pure functions of their parameters, and a possession
 * fires the same shot dozens of times. Quantising the continuous parameters
 * keeps the table small while still letting release quality change the form.
 */
const CACHE = new Map<string, Clip>();
function memo(k: string, make: () => Clip): Clip {
  let c = CACHE.get(k);
  if (!c) {
    c = make();
    CACHE.set(k, c);
  }
  return c;
}

/** Drops every cached clip. Only needed if a builder is edited at runtime. */
export function clearClipCache(): void {
  for (const c of CACHE.values()) invalidateClip(c);
  CACHE.clear();
}

const qBucket = (v: number): number => Math.round(clamp01(v) * 4) / 4;

/** Builds the clip for a discrete action, parameterised by how it was thrown. */
export function actionClip(kind: ActionKind, params: ActionParams, opts: AnimatorOptions): Clip | null {
  const hand = params.hand ?? opts.dominantHand;
  const q = qBucket(params.quality ?? 0.8);
  const power = qBucket(params.power ?? 0.7);
  const dir: 1 | -1 = (params.direction ?? 0) >= 0 ? 1 : -1;
  const style = params.style;

  switch (kind) {
    case 'shoot':
    case 'jumpShot':
      return memo(`shot-${q}-${hand}-${style ?? 'set'}`, () =>
        jumpShot(q, hand, style === 'quick' ? 'quick' : 'set'));
    case 'fadeaway':
      return memo(`fade-${q}-${hand}`, () => jumpShot(q, hand, 'fade'));
    case 'floater':
      return memo(`floater-${hand}`, () => floater(hand));
    case 'hookShot':
      return memo(`hook-${hand}`, () => hookShot(hand));
    case 'layup':
      return memo(`layup-${hand}`, () => layup(hand, 'overhand'));
    case 'fingerRoll':
      return memo(`roll-${hand}`, () => layup(hand, 'fingerRoll'));
    case 'reverseLayup':
      return memo(`reverse-${hand}`, () => layup(hand, 'reverse'));
    case 'euroStep':
      return memo(`euro-${hand}-${dir}`, () => euroStep(hand, dir));
    case 'dunk':
      return memo(`dunk-${power}-${hand}-${params.twoFoot ? 2 : 1}`, () =>
        dunk(power, hand, 'oneHand', params.twoFoot ?? false));
    case 'dunkTwoHand':
      return memo(`dunk2-${power}-${params.twoFoot === false ? 1 : 2}`, () =>
        dunk(power, hand, 'twoHand', params.twoFoot ?? true));
    case 'dunkTomahawk':
      return memo(`dunkT-${power}-${hand}-${params.twoFoot ? 2 : 1}`, () =>
        dunk(power, hand, 'tomahawk', params.twoFoot ?? false));
    case 'pass':
      return memo(`pass-${hand}`, () => pass('chest', hand));
    case 'bouncePass':
      return memo(`bpass-${hand}`, () => pass('bounce', hand));
    case 'overheadPass':
      return memo('opass', () => pass('overhead', hand));
    case 'block':
      return memo(`block-${hand}`, () => block(hand));
    case 'contest':
      return memo(`contest-${hand}-${params.twoFoot ? 1 : 0}`, () => contest(hand, params.twoFoot ?? false));
    case 'closeout':
      return memo('closeout', closeout);
    case 'boxOut':
      return memo('boxout', boxOut);
    case 'steal':
      return memo(`steal-${hand}`, () => steal(hand));
    case 'rebound':
      return memo('rebound', rebound);
    case 'postUp':
      return memo(`postup-${hand}`, () => postUp(hand));
    case 'dropStep':
      return memo(`dropstep-${hand}`, () => dropStep(hand));
    case 'crossover':
      return memo(`cross-${hand}-${style === 'betweenLegs' ? 1 : 0}`, () =>
        crossover(hand, style === 'betweenLegs'));
    case 'stepback':
      return memo(`stepback-${hand}`, () => stepback(hand));
    case 'spin':
      return memo(`spin-${dir}`, () => spinMove(dir));
    case 'cut':
      return memo(`cut-${dir}`, () => cut(dir));
    case 'jumpStop':
      return memo('jumpstop', jumpStop);
    case 'hardStop':
      return memo('hardstop', hardStop);
    case 'pivot':
      return memo(`pivot-${dir}`, () => pivot(dir));
    case 'accelBurst':
      return memo(`accel-${dir}`, () => accelBurst(dir));
    case 'celebrate':
      return memo(`celebrate-${opts.seed % 2}`, () => celebrate(opts.seed));
    case 'dejected':
      return memo('dejected', dejected);
    default:
      return null;
  }
}

/** Diagnostics: the measured stride table, used by the animation harness. */
export function strideTable(): Array<{ name: string; refSpeed: number; strideLength: number; stance: number }> {
  return GAITS.map((g) => ({
    name: g.name,
    refSpeed: g.refSpeed ?? 0,
    strideLength: g.strideLength ?? 0,
    stance: g.stanceFraction ?? 0,
  }));
}

