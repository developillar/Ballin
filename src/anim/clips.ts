/**
 * Code-authored animation clips.
 *
 * There is no animation file format in this project, so every clip is a small
 * set of keyframed joint rotations built here. The upside is that clips are
 * *parametric*: a jump shot can be rebuilt for a different release height, a
 * dunk for a different approach, without an artist round-trip.
 *
 * Convention (matches Skeleton.ts): each bone's +Y runs down the bone toward
 * its child and +Z faces forward, so for the limbs:
 *   - thigh / upperArm **negative X** swings the limb forward
 *   - shin **positive X** bends the knee (heel toward the seat)
 *   - foreArm **negative X** bends the elbow
 * Angles are radians.
 *
 * Owned by the animation agent.
 */

import type { BoneName } from '../entities/Skeleton';
import type { Clip, Keyframe } from './Pose';
import type { ActionKind, ActionParams, AnimatorOptions } from './AnimatorTypes';
import { clamp01 } from '../core/MathX';

type PoseMap = Partial<Record<BoneName, [number, number, number]>>;

function key(t: number, pose: PoseMap, extra: Partial<Keyframe> = {}): Keyframe {
  return { t, pose, ...extra };
}

function clip(name: string, duration: number, loop: boolean, keys: Keyframe[], events?: Clip['events']): Clip {
  return { name, duration, loop, keys, events };
}

/** Mirrors a pose's left/right limbs — halves the authoring for gait cycles. */
function mirror(p: PoseMap): PoseMap {
  const out: PoseMap = {};
  const swap: Record<string, string> = {
    thighL: 'thighR', thighR: 'thighL',
    shinL: 'shinR', shinR: 'shinL',
    footL: 'footR', footR: 'footL',
    upperArmL: 'upperArmR', upperArmR: 'upperArmL',
    foreArmL: 'foreArmR', foreArmR: 'foreArmL',
    handL: 'handR', handR: 'handL',
    clavicleL: 'clavicleR', clavicleR: 'clavicleL',
  };
  for (const [k, v] of Object.entries(p) as Array<[BoneName, [number, number, number]]>) {
    const name = (swap[k] ?? k) as BoneName;
    // Mirroring flips yaw and roll, keeps pitch.
    out[name] = swap[k] ? [v[0], -v[1], -v[2]] : [v[0], -v[1], -v[2]];
  }
  return out;
}

// ---------------------------------------------------------------------------
// Locomotion
// ---------------------------------------------------------------------------

/**
 * A gait cycle from one set of parameters. `swing` is how far the thighs
 * travel, `knee` how much the trailing knee folds, `arm` the counter-swing,
 * and `lean` the forward pitch of the torso — the four numbers that separate a
 * walk from a sprint.
 */
function gait(
  name: string,
  duration: number,
  swing: number,
  knee: number,
  arm: number,
  lean: number,
  bounce: number,
): Clip {
  // Contact → mid-stance → contact, then the mirrored half.
  const contact: PoseMap = {
    spine: [lean * 0.35, 0, 0],
    chest: [lean * 0.35, 0, 0],
    upperChest: [lean * 0.3, 0, 0],
    thighL: [-swing, 0.03, 0],
    shinL: [knee * 0.25, 0, 0],
    footL: [-0.14, 0, 0],
    thighR: [swing * 0.85, -0.03, 0],
    shinR: [knee, 0, 0],
    footR: [0.2, 0, 0],
    upperArmL: [arm * 0.9, 0, -0.16],
    foreArmL: [-arm * 0.75 - 0.5, 0, 0],
    upperArmR: [-arm, 0, 0.16],
    foreArmR: [-arm * 0.5 - 0.55, 0, 0],
    neck: [-lean * 0.5, 0, 0],
    head: [-lean * 0.35, 0, 0],
  };
  const pass: PoseMap = {
    spine: [lean * 0.4, 0, 0],
    chest: [lean * 0.4, 0, 0],
    upperChest: [lean * 0.32, 0, 0],
    thighL: [-swing * 0.15, 0.02, 0],
    shinL: [knee * 0.65, 0, 0],
    footL: [0.06, 0, 0],
    thighR: [swing * 0.2, -0.02, 0],
    shinR: [knee * 0.55, 0, 0],
    footR: [-0.02, 0, 0],
    upperArmL: [arm * 0.2, 0, -0.16],
    foreArmL: [-arm * 0.6 - 0.55, 0, 0],
    upperArmR: [-arm * 0.2, 0, 0.16],
    foreArmR: [-arm * 0.6 - 0.55, 0, 0],
    neck: [-lean * 0.5, 0, 0],
    head: [-lean * 0.35, 0, 0],
  };

  return clip(name, duration, true, [
    key(0, contact, { root: [0, 0, 0] }),
    key(0.25, pass, { root: [0, bounce, 0] }),
    key(0.5, mirror(contact), { root: [0, 0, 0] }),
    key(0.75, mirror(pass), { root: [0, bounce, 0] }),
    key(1, contact, { root: [0, 0, 0] }),
  ]);
}

const idle = clip('idle', 3.4, true, [
  key(0, {
    thighL: [-0.05, 0.04, 0.02], shinL: [0.1, 0, 0], footL: [-0.05, 0, 0],
    thighR: [-0.03, -0.04, -0.02], shinR: [0.08, 0, 0], footR: [-0.04, 0, 0],
    spine: [0.04, 0.02, 0], chest: [0.03, 0.01, 0], upperChest: [0.02, 0, 0],
    upperArmL: [0.06, 0, -0.09], foreArmL: [-0.3, 0, 0],
    upperArmR: [0.06, 0, 0.09], foreArmR: [-0.3, 0, 0],
    head: [0.02, 0.05, 0],
  }, { root: [0, 0, 0] }),
  key(0.5, {
    thighL: [-0.03, 0.04, 0.02], shinL: [0.08, 0, 0], footL: [-0.04, 0, 0],
    thighR: [-0.05, -0.04, -0.02], shinR: [0.1, 0, 0], footR: [-0.05, 0, 0],
    spine: [0.04, -0.02, 0], chest: [0.03, -0.01, 0], upperChest: [0.02, 0, 0],
    upperArmL: [0.09, 0, -0.1], foreArmL: [-0.34, 0, 0],
    upperArmR: [0.04, 0, 0.08], foreArmR: [-0.27, 0, 0],
    head: [0.01, -0.05, 0],
  }, { root: [0, -0.008, 0] }),
  key(1, {
    thighL: [-0.05, 0.04, 0.02], shinL: [0.1, 0, 0], footL: [-0.05, 0, 0],
    thighR: [-0.03, -0.04, -0.02], shinR: [0.08, 0, 0], footR: [-0.04, 0, 0],
    spine: [0.04, 0.02, 0], chest: [0.03, 0.01, 0], upperChest: [0.02, 0, 0],
    upperArmL: [0.06, 0, -0.09], foreArmL: [-0.3, 0, 0],
    upperArmR: [0.06, 0, 0.09], foreArmR: [-0.3, 0, 0],
    head: [0.02, 0.05, 0],
  }, { root: [0, 0, 0] }),
]);

/** Defensive stance: wide base, hips low, hands active. */
const stance = clip('stance', 1.9, true, [
  key(0, {
    thighL: [-0.42, 0.3, 0.1], shinL: [0.82, 0, 0], footL: [-0.3, 0.24, 0],
    thighR: [-0.42, -0.3, -0.1], shinR: [0.82, 0, 0], footR: [-0.3, -0.24, 0],
    spine: [0.24, 0, 0], chest: [0.16, 0, 0], upperChest: [0.1, 0, 0],
    upperArmL: [0.15, 0, -0.85], foreArmL: [-0.35, 0, 0],
    upperArmR: [0.15, 0, 0.85], foreArmR: [-0.35, 0, 0],
    head: [-0.12, 0, 0],
  }, { root: [0, -0.15, 0] }),
  key(0.5, {
    thighL: [-0.46, 0.3, 0.1], shinL: [0.88, 0, 0], footL: [-0.32, 0.24, 0],
    thighR: [-0.46, -0.3, -0.1], shinR: [0.88, 0, 0], footR: [-0.32, -0.24, 0],
    spine: [0.26, 0, 0], chest: [0.17, 0, 0], upperChest: [0.11, 0, 0],
    upperArmL: [0.1, 0, -0.95], foreArmL: [-0.28, 0, 0],
    upperArmR: [0.1, 0, 0.95], foreArmR: [-0.28, 0, 0],
    head: [-0.12, 0, 0],
  }, { root: [0, -0.18, 0] }),
  key(1, {
    thighL: [-0.42, 0.3, 0.1], shinL: [0.82, 0, 0], footL: [-0.3, 0.24, 0],
    thighR: [-0.42, -0.3, -0.1], shinR: [0.82, 0, 0], footR: [-0.3, -0.24, 0],
    spine: [0.24, 0, 0], chest: [0.16, 0, 0], upperChest: [0.1, 0, 0],
    upperArmL: [0.15, 0, -0.85], foreArmL: [-0.35, 0, 0],
    upperArmR: [0.15, 0, 0.85], foreArmR: [-0.35, 0, 0],
    head: [-0.12, 0, 0],
  }, { root: [0, -0.15, 0] }),
]);

/** Defensive slide: the feet never cross. */
const slide = clip('slide', 0.66, true, [
  key(0, {
    thighL: [-0.4, 0.5, 0.12], shinL: [0.78, 0, 0], footL: [-0.28, 0.3, 0],
    thighR: [-0.38, -0.2, -0.1], shinR: [0.72, 0, 0], footR: [-0.26, -0.2, 0],
    spine: [0.26, 0, 0.04], upperChest: [0.1, 0, 0],
    upperArmL: [0.12, 0, -1.0], foreArmL: [-0.3, 0, 0],
    upperArmR: [0.12, 0, 0.8], foreArmR: [-0.3, 0, 0],
  }, { root: [0, -0.17, 0] }),
  key(0.5, {
    thighL: [-0.36, 0.22, 0.1], shinL: [0.7, 0, 0], footL: [-0.24, 0.2, 0],
    thighR: [-0.42, -0.48, -0.12], shinR: [0.8, 0, 0], footR: [-0.3, -0.3, 0],
    spine: [0.26, 0, -0.04], upperChest: [0.1, 0, 0],
    upperArmL: [0.12, 0, -0.8], foreArmL: [-0.3, 0, 0],
    upperArmR: [0.12, 0, 1.0], foreArmR: [-0.3, 0, 0],
  }, { root: [0, -0.19, 0] }),
  key(1, {
    thighL: [-0.4, 0.5, 0.12], shinL: [0.78, 0, 0], footL: [-0.28, 0.3, 0],
    thighR: [-0.38, -0.2, -0.1], shinR: [0.72, 0, 0], footR: [-0.26, -0.2, 0],
    spine: [0.26, 0, 0.04], upperChest: [0.1, 0, 0],
    upperArmL: [0.12, 0, -1.0], foreArmL: [-0.3, 0, 0],
    upperArmR: [0.12, 0, 0.8], foreArmR: [-0.3, 0, 0],
  }, { root: [0, -0.17, 0] }),
]);

/** Airborne: legs tuck, arms rise. */
const air = clip('air', 0.9, true, [
  key(0, {
    thighL: [-0.55, 0.05, 0], shinL: [0.7, 0, 0], footL: [0.25, 0, 0],
    thighR: [-0.2, -0.05, 0], shinR: [0.35, 0, 0], footR: [0.2, 0, 0],
    spine: [-0.06, 0, 0],
    upperArmL: [-0.5, 0, -0.3], foreArmL: [-0.6, 0, 0],
    upperArmR: [-0.7, 0, 0.3], foreArmR: [-0.5, 0, 0],
  }),
  key(1, {
    thighL: [-0.55, 0.05, 0], shinL: [0.7, 0, 0], footL: [0.25, 0, 0],
    thighR: [-0.2, -0.05, 0], shinR: [0.35, 0, 0], footR: [0.2, 0, 0],
    spine: [-0.06, 0, 0],
    upperArmL: [-0.5, 0, -0.3], foreArmL: [-0.6, 0, 0],
    upperArmR: [-0.7, 0, 0.3], foreArmR: [-0.5, 0, 0],
  }),
]);

/** Adds a dribble pump on the ball hand to any gait. */
function withDribble(base: Clip, name: string, hand: 'left' | 'right' = 'right'): Clip {
  const arm: BoneName = hand === 'right' ? 'upperArmR' : 'upperArmL';
  const fore: BoneName = hand === 'right' ? 'foreArmR' : 'foreArmL';
  const keys = base.keys.map((k, i) => {
    const phase = i / Math.max(1, base.keys.length - 1);
    // Two pumps per stride cycle keeps the ball under the hip.
    const pump = Math.sin(phase * Math.PI * 4);
    return key(
      k.t,
      {
        ...k.pose,
        [arm]: [0.35 + pump * 0.34, 0, hand === 'right' ? 0.28 : -0.28],
        [fore]: [-0.75 - pump * 0.42, 0, 0],
      },
      { root: k.root, rootYaw: k.rootYaw },
    );
  });
  return clip(name, base.duration, true, keys);
}

const jog = gait('jog', 0.72, 0.42, 0.62, 0.34, 0.1, 0.022);
const run = gait('run', 0.62, 0.62, 0.95, 0.52, 0.19, 0.036);
const sprint = gait('sprint', 0.52, 0.82, 1.35, 0.72, 0.31, 0.05);

export const LOCOMOTION = {
  idle,
  jog,
  run,
  sprint,
  stance,
  slide,
  air,
  dribbleIdle: withDribble(idle, 'dribbleIdle'),
  dribbleRun: withDribble(run, 'dribbleRun'),
  dribbleSprint: withDribble(sprint, 'dribbleSprint'),
} as const;

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/**
 * Jump shot. `quality` widens the follow-through and squares the base: a
 * well-timed release looks composed, a rushed one is short and off-balance.
 */
function jumpShot(quality: number, hand: 'left' | 'right'): Clip {
  const q = clamp01(quality);
  const L = hand === 'left';
  const shootArm: BoneName = L ? 'upperArmL' : 'upperArmR';
  const shootFore: BoneName = L ? 'foreArmL' : 'foreArmR';
  const shootHand: BoneName = L ? 'handL' : 'handR';
  const guideArm: BoneName = L ? 'upperArmR' : 'upperArmL';
  const guideFore: BoneName = L ? 'foreArmR' : 'foreArmL';
  const s = L ? -1 : 1;

  const dip: PoseMap = {
    thighL: [-0.5, 0.06, 0], shinL: [0.95, 0, 0], footL: [-0.36, 0, 0],
    thighR: [-0.5, -0.06, 0], shinR: [0.95, 0, 0], footR: [-0.36, 0, 0],
    spine: [0.2, 0, 0], chest: [0.12, 0, 0], upperChest: [0.06, 0, 0],
    [shootArm]: [0.5, 0, 0.3 * s], [shootFore]: [-1.5, 0, 0],
    [guideArm]: [0.45, 0, -0.42 * s], [guideFore]: [-1.45, 0, 0],
    head: [-0.16, 0, 0],
  };
  const rise: PoseMap = {
    thighL: [-0.12, 0.04, 0], shinL: [0.22, 0, 0], footL: [0.16, 0, 0],
    thighR: [-0.12, -0.04, 0], shinR: [0.22, 0, 0], footR: [0.16, 0, 0],
    spine: [0.02, 0, 0], upperChest: [-0.04, 0, 0],
    [shootArm]: [-0.9, 0, 0.16 * s], [shootFore]: [-1.7, 0, 0], [shootHand]: [-0.3, 0, 0],
    [guideArm]: [-0.7, 0, -0.34 * s], [guideFore]: [-1.6, 0, 0],
    head: [-0.24, 0, 0],
  };
  const release: PoseMap = {
    thighL: [-0.3, 0.04, 0], shinL: [0.5, 0, 0], footL: [0.34, 0, 0],
    thighR: [-0.24, -0.04, 0], shinR: [0.44, 0, 0], footR: [0.32, 0, 0],
    spine: [-0.06, 0, 0], upperChest: [-0.1, 0, 0],
    [shootArm]: [-2.05 - q * 0.16, 0, 0.1 * s],
    [shootFore]: [-0.42 + q * 0.28, 0, 0],
    [shootHand]: [0.62 + q * 0.3, 0, 0],
    [guideArm]: [-1.5, 0, -0.5 * s], [guideFore]: [-1.1, 0, 0],
    head: [-0.28, 0, 0],
  };
  const follow: PoseMap = {
    ...release,
    [shootArm]: [-2.2 - q * 0.2, 0, 0.08 * s],
    [shootFore]: [-0.28 + q * 0.2, 0, 0],
    [shootHand]: [0.78 + q * 0.28, 0, 0],
    thighL: [-0.42, 0.05, 0], shinL: [0.72, 0, 0],
    thighR: [-0.4, -0.05, 0], shinR: [0.7, 0, 0],
  };

  return clip(
    'jumpShot',
    1.05,
    false,
    [
      key(0, {}),
      key(0.22, dip, { root: [0, -0.19, 0] }),
      key(0.5, rise, { root: [0, 0.3, 0] }),
      key(0.62, release, { root: [0, 0.38, 0] }),
      key(0.78, follow, { root: [0, 0.24, 0] }),
      key(1, {}, { root: [0, 0, 0] }),
    ],
    [
      { t: 0.5, name: 'apex' },
      { t: 0.62, name: 'release' },
    ],
  );
}

function layup(hand: 'left' | 'right'): Clip {
  const L = hand === 'left';
  const arm: BoneName = L ? 'upperArmL' : 'upperArmR';
  const fore: BoneName = L ? 'foreArmL' : 'foreArmR';
  const driveThigh: BoneName = L ? 'thighR' : 'thighL';
  const s = L ? -1 : 1;
  return clip(
    'layup',
    0.95,
    false,
    [
      key(0, {}),
      key(0.25, {
        [driveThigh]: [-1.15, 0, 0], spine: [0.16, 0.1 * s, 0],
        [arm]: [0.2, 0, 0.3 * s], [fore]: [-1.3, 0, 0],
      }, { root: [0, -0.1, 0] }),
      key(0.55, {
        [driveThigh]: [-1.55, 0, 0], spine: [-0.05, 0.16 * s, 0],
        [arm]: [-2.3, 0, 0.2 * s], [fore]: [-0.5, 0, 0],
        head: [-0.35, 0.1 * s, 0],
      }, { root: [0, 0.44, 0] }),
      key(0.7, {
        [driveThigh]: [-1.3, 0, 0],
        [arm]: [-2.5, 0, 0.16 * s], [fore]: [-0.28, 0, 0],
      }, { root: [0, 0.4, 0] }),
      key(1, {}, { root: [0, 0, 0] }),
    ],
    [{ t: 0.55, name: 'apex' }, { t: 0.58, name: 'release' }],
  );
}

function dunk(power: number, hand: 'left' | 'right'): Clip {
  const p = clamp01(power);
  const L = hand === 'left';
  const arm: BoneName = L ? 'upperArmL' : 'upperArmR';
  const fore: BoneName = L ? 'foreArmL' : 'foreArmR';
  const s = L ? -1 : 1;
  return clip(
    'dunk',
    1.25,
    false,
    [
      key(0, {}),
      key(0.2, {
        thighL: [-0.62, 0.06, 0], shinL: [1.15, 0, 0],
        thighR: [-0.62, -0.06, 0], shinR: [1.15, 0, 0],
        spine: [0.3, 0, 0],
        [arm]: [0.6, 0, 0.3 * s], [fore]: [-1.2, 0, 0],
      }, { root: [0, -0.26, 0] }),
      key(0.48, {
        thighL: [-0.5, 0.05, 0], shinL: [0.6, 0, 0],
        thighR: [-0.2, -0.05, 0], shinR: [0.3, 0, 0],
        spine: [-0.14, 0, 0],
        [arm]: [-2.45 - p * 0.2, 0, 0.12 * s], [fore]: [-0.34, 0, 0],
        head: [-0.4, 0, 0],
      }, { root: [0, 0.62 + p * 0.14, 0] }),
      key(0.62, {
        thighL: [-0.7, 0.05, 0], shinL: [0.9, 0, 0],
        thighR: [-0.35, -0.05, 0], shinR: [0.5, 0, 0],
        spine: [0.05, 0, 0],
        [arm]: [-2.05, 0, 0.1 * s], [fore]: [-0.2, 0, 0],
      }, { root: [0, 0.58, 0] }),
      key(0.85, {
        thighL: [-0.55, 0.06, 0], shinL: [1.0, 0, 0],
        thighR: [-0.55, -0.06, 0], shinR: [1.0, 0, 0],
        spine: [0.22, 0, 0],
      }, { root: [0, -0.14, 0] }),
      key(1, {}, { root: [0, 0, 0] }),
    ],
    [{ t: 0.48, name: 'apex' }, { t: 0.52, name: 'release' }],
  );
}

function pass(bounce: boolean, hand: 'left' | 'right'): Clip {
  const s = hand === 'left' ? -1 : 1;
  const wind: PoseMap = {
    upperArmL: [0.5, 0, -0.5], foreArmL: [-1.7, 0, 0],
    upperArmR: [0.5, 0, 0.5], foreArmR: [-1.7, 0, 0],
    spine: [0.06, -0.12 * s, 0], upperChest: [0.04, -0.1 * s, 0],
  };
  const throwPose: PoseMap = {
    upperArmL: [bounce ? 0.1 : -0.34, 0, -0.2], foreArmL: [-0.28, 0, 0],
    upperArmR: [bounce ? 0.1 : -0.34, 0, 0.2], foreArmR: [-0.28, 0, 0],
    spine: [bounce ? 0.14 : -0.02, 0.14 * s, 0], upperChest: [0.02, 0.12 * s, 0],
  };
  return clip(
    bounce ? 'bouncePass' : 'pass',
    0.44,
    false,
    [
      key(0, {}),
      key(0.32, wind),
      key(0.58, throwPose),
      key(1, {}),
    ],
    [{ t: 0.58, name: 'release' }],
  );
}

function block(hand: 'left' | 'right'): Clip {
  const arm: BoneName = hand === 'left' ? 'upperArmL' : 'upperArmR';
  const fore: BoneName = hand === 'left' ? 'foreArmL' : 'foreArmR';
  const s = hand === 'left' ? -1 : 1;
  return clip('block', 0.85, false, [
    key(0, {}),
    key(0.18, {
      thighL: [-0.55, 0.06, 0], shinL: [1.0, 0, 0],
      thighR: [-0.55, -0.06, 0], shinR: [1.0, 0, 0],
      spine: [0.24, 0, 0],
    }, { root: [0, -0.2, 0] }),
    key(0.46, {
      thighL: [-0.12, 0.04, 0], shinL: [0.2, 0, 0],
      thighR: [-0.12, -0.04, 0], shinR: [0.2, 0, 0],
      [arm]: [-2.7, 0, 0.1 * s], [fore]: [-0.12, 0, 0],
      spine: [-0.1, 0, 0], head: [-0.45, 0, 0],
    }, { root: [0, 0.56, 0] }),
    key(1, {}, { root: [0, 0, 0] }),
  ]);
}

function quickAction(name: string, duration: number, peak: PoseMap, root?: [number, number, number]): Clip {
  return clip(name, duration, false, [
    key(0, {}),
    key(0.42, peak, { root }),
    key(1, {}, { root: [0, 0, 0] }),
  ]);
}

/** Builds the clip for a discrete action, parameterised by how it was thrown. */
export function actionClip(
  kind: ActionKind,
  params: ActionParams,
  opts: AnimatorOptions,
): Clip | null {
  const hand = params.hand ?? opts.dominantHand;
  switch (kind) {
    case 'shoot':
    case 'jumpShot':
      return jumpShot(params.quality ?? 0.8, hand);
    case 'layup':
      return layup(hand);
    case 'dunk':
      return dunk(params.power ?? 0.7, hand);
    case 'pass':
      return pass(false, hand);
    case 'bouncePass':
      return pass(true, hand);
    case 'block':
      return block(hand);
    case 'steal':
      return quickAction('steal', 0.42, {
        [hand === 'left' ? 'upperArmL' : 'upperArmR']: [-1.1, 0, hand === 'left' ? -0.5 : 0.5],
        [hand === 'left' ? 'foreArmL' : 'foreArmR']: [-0.3, 0, 0],
        spine: [0.18, hand === 'left' ? -0.28 : 0.28, 0],
      });
    case 'rebound':
      return clip('rebound', 0.9, false, [
        key(0, {}),
        key(0.2, {
          thighL: [-0.58, 0.06, 0], shinL: [1.05, 0, 0],
          thighR: [-0.58, -0.06, 0], shinR: [1.05, 0, 0],
          spine: [0.26, 0, 0],
        }, { root: [0, -0.22, 0] }),
        key(0.5, {
          thighL: [-0.3, 0.06, 0], shinL: [0.5, 0, 0],
          thighR: [-0.3, -0.06, 0], shinR: [0.5, 0, 0],
          upperArmL: [-2.5, 0, -0.24], foreArmL: [-0.2, 0, 0],
          upperArmR: [-2.5, 0, 0.24], foreArmR: [-0.2, 0, 0],
          head: [-0.4, 0, 0],
        }, { root: [0, 0.5, 0] }),
        key(1, {}, { root: [0, 0, 0] }),
      ], [{ t: 0.5, name: 'apex' }]);
    case 'crossover':
      return quickAction('crossover', 0.5, {
        spine: [0.2, 0.3, 0],
        thighL: [-0.5, 0.34, 0.14], shinL: [0.8, 0, 0],
        thighR: [-0.3, -0.2, -0.1], shinR: [0.55, 0, 0],
        upperArmR: [0.5, 0, 0.7], foreArmR: [-1.1, 0, 0],
        upperArmL: [0.2, 0, -0.5], foreArmL: [-0.7, 0, 0],
      }, [0, -0.13, 0]);
    case 'stepback':
      return quickAction('stepback', 0.55, {
        spine: [-0.16, 0, 0],
        thighL: [0.5, 0.06, 0], shinL: [0.5, 0, 0],
        thighR: [-0.35, -0.06, 0], shinR: [0.6, 0, 0],
        upperArmL: [0.3, 0, -0.45], foreArmL: [-1.2, 0, 0],
        upperArmR: [0.3, 0, 0.45], foreArmR: [-1.2, 0, 0],
      }, [0, -0.1, 0]);
    case 'spin':
      return clip('spin', 0.62, false, [
        key(0, {}),
        key(0.4, {
          spine: [0.16, 0.4, 0],
          thighL: [-0.6, 0.3, 0], shinL: [0.9, 0, 0],
          thighR: [-0.2, -0.3, 0], shinR: [0.5, 0, 0],
        }, { root: [0, -0.1, 0], rootYaw: Math.PI }),
        key(1, {}, { root: [0, 0, 0], rootYaw: Math.PI * 2 }),
      ]);
    case 'celebrate':
      return clip('celebrate', 1.4, false, [
        key(0, {}),
        key(0.3, {
          upperArmL: [-2.3, 0, -0.5], foreArmL: [-0.4, 0, 0],
          upperArmR: [-2.3, 0, 0.5], foreArmR: [-0.4, 0, 0],
          spine: [-0.18, 0, 0], head: [-0.3, 0, 0],
        }, { root: [0, 0.05, 0] }),
        key(0.62, {
          upperArmL: [-1.9, 0, -0.75], foreArmL: [-0.9, 0, 0],
          upperArmR: [-1.9, 0, 0.75], foreArmR: [-0.9, 0, 0],
          spine: [-0.1, 0.14, 0], head: [-0.2, 0.16, 0],
        }),
        key(1, {}, { root: [0, 0, 0] }),
      ]);
    case 'dejected':
      return quickAction('dejected', 1.1, {
        spine: [0.26, 0, 0], chest: [0.16, 0, 0], neck: [0.24, 0, 0], head: [0.3, 0, 0],
        upperArmL: [0.2, 0, -0.06], foreArmL: [-0.2, 0, 0],
        upperArmR: [0.2, 0, 0.06], foreArmR: [-0.2, 0, 0],
      });
    default:
      return null;
  }
}
