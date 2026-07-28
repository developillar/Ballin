/**
 * Humanoid skeleton definition.
 *
 * Proportions are expressed as fractions of standing height so one table drives
 * every body type on the roster. The numbers are drawn from athlete anthropometry
 * rather than generic human averages — NBA players are long-limbed relative to
 * their height (mean wingspan runs about 1.06× height), and getting that ratio
 * right is most of what makes a silhouette read as a basketball player instead
 * of a generic game character.
 *
 * Rest pose is a relaxed A-pose: arms down and slightly out, palms inward, feet
 * shoulder-width. Every joint's local axes are consistent — +Y points down the
 * bone toward its child, +Z faces forward — so IK and clip authoring can assume
 * one convention.
 */

import { Bone, Matrix4, Quaternion, Skeleton as ThreeSkeleton, Vector3 } from 'three';

export type BoneName =
  | 'root'
  | 'hips'
  | 'spine'
  | 'chest'
  | 'upperChest'
  | 'neck'
  | 'head'
  | 'clavicleL'
  | 'upperArmL'
  | 'foreArmL'
  | 'handL'
  | 'clavicleR'
  | 'upperArmR'
  | 'foreArmR'
  | 'handR'
  | 'thighL'
  | 'shinL'
  | 'footL'
  | 'toeL'
  | 'thighR'
  | 'shinR'
  | 'footR'
  | 'toeR';

export interface BoneDef {
  name: BoneName;
  parent: BoneName | null;
  /**
   * Offset from the parent joint, in fractions of standing height. X is right,
   * Y is up, Z is forward. Mirrored limbs share magnitudes with flipped X.
   */
  offset: [number, number, number];
  /** Radius of the limb at this joint, again as a fraction of height. */
  radius: number;
  /** Bones the skinning solver should blend across at this joint. */
  smooth?: number;
}

/**
 * Landmark heights as fractions of standing height, for reference and for any
 * system that needs to reason about the body without walking the hierarchy.
 */
export const LANDMARK = {
  ankle: 0.039,
  knee: 0.285,
  crotch: 0.468,
  hip: 0.53,
  navel: 0.6,
  sternum: 0.72,
  shoulder: 0.818,
  chin: 0.87,
  eye: 0.935,
  crown: 1.0,
  /** Half the biacromial (shoulder-to-shoulder) breadth. */
  shoulderHalfWidth: 0.1105,
  hipHalfWidth: 0.0745,
  /** Long-limb athlete build. */
  upperArm: 0.196,
  foreArm: 0.158,
  hand: 0.112,
  thigh: 0.245,
  shin: 0.246,
  footLength: 0.16,
} as const;

const L = LANDMARK;

/**
 * The hierarchy. Offsets are parent-relative, so each entry reads as "how far
 * this joint sits from the one it hangs off".
 */
export const BONES: readonly BoneDef[] = [
  { name: 'root', parent: null, offset: [0, 0, 0], radius: 0 },
  { name: 'hips', parent: 'root', offset: [0, L.hip, 0], radius: 0.078 },
  { name: 'spine', parent: 'hips', offset: [0, L.navel - L.hip, 0], radius: 0.071, smooth: 0.055 },
  { name: 'chest', parent: 'spine', offset: [0, 0.06, 0], radius: 0.079, smooth: 0.05 },
  {
    name: 'upperChest',
    parent: 'chest',
    offset: [0, L.sternum - L.navel - 0.06, 0],
    radius: 0.086,
    smooth: 0.05,
  },
  {
    name: 'neck',
    parent: 'upperChest',
    offset: [0, L.shoulder - L.sternum + 0.018, 0],
    radius: 0.039,
    smooth: 0.03,
  },
  { name: 'head', parent: 'neck', offset: [0, L.chin - L.shoulder - 0.006, 0], radius: 0.072 },

  // --- Left arm -----------------------------------------------------------
  {
    name: 'clavicleL',
    parent: 'upperChest',
    offset: [0.032, L.shoulder - L.sternum - 0.012, 0.006],
    radius: 0.044,
    smooth: 0.04,
  },
  {
    name: 'upperArmL',
    parent: 'clavicleL',
    offset: [L.shoulderHalfWidth - 0.032, 0.004, 0],
    radius: 0.047,
    smooth: 0.045,
  },
  // A-pose: the arm hangs down and about 9° out from vertical.
  {
    name: 'foreArmL',
    parent: 'upperArmL',
    offset: [L.upperArm * 0.156, -L.upperArm * 0.988, 0],
    radius: 0.039,
    smooth: 0.038,
  },
  {
    name: 'handL',
    parent: 'foreArmL',
    offset: [L.foreArm * 0.1, -L.foreArm * 0.995, 0],
    radius: 0.032,
    smooth: 0.026,
  },

  // --- Right arm (mirrored) -----------------------------------------------
  {
    name: 'clavicleR',
    parent: 'upperChest',
    offset: [-0.032, L.shoulder - L.sternum - 0.012, 0.006],
    radius: 0.044,
    smooth: 0.04,
  },
  {
    name: 'upperArmR',
    parent: 'clavicleR',
    offset: [-(L.shoulderHalfWidth - 0.032), 0.004, 0],
    radius: 0.047,
    smooth: 0.045,
  },
  {
    name: 'foreArmR',
    parent: 'upperArmR',
    offset: [-L.upperArm * 0.156, -L.upperArm * 0.988, 0],
    radius: 0.039,
    smooth: 0.038,
  },
  {
    name: 'handR',
    parent: 'foreArmR',
    offset: [-L.foreArm * 0.1, -L.foreArm * 0.995, 0],
    radius: 0.032,
    smooth: 0.026,
  },

  // --- Left leg -----------------------------------------------------------
  {
    name: 'thighL',
    parent: 'hips',
    offset: [L.hipHalfWidth, -(L.hip - L.crotch) * 0.42, 0],
    radius: 0.064,
    smooth: 0.06,
  },
  { name: 'shinL', parent: 'thighL', offset: [0.004, -L.thigh, 0], radius: 0.048, smooth: 0.045 },
  { name: 'footL', parent: 'shinL', offset: [0, -L.shin, 0], radius: 0.038, smooth: 0.03 },
  { name: 'toeL', parent: 'footL', offset: [0, -L.ankle * 0.55, L.footLength * 0.62], radius: 0.03 },

  // --- Right leg (mirrored) -----------------------------------------------
  {
    name: 'thighR',
    parent: 'hips',
    offset: [-L.hipHalfWidth, -(L.hip - L.crotch) * 0.42, 0],
    radius: 0.064,
    smooth: 0.06,
  },
  { name: 'shinR', parent: 'thighR', offset: [-0.004, -L.thigh, 0], radius: 0.048, smooth: 0.045 },
  { name: 'footR', parent: 'shinR', offset: [0, -L.shin, 0], radius: 0.038, smooth: 0.03 },
  { name: 'toeR', parent: 'footR', offset: [0, -L.ankle * 0.55, L.footLength * 0.62], radius: 0.03 },
];

export const BONE_INDEX: Record<BoneName, number> = (() => {
  const m = {} as Record<BoneName, number>;
  BONES.forEach((b, i) => (m[b.name] = i));
  return m;
})();

/** Chains the IK solver and the animation blend tree operate on. */
export const CHAINS = {
  armL: ['upperArmL', 'foreArmL', 'handL'] as const,
  armR: ['upperArmR', 'foreArmR', 'handR'] as const,
  legL: ['thighL', 'shinL', 'footL'] as const,
  legR: ['thighR', 'shinR', 'footR'] as const,
  spine: ['hips', 'spine', 'chest', 'upperChest', 'neck', 'head'] as const,
} as const;

export interface BuiltSkeleton {
  bones: Bone[];
  byName: Record<BoneName, Bone>;
  skeleton: ThreeSkeleton;
  /** Standing height this instance was built at, in metres. */
  height: number;
  /** World-space rest position of each joint, scaled to `height`. */
  restWorld: Vector3[];
  /** Bone length toward the primary child, in metres. Leaf bones report 0. */
  lengths: number[];
  /** Joint radius in metres. */
  radii: number[];
}

export interface BodyShape {
  /** Standing height in metres. */
  height: number;
  /** Scales every limb radius — 0.9 is wiry, 1.15 is a heavy interior body. */
  build: number;
  /** Extra shoulder breadth multiplier on top of `build`. */
  shoulders: number;
  /** Wingspan relative to height; NBA mean is ~1.06. */
  wingspan: number;
  /** Leg length as a share of standing height; taller share = longer strider. */
  legRatio: number;
}

export const DEFAULT_SHAPE: BodyShape = {
  height: 1.98,
  build: 1,
  shoulders: 1,
  wingspan: 1.06,
  legRatio: 1,
};

/**
 * Instantiates the hierarchy at a given height and body shape, returning both
 * the Three bones and the derived measurements the mesh generator needs.
 */
export function buildSkeleton(shape: Partial<BodyShape> = {}): BuiltSkeleton {
  const s: BodyShape = { ...DEFAULT_SHAPE, ...shape };
  const H = s.height;

  const bones: Bone[] = [];
  const byName = {} as Record<BoneName, Bone>;
  const radii: number[] = [];

  // Wingspan is applied to the arm segments; leg ratio to thigh and shin. Both
  // are compensated in the torso so total height stays exactly `height`.
  const armScale = s.wingspan / DEFAULT_SHAPE.wingspan;

  for (const def of BONES) {
    const bone = new Bone();
    bone.name = def.name;

    let [ox, oy, oz] = def.offset;
    const isArm = /^(upperArm|foreArm|hand)/.test(def.name);
    const isLeg = /^(thigh|shin)/.test(def.name);
    if (isArm) {
      ox *= armScale;
      oy *= armScale;
      oz *= armScale;
    }
    if (isLeg) {
      oy *= s.legRatio;
    }
    if (def.name === 'clavicleL' || def.name === 'clavicleR') {
      ox *= s.shoulders;
    }
    if (def.name === 'upperArmL' || def.name === 'upperArmR') {
      ox *= s.shoulders;
    }

    bone.position.set(ox * H, oy * H, oz * H);
    bones.push(bone);
    byName[def.name] = bone;

    let r = def.radius * H * s.build;
    if (/^(upperChest|chest|clavicle)/.test(def.name)) r *= s.shoulders;
    if (def.name === 'head' || def.name === 'neck') r = def.radius * H; // heads do not bulk up
    radii.push(r);
  }

  for (const def of BONES) {
    if (def.parent) byName[def.parent].add(byName[def.name]);
  }

  const root = byName.root;
  root.updateMatrixWorld(true);

  const restWorld = bones.map((b) => new Vector3().setFromMatrixPosition(b.matrixWorld));
  const lengths = bones.map((b) => {
    const child = b.children.find((c) => c instanceof Bone) as Bone | undefined;
    return child ? child.position.length() : 0;
  });

  const inverses = bones.map((b) => new Matrix4().copy(b.matrixWorld).invert());
  const skeleton = new ThreeSkeleton(bones, inverses);

  return { bones, byName, skeleton, height: H, restWorld, lengths, radii };
}

/** Convenience: the rest-pose world height of a named joint, in metres. */
export function restHeight(sk: BuiltSkeleton, name: BoneName): number {
  return sk.restWorld[BONE_INDEX[name]].y;
}

const _q = new Quaternion();

/** Resets every bone to its rest rotation. Positions are never animated. */
export function resetPose(sk: BuiltSkeleton): void {
  _q.identity();
  for (const b of sk.bones) b.quaternion.copy(_q);
}
