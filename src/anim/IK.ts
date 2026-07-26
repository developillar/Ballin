/**
 * Inverse kinematics.
 *
 * Basketball animation lives or dies on contact: the shooting hand has to end
 * up under the ball with the fingers spread across the leather, the guide hand
 * on its side, the plant foot has to meet the floor without sliding, and the
 * head has to track the rim without detaching from the neck. All of that is IK
 * layered on top of whatever the blend tree produced, which is why these
 * solvers operate on an already-posed skeleton rather than generating poses.
 *
 * Everything here is allocation-free on the hot path. Ten players × four chains
 * × sixty frames is 2400 solves a second; a `new Vector3()` inside one of them
 * is a garbage collector pause you can see.
 */

import { Bone, Euler, Matrix4, Quaternion, Vector3 } from 'three';
import { clamp, clamp01 } from '../core/MathX';

const _a = new Vector3();
const _b = new Vector3();
const _c = new Vector3();
const _d = new Vector3();
const _t = new Vector3();
const _axis = new Vector3();
const _pRoot = new Vector3();
const _pMid = new Vector3();
const _pTip = new Vector3();
const _target = new Vector3();
const _pole = new Vector3();
const _palm = new Vector3();
const _fing = new Vector3();
const _out = new Vector3();
const _q = new Quaternion();
const _q2 = new Quaternion();
const _qi = new Quaternion();
const _qp = new Quaternion();
const _m = new Matrix4();
const _scale = new Vector3();
const _euler = new Euler();
const UP = new Vector3(0, 1, 0);

/** World-space position of a bone's joint. */
export function boneWorld(bone: Bone, out = new Vector3()): Vector3 {
  return out.setFromMatrixPosition(bone.matrixWorld);
}

/** World-space rotation of a bone. */
export function boneWorldQuat(bone: Bone, out = new Quaternion()): Quaternion {
  bone.matrixWorld.decompose(_a, out, _scale);
  return out;
}

/**
 * Applies a world-space rotation delta to a bone, converting into its parent's
 * space so the rest of the hierarchy follows correctly.
 */
export function applyWorldDelta(bone: Bone, worldDelta: Quaternion): void {
  const parent = bone.parent;
  if (parent) {
    parent.updateWorldMatrix(true, false);
    parent.matrixWorld.decompose(_a, _qp, _scale);
    // local' = (P⁻¹ · Δ · P) · local
    _qi.copy(_qp).invert().multiply(worldDelta).multiply(_qp);
    bone.quaternion.premultiply(_qi);
  } else {
    bone.quaternion.premultiply(worldDelta);
  }
  bone.updateMatrixWorld(true);
}

/** Forces a bone to a world-space orientation. */
export function setBoneWorldQuat(bone: Bone, worldQuat: Quaternion, weight = 1): void {
  if (weight <= 0.0001) return;
  const parent = bone.parent;
  if (parent) {
    parent.updateWorldMatrix(true, false);
    parent.matrixWorld.decompose(_a, _qp, _scale);
    _qi.copy(_qp).invert().multiply(worldQuat);
  } else {
    _qi.copy(worldQuat);
  }
  if (weight < 1) bone.quaternion.slerp(_qi, weight);
  else bone.quaternion.copy(_qi);
  bone.updateMatrixWorld(true);
}

/**
 * Soft reach limit. A hard clamp at full extension makes a limb *snap* straight
 * the instant the target leaves the reachable sphere, which is one of the most
 * recognisable IK artefacts there is. This eases into the limit asymptotically
 * so the last few centimetres of extension are gradual and the joint keeps a
 * few degrees of residual bend — nothing living locks a knee.
 */
function softReach(dist: number, maxReach: number): number {
  // Narrow band: wider than ~4% and the compression is large enough to show up
  // as a foot that never quite reaches its lock point.
  const band = maxReach * 0.035;
  const knee = maxReach - band;
  if (dist <= knee) return dist;
  return knee + band * (1 - Math.exp(-(dist - knee) / band));
}

/**
 * Clamps a hinge joint's local X rotation. Knees only bend one way and elbows
 * only bend the other; without this the pole solve can and will invert one
 * during a fast blend, and a backwards knee is instantly disqualifying.
 */
export function clampHinge(bone: Bone, minX: number, maxX: number): void {
  _euler.setFromQuaternion(bone.quaternion, 'XYZ');
  if (_euler.x >= minX && _euler.x <= maxX) return;
  _euler.x = clamp(_euler.x, minX, maxX);
  bone.quaternion.setFromEuler(_euler);
  bone.updateMatrixWorld(true);
}

export const HINGE = {
  /** Knee: flexion only. Never negative — that is a knee bending backward. */
  knee: [0.015, 2.55] as const,
  /** Elbow: flexion only, in the opposite sense. Never positive. */
  elbow: [-2.62, -0.015] as const,
};

/**
 * Analytic two-bone IK — the workhorse for arms and legs.
 *
 * Solves for the interior angle with the law of cosines, swings the whole chain
 * so the end effector lands on the target, then rolls the chain about its own
 * axis so the mid joint faces the pole. The pole vector controls which way the
 * joint bends: for a knee that is forward and slightly out, for an elbow back
 * and down, and getting it wrong is instantly readable as "broken".
 *
 * @param root   upper bone (shoulder / hip)
 * @param mid    middle bone (elbow / knee)
 * @param tip    end bone (hand / foot) — only read for its position
 * @param target world-space position the tip joint should reach
 * @param pole   world-space hint the mid joint bends toward
 * @param weight 0 leaves the pose untouched, 1 fully solves
 * @param hinge  optional local-X limits applied to `mid` after the solve
 */
export function solveTwoBoneIK(
  root: Bone,
  mid: Bone,
  tip: Bone,
  target: Vector3,
  pole: Vector3,
  weight = 1,
  hinge?: readonly [number, number],
): void {
  if (weight <= 0.0001) return;

  root.updateWorldMatrix(true, false);
  mid.updateWorldMatrix(false, false);
  tip.updateWorldMatrix(false, false);

  boneWorld(root, _pRoot);
  boneWorld(mid, _pMid);
  boneWorld(tip, _pTip);

  const lenUpper = _pRoot.distanceTo(_pMid);
  const lenLower = _pMid.distanceTo(_pTip);
  if (lenUpper < 1e-5 || lenLower < 1e-5) return;
  const maxReach = (lenUpper + lenLower) * 0.995;
  const minReach = Math.abs(lenUpper - lenLower) * 1.02 + 1e-3;

  _t.copy(target).sub(_pRoot);
  const rawDist = _t.length();
  if (rawDist < 1e-5) return;
  _d.copy(_t).divideScalar(rawDist);
  const dist = clamp(softReach(rawDist, maxReach), minReach, maxReach);
  _target.copy(_pRoot).addScaledVector(_d, dist);

  // --- Interior (bend) angle ---------------------------------------------
  const denom = 2 * lenUpper * lenLower;
  const wantMid = Math.acos(
    clamp((lenUpper * lenUpper + lenLower * lenLower - dist * dist) / denom, -1, 1),
  );
  const curDist = _pRoot.distanceTo(_pTip);
  const curMid = Math.acos(
    clamp((lenUpper * lenUpper + lenLower * lenLower - curDist * curDist) / denom, -1, 1),
  );

  // Bend axis: normal to the plane the chain currently occupies. Near full
  // extension that plane is numerically meaningless, so fall back to the pole —
  // this is the wrap-around case where a naive solver folds the joint into a
  // random direction and pops.
  _a.copy(_pMid).sub(_pRoot);
  _b.copy(_pTip).sub(_pMid);
  _axis.copy(_a).cross(_b);
  const straightness = _axis.length() / Math.max(1e-9, _a.length() * _b.length());
  if (straightness < 0.06) {
    _c.copy(pole).sub(_pRoot);
    _axis.copy(_c).cross(_d);
    if (_axis.lengthSq() < 1e-10) {
      _axis.copy(_d).cross(UP);
      if (_axis.lengthSq() < 1e-10) _axis.set(1, 0, 0);
    }
  }
  _axis.normalize();

  // Sign: `_axis` is (root→mid) × (mid→tip). A *positive* rotation about it
  // opens the angle between the two segments, which *closes* the interior angle
  // at the mid joint — hence `cur − want`, not `want − cur`. Getting this
  // backwards leaves the chain a few centimetres short and, worse, makes the
  // solver diverge under iteration instead of converging.
  applyWorldDelta(mid, _q.setFromAxisAngle(_axis, (curMid - wantMid) * weight));

  // --- Swing the chain so the tip lands on the target ----------------------
  root.updateWorldMatrix(true, false);
  mid.updateWorldMatrix(false, false);
  tip.updateWorldMatrix(false, false);
  boneWorld(root, _pRoot);
  boneWorld(tip, _pTip);

  _a.copy(_pTip).sub(_pRoot);
  if (_a.lengthSq() > 1e-10) {
    _a.normalize();
    _b.copy(_target).sub(_pRoot).normalize();
    _q.setFromUnitVectors(_a, _b);
    if (weight < 1) _q.slerp(_qi.identity(), 1 - weight);
    applyWorldDelta(root, _q);
  }

  // --- Roll the chain around the root→tip axis to satisfy the pole ---------
  root.updateWorldMatrix(true, false);
  mid.updateWorldMatrix(false, false);
  tip.updateWorldMatrix(false, false);
  boneWorld(root, _pRoot);
  boneWorld(mid, _pMid);
  boneWorld(tip, _pTip);

  _axis.copy(_pTip).sub(_pRoot);
  const axisLen = _axis.length();
  if (axisLen > 1e-5) {
    _axis.divideScalar(axisLen);
    // Project the current mid joint and the pole onto the plane normal to the
    // chain axis; the angle between them is the roll we need.
    _b.copy(_pMid).sub(_pRoot);
    _b.addScaledVector(_axis, -_b.dot(_axis));
    _c.copy(pole).sub(_pRoot);
    _c.addScaledVector(_axis, -_c.dot(_axis));
    // A mid joint sitting exactly on the chain axis has no defined roll, and a
    // pole that has collapsed onto it has nothing to say — skip both.
    if (_b.lengthSq() > 1e-7 && _c.lengthSq() > 1e-7) {
      _b.normalize();
      _c.normalize();
      let angle = Math.acos(clamp(_b.dot(_c), -1, 1));
      if (_a.copy(_b).cross(_c).dot(_axis) < 0) angle = -angle;
      applyWorldDelta(root, _q.setFromAxisAngle(_axis, angle * weight));
      if (hinge) clampHinge(mid, hinge[0], hinge[1]);
    }
  }
}

/**
 * Aims a bone's forward axis at a target.
 *
 * The limits are **absolute, measured against the parent**, not applied to the
 * per-frame correction. That distinction matters: clamping the delta lets a
 * clip that already has the head turned 60° add another 60° of look-at and end
 * up with the chin over the shoulder blade. `rate` caps how fast the aim can
 * move, so a target that teleports does not snap the neck with it.
 */
export function solveLookAt(
  bone: Bone,
  target: Vector3,
  forward: Vector3,
  limits: { yaw: number; pitch: number },
  weight = 1,
): void {
  if (weight <= 0.0001) return;
  const parent = bone.parent;
  if (!parent) return;

  bone.updateWorldMatrix(true, false);
  boneWorld(bone, _a);
  _t.copy(target).sub(_a);
  if (_t.lengthSq() < 1e-8) return;
  _t.normalize();

  // Work in the parent's frame so the clamp is anatomical.
  parent.matrixWorld.decompose(_b, _qp, _scale);
  _qi.copy(_qp).invert();
  _d.copy(_t).applyQuaternion(_qi).normalize();

  // `forward` is the bone-local axis that should end up pointing at the target.
  // In the parent's frame the rest orientation of that axis is `forward` itself,
  // because bones rest at identity.
  let yaw = Math.atan2(-_d.x, _d.z) - Math.atan2(-forward.x, forward.z);
  if (yaw > Math.PI) yaw -= Math.PI * 2;
  if (yaw < -Math.PI) yaw += Math.PI * 2;
  const pitch = Math.asin(clamp(_d.y, -1, 1)) - Math.asin(clamp(forward.y, -1, 1));

  yaw = clamp(yaw, -limits.yaw, limits.yaw) * weight;
  const p = clamp(pitch, -limits.pitch, limits.pitch) * weight;

  // Yaw about the parent's up, then pitch about the resulting right — the order
  // a neck actually works in.
  _q.setFromAxisAngle(UP, yaw);
  _c.set(1, 0, 0);
  _q2.setFromAxisAngle(_c, -p);
  _q.multiply(_q2);
  bone.quaternion.copy(_q);
  bone.updateMatrixWorld(true);
}

/**
 * Foot planting. Given a world point the ankle should sit at, re-solves the leg
 * and rolls the ankle so the sole is flat on the floor.
 */
export interface FootPlant {
  /** World point the ankle should sit at. */
  target: Vector3;
  /** Floor normal, for rolling the foot onto a slope. */
  normal: Vector3;
  /** 0 = airborne (ignore), 1 = fully planted. */
  weight: number;
}

export function solveLegPlant(
  thigh: Bone,
  shin: Bone,
  foot: Bone,
  plant: FootPlant,
  kneeForward: Vector3,
  /** Outward (lateral) direction for this leg — knees track slightly out, never in. */
  kneeOut?: Vector3,
): void {
  if (plant.weight <= 0.001) return;
  thigh.updateWorldMatrix(true, false);
  boneWorld(thigh, _a);

  // The pole sits forward of the hip and roughly level with the knee. Putting
  // it at ankle height (a common shortcut) drags the knee down into the shin
  // and produces the classic "sitting in an invisible chair" plant.
  const legLen = Math.max(0.2, _a.y - plant.target.y);
  _pole.copy(_a).addScaledVector(kneeForward, legLen * 1.15);
  if (kneeOut) _pole.addScaledVector(kneeOut, legLen * 0.16);
  _pole.y = plant.target.y + legLen * 0.55;

  solveTwoBoneIK(thigh, shin, foot, plant.target, _pole, plant.weight, HINGE.knee);

  // Roll the ankle so the sole matches the floor normal.
  if (plant.normal.lengthSq() > 1e-6) {
    foot.updateWorldMatrix(true, false);
    boneWorldQuat(foot, _q);
    _c.set(0, -1, 0).applyQuaternion(_q).normalize();
    _d.copy(plant.normal).normalize().negate();
    _q2.setFromUnitVectors(_c, _d);
    if (plant.weight < 1) _q2.slerp(_qi.identity(), 1 - plant.weight);
    applyWorldDelta(foot, _q2);
  }
}

/**
 * Builds a world quaternion for a hand from a palm normal and a finger
 * direction, and writes it onto the bone.
 *
 * Hand-local axes follow the skeleton convention: the fingers run down local
 * **−Y**, and the palm faces **−X on the left hand / +X on the right** (the
 * rest pose is an A-pose with the palms turned inward). Position alone is not
 * enough for a ball hold — a wrist in the right place with the palm facing the
 * sky reads as a mannequin holding an invisible tray.
 */
export function orientHand(
  hand: Bone,
  palmNormal: Vector3,
  fingerDir: Vector3,
  isLeft: boolean,
  weight = 1,
): void {
  if (weight <= 0.0001) return;
  _a.copy(palmNormal);
  if (_a.lengthSq() < 1e-8) return;
  _a.normalize();
  // Orthogonalise the finger direction against the palm normal.
  _b.copy(fingerDir).addScaledVector(_a, -fingerDir.dot(_a));
  if (_b.lengthSq() < 1e-8) {
    _b.copy(UP).addScaledVector(_a, -UP.dot(_a));
    if (_b.lengthSq() < 1e-8) return;
  }
  _b.normalize();

  // Column 0 is the local +X axis in world space, column 1 the local +Y.
  const sx = isLeft ? -1 : 1;
  _c.copy(_a).multiplyScalar(sx); // local +X
  _d.copy(_b).multiplyScalar(-1); // local +Y (fingers run along −Y)
  _t.copy(_c).cross(_d).normalize(); // local +Z
  // Re-orthogonalise X against the other two so the basis is exactly rigid.
  _c.copy(_d).cross(_t).normalize();

  _m.makeBasis(_c, _d, _t);
  _q.setFromRotationMatrix(_m);
  setBoneWorldQuat(hand, _q, weight);
}

/**
 * Two-handed ball hold.
 *
 * The shooting hand goes **under and behind** the ball with the fingers spread
 * up its back face and the wrist cocked; the guide hand sits on the side with
 * the palm turned in and the fingers up. Both hands are oriented, not just
 * positioned, and the wrists are set back from the leather by a palm thickness
 * so the fingers rest on the surface instead of through it.
 */
export function solveBallHold(
  shootArm: readonly [Bone, Bone, Bone],
  guideArm: readonly [Bone, Bone, Bone],
  ballCentre: Vector3,
  ballRadius: number,
  chestForward: Vector3,
  /** The character's LEFT direction in world space. */
  chestLeft: Vector3,
  shootIsLeft: boolean,
  weight = 1,
  /** Wrist-to-palm-centre distance, metres. Scales with the player. */
  palmReach = 0.062,
): void {
  if (weight <= 0.0001) return;
  const shootSide = shootIsLeft ? 1 : -1;

  // --- Shooting hand: under and slightly behind ---------------------------
  // Outward direction from the ball centre to the middle of that palm.
  _out.copy(UP).multiplyScalar(-0.78)
    .addScaledVector(chestForward, -0.5)
    .addScaledVector(chestLeft, 0.22 * shootSide)
    .normalize();
  // Palm normal points from the hand into the ball.
  _palm.copy(_out).multiplyScalar(-1);
  // Fingers run up the back of the ball toward the top.
  _fing.copy(UP).addScaledVector(_palm, -UP.dot(_palm));
  if (_fing.lengthSq() < 1e-8) _fing.copy(chestForward);
  _fing.normalize();

  _target
    .copy(ballCentre)
    .addScaledVector(_out, ballRadius + 0.028)
    .addScaledVector(_fing, -palmReach);
  // Elbow tucks under the ball and slightly inboard — the whole point of a
  // repeatable jump shot.
  _pole
    .copy(ballCentre)
    .addScaledVector(chestForward, -0.5)
    .addScaledVector(chestLeft, 0.12 * shootSide);
  _pole.y -= 0.62;
  solveTwoBoneIK(shootArm[0], shootArm[1], shootArm[2], _target, _pole, weight, HINGE.elbow);
  orientHand(shootArm[2], _palm, _fing, shootIsLeft, weight);

  // --- Guide hand: on the far side, palm turned in ------------------------
  _out.copy(chestLeft).multiplyScalar(-shootSide)
    .addScaledVector(chestForward, -0.24)
    .addScaledVector(UP, 0.08)
    .normalize();
  _palm.copy(_out).multiplyScalar(-1);
  _fing.copy(UP).addScaledVector(_palm, -UP.dot(_palm));
  if (_fing.lengthSq() < 1e-8) _fing.copy(chestForward);
  _fing.normalize();

  _target
    .copy(ballCentre)
    .addScaledVector(_out, ballRadius + 0.024)
    .addScaledVector(_fing, -palmReach * 0.85);
  // The guide elbow flares out and down, unlike the shooting elbow.
  _pole
    .copy(ballCentre)
    .addScaledVector(chestForward, -0.4)
    .addScaledVector(chestLeft, -0.72 * shootSide);
  _pole.y -= 0.4;
  solveTwoBoneIK(guideArm[0], guideArm[1], guideArm[2], _target, _pole, weight * 0.94, HINGE.elbow);
  orientHand(guideArm[2], _palm, _fing, !shootIsLeft, weight * 0.94);
}

/**
 * One-handed ball control — a palm-down dribble ride, a cradle on a drive, a
 * dunk hand under the ball. `over` puts the palm on top instead of underneath.
 */
export function solveOneHandBall(
  arm: readonly [Bone, Bone, Bone],
  ballCentre: Vector3,
  ballRadius: number,
  chestForward: Vector3,
  chestLeft: Vector3,
  isLeft: boolean,
  over: boolean,
  weight = 1,
  palmReach = 0.062,
): void {
  if (weight <= 0.0001) return;
  const side = isLeft ? 1 : -1;
  _out.copy(UP).multiplyScalar(over ? 0.9 : -0.72)
    .addScaledVector(chestForward, over ? -0.18 : -0.42)
    .addScaledVector(chestLeft, 0.2 * side)
    .normalize();
  _palm.copy(_out).multiplyScalar(-1);
  const ref = over ? chestForward : UP;
  _fing.copy(ref).addScaledVector(_palm, -ref.dot(_palm));
  if (_fing.lengthSq() < 1e-8) _fing.copy(chestForward);
  _fing.normalize();

  _target
    .copy(ballCentre)
    .addScaledVector(_out, ballRadius + 0.026)
    .addScaledVector(_fing, -palmReach * 0.8);
  _pole
    .copy(ballCentre)
    .addScaledVector(chestForward, -0.46)
    .addScaledVector(chestLeft, 0.34 * side);
  _pole.y -= 0.5;
  solveTwoBoneIK(arm[0], arm[1], arm[2], _target, _pole, weight, HINGE.elbow);
  orientHand(arm[2], _palm, _fing, isLeft, weight);
}

/** Straight reach for a single hand — blocks, rebounds, dunk finishes. */
export function solveArmReach(
  arm: readonly [Bone, Bone, Bone],
  target: Vector3,
  chestForward: Vector3,
  chestLeft: Vector3,
  isLeft: boolean,
  weight = 1,
): void {
  if (weight <= 0.0001) return;
  arm[0].updateWorldMatrix(true, false);
  boneWorld(arm[0], _a);
  _pole.copy(_a).addScaledVector(chestForward, -0.55).addScaledVector(chestLeft, (isLeft ? 1 : -1) * 0.45);
  _pole.y -= 0.3;
  solveTwoBoneIK(arm[0], arm[1], arm[2], target, _pole, clamp01(weight), HINGE.elbow);
}
