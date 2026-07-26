/**
 * Inverse kinematics.
 *
 * Basketball animation lives or dies on contact: the shooting hand has to end
 * up under the ball, the guide hand on its side, the plant foot has to meet the
 * floor without sliding, and the head has to track the rim. All of that is IK
 * layered on top of whatever the blend tree produced, which is why these
 * solvers operate on an already-posed skeleton rather than generating poses.
 */

import { Bone, Matrix4, Quaternion, Vector3 } from 'three';
import { clamp } from '../core/MathX';

const _a = new Vector3();
const _b = new Vector3();
const _c = new Vector3();
const _t = new Vector3();
const _axis = new Vector3();
const _q = new Quaternion();
const _qi = new Quaternion();
const _m = new Matrix4();

/** World-space position of a bone's joint. */
export function boneWorld(bone: Bone, out = new Vector3()): Vector3 {
  return out.setFromMatrixPosition(bone.matrixWorld);
}

/** World-space rotation of a bone. */
export function boneWorldQuat(bone: Bone, out = new Quaternion()): Quaternion {
  bone.matrixWorld.decompose(_a, out, _b);
  return out;
}

/**
 * Analytic two-bone IK — the workhorse for arms and legs.
 *
 * Solves for the interior angle with the law of cosines, then swings the whole
 * chain so the end effector lands on the target. The pole vector controls which
 * way the joint bends: for a knee that is forward, for an elbow it is back and
 * slightly out, and getting it wrong is instantly readable as "broken".
 *
 * @param root   upper bone (shoulder / hip)
 * @param mid    middle bone (elbow / knee)
 * @param tip    end bone (hand / foot) — only read for its length
 * @param target world-space position the tip joint should reach
 * @param pole   world-space hint the mid joint bends toward
 * @param weight 0 leaves the pose untouched, 1 fully solves
 */
export function solveTwoBoneIK(
  root: Bone,
  mid: Bone,
  tip: Bone,
  target: Vector3,
  pole: Vector3,
  weight = 1,
): void {
  if (weight <= 0.0001) return;

  root.updateWorldMatrix(true, false);
  mid.updateWorldMatrix(false, false);
  tip.updateWorldMatrix(false, false);

  const pRoot = boneWorld(root, _a.clone());
  const pMid = boneWorld(mid, _b.clone());
  const pTip = boneWorld(tip, _c.clone());

  const lenUpper = pRoot.distanceTo(pMid);
  const lenLower = pMid.distanceTo(pTip);
  const maxReach = (lenUpper + lenLower) * 0.9995;

  _t.copy(target).sub(pRoot);
  let dist = _t.length();
  if (dist < 1e-5) return;
  // Clamp inside the reachable annulus so the solver never snaps.
  const minReach = Math.abs(lenUpper - lenLower) * 1.0005 + 1e-4;
  dist = clamp(dist, minReach, maxReach);
  const dir = _t.clone().normalize();
  const clampedTarget = pRoot.clone().addScaledVector(dir, dist);

  // --- Interior (bend) angle ---------------------------------------------
  const cosMid = clamp(
    (lenUpper * lenUpper + lenLower * lenLower - dist * dist) / (2 * lenUpper * lenLower),
    -1,
    1,
  );
  const wantMid = Math.acos(cosMid);

  const curDist = pRoot.distanceTo(pTip);
  const curCosMid = clamp(
    (lenUpper * lenUpper + lenLower * lenLower - curDist * curDist) / (2 * lenUpper * lenLower),
    -1,
    1,
  );
  const curMid = Math.acos(curCosMid);

  // Bend axis: perpendicular to the plane the chain currently occupies, biased
  // by the pole so a straight chain still knows which way to fold.
  _axis.copy(pMid).sub(pRoot).cross(_c.copy(pTip).sub(pMid));
  if (_axis.lengthSq() < 1e-8) {
    _axis.copy(pole).sub(pRoot).cross(dir);
    if (_axis.lengthSq() < 1e-8) _axis.set(1, 0, 0);
  }
  _axis.normalize();

  applyWorldDelta(mid, _q.setFromAxisAngle(_axis, (wantMid - curMid) * weight));

  // --- Swing the chain so the tip lands on the target ----------------------
  root.updateWorldMatrix(true, false);
  mid.updateWorldMatrix(false, false);
  tip.updateWorldMatrix(false, false);
  boneWorld(root, pRoot);
  boneWorld(tip, pTip);

  _a.copy(pTip).sub(pRoot).normalize();
  _b.copy(clampedTarget).sub(pRoot).normalize();
  _q.setFromUnitVectors(_a, _b);
  if (weight < 1) _q.slerp(_qi.identity(), 1 - weight);
  applyWorldDelta(root, _q);

  // --- Roll the chain around the root→tip axis to satisfy the pole ---------
  root.updateWorldMatrix(true, false);
  mid.updateWorldMatrix(false, false);
  tip.updateWorldMatrix(false, false);
  boneWorld(root, pRoot);
  boneWorld(mid, pMid);
  boneWorld(tip, pTip);

  _axis.copy(pTip).sub(pRoot);
  const axisLen = _axis.length();
  if (axisLen > 1e-5) {
    _axis.divideScalar(axisLen);
    // Project current mid and the pole onto the plane normal to the chain axis.
    const curOff = _b.copy(pMid).sub(pRoot);
    curOff.addScaledVector(_axis, -curOff.dot(_axis));
    const poleOff = _c.copy(pole).sub(pRoot);
    poleOff.addScaledVector(_axis, -poleOff.dot(_axis));
    if (curOff.lengthSq() > 1e-8 && poleOff.lengthSq() > 1e-8) {
      curOff.normalize();
      poleOff.normalize();
      let angle = Math.acos(clamp(curOff.dot(poleOff), -1, 1));
      if (curOff.clone().cross(poleOff).dot(_axis) < 0) angle = -angle;
      applyWorldDelta(root, _q.setFromAxisAngle(_axis, angle * weight));
    }
  }
}

/**
 * Applies a world-space rotation delta to a bone, converting into its parent's
 * space so the rest of the hierarchy follows correctly.
 */
export function applyWorldDelta(bone: Bone, worldDelta: Quaternion): void {
  const parent = bone.parent;
  if (parent) {
    parent.updateWorldMatrix(true, false);
    _m.copy(parent.matrixWorld);
    const parentQuat = new Quaternion();
    _m.decompose(new Vector3(), parentQuat, new Vector3());
    const inv = parentQuat.clone().invert();
    // local' = (P⁻¹ · Δ · P) · local
    const localDelta = inv.multiply(worldDelta).multiply(parentQuat);
    bone.quaternion.premultiply(localDelta);
  } else {
    bone.quaternion.premultiply(worldDelta);
  }
  bone.updateMatrixWorld(true);
}

/**
 * Aims a bone's forward axis at a target, with per-axis limits so a head does
 * not spin off its neck. Angles are in radians.
 */
export function solveLookAt(
  bone: Bone,
  target: Vector3,
  forward: Vector3,
  limits: { yaw: number; pitch: number },
  weight = 1,
): void {
  if (weight <= 0.0001) return;
  bone.updateWorldMatrix(true, false);
  const pos = boneWorld(bone, _a.clone());
  const worldQuat = boneWorldQuat(bone, _q.clone());

  // Desired direction in the bone's parent space.
  const desired = _b.copy(target).sub(pos);
  if (desired.lengthSq() < 1e-8) return;
  desired.normalize();

  const current = _c.copy(forward).applyQuaternion(worldQuat).normalize();

  // Split the correction into yaw (about world up) and pitch (about the
  // bone's right) so each can be clamped independently.
  const up = _t.set(0, 1, 0);
  const flatCur = new Vector3(current.x, 0, current.z);
  const flatDes = new Vector3(desired.x, 0, desired.z);
  if (flatCur.lengthSq() > 1e-8 && flatDes.lengthSq() > 1e-8) {
    flatCur.normalize();
    flatDes.normalize();
    let yaw = Math.acos(clamp(flatCur.dot(flatDes), -1, 1));
    if (flatCur.clone().cross(flatDes).dot(up) < 0) yaw = -yaw;
    yaw = clamp(yaw, -limits.yaw, limits.yaw) * weight;
    applyWorldDelta(bone, _qi.setFromAxisAngle(up, yaw));
  }

  bone.updateWorldMatrix(true, false);
  boneWorldQuat(bone, worldQuat);
  current.copy(forward).applyQuaternion(worldQuat).normalize();
  const right = new Vector3().crossVectors(current, up);
  if (right.lengthSq() > 1e-8) {
    right.normalize();
    let pitch = Math.asin(clamp(desired.y, -1, 1)) - Math.asin(clamp(current.y, -1, 1));
    pitch = clamp(pitch, -limits.pitch, limits.pitch) * weight;
    applyWorldDelta(bone, _qi.setFromAxisAngle(right, pitch));
  }
}

/**
 * Foot planting. Given the floor height under a foot, lifts or drops the hip
 * and re-solves the leg so both feet stay in contact without the pelvis
 * popping — the difference between a character that walks and one that skates.
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
): void {
  if (plant.weight <= 0.001) return;
  // Pole target sits well in front of the knee so it never inverts backward.
  const hip = boneWorld(thigh, new Vector3());
  const pole = hip.clone().addScaledVector(kneeForward, 1.2).setY(plant.target.y + 0.42);
  solveTwoBoneIK(thigh, shin, foot, plant.target, pole, plant.weight);

  // Roll the ankle so the sole matches the floor normal.
  if (plant.normal.lengthSq() > 1e-6) {
    foot.updateWorldMatrix(true, false);
    const q = boneWorldQuat(foot, new Quaternion());
    const soleDown = new Vector3(0, -1, 0).applyQuaternion(q).normalize();
    const wanted = plant.normal.clone().negate().normalize();
    const delta = new Quaternion().setFromUnitVectors(soleDown, wanted);
    if (plant.weight < 1) delta.slerp(new Quaternion(), 1 - plant.weight);
    applyWorldDelta(foot, delta);
  }
}

/**
 * Two-handed ball hold. Places the shooting hand under the ball and the guide
 * hand on its side, both oriented to the ball's centre — the pose everyone
 * recognises from a jump shot.
 */
export function solveBallHold(
  shootArm: readonly [Bone, Bone, Bone],
  guideArm: readonly [Bone, Bone, Bone],
  ballCentre: Vector3,
  ballRadius: number,
  chestForward: Vector3,
  chestRight: Vector3,
  weight = 1,
): void {
  const under = ballCentre.clone().addScaledVector(chestForward, -0.012).setY(ballCentre.y - ballRadius * 0.92);
  const side = ballCentre.clone().addScaledVector(chestRight, -ballRadius * 0.95).addScaledVector(chestForward, -0.01);

  // Elbows tuck in and down under the ball; the guide elbow flares out.
  const shootPole = ballCentre
    .clone()
    .addScaledVector(chestForward, -0.55)
    .addScaledVector(chestRight, 0.1);
  shootPole.y -= 0.5;
  const guidePole = ballCentre
    .clone()
    .addScaledVector(chestForward, -0.42)
    .addScaledVector(chestRight, -0.55);
  guidePole.y -= 0.34;

  solveTwoBoneIK(shootArm[0], shootArm[1], shootArm[2], under, shootPole, weight);
  solveTwoBoneIK(guideArm[0], guideArm[1], guideArm[2], side, guidePole, weight * 0.92);
}
