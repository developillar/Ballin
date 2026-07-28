/**
 * Court knowledge for the team AI: where the basket is, where the lane is, and
 * the five places an offence stands when it is not doing anything else.
 *
 * Everything here is pure — no engine, no systems, no state. It exists so
 * `TeamAI.ts` reads as behaviour rather than as arithmetic, and so the spacing
 * geometry can be checked against `Constants.ts` in one place instead of being
 * scattered through the steering code.
 *
 * Owned by the team-AI agent.
 */

import { Vector3 } from 'three';
import { COURT, HOOP, basketX } from '../core/Constants';

/**
 * The live basket.
 *
 * This match is street rules — `RULES.streetTarget` says so, and
 * `GameSystem.releaseShot` always solves its arc toward `basketX(1)` whoever is
 * shooting. So both teams attack the same rim and both teams defend it, and
 * "goal-side" means the same thing for everybody. If a full-court rules layer
 * ever arrives, this is the one constant that has to become a function of team.
 */
export const RIM = new Vector3(basketX(1), HOOP.rimHeight, 0);

/** The same basket, on the floor — every steering decision is 2D. */
export const RIM_GROUND = new Vector3(basketX(1), 0, 0);

/** The painted lane, as a box. Offence does not loiter in here. */
export const LANE = {
  xMin: COURT.halfLength - COURT.key.length,
  xMax: COURT.halfLength,
  halfZ: COURT.key.width * 0.5,
} as const;

/** Playable floor: the boundary lines plus half the painted apron. */
export const BOUNDS = {
  x: COURT.halfLength + COURT.apronX * 0.5,
  z: COURT.halfWidth + COURT.apronZ * 0.5,
} as const;

/**
 * Five standing spots — top of the key, two wings, two corners.
 *
 * They are stated as absolute court coordinates rather than as offsets so the
 * numbers can be read against `COURT` directly. Every one of them is outside
 * the three-point line (the corners sit 0.25 m behind the corner line at
 * `cornerFromCentre`), none of them is inside `LANE`, and the *minimum pairwise
 * distance across the whole set is 5.21 m* — which is the spacing floor the
 * offence cannot violate by standing still, whichever four of the five are
 * occupied. Clumping can then only come from steering, and that is what the
 * separation term in `TeamAI` is for.
 */
export const SPOTS: readonly Readonly<{ x: number; z: number; name: string }>[] = [
  { x: basketX(1) - 7.9, z: 0, name: 'top' },
  { x: basketX(1) - 6.0, z: -5.5, name: 'leftWing' },
  { x: basketX(1) - 6.0, z: 5.5, name: 'rightWing' },
  { x: basketX(1) - 1.0, z: -6.95, name: 'leftCorner' },
  { x: basketX(1) - 1.0, z: 6.95, name: 'rightCorner' },
];

/** Flat (XZ) distance between two points. */
export function flatDistance(a: Vector3, b: Vector3): number {
  return Math.hypot(a.x - b.x, a.z - b.z);
}

/**
 * Flat distance from a point to the segment `a`→`b`.
 *
 * This is what makes catching a ball a *swept* test rather than a point test,
 * and it is not a nicety. A pass travels 12 m/s; at 60 fps that is 200 mm
 * between frames, but the frame budget is not guaranteed and this project's own
 * headless harness runs at ~100 ms a frame, where the ball jumps 1.2 m — further
 * than anybody's reach. A point test then samples the ball just short of the
 * receiver and again just past him, and he never touches it: measured, that
 * turned a 5-of-6 completion rate into 1-of-5 purely as a function of frame
 * time. Catching has to depend on where the ball *went*, not on where it
 * happened to be when someone looked.
 */
export function segmentDistanceXZ(px: number, pz: number, a: Vector3, b: Vector3): number {
  const dx = b.x - a.x;
  const dz = b.z - a.z;
  const len2 = dx * dx + dz * dz;
  if (len2 < 1e-8) return Math.hypot(px - a.x, pz - a.z);
  let t = ((px - a.x) * dx + (pz - a.z) * dz) / len2;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return Math.hypot(px - (a.x + dx * t), pz - (a.z + dz * t));
}

/** True when a floor point is inside the painted lane. */
export function inLane(x: number, z: number): boolean {
  return x > LANE.xMin && x < LANE.xMax && Math.abs(z) < LANE.halfZ;
}

/**
 * Shortest push that takes a point out of the lane, written into `out`.
 *
 * Sideways whenever sideways is the shorter way out, which it is everywhere
 * except right on the free-throw line — a cutter who has to clear the paint
 * steps out to the wing, he does not walk backwards up the middle.
 */
export function laneEscape(x: number, z: number, out: Vector3): Vector3 {
  out.set(0, 0, 0);
  if (!inLane(x, z)) return out;
  const outZ = LANE.halfZ - Math.abs(z);
  const outX = x - LANE.xMin;
  if (outZ <= outX) out.z = Math.sign(z || 1) * (outZ + 0.6);
  else out.x = -(outX + 0.6);
  return out;
}

/**
 * The goal-side point for a defender: `gap` metres from `man` along the line
 * from `man` to the basket, shaded `shade` metres toward `ball`.
 *
 * The shade is applied *perpendicular* to the man→basket line, so it can move
 * the defender across to help without ever moving him off the ball side of his
 * man — the along-the-line component stays exactly `gap`, and `gap` is capped
 * at 55% of the man's own distance from the rim so a defender guarding someone
 * in the restricted area does not end up under the backboard.
 */
export function goalSidePoint(
  man: Vector3,
  ball: Vector3,
  gap: number,
  shade: number,
  out: Vector3,
): Vector3 {
  let ux = RIM_GROUND.x - man.x;
  let uz = RIM_GROUND.z - man.z;
  const toRim = Math.hypot(ux, uz) || 1e-4;
  ux /= toRim;
  uz /= toRim;

  const g = Math.min(gap, toRim * 0.55);

  // Perpendicular in the floor plane, and how far the ball sits along it.
  const px = uz;
  const pz = -ux;
  let bx = ball.x - man.x;
  let bz = ball.z - man.z;
  const toBall = Math.hypot(bx, bz) || 1e-4;
  bx /= toBall;
  bz /= toBall;
  const s = Math.max(-1, Math.min(1, bx * px + bz * pz)) * shade;

  out.set(man.x + ux * g + px * s, 0, man.z + uz * g + pz * s);
  return out;
}

/**
 * Signed goal-side depth: how far `defender` sits along the man→basket line.
 *
 * Positive means goal-side. This is the number the verification probe asserts
 * on, so it lives next to the code that produces it rather than being
 * re-derived in the test.
 */
export function goalSideDepth(defender: Vector3, man: Vector3): number {
  let ux = RIM_GROUND.x - man.x;
  let uz = RIM_GROUND.z - man.z;
  const len = Math.hypot(ux, uz) || 1e-4;
  ux /= len;
  uz /= len;
  return (defender.x - man.x) * ux + (defender.z - man.z) * uz;
}

/**
 * Where a ball in flight comes down to catchable height, written into `out`.
 *
 * Ballistic and drag-free on purpose: it is a chase target that is re-solved
 * every frame, so a 10% error at 1.5 s out costs a step and is gone by the time
 * it matters. Modelling drag here would be precision the chase cannot use.
 */
export function landingPoint(
  position: Vector3,
  velocity: Vector3,
  catchHeight: number,
  gravity: number,
  out: Vector3,
): Vector3 {
  const g = Math.abs(gravity);
  const dy = position.y - catchHeight;
  let t: number;
  if (dy <= 0) {
    t = 0;
  } else {
    const disc = velocity.y * velocity.y + 2 * g * dy;
    t = disc <= 0 ? 0 : (velocity.y + Math.sqrt(disc)) / g;
  }
  t = Math.max(0, Math.min(t, 2.5));
  out.set(position.x + velocity.x * t, 0, position.z + velocity.z * t);
  out.x = Math.max(-BOUNDS.x, Math.min(BOUNDS.x, out.x));
  out.z = Math.max(-BOUNDS.z, Math.min(BOUNDS.z, out.z));
  return out;
}
