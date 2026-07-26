/**
 * Composition solver.
 *
 * The rubric states framing as screen positions — "rim centre at 30–38% of
 * frame height from the top", "ball-handler's feet at 62–74%". Hand-tuned
 * camera offsets cannot hold those targets, because the moment the subject
 * moves, the distance changes and the composition drifts. So instead of
 * choosing a position and hoping the framing lands, this solves for the
 * position that produces the framing.
 *
 * Two unknowns, two constraints:
 *
 *   distance  is solved so the *angular separation* between the two subjects
 *             matches the separation the desired screen positions imply. This
 *             is monotonic in distance — further away means a smaller
 *             separation — so a bisection converges quickly and cannot
 *             oscillate.
 *   pitch     then follows directly: aim so the lower subject lands on its
 *             target fraction.
 *
 * Camera height is an input, not a solved value. It is what sets the *feel* of
 * the shot — courtside eyeline versus an RTS view — and it should be chosen
 * deliberately rather than falling out of an optimiser.
 */

import { Vector3 } from 'three';
import { clamp } from '../core/MathX';

export interface FramingRequest {
  /** The subject that should sit high in frame — usually the rim. */
  upper: Vector3;
  /** The subject that should sit low in frame — usually the ball-handler. */
  lower: Vector3;
  /** Desired screen position of `upper`, as a fraction of frame height from the top. */
  upperFraction: number;
  /** Desired screen position of `lower`, same convention. */
  lowerFraction: number;
  /** Vertical field of view, in degrees. */
  fov: number;
  /** Camera eye height above the floor, in metres. */
  height: number;
  /** Horizontal direction from the lower subject toward the camera, normalised. */
  back: Vector3;
  /** Bounds on the solved ground distance. */
  minDistance: number;
  maxDistance: number;
}

export interface FramingResult {
  position: Vector3;
  /** World point the camera should aim at to land the composition. */
  lookAt: Vector3;
  distance: number;
  /** Downward pitch in degrees; positive means looking down. */
  pitchDegrees: number;
}

const _toUpper = new Vector3();
const _toLower = new Vector3();
const _eye = new Vector3();

/**
 * NDC y for a point at angle `theta` above the camera's forward axis, and its
 * inverse. Screen fraction from the top is `(1 - ndc) / 2`.
 */
function fractionToTanAngle(fraction: number, halfFovTan: number): number {
  const ndc = 1 - 2 * fraction;
  return ndc * halfFovTan;
}

/** Signed angle of `point` above the horizontal, seen from `eye`. */
function elevation(eye: Vector3, point: Vector3, out: Vector3): number {
  out.copy(point).sub(eye);
  const horizontal = Math.hypot(out.x, out.z);
  return Math.atan2(out.y, Math.max(horizontal, 1e-4));
}

export function solveFraming(req: FramingRequest): FramingResult {
  const halfFovTan = Math.tan((req.fov * Math.PI) / 360);

  // The angular separation the requested composition implies. Both targets are
  // converted through the same projection, so this is exact rather than a
  // small-angle approximation.
  const tanUpper = fractionToTanAngle(req.upperFraction, halfFovTan);
  const tanLower = fractionToTanAngle(req.lowerFraction, halfFovTan);
  const wantedSeparation = Math.atan(tanUpper) - Math.atan(tanLower);

  // Separation as a function of ground distance, measured from a trial eye.
  const separationAt = (distance: number): number => {
    _eye.copy(req.lower).addScaledVector(req.back, distance);
    _eye.y = req.height;
    return elevation(_eye, req.upper, _toUpper) - elevation(_eye, req.lower, _toLower);
  };

  // Bisection. Separation shrinks monotonically as the camera pulls back, so
  // the bracket is [min, max] and 22 halvings resolve to well under a
  // millimetre over any distance this game uses.
  let lo = req.minDistance;
  let hi = req.maxDistance;
  const sepLo = separationAt(lo);
  const sepHi = separationAt(hi);

  let distance: number;
  if (wantedSeparation >= sepLo) {
    // Even at the closest allowed distance the subjects are not far enough
    // apart on screen. Take the closest position and accept the compromise.
    distance = lo;
  } else if (wantedSeparation <= sepHi) {
    distance = hi;
  } else {
    for (let i = 0; i < 22; i++) {
      const mid = (lo + hi) * 0.5;
      if (separationAt(mid) > wantedSeparation) lo = mid;
      else hi = mid;
    }
    distance = (lo + hi) * 0.5;
  }

  const position = new Vector3().copy(req.lower).addScaledVector(req.back, distance);
  position.y = req.height;

  // With distance fixed, aim so the lower subject lands on its fraction. The
  // look-at point is placed at the lower subject's horizontal distance so the
  // resulting pitch is exact.
  const lowerElevation = elevation(position, req.lower, _toLower);
  const pitch = lowerElevation - Math.atan(tanLower);

  const horizontal = Math.hypot(req.lower.x - position.x, req.lower.z - position.z);
  const lookAt = new Vector3()
    .copy(req.lower)
    .sub(position)
    .setY(0)
    .normalize()
    .multiplyScalar(horizontal)
    .add(position);
  lookAt.y = position.y + Math.tan(pitch) * horizontal;

  return {
    position,
    lookAt,
    distance,
    pitchDegrees: clamp((-pitch * 180) / Math.PI, -89, 89),
  };
}

/**
 * Two octaves of smooth value noise in [-1, 1].
 *
 * Used for the micro-handheld shake. It has to be *smooth* — sampled random
 * numbers produce a jitter that reads as a broken frame rather than as an
 * operator breathing.
 */
export function handheldNoise(t: number, seed: number): number {
  const wave = (freq: number, phase: number): number => {
    const x = t * freq + phase;
    const i = Math.floor(x);
    const f = x - i;
    // Hash two lattice points and smoothstep between them.
    const h = (n: number): number => {
      const s = Math.sin((n + seed) * 127.1) * 43758.5453;
      return (s - Math.floor(s)) * 2 - 1;
    };
    const u = f * f * (3 - 2 * f);
    return h(i) * (1 - u) + h(i + 1) * u;
  };
  // The second octave is quieter and faster: a slow sway with a fine tremor on
  // top, which is what a shoulder-mounted camera actually does.
  return wave(0.55, 0) * 0.72 + wave(1.7, 31.4) * 0.28;
}
