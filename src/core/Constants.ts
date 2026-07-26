/**
 * Regulation NBA geometry, in metres. Every gameplay and rendering system reads
 * from here so the court, the physics colliders and the AI navigation all agree
 * on one source of truth.
 *
 * Court axes:
 *   +X  → toward the far baseline (length of the court)
 *   +Y  → up
 *   +Z  → toward the near sideline (width of the court)
 *   origin = centre circle, floor level
 */

export const FT = 0.3048;
export const IN = 0.0254;

export const COURT = {
  /** 94 ft */
  length: 94 * FT,
  /** 50 ft */
  width: 50 * FT,
  halfLength: 47 * FT,
  halfWidth: 25 * FT,
  /** Painted apron beyond the boundary lines before the stands begin. */
  apronX: 2.9,
  apronZ: 2.3,
  lineWidth: 2 * IN,
  centreCircleRadius: 6 * FT,
  /** Distance from baseline to the centre of the rim. */
  basketFromBaseline: 5.25 * FT,
  /** Restricted-area arc under the rim. */
  restrictedRadius: 4 * FT,
  key: {
    /** 16 ft wide lane. */
    width: 16 * FT,
    /** Baseline → free-throw line. */
    length: 19 * FT,
    circleRadius: 6 * FT,
  },
  threePoint: {
    /** 23 ft 9 in from the centre of the rim. */
    radius: 23.75 * FT,
    /** 22 ft from the centre of the rim in the corners. */
    cornerFromCentre: 22 * FT,
    /** Corner-three straightaway is 3 ft from the sideline. */
    cornerInsetFromSideline: 3 * FT,
    /** Length of the straight corner section, measured from the baseline. */
    cornerRunLength: 14 * FT,
  },
} as const;

export const HOOP = {
  /** Rim height, 10 ft. */
  rimHeight: 10 * FT,
  /** Inside radius of the ring: 18 in diameter. */
  rimRadius: 9 * IN,
  /** Ring stock is 5/8 in bar. */
  rimTubeRadius: 0.625 * 0.5 * IN,
  /** Rim centre stands 6 in out from the backboard face. */
  rimOffsetFromBoard: 6 * IN,
  board: {
    width: 6 * FT,
    height: 3.5 * FT,
    thickness: 1.5 * IN,
    /** Bottom edge of the glass. */
    bottomHeight: 9.5 * FT - 0.5 * FT,
    innerSquare: { width: 24 * IN, height: 18 * IN, borderWidth: 2 * IN },
  },
  net: {
    length: 15 * IN,
    /** 12 loops around the ring. */
    strands: 12,
    /** Rings of mesh down the net. */
    segments: 9,
    /** Net mouth tapers in toward the bottom. */
    bottomRadiusScale: 0.72,
  },
} as const;

export const BALL = {
  /** Size 7: 29.5 in circumference → 0.1192 m radius. */
  radius: 0.11925,
  /** 22 oz. */
  mass: 0.6237,
  /** Moment of inertia for a thin spherical shell: (2/3) m r². */
  inertiaFactor: 2 / 3,
  /** Coefficient of restitution off hardwood — an NBA ball rebounds to ~54%. */
  restitutionFloor: 0.735,
  restitutionRim: 0.55,
  restitutionBoard: 0.62,
  restitutionBody: 0.35,
  /** Tangential friction on contact. */
  frictionFloor: 0.52,
  frictionRim: 0.42,
  frictionBoard: 0.31,
  /** Aerodynamics. */
  dragCoefficient: 0.54,
  magnusCoefficient: 0.22,
  /** Angular velocity bleed per second in flight. */
  spinDecay: 0.055,
} as const;

export const PHYSICS = {
  gravity: -9.80665,
  /** Sea-level air density (kg/m³). */
  airDensity: 1.2041,
  /** Fixed simulation step — 240 Hz keeps rim contacts stable. */
  fixedStep: 1 / 240,
  maxSubSteps: 8,
  sleepLinear: 0.018,
  sleepAngular: 0.12,
} as const;

export const PLAYER = {
  /** Average NBA height, in metres (6 ft 6 in). */
  height: 1.98,
  shoulderWidth: 0.52,
  reach: 2.6,
  /** Standing vertical, metres. */
  vertical: 0.86,
  topSpeed: 7.3,
  sprintSpeed: 8.6,
  acceleration: 22,
  deceleration: 30,
  turnRate: 11,
  releaseHeightFactor: 1.28,
} as const;

export const RULES = {
  quarterSeconds: 12 * 60,
  shotClockSeconds: 24,
  shotClockOffensiveRebound: 14,
  quarters: 4,
  /** Street-rules pickup mode used by the default match. */
  streetTarget: 21,
} as const;

/** Signed X position of a basket. side = 1 → +X baseline. */
export function basketX(side: 1 | -1): number {
  return side * (COURT.halfLength - COURT.basketFromBaseline);
}

export const TEAM_HOME = 0;
export const TEAM_AWAY = 1;
