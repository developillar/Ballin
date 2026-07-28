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

/**
 * Shot release timing.
 *
 * Lives here because three places have to agree on it and cannot import each
 * other: the gameplay code that scores a release, the HUD that draws the green
 * window, and the stylesheet that positions it. When those drift, the window is
 * drawn somewhere other than where it actually is, and the player is being lied
 * to by the interface — the worst possible bug in a timing mechanic.
 *
 * `src/ui/hudStyle.ts` positions `.meter .window` from these values; if you
 * change them, change the percentages there to match.
 */
export const SHOT = {
  /** Meter position, 0..1, where the release is perfect. */
  windowCentre: 0.84,
  /** Half-width of the perfect window, in meter units. */
  windowHalf: 0.06,
  /** Seconds for the meter to travel bottom to top. */
  chargeSeconds: 0.62,
} as const;

/** Signed X position of a basket. side = 1 → +X baseline. */
export function basketX(side: 1 | -1): number {
  return side * (COURT.halfLength - COURT.basketFromBaseline);
}

export const TEAM_HOME = 0;
export const TEAM_AWAY = 1;

/**
 * Overhead lighting rig. Added by the lighting agent so the arena's rafters,
 * catwalks and truss work can be hung around exactly the fixtures that the
 * light rig — and the baked environment map — are using. Purely additive: no
 * existing constant changes.
 *
 * Real NBA arenas light the floor from two long catwalk runs parallel to the
 * sidelines, roughly 17–18 m up, with the fixtures aimed steeply down so the
 * playing surface gets 1500–2000 lux while the bowl is held several stops
 * under. The layout below is that plot, expressed in court coordinates.
 */
export const LIGHT_RIG = {
  /**
   * Catwalk / fixture plane. Mirrors `ARENA.riggingY` in
   * `src/world/arenaGeometry.ts` — the arena hangs the visible truss, catwalks
   * and emissive pods here, and the analytic rig plus the environment bake put
   * their emitters at the same coordinates so a highlight in the backboard
   * belongs to a fixture that is physically in the room.
   */
  bankHeight: 17.8,
  /** The two long sideline runs, over the +Z and -Z catwalks. */
  sideline: { z: 10.8, length: 44, pods: 18 },
  /** Shorter cross banks over each basket. */
  cross: { x: 11.6, length: 15, pods: 6, drop: 1.1 },
  /** Outer wash over the lower bowl — keeps the crowd off pure black. */
  wash: { z: 22, length: 40, pods: 9, rise: 1.2 },
  /** A single fixture pod's lens. */
  pod: { width: 1.05, depth: 0.78 },
  /** Bank colour temperature — broadcast neutral. */
  kelvin: 5600,
  /** Height of the LED ribbon band around the lip of the lower bowl. */
  ribbonHeight: 3.2,
  /** Jumbotron hangs over centre court. */
  jumbotron: { centreHeight: 17.4, width: 7.4, depth: 5.2, faceHeight: 3.2 },
} as const;
