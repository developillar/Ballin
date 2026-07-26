/**
 * The crowd.
 *
 * Two things decide whether a seating bowl reads as a photographed arena or as
 * a lit gym, and neither of them is polygon count.
 *
 * The first is **exposure**. Rubric §1.1 makes the court-to-bowl ratio the
 * master criterion: hardwood at 95–140 mean sRGB, bowl at 18–45, i.e. 2.5–4
 * stops down. That is why this shader carries its own lighting model rather
 * than sitting on the standard rig — the ratio is then a constant we set, not
 * an accident of how many lights happened to reach row 9. Dark is not enough on
 * its own though: §1.1 also demands structure (stddev > 6 in any 200 px window)
 * and a floor off true black (6–16), so the bowl is built as *dark with
 * readable silhouettes* — mostly near-black outerwear, a minority in team
 * colour, a thin scattering of light shirts that catch the spill, a directional
 * rim that separates one row from the next, and a spill term that falls off as
 * the rake climbs so the top of the bowl sits a stop under the front row.
 *
 * The second is **silhouette**. At 12–40 px a face is meaningless but a
 * shoulder line is not. So a spectator is a rounded skull over a neck, a chest
 * that tapers into sloped shoulders, a waist, arms that read as arms against
 * the body, and a lap with a knee break. Pose is per-person and resolved in the
 * vertex shader from a hash — leaning forward, arms crossed, on a phone, hands
 * on the lap, arms wide on the seat back, standing — so the variety costs an
 * attribute read rather than a draw call.
 *
 * The economics: one InstancedMesh per bowl block, where an *instance* is a pod
 * of several adjacent seats rather than one person. Every per-person difference
 * — height, girth, yaw, pose, clothing, skin, hair, phone, idle phase — is
 * derived in the vertex shader from a hash of (instance seed, seat index).
 * Nothing is a per-frame CPU matrix write; the only thing the CPU touches each
 * frame is one float per pod, the excitement level, which drives the wave.
 *
 * Owned by the crowd agent.
 */

import {
  Color,
  DynamicDrawUsage,
  InstancedBufferAttribute,
  InstancedMesh,
  Object3D,
  ShaderMaterial,
  StaticDrawUsage,
  Vector2,
  Vector3,
} from 'three';
import { FACE_OPEN_BOTTOM, FACE_SIDES, FACE_TOP, MeshBuilder } from './arenaGeometry';

// Vertex part tags. Kept as floats because they ride in an attribute.
const PART_SEAT = 0;
const PART_THIGH = 1;
const PART_TORSO = 2;
const PART_HEAD = 3;
const PART_HAIR = 4;
const PART_UPPERARM = 5;
const PART_PHONE = 6;
const PART_SHIN = 7;
const PART_FOREARM = 8;
const PART_NECK = 9;

// -----------------------------------------------------------------------------
// Skeleton
//
// One rest pose, seated, local origin on the row tread at the seat centre,
// facing local −Z (toward the court). Every other pose is a rotation of this
// one about a named joint, applied in the vertex shader, so the geometry is
// built once and posed 8 000 times for free.
// -----------------------------------------------------------------------------

const HIP_Y = 0.500;
const HIP_Z = 0.230;
const KNEE_Y = 0.470;
const KNEE_Z = -0.180;
const FOOT_Y = 0.045;
const FOOT_Z = -0.140;
const SHO_X = 0.168;
const SHO_Y = 0.975;
const SHO_Z = 0.185;
const ELB_X = 0.178;
const ELB_Y = 0.745;
const ELB_Z = 0.170;
const HAND_X = 0.150;
const HAND_Y = 0.588;
const HAND_Z = -0.020;
const NECK_Y = 1.010;
const NECK_Z = 0.178;
/** Bottom of the hair cap — the pivot the per-person hair height scales about. */
const HAIR_BASE = 1.150;

/**
 * Standing. The thigh swings down under the pelvis and the shin follows, which
 * gives back exactly `STAND_RISE` of height; the body then steps forward into
 * the gap in front of the seat rather than standing on its own seat pan.
 */
const HIP_STAND = -1.498;
const KNEE_STAND = 1.592;
const STAND_RISE = 0.383;
const STAND_STEP = 0.22;

export type CrowdDetail = 0 | 1 | 2 | 3;

export interface PodPlacement {
  x: number;
  y: number;
  z: number;
  rotY: number;
  /** Local exposure multiplier: courtside sits closer to the floor spill. */
  tone: number;
}

export interface CrowdBlockSpec {
  name: string;
  detail: CrowdDetail;
  /** Seats per instance. */
  podSize: number;
  /** Seat pitch along the row, metres. */
  pitch: number;
  /** 0–1 fraction of seats that hold a person. */
  occupancy: number;
  /** Fraction of spectators holding a phone. */
  phoneRate: number;
  /** Base seat shell colour. */
  seatColor: number;
  placements: PodPlacement[];
}

// -----------------------------------------------------------------------------
// Geometry
// -----------------------------------------------------------------------------

/**
 * A four-sided prism between two joints, tapering from (w0, t0) to (w1, t1).
 *
 * This is the whole reason limbs read as limbs: a `box` can only be axis
 * aligned, so a thigh built from one is a slab that either floats horizontally
 * or stands vertically, and the knee break — the single cue that says *seated*
 * — is impossible. A prism laid along an arbitrary joint-to-joint axis gives
 * the thigh its forward-and-down run and the shin its drop, for the same eight
 * triangles.
 */
function limb(
  b: MeshBuilder,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  w0: number, t0: number,
  w1: number, t1: number,
  capEnd = false,
): void {
  const dx = bx - ax;
  const dy = by - ay;
  const dz = bz - az;
  const len = Math.hypot(dx, dy, dz) || 1;
  const dxn = dx / len;
  const dyn = dy / len;
  const dzn = dz / len;

  // Pick the reference axis furthest from the limb direction so the cross
  // product never degenerates on a vertical thigh or a horizontal forearm.
  const rx = 0;
  const ry = Math.abs(dyn) > Math.abs(dzn) ? 0 : 1;
  const rz = Math.abs(dyn) > Math.abs(dzn) ? 1 : 0;

  let ux = dyn * rz - dzn * ry;
  let uy = dzn * rx - dxn * rz;
  let uz = dxn * ry - dyn * rx;
  const ul = Math.hypot(ux, uy, uz) || 1;
  ux /= ul;
  uy /= ul;
  uz /= ul;

  const vx = uy * dzn - uz * dyn;
  const vy = uz * dxn - ux * dzn;
  const vz = ux * dyn - uy * dxn;

  // Corner order runs around the section, so consecutive pairs bound one face.
  const su = [-1, 1, 1, -1];
  const sv = [-1, -1, 1, 1];
  const A: number[][] = [];
  const B: number[][] = [];
  for (let i = 0; i < 4; i++) {
    A.push([
      ax + su[i] * w0 * ux + sv[i] * t0 * vx,
      ay + su[i] * w0 * uy + sv[i] * t0 * vy,
      az + su[i] * w0 * uz + sv[i] * t0 * vz,
    ]);
    B.push([
      bx + su[i] * w1 * ux + sv[i] * t1 * vx,
      by + su[i] * w1 * uy + sv[i] * t1 * vy,
      bz + su[i] * w1 * uz + sv[i] * t1 * vz,
    ]);
  }
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4;
    b.quad(
      A[i][0], A[i][1], A[i][2],
      B[i][0], B[i][1], B[i][2],
      B[j][0], B[j][1], B[j][2],
      A[j][0], A[j][1], A[j][2],
    );
  }
  if (capEnd) {
    b.quad(
      B[0][0], B[0][1], B[0][2],
      B[3][0], B[3][1], B[3][2],
      B[2][0], B[2][1], B[2][2],
      B[1][0], B[1][1], B[1][2],
    );
  }
}

/**
 * One pod: `podSize` seats side by side along local +X, all facing local −Z.
 *
 * Triangle budget per seat: 18 at detail 0 (upper bowl, ~10 px figures), 48 at
 * detail 1 (far lower bowl), 108 at detail 2 (near lower bowl) and 140 at
 * detail 3 (courtside, where the figures are 60–120 px and every one of those
 * triangles is doing visible work).
 */
function buildPod(podSize: number, pitch: number, detail: CrowdDetail): MeshBuilder {
  const b = new MeshBuilder();
  b.attribute('aPart', 1, [PART_SEAT]);
  b.attribute('aSeat', 1, [0]);
  b.attribute('aSide', 1, [0]);

  const halfSeat = pitch * 0.5;
  for (let k = 0; k < podSize; k++) {
    const bx = (k - (podSize - 1) * 0.5) * pitch;
    b.set('aSeat', k);
    b.set('aSide', 0);

    // --- the seat itself -----------------------------------------------------
    if (detail >= 1) {
      b.set('aPart', PART_SEAT);
      // Pan: a single upward quad. The underside is never seen from the court.
      const pw = halfSeat * 0.86;
      b.quad(
        bx - pw, 0.44, 0.03,
        bx + pw, 0.44, 0.03,
        bx + pw, 0.435, 0.42,
        bx - pw, 0.435, 0.42,
      );
      // Back, raked back 8°, with a visible gap to its neighbour.
      b.box(bx, 0.665, 0.425, pw, 0.225, 0.035, {
        faces: detail >= 2 ? FACE_SIDES | FACE_TOP : FACE_SIDES,
        shearZ: 0.055,
      });
    }

    // --- legs ----------------------------------------------------------------
    // Two thighs from detail 2 up: a single slab across both legs is the one
    // remaining piece of the seated silhouette that reads as furniture rather
    // than as a person, and at six rows out it is directly in shot. Shins stay
    // at detail 3 only — in a raked bowl the row in front eats everything below
    // mid-thigh, so they are pure cost anywhere but courtside.
    if (detail >= 2) {
      b.set('aPart', PART_THIGH);
      for (const s of [-1, 1]) {
        b.set('aSide', s);
        limb(b, bx + s * 0.080, HIP_Y, HIP_Z, bx + s * 0.090, KNEE_Y, KNEE_Z,
          0.072, 0.084, 0.063, 0.074);
      }
      if (detail >= 3) {
        b.set('aPart', PART_SHIN);
        for (const s of [-1, 1]) {
          b.set('aSide', s);
          limb(b, bx + s * 0.090, KNEE_Y, KNEE_Z, bx + s * 0.086, FOOT_Y, FOOT_Z,
            0.059, 0.066, 0.050, 0.058, true);
        }
      }
      b.set('aSide', 0);
    } else if (detail >= 1) {
      b.set('aPart', PART_THIGH);
      limb(b, bx, HIP_Y, HIP_Z, bx, KNEE_Y, KNEE_Z, 0.158, 0.086, 0.138, 0.076);
    }

    // --- torso ---------------------------------------------------------------
    // Two stacked frusta. The lower one is the waist, the upper one narrows
    // hard toward the neck, and that slope *is* the shoulder line — without it
    // the top of the body is a flat slab and a bowl full of them reads as a
    // field of envelopes.
    b.set('aPart', PART_TORSO);
    if (detail >= 2) {
      b.box(bx, 0.683, 0.196, 0.170, 0.117, 0.112, {
        faces: FACE_SIDES,
        bottomScaleX: 0.90,
        bottomScaleZ: 0.94,
        topScaleX: 1.02,
      });
      b.box(bx, 0.883, 0.192, 0.176, 0.083, 0.114, {
        faces: FACE_SIDES | FACE_TOP,
        topScaleX: 0.64,
        topScaleZ: 0.76,
        shearZ: 0.010,
      });
    } else if (detail >= 1) {
      b.box(bx, 0.775, 0.194, 0.172, 0.190, 0.113, {
        faces: FACE_OPEN_BOTTOM,
        topScaleX: 0.78,
        topScaleZ: 0.84,
        bottomScaleX: 0.90,
        shearZ: 0.030,
      });
    } else {
      b.box(bx, 0.790, 0.194, 0.168, 0.205, 0.110, {
        faces: FACE_OPEN_BOTTOM,
        topScaleX: 0.76,
        topScaleZ: 0.84,
        bottomScaleX: 0.88,
      });
    }

    // --- neck ----------------------------------------------------------------
    if (detail >= 2) {
      b.set('aPart', PART_NECK);
      b.box(bx, NECK_Y - 0.010, NECK_Z + 0.002, 0.052, 0.043, 0.050, {
        faces: FACE_SIDES,
        topScaleX: 0.94,
        topScaleZ: 0.94,
      });
    }

    // --- head ----------------------------------------------------------------
    b.set('aPart', PART_HEAD);
    if (detail >= 2) {
      // Jaw: widens upward off the neck.
      b.box(bx, 1.080, 0.174, 0.068, 0.041, 0.074, {
        faces: FACE_SIDES,
        bottomScaleX: 0.70,
        bottomScaleZ: 0.76,
      });
      // Cranium: narrows again toward the crown, so the profile is round-ish
      // instead of the cube that a single box gives.
      b.box(bx, 1.156, 0.174, 0.074, 0.036, 0.079, {
        faces: FACE_SIDES | FACE_TOP,
        bottomScaleX: 0.94,
        bottomScaleZ: 0.96,
        topScaleX: 0.76,
        topScaleZ: 0.80,
      });
    } else if (detail >= 1) {
      b.box(bx, 1.120, 0.174, 0.076, 0.078, 0.081, {
        faces: FACE_SIDES | FACE_TOP,
        bottomScaleX: 0.76,
        bottomScaleZ: 0.82,
        topScaleX: 0.84,
        topScaleZ: 0.88,
      });
    } else {
      b.octa(bx, 1.118, 0.180, 0.080, 0.096, 0.084);
    }

    // --- hair ----------------------------------------------------------------
    if (detail >= 1) {
      b.set('aPart', PART_HAIR);
      b.box(bx, HAIR_BASE + 0.046, 0.174, 0.079, 0.046, 0.084, {
        faces: FACE_SIDES | FACE_TOP,
        bottomScaleX: 1.03,
        bottomScaleZ: 1.03,
        topScaleX: 0.62,
        topScaleZ: 0.68,
        shearZ: -0.006,
      });
    }

    // --- arms ----------------------------------------------------------------
    if (detail >= 2) {
      b.set('aPart', PART_UPPERARM);
      for (const s of [-1, 1]) {
        b.set('aSide', s);
        limb(b, bx + s * SHO_X, SHO_Y, SHO_Z, bx + s * ELB_X, ELB_Y, ELB_Z,
          0.050, 0.058, 0.044, 0.052);
      }
      b.set('aPart', PART_FOREARM);
      for (const s of [-1, 1]) {
        b.set('aSide', s);
        limb(b, bx + s * ELB_X, ELB_Y, ELB_Z, bx + s * HAND_X, HAND_Y, HAND_Z,
          0.042, 0.050, 0.037, 0.044, detail >= 3);
      }

      // A phone, parented to the right forearm so it swings up with the pose.
      // Small on purpose: 3–5 px at bowl distance, per §6.3.
      b.set('aPart', PART_PHONE);
      b.set('aSide', 1);
      const px = bx + HAND_X * 0.92;
      b.quad(
        px - 0.018, HAND_Y + 0.010, HAND_Z - 0.040,
        px + 0.018, HAND_Y + 0.010, HAND_Z - 0.040,
        px + 0.018, HAND_Y + 0.062, HAND_Z - 0.028,
        px - 0.018, HAND_Y + 0.062, HAND_Z - 0.028,
      );
      b.set('aSide', 0);
    }
  }
  return b;
}

// -----------------------------------------------------------------------------
// Material
// -----------------------------------------------------------------------------

const f = (n: number): string => n.toFixed(4);

const CROWD_VERT = /* glsl */ `
  attribute float aPart;
  attribute float aSeat;
  attribute float aSide;
  attribute float iSeed;
  attribute float iReact;
  attribute float iTone;

  uniform float uTime;
  uniform float uPodSize;
  uniform float uPitch;
  uniform float uOccupancy;
  uniform float uPhoneRate;
  uniform float uPhoneLit;
  uniform float uExposure;
  uniform float uPhoneGain;
  uniform float uStandRate;
  uniform float uHasShin;
  uniform float uStandRise;
  uniform vec3  uSeatCol;
  uniform vec3  uKeyDir;
  uniform vec3  uKeyCol;
  uniform vec3  uFillDir;
  uniform vec3  uFillCol;
  uniform vec3  uAmbTop;
  uniform vec3  uAmbBot;
  uniform vec3  uSpillCol;
  uniform vec3  uRimDir;
  uniform vec3  uRimCol;
  uniform vec3  uHazeCol;
  uniform vec3  uFloorCol;
  uniform vec2  uHazeRange;
  uniform float uHazeAmount;
  uniform float uRim;
  uniform float uSpec;
  uniform float uClimbFall;
  uniform float uCap;
  uniform float uAnimate;

  varying vec3 vCol;

  const vec3 HIP   = vec3( 0.0, ${f(HIP_Y)}, ${f(HIP_Z)} );
  const vec3 KNEE  = vec3( 0.0, ${f(KNEE_Y)}, ${f(KNEE_Z)} );
  const vec3 NECKP = vec3( 0.0, ${f(NECK_Y)}, ${f(NECK_Z)} );

  // No sin() — this runs on every vertex of every spectator and the transcendental
  // version of the same hash costs more than the whole pose solve.
  float rnd( float n ) {
    n = fract( n * 0.1031 );
    n *= n + 33.33;
    n *= n + n;
    return fract( n );
  }

  vec3 rotX( vec3 p, float c, float s ) { return vec3( p.x, p.y * c - p.z * s, p.y * s + p.z * c ); }
  vec3 rotY( vec3 p, float c, float s ) { return vec3( p.x * c + p.z * s, p.y, -p.x * s + p.z * c ); }
  vec3 rotZ( vec3 p, float c, float s ) { return vec3( p.x * c - p.y * s, p.x * s + p.y * c, p.z ); }

  // Clothing. A real bowl is overwhelmingly dark — coats, hoodies, charcoal —
  // with a minority in team colour and a thin scattering of light shirts. Get
  // that distribution wrong in either direction and the crowd reads as confetti
  // or as a black wall. Eight clusters, per §6.3, but the *range* is held to
  // about 8:1 so no shirt can climb out of the §1.1 band on its own.
  vec3 clothColour( float k, float v ) {
    vec3 c;
    if ( k < 0.40 ) {
      // Near-black outerwear, faintly cool.
      float g = 0.0155 + v * 0.0300;
      c = vec3( g * 0.92, g * 0.96, g * 1.20 );
    } else if ( k < 0.62 ) {
      // Charcoal / denim / olive.
      float g = 0.0460 + v * 0.0450;
      c = mix( vec3( g * 0.88, g * 0.97, g * 1.32 ), vec3( g * 1.16, g * 1.0, g * 0.72 ), step( 0.5, v ) );
    } else if ( k < 0.745 ) {
      c = vec3( 0.020, 0.048, 0.148 ) * ( 0.55 + v * 0.92 );      // home blue
    } else if ( k < 0.830 ) {
      c = vec3( 0.170, 0.038, 0.023 ) * ( 0.55 + v * 0.92 );      // away red
    } else if ( k < 0.878 ) {
      c = vec3( 0.185, 0.115, 0.024 ) * ( 0.55 + v * 0.85 );      // gold
    } else if ( k < 0.906 ) {
      c = vec3( 0.030 + v * 0.050, 0.078 + v * 0.060, 0.052 + v * 0.040 ); // green
    } else if ( k < 0.930 ) {
      c = vec3( 0.140, 0.108, 0.070 ) * ( 0.62 + v * 0.70 );      // tan / camel
    } else {
      float g = 0.095 + v * 0.080;                                // light shirts
      c = vec3( g, g * 0.985, g * 0.95 );
    }
    return c;
  }

  void main() {
    float part = aPart;
    bool isSeat = part < 0.5;

    // --- per-person identity ------------------------------------------------
    float id    = iSeed * 61.0 + aSeat * 1.37;
    float kOcc  = rnd( id + 1.0 );
    float kSize = rnd( id + 2.0 );
    float kYaw  = rnd( id + 3.0 );
    float kLat  = rnd( id + 4.0 );
    float kPh   = rnd( id + 5.0 );
    float kFreq = rnd( id + 6.0 );
    float kCol  = rnd( id + 7.0 );
    float kVal  = rnd( id + 8.0 );
    float kSkin = rnd( id + 9.0 );
    float kHair = rnd( id + 10.0 );
    float kStand= rnd( id + 11.0 );
    float kPhone= rnd( id + 12.0 );
    float kArms = rnd( id + 13.0 );
    float kPose = rnd( id + 14.0 );
    float kFine = rnd( id + 15.0 );
    float kNod  = rnd( id + 16.0 );
    float kPerm = rnd( id + 17.0 );
    float kLean = rnd( id + 18.0 );

    float baseX = ( aSeat - ( uPodSize - 1.0 ) * 0.5 ) * uPitch;

    vec3 p = position;
    vec3 n = normal;
    float rel = 0.0;
    float stand = 0.0;
    bool isPhone = part > 5.5 && part < 6.5;
    bool isFore  = ( part > 7.5 && part < 8.5 ) || isPhone;
    bool isUpper = part > 4.5 && part < 5.5;
    bool isArm   = isFore || isUpper;
    bool isLeg   = ( part > 0.5 && part < 1.5 ) || ( part > 6.5 && part < 7.5 );
    bool isHead  = part > 2.5 && part < 4.5;

    if ( !isSeat ) {
      float present  = step( kOcc, uOccupancy );
      float hasPhone = step( kPhone, uPhoneRate );
      float keep = present * ( isPhone ? hasPhone : 1.0 );

      vec3 q = vec3( p.x - baseX, p.y, p.z );

      // --- hair volume, before anything moves ------------------------------
      if ( part > 3.5 && part < 4.5 ) {
        float hv = kHair < 0.10 ? 0.35 : ( 0.72 + kHair * 0.95 );
        q.y = ${f(HAIR_BASE)} + ( q.y - ${f(HAIR_BASE)} ) * hv;
      }

      // --- who is on their feet --------------------------------------------
      // A share of the bowl is standing at any moment and that share is much
      // higher near the floor. On top of that, a reaction sweeps people up —
      // each with their own trigger level, so a section rises raggedly instead
      // of flipping as one.
      float standRate = uStandRate * ( 0.30 + 0.70 * clamp( ( iTone - 1.0 ) * 2.4, 0.0, 1.0 ) );
      float perm  = step( kPerm, standRate );
      float react = smoothstep( kStand * 0.80, kStand * 0.80 + 0.30, iReact );
      stand = max( perm, react ) * present;
      float cheer = react * step( kArms, 0.46 ) * present;

      // --- pose -------------------------------------------------------------
      float crossed = smoothstep( 0.30, 0.36, kPose ) * ( 1.0 - smoothstep( 0.50, 0.56, kPose ) );
      float leanF   = smoothstep( 0.56, 0.61, kPose ) * ( 1.0 - smoothstep( 0.72, 0.77, kPose ) );
      float wide    = smoothstep( 0.86, 0.92, kPose );

      float foreLift = ( kFine - 0.5 ) * 0.24
                     + crossed * 1.95
                     + leanF * 0.60
                     + hasPhone * 1.50
                     - wide * 0.30;
      foreLift = mix( foreLift, -0.80, cheer );

      float foreIn = crossed * 0.93 + hasPhone * 0.55;
      foreIn = mix( foreIn, 0.0, cheer );

      float armSwing = -0.04 - crossed * 0.16 + leanF * 0.30 - wide * 0.20 + hasPhone * 0.10;
      float armRaise = cheer * ( 1.55 + kArms * 2.4 ) * smoothstep( 0.18, 0.55, iReact );
      float lean     = 0.05 + ( kLean - 0.5 ) * 0.12 + leanF * 0.30 + stand * 0.06;
      float nod      = -0.02 + hasPhone * 0.28 + leanF * 0.14 + ( kNod - 0.5 ) * 0.16;
      float headYaw  = ( kNod - 0.5 ) * 0.50 + sin( uTime * ( 0.31 + kFreq * 0.4 ) + kPh * 17.0 ) * 0.11 * uAnimate;

      // --- pose solve -------------------------------------------------------
      if ( isLeg ) {
        float kneeA = ${f(KNEE_STAND)} * stand;
        if ( part > 6.5 ) {
          float c = cos( kneeA ), s = sin( kneeA );
          q = KNEE + rotX( q - KNEE, c, s );
          n = rotX( n, c, s );
        }
        float hipA = ${f(HIP_STAND)} * stand;
        float c = cos( hipA ), s = sin( hipA );
        q = HIP + rotX( q - HIP, c, s );
        n = rotX( n, c, s );
        // Blocks without a modelled shin have to let the thigh become the
        // whole leg, or a standing spectator is a torso on two stumps.
        q.y = HIP.y + ( q.y - HIP.y ) * mix( 1.0, 1.95, stand * ( 1.0 - uHasShin ) );
      } else {
        if ( isFore ) {
          vec3 elb = vec3( aSide * ${f(ELB_X)}, ${f(ELB_Y)}, ${f(ELB_Z)} );
          float c = cos( foreLift ), s = sin( foreLift );
          q = elb + rotX( q - elb, c, s );
          n = rotX( n, c, s );
          float a = aSide * foreIn;
          float c2 = cos( a ), s2 = sin( a );
          q = elb + rotZ( q - elb, c2, s2 );
          n = rotZ( n, c2, s2 );
        }
        if ( isArm ) {
          vec3 sho = vec3( aSide * ${f(SHO_X)}, ${f(SHO_Y)}, ${f(SHO_Z)} );
          float c = cos( armSwing ), s = sin( armSwing );
          q = sho + rotX( q - sho, c, s );
          n = rotX( n, c, s );
          float a = aSide * armRaise;
          float c2 = cos( a ), s2 = sin( a );
          q = sho + rotZ( q - sho, c2, s2 );
          n = rotZ( n, c2, s2 );
        }
        if ( isHead ) {
          float c = cos( -nod ), s = sin( -nod );
          q = NECKP + rotX( q - NECKP, c, s );
          n = rotX( n, c, s );
          float c2 = cos( headYaw ), s2 = sin( headYaw );
          q = NECKP + rotY( q - NECKP, c2, s2 );
          n = rotY( n, c2, s2 );
        }
        // Everything above the hip pitches forward together.
        float c = cos( -lean ), s = sin( -lean );
        q = HIP + rotX( q - HIP, c, s );
        n = rotX( n, c, s );
      }

      // Stand up and step into the gap in front of the seat.
      q.y += uStandRise * stand;
      q.z -= ${f(STAND_STEP)} * stand * ( uStandRise * ${f(1 / STAND_RISE)} );

      // Size: height and girth vary independently, so the row is not a wave of
      // one silhouette scaled up and down.
      float hScale = 0.88 + kSize * 0.26;
      float wScale = 0.90 + kVal * 0.22;
      q.y *= hScale;
      q.x *= wScale;
      q.z *= mix( 0.94, 1.10, kSize );

      // Yaw, so neighbours do not present identical faces to the camera.
      float yaw = ( kYaw - 0.5 ) * 0.72 + sin( uTime * ( 0.19 + kFreq * 0.2 ) + kPh * 12.0 ) * 0.10 * uAnimate;
      float cy = cos( yaw ), sy = sin( yaw );
      vec3 r = rotY( q, cy, sy );
      n = rotY( n, cy, sy );

      // Idle life: a slow lean plus a small bob, every one at its own phase and
      // rate. Amplitude scales with height off the seat so the lap stays put.
      float ph = kPh * 6.2831853;
      float fr = 0.62 + kFreq * 0.62;
      float sway = sin( uTime * fr + ph ) * 0.026 * uAnimate;
      float bob  = sin( uTime * fr * 1.7 + ph * 1.9 ) * 0.010 * uAnimate;
      // Standing people bounce noticeably harder.
      sway *= 1.0 + stand * 2.6;
      bob  *= 1.0 + stand * 3.4;
      float lever = max( r.y - 0.45, 0.0 );
      r.x += sway * lever;
      r.z += sway * 0.45 * lever;
      r.y += bob * min( lever * 2.0, 1.0 );

      // Break the lattice. Rows and columns of evenly spaced figures is a named
      // tell (§10.30), and a jitter of a fifth of the seat pitch is enough to
      // destroy the column without anyone sitting in a neighbour's lap.
      r.x += ( kLat - 0.5 ) * uPitch * 0.52;
      r.z += ( kSize - 0.5 ) * 0.11;

      p = vec3( baseX + r.x, r.y, r.z ) * keep;
      p += vec3( baseX, 0.42, 0.24 ) * ( 1.0 - keep );
      rel = r.y;
    }

    vec3 albedo;
    float emissive = 0.0;
    float sheen = 0.25;

    if ( isSeat ) {
      // Seat shells carry their own scatter so a bank of empties is not a
      // single flat value.
      albedo = uSeatCol * ( 0.72 + rnd( id + 21.0 ) * 0.52 );
      sheen = 0.55;
    } else if ( part > 2.5 && part < 3.5 ) {
      // Skin. Range of tones, and the SSS-ish warmth is baked into the hue.
      float tone = 0.26 + kSkin * kSkin * 0.82;
      albedo = vec3( 0.215, 0.126, 0.088 ) * tone + vec3( 0.010, 0.005, 0.004 );
      sheen = 0.45;
    } else if ( part > 8.5 ) {
      // Neck: skin, but it sits in the shadow of the jaw.
      float tone = 0.26 + kSkin * kSkin * 0.82;
      albedo = vec3( 0.215, 0.126, 0.088 ) * tone * 0.72;
    } else if ( part > 3.5 && part < 4.5 ) {
      if ( kHair > 0.90 ) {
        albedo = clothColour( rnd( id + 22.0 ), kVal ) * 0.8;      // a cap
      } else {
        float g = kHair < 0.70 ? 0.010 + kHair * 0.022 : ( kHair < 0.855 ? 0.046 : 0.135 );
        albedo = vec3( g * 1.14, g * 0.95, g * 0.84 );
        sheen = 0.40;
      }
    } else if ( isPhone ) {
      albedo = vec3( 0.40, 0.47, 0.66 );
      // Sparse on purpose: a bowl where every phone is lit is a string of
      // fairy lights, not an arena.
      emissive = uPhoneGain * step( rnd( id + 31.0 ), uPhoneLit )
               * ( 0.80 + 0.20 * sin( uTime * 3.1 + kPh * 20.0 ) );
    } else if ( isArm ) {
      // Sleeved for most, bare skin for the rest.
      if ( rnd( id + 23.0 ) < 0.72 ) {
        albedo = clothColour( kCol, kVal ) * 0.92;
      } else {
        float tone = 0.26 + kSkin * kSkin * 0.82;
        albedo = vec3( 0.215, 0.126, 0.088 ) * tone * 0.88;
      }
    } else {
      albedo = clothColour( kCol, kVal );
      sheen = 0.20 + step( 0.62, kCol ) * 0.5;
      // Trousers read darker than tops almost universally.
      if ( isLeg ) albedo *= 0.52;
    }

    // --- transform ----------------------------------------------------------
    vec4 world = modelMatrix * instanceMatrix * vec4( p, 1.0 );
    mat3 im = mat3( instanceMatrix );
    vec3 N = normalize( mat3( modelMatrix ) * ( im * n ) );

    // --- lighting -----------------------------------------------------------
    // The bowl is not lit by the court banks; it catches their spill, the
    // house lights and a warm kick off the hardwood. Deliberately shaped so
    // shoulders and the tops of heads separate and laps fall away.
    float ndK = max( dot( N, uKeyDir ), 0.0 );
    float ndF = max( dot( N, uFillDir ), 0.0 );
    float hemi = 0.5 + 0.5 * N.y;
    // Self-occlusion down the body: rows shadow each other and a seated torso
    // shadows its own lap.
    float ao = mix( 0.20, 1.0, smoothstep( 0.08, 1.05, rel ) );
    if ( isSeat ) ao = 0.42;

    vec3 lit = albedo * (
        uKeyCol * ndK
      + uFillCol * ndF
      + uAmbTop * hemi
      + uAmbBot * ( 1.0 - hemi )
    ) * ao;

    // Courtside LED / hardwood spill: strongest on the lowest rows, which is
    // what iTone encodes.
    lit += albedo * uSpillCol * max( iTone - 1.0, 0.0 ) * ( 0.30 + 0.70 * ndF );

    vec3 V = normalize( cameraPosition - world.xyz );

    // Nothing in frame has zero specular (§1.4) — a broad, low lobe, stronger
    // on the shell jackets than on knitwear.
    vec3 H = normalize( uKeyDir + V );
    lit += uKeyCol * pow( max( dot( N, H ), 0.0 ), 16.0 ) * uSpec * sheen * ao;

    // Rim. Directional, not a Fresnel halo (§10.4): it only fires where the
    // surface turns up and away from the court, so it lands on the tops of
    // shoulders and skulls and cuts each row off the one behind it.
    float fres = pow( 1.0 - clamp( dot( N, V ), 0.0, 1.0 ), 3.2 );
    float back = smoothstep( -0.05, 0.75, dot( N, uRimDir ) );
    lit += uRimCol * ( fres * back * uRim ) * ( 0.30 + albedo * 2.0 );

    // The bowl's light comes off the floor and the lower fixtures, so it falls
    // away as the rake climbs. Front row to the top of the upper deck is about
    // a stop and a half, which is what makes the bowl read as a volume rather
    // than as a wall of evenly lit people.
    // The bowl's whole tonal range has to fit inside §1.1's 18–45 band, which
    // is 1.3 stops wide, so the front-row-to-upper-deck gradient is deliberately
    // gentle: present, readable, and about half a stop end to end.
    float climb = 1.0 / ( 1.0 + max( world.y - 0.9, 0.0 ) * uClimbFall );
    lit *= uExposure * mix( 1.0, iTone, 0.72 ) * climb;

    // §6.3, stated flatly: the crowd must never be brighter than the near
    // hardwood. A soft knee rather than a clamp, so a white shirt catching the
    // spill still reads as a white shirt instead of turning into a grey card.
    float lum = dot( lit, vec3( 0.2126, 0.7152, 0.0722 ) );
    if ( lum > uCap ) {
      float over = lum - uCap;
      lit *= ( uCap + over / ( 1.0 + over / uCap ) ) / max( lum, 1e-5 );
    }

    // Phone screens are the one thing in the bowl allowed above that ceiling.
    lit += albedo * emissive;

    // Depth cueing — far stands lose contrast, which is most of what makes a
    // big room read as big.
    float d = length( cameraPosition - world.xyz );
    float haze = smoothstep( uHazeRange.x, uHazeRange.y, d ) * uHazeAmount;
    // Floor the bowl off the black point: crushed regions with no detail are a
    // named tell, and a real building always has some ambient spill.
    vCol = max( mix( lit, uHazeCol, haze ), uFloorCol );

    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const CROWD_FRAG = /* glsl */ `
  #include <common>
  varying vec3 vCol;
  void main() {
    gl_FragColor = vec4( max( vCol, vec3( 0.0 ) ), 1.0 );
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`;

export interface CrowdUniformSet {
  uTime: { value: number };
  uPodSize: { value: number };
  uPitch: { value: number };
  uOccupancy: { value: number };
  uPhoneRate: { value: number };
  uPhoneLit: { value: number };
  uExposure: { value: number };
  uPhoneGain: { value: number };
  uStandRate: { value: number };
  uHasShin: { value: number };
  uStandRise: { value: number };
  uSeatCol: { value: Color };
  uKeyDir: { value: Vector3 };
  uKeyCol: { value: Color };
  uFillDir: { value: Vector3 };
  uFillCol: { value: Color };
  uAmbTop: { value: Color };
  uAmbBot: { value: Color };
  uSpillCol: { value: Color };
  uRimDir: { value: Vector3 };
  uRimCol: { value: Color };
  uHazeCol: { value: Color };
  uFloorCol: { value: Color };
  uHazeRange: { value: Vector2 };
  uHazeAmount: { value: number };
  uRim: { value: number };
  uSpec: { value: number };
  uClimbFall: { value: number };
  uCap: { value: number };
  uAnimate: { value: number };
}

function makeCrowdMaterial(spec: CrowdBlockSpec, animate: boolean): ShaderMaterial {
  const mat = new ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uPodSize: { value: spec.podSize },
      uPitch: { value: spec.pitch },
      uOccupancy: { value: spec.occupancy },
      uPhoneRate: { value: spec.phoneRate },
      // Of the people holding a phone, only this share have the screen awake.
      uPhoneLit: { value: 0.20 },
      // §1.1 is the master criterion: with the hardwood at 95–140 the bowl has
      // to land at 18–45, i.e. 2.5–4 stops down. This number is that ratio.
      uExposure: { value: 4.6 },
      uPhoneGain: { value: 1.2 },
      uStandRate: { value: 0.16 },
      uHasShin: { value: spec.detail >= 3 ? 1 : 0 },
      // Blocks with no leg geometry at all cannot stand all the way up without
      // leaving a floating torso, so up there a reaction is a rise in the seat.
      uStandRise: { value: spec.detail >= 1 ? STAND_RISE : 0.12 },
      uSeatCol: { value: new Color(spec.seatColor) },
      uKeyDir: { value: new Vector3(0.36, 0.86, 0.36).normalize() },
      uKeyCol: { value: new Color(0.30, 0.325, 0.40) },
      uFillDir: { value: new Vector3(-0.5, 0.42, -0.75).normalize() },
      uFillCol: { value: new Color(0.13, 0.128, 0.155) },
      uAmbTop: { value: new Color(0.088, 0.102, 0.146) },
      uAmbBot: { value: new Color(0.072, 0.058, 0.044) },
      uSpillCol: { value: new Color(0.44, 0.37, 0.30) },
      uRimDir: { value: new Vector3(0, 0.86, 0.51).normalize() },
      uRimCol: { value: new Color(0.52, 0.60, 0.86) },
      uHazeCol: { value: new Color(0.0205, 0.0250, 0.0350) },
      uFloorCol: { value: new Color(0.0112, 0.0126, 0.0168) },
      uHazeRange: { value: new Vector2(22, 74) },
      uHazeAmount: { value: 0.55 },
      uRim: { value: 0.055 },
      uSpec: { value: 0.10 },
      uClimbFall: { value: 0.028 },
      uCap: { value: 0.185 },
      uAnimate: { value: animate ? 1 : 0 },
    },
    vertexShader: CROWD_VERT,
    fragmentShader: CROWD_FRAG,
  });
  mat.name = 'arena.crowd';
  return mat;
}

// -----------------------------------------------------------------------------
// System-facing crowd
// -----------------------------------------------------------------------------

interface Block {
  mesh: InstancedMesh;
  react: InstancedBufferAttribute;
  /** Pod world XZ, for the reaction wave. */
  px: Float32Array;
  pz: Float32Array;
  /** Current excitement, 0–1. */
  level: Float32Array;
  /** Scheduled wave arrival time and amplitude. */
  waveAt: Float32Array;
  waveAmp: Float32Array;
}

export class Crowd {
  readonly blocks: Block[] = [];
  readonly materials: ShaderMaterial[] = [];

  private time = 0;
  private baseline = 0.055;
  private _dummy = new Object3D();

  /** Total spectators actually modelled, for the perf report. */
  people = 0;
  seats = 0;

  add(spec: CrowdBlockSpec, animated: boolean): InstancedMesh {
    const count = spec.placements.length;
    if (count === 0) throw new Error('crowd block with no placements');

    const geo = buildPod(spec.podSize, spec.pitch, spec.detail).build(`crowd.${spec.name}`);
    const mat = makeCrowdMaterial(spec, animated);
    this.materials.push(mat);

    const mesh = new InstancedMesh(geo, mat, count);
    mesh.name = `crowd.${spec.name}`;
    mesh.frustumCulled = true;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    mesh.instanceMatrix.setUsage(StaticDrawUsage);

    const seed = new Float32Array(count);
    const react = new Float32Array(count);
    const tone = new Float32Array(count);
    const px = new Float32Array(count);
    const pz = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      const p = spec.placements[i];
      this._dummy.position.set(p.x, p.y, p.z);
      this._dummy.rotation.set(0, p.rotY, 0);
      this._dummy.scale.setScalar(1);
      this._dummy.updateMatrix();
      mesh.setMatrixAt(i, this._dummy.matrix);
      seed[i] = ((i * 0.6180339887 + 0.1237) % 1) * 0.97 + 0.013;
      tone[i] = p.tone;
      px[i] = p.x;
      pz[i] = p.z;
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.computeBoundingSphere();

    geo.setAttribute('iSeed', new InstancedBufferAttribute(seed, 1));
    const reactAttr = new InstancedBufferAttribute(react, 1);
    reactAttr.setUsage(DynamicDrawUsage);
    geo.setAttribute('iReact', reactAttr);
    geo.setAttribute('iTone', new InstancedBufferAttribute(tone, 1));

    this.blocks.push({
      mesh,
      react: reactAttr,
      px,
      pz,
      level: new Float32Array(count),
      waveAt: new Float32Array(count).fill(-1),
      waveAmp: new Float32Array(count),
    });

    this.people += count * spec.podSize * spec.occupancy;
    this.seats += count * spec.podSize;
    return mesh;
  }

  /**
   * Fire a reaction. The wave leaves the epicentre at `speed` m/s so the
   * section under the play comes up first and the far corner follows about a
   * second later, which is exactly how a real building sounds and looks.
   */
  excite(epicentre: Vector3, amplitude: number, speed = 26, spread = 1): void {
    for (const b of this.blocks) {
      for (let i = 0; i < b.px.length; i++) {
        const dx = b.px[i] - epicentre.x;
        const dz = b.pz[i] - epicentre.z;
        const d = Math.hypot(dx, dz);
        const falloff = 1 / (1 + Math.pow(d / (26 * spread), 2.1));
        const amp = amplitude * (0.32 + 0.68 * falloff);
        const at = this.time + d / speed;
        // A stronger, earlier wave wins.
        if (b.waveAt[i] < 0 || amp > b.waveAmp[i] * 0.9) {
          b.waveAt[i] = at;
          b.waveAmp[i] = amp;
        }
      }
    }
  }

  /** Slow ambient excitement, e.g. a tight game late. */
  setBaseline(v: number): void {
    this.baseline = v;
  }

  update(dt: number, elapsed: number): void {
    this.time = elapsed;
    for (const m of this.materials) m.uniforms.uTime.value = elapsed;

    const attack = 1 - Math.exp(-9 * dt);
    const release = 1 - Math.exp(-1.3 * dt);

    for (const b of this.blocks) {
      const n = b.level.length;
      const arr = b.react.array as Float32Array;
      let dirty = false;
      for (let i = 0; i < n; i++) {
        let target = this.baseline;
        const at = b.waveAt[i];
        if (at >= 0) {
          const age = elapsed - at;
          if (age >= 0) {
            // Fast rise, ~1.6 s hold, slow settle back into the seats.
            const env =
              age < 0.22 ? age / 0.22 : Math.exp(-Math.max(0, age - 0.22) * 0.62);
            target = Math.max(target, b.waveAmp[i] * env);
            if (age > 6) b.waveAt[i] = -1;
          }
        }
        const cur = b.level[i];
        const next = cur + (target - cur) * (target > cur ? attack : release);
        if (Math.abs(next - arr[i]) > 0.002) dirty = true;
        b.level[i] = next;
        arr[i] = next;
      }
      if (dirty) b.react.needsUpdate = true;
    }
  }

  /** Bulk uniform poke — used to tune bowl exposure from one place. */
  setUniform(name: string, value: unknown): void {
    for (const m of this.materials) {
      const u = m.uniforms[name];
      if (u) u.value = value as never;
    }
  }

  dispose(): void {
    for (const b of this.blocks) {
      b.mesh.geometry.dispose();
      b.mesh.dispose();
    }
    for (const m of this.materials) m.dispose();
    this.blocks.length = 0;
    this.materials.length = 0;
  }
}
