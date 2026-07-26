/**
 * Procedural bodies: one continuous skinned surface per athlete.
 *
 * The thing that instantly identifies a procedurally generated character as a
 * toy is a body assembled from disconnected capsules — you can see where the
 * upper arm ends and the forearm begins, the shoulder is a ball intersecting a
 * tube, and the armpit is a hole. So nothing here is a separate limb. The whole
 * figure is a single implicit surface: every anatomical part contributes a
 * signed distance field with an *elliptical, profiled* cross-section, they are
 * combined with a smooth union whose blend radius is tuned per joint, and the
 * result is polygonised in one pass. What comes out is watertight, continuous,
 * and carries real deltoid / pectoral / lat / quad / calf landmarks on its
 * outline — because those landmarks are geometry, not a texture.
 *
 * Pipeline:
 *
 *  1. **Primitives** from the rest-pose skeleton. Cross-sections are
 *     superellipses with independent medial-lateral and anterior-posterior
 *     radii swept along each bone by a profile function — a thigh is wider than
 *     deep, a forearm turns from an oval at the elbow into a flattened wrist, a
 *     torso tapers ribcage → waist and flares at the shoulders.
 *  2. **Rasterise** into a voxel grid, each primitive touching only the cells
 *     inside its own bounds, so cost scales with surface area rather than with
 *     (primitives × grid).
 *  3. **Surface-net** the grid, then Newton-project every vertex onto the exact
 *     isosurface and take its normal from the analytic gradient. That removes
 *     the blobbiness naive surface nets are known for and needs no smoothing.
 *  4. **Parameterise** each vertex against the bone that owns it — distance
 *     along the bone, angle around it — which is both the skin-atlas UV and the
 *     basis for the anatomical vertex colouring and the field-derived AO.
 *  5. **Skin** by *relative* distance to the bone segments using the falloff
 *     width `Skeleton.ts` declares per joint, so an elbow or knee past 90°
 *     rolls instead of pinching.
 *
 * Garments are never painted onto this surface. Jersey, shorts, shoes and hair
 * are separate lofted shells built off the same skeleton, offset with their own
 * drape, and skinned the same way.
 *
 * Owned by the players agent.
 */

import {
  BufferGeometry,
  Float32BufferAttribute,
  Sphere,
  Uint16BufferAttribute,
  Vector3,
} from 'three';
import {
  BONES,
  BONE_INDEX,
  type BodyShape,
  type BoneName,
  type BuiltSkeleton,
} from './Skeleton';
import { clamp01, fbm2, lerp, makeRng } from '../core/MathX';
import { skinPartUv, type SkinPart } from '../textures/skinTextures';
import { KIT_UV } from '../textures/jerseyTextures';

// ---------------------------------------------------------------------------
// Signed distance primitives
// ---------------------------------------------------------------------------

type PrimGroup = 'torso' | 'arm' | 'leg' | 'head';

interface Prim {
  minX: number;
  minY: number;
  minZ: number;
  maxX: number;
  maxY: number;
  maxZ: number;
  /** Smooth-union blend radius against everything already accumulated. */
  k: number;
  group: PrimGroup;
  dist(x: number, y: number, z: number): number;
}

/** Quadratic polynomial smooth minimum. */
function smin(a: number, b: number, k: number): number {
  if (k <= 1e-6) return a < b ? a : b;
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return (a < b ? a : b) - h * h * k * 0.25;
}

type Profile = (t: number) => number;

/** Smooth-interpolated radius profile from control points [t, radius]. */
function prof(points: ReadonlyArray<readonly [number, number]>): Profile {
  return (t: number): number => {
    if (t <= points[0][0]) return points[0][1];
    for (let i = 1; i < points.length; i++) {
      if (t <= points[i][0]) {
        const a = points[i - 1];
        const b = points[i];
        const f = (t - a[0]) / Math.max(1e-6, b[0] - a[0]);
        const s = f * f * (3 - 2 * f);
        return a[1] + (b[1] - a[1]) * s;
      }
    }
    return points[points.length - 1][1];
  };
}

/**
 * A generalised elliptical capsule: a superellipse cross-section with separate
 * lateral and forward radii, swept along a bone segment.
 */
function limbPrim(
  a: Vector3,
  b: Vector3,
  side: Vector3,
  fwd: Vector3,
  rSide: Profile,
  rFwd: Profile,
  exponent: number,
  k: number,
  group: PrimGroup,
): Prim {
  const dxL = b.x - a.x;
  const dyL = b.y - a.y;
  const dzL = b.z - a.z;
  const L = Math.sqrt(dxL * dxL + dyL * dyL + dzL * dzL) || 1e-4;
  const wx = dxL / L;
  const wy = dyL / L;
  const wz = dzL / L;
  const sx = side.x;
  const sy = side.y;
  const sz = side.z;
  const fx = fwd.x;
  const fy = fwd.y;
  const fz = fwd.z;
  const ax0 = a.x;
  const ay0 = a.y;
  const az0 = a.z;
  const invE = 1 / exponent;

  let rmax = 0;
  for (let i = 0; i <= 12; i++) rmax = Math.max(rmax, rSide(i / 12), rFwd(i / 12));
  rmax += k;

  return {
    minX: Math.min(a.x, b.x) - rmax,
    minY: Math.min(a.y, b.y) - rmax,
    minZ: Math.min(a.z, b.z) - rmax,
    maxX: Math.max(a.x, b.x) + rmax,
    maxY: Math.max(a.y, b.y) + rmax,
    maxZ: Math.max(a.z, b.z) + rmax,
    k,
    group,
    dist(px: number, py: number, pz: number): number {
      const dx = px - ax0;
      const dy = py - ay0;
      const dz = pz - az0;
      const along = dx * wx + dy * wy + dz * wz;
      const tRaw = along / L;
      const t = tRaw < 0 ? 0 : tRaw > 1 ? 1 : tRaw;
      const axial = tRaw < 0 ? -along : tRaw > 1 ? along - L : 0;
      const footAlong = t * L;
      const px2 = dx - wx * footAlong;
      const py2 = dy - wy * footAlong;
      const pz2 = dz - wz * footAlong;
      const u = px2 * sx + py2 * sy + pz2 * sz;
      const v = px2 * fx + py2 * fy + pz2 * fz;
      const rr = Math.sqrt(u * u + v * v);
      const rs = rSide(t);
      const rf = rFwd(t);
      let rEff: number;
      if (rr < 1e-6) {
        rEff = rs < rf ? rs : rf;
      } else {
        const q = Math.pow(Math.abs(u) / rs, exponent) + Math.pow(Math.abs(v) / rf, exponent);
        rEff = rr / Math.pow(q, invE);
      }
      return Math.sqrt(rr * rr + axial * axial) - rEff;
    },
  };
}

/** Ellipsoid in an arbitrary basis; the workhorse for muscle bellies. */
function ballPrim(
  c: Vector3,
  rx: number,
  ry: number,
  rz: number,
  k: number,
  basis: { x: Vector3; y: Vector3; z: Vector3 },
  group: PrimGroup,
): Prim {
  const cx = c.x;
  const cy = c.y;
  const cz = c.z;
  const bx = basis.x;
  const by = basis.y;
  const bz = basis.z;
  const rmax = Math.max(rx, ry, rz) + k;
  const rmin = Math.min(rx, ry, rz);
  return {
    minX: cx - rmax,
    minY: cy - rmax,
    minZ: cz - rmax,
    maxX: cx + rmax,
    maxY: cy + rmax,
    maxZ: cz + rmax,
    k,
    group,
    dist(px: number, py: number, pz: number): number {
      const dx = px - cx;
      const dy = py - cy;
      const dz = pz - cz;
      const u = (dx * bx.x + dy * bx.y + dz * bx.z) / rx;
      const v = (dx * by.x + dy * by.y + dz * by.z) / ry;
      const w = (dx * bz.x + dy * bz.y + dz * bz.z) / rz;
      return (Math.sqrt(u * u + v * v + w * w) - 1) * rmin;
    },
  };
}

/** Rounded box; a hand needs a palm plane and a foot needs a sole. */
function slabPrim(
  c: Vector3,
  hx: number,
  hy: number,
  hz: number,
  round: number,
  k: number,
  basis: { x: Vector3; y: Vector3; z: Vector3 },
  group: PrimGroup,
): Prim {
  const rmax = Math.max(hx, hy, hz) + round + k;
  return {
    minX: c.x - rmax,
    minY: c.y - rmax,
    minZ: c.z - rmax,
    maxX: c.x + rmax,
    maxY: c.y + rmax,
    maxZ: c.z + rmax,
    k,
    group,
    dist(px: number, py: number, pz: number): number {
      const dx = px - c.x;
      const dy = py - c.y;
      const dz = pz - c.z;
      const u = Math.abs(dx * basis.x.x + dy * basis.x.y + dz * basis.x.z) - hx;
      const v = Math.abs(dx * basis.y.x + dy * basis.y.y + dz * basis.y.z) - hy;
      const w = Math.abs(dx * basis.z.x + dy * basis.z.y + dz * basis.z.z) - hz;
      const ox = Math.max(u, 0);
      const oy = Math.max(v, 0);
      const oz = Math.max(w, 0);
      return (
        Math.sqrt(ox * ox + oy * oy + oz * oz) + Math.min(Math.max(u, Math.max(v, w)), 0) - round
      );
    },
  };
}

/** Smooth union of a primitive list, with an AABB rejection per primitive. */
function evalPrims(prims: readonly Prim[], x: number, y: number, z: number): number {
  let d = 1;
  let first = true;
  for (let i = 0; i < prims.length; i++) {
    const p = prims[i];
    if (x < p.minX || x > p.maxX || y < p.minY || y > p.maxY || z < p.minZ || z > p.maxZ) continue;
    const v = p.dist(x, y, z);
    if (first) {
      d = v;
      first = false;
    } else {
      d = smin(d, v, p.k);
    }
  }
  return first ? 1 : d;
}

// ---------------------------------------------------------------------------
// Skeleton-derived helpers
// ---------------------------------------------------------------------------

/** Per-bone skin-weight falloff width, as a fraction of standing height. */
const SIGMA: Partial<Record<BoneName, number>> = {
  hips: 0.062,
  spine: 0.05,
  chest: 0.05,
  upperChest: 0.05,
  neck: 0.032,
  head: 0.048,
  clavicleL: 0.038,
  clavicleR: 0.038,
  upperArmL: 0.036,
  upperArmR: 0.036,
  foreArmL: 0.03,
  foreArmR: 0.03,
  handL: 0.024,
  handR: 0.024,
  thighL: 0.052,
  thighR: 0.052,
  shinL: 0.04,
  shinR: 0.04,
  footL: 0.03,
  footR: 0.03,
  toeL: 0.024,
  toeR: 0.024,
};

/** Which atlas cell a bone's vertices sample from. */
const PART_OF_BONE: Partial<Record<BoneName, SkinPart>> = {
  hips: 'torso',
  spine: 'torso',
  chest: 'torso',
  upperChest: 'torso',
  clavicleL: 'torso',
  clavicleR: 'torso',
  neck: 'neck',
  head: 'head',
  upperArmL: 'upperArm',
  upperArmR: 'upperArm',
  foreArmL: 'foreArm',
  foreArmR: 'foreArm',
  handL: 'hand',
  handR: 'hand',
  thighL: 'thigh',
  thighR: 'thigh',
  shinL: 'shin',
  shinR: 'shin',
  footL: 'foot',
  footR: 'foot',
  toeL: 'foot',
  toeR: 'foot',
};

interface Segment {
  a: Vector3;
  b: Vector3;
  w: Vector3;
  len: number;
  fwd: Vector3;
  side: Vector3;
  sigma: number;
}

const _ref = new Vector3();

/** A stable frame around an axis: `fwd` is world-forward projected off it. */
function frameFor(w: Vector3, fwd: Vector3, side: Vector3): void {
  const ref = Math.abs(w.z) < 0.85 ? _ref.set(0, 0, 1) : _ref.set(0, 1, 0);
  fwd.copy(ref).addScaledVector(w, -ref.dot(w));
  if (fwd.lengthSq() < 1e-8) fwd.set(1, 0, 0);
  fwd.normalize();
  side.crossVectors(fwd, w).normalize();
}

function makeSegment(a: Vector3, b: Vector3, sigma: number): Segment {
  const w = b.clone().sub(a);
  const len = w.length() || 1e-4;
  w.divideScalar(len);
  const fwd = new Vector3();
  const side = new Vector3();
  frameFor(w, fwd, side);
  return { a: a.clone(), b: b.clone(), w, len, fwd, side, sigma };
}

function buildSegments(sk: BuiltSkeleton): Segment[] {
  const H = sk.height;
  const segs: Segment[] = [];
  for (let i = 0; i < BONES.length; i++) {
    const def = BONES[i];
    const a = sk.restWorld[i];
    let b: Vector3;
    const childIdx = BONES.findIndex((d) => d.parent === def.name);
    if (childIdx >= 0) {
      b = sk.restWorld[childIdx].clone();
    } else {
      // Leaves get a synthetic extension so they own a real volume.
      const parentIdx = def.parent ? BONE_INDEX[def.parent] : -1;
      const dir =
        parentIdx >= 0
          ? a.clone().sub(sk.restWorld[parentIdx]).normalize()
          : new Vector3(0, 1, 0);
      const ext = def.name === 'head' ? 0.115 : def.name.startsWith('hand') ? 0.1 : 0.05;
      b = a.clone().addScaledVector(dir, ext * H);
    }
    if (b.distanceToSquared(a) < 1e-8) b = a.clone().add(new Vector3(0, 0.01, 0));
    segs.push(makeSegment(a, b, (SIGMA[def.name] ?? 0.03) * H));
  }
  return segs;
}

function segDistance(s: Segment, x: number, y: number, z: number): number {
  const dx = x - s.a.x;
  const dy = y - s.a.y;
  const dz = z - s.a.z;
  let t = (dx * s.w.x + dy * s.w.y + dz * s.w.z) / s.len;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const px = dx - s.w.x * t * s.len;
  const py = dy - s.w.y * t * s.len;
  const pz = dz - s.w.z * t * s.len;
  return Math.sqrt(px * px + py * py + pz * pz);
}

const BASIS = { x: new Vector3(1, 0, 0), y: new Vector3(0, 1, 0), z: new Vector3(0, 0, 1) };

// ---------------------------------------------------------------------------
// The athlete, as a field
// ---------------------------------------------------------------------------

/**
 * Radii are quoted in metres for a 1.98 m frame at build 1 and scaled by height
 * and build, so a 2.13 m centre at build 1.16 is a genuinely different body
 * rather than the same mesh scaled up.
 */
function buildPrims(sk: BuiltSkeleton, shape: BodyShape): Prim[] {
  const H = sk.height;
  const S = H / 1.98;
  const B = shape.build;
  const SH = shape.shoulders;
  const P = sk.restWorld;
  const I = BONE_INDEX;
  const prims: Prim[] = [];
  const m = (x: number): number => x * S * B;
  const ms = (x: number): number => x * S * B * SH;

  const seg = (
    ai: number,
    bi: number,
    rs: ReadonlyArray<readonly [number, number]>,
    rf: ReadonlyArray<readonly [number, number]>,
    e: number,
    k: number,
    group: PrimGroup,
    scale: (x: number) => number = m,
  ): void => {
    const a = P[ai];
    const b = P[bi];
    const w = b.clone().sub(a).normalize();
    const fwd = new Vector3();
    const side = new Vector3();
    frameFor(w, fwd, side);
    prims.push(
      limbPrim(
        a,
        b,
        side,
        fwd,
        prof(rs.map((p) => [p[0], scale(p[1])] as const)),
        prof(rf.map((p) => [p[0], scale(p[1])] as const)),
        e,
        k * S,
        group,
      ),
    );
  };

  // --- Torso -------------------------------------------------------------
  seg(I.hips, I.spine, [[0, 0.163], [1, 0.14]], [[0, 0.118], [1, 0.104]], 3.0, 0.05, 'torso');
  seg(I.spine, I.chest, [[0, 0.14], [1, 0.157]], [[0, 0.104], [1, 0.115]], 3.2, 0.05, 'torso');
  seg(
    I.chest,
    I.upperChest,
    [[0, 0.157], [1, 0.182]],
    [[0, 0.115], [1, 0.129]],
    3.4,
    0.05,
    'torso',
    ms,
  );
  // The yoke. This taper from the acromion line into the neck is the
  // trapezius, and it is why a neck must never be a cylinder on a sphere.
  seg(
    I.upperChest,
    I.neck,
    [[0, 0.182], [0.45, 0.148], [1, 0.076]],
    [[0, 0.129], [0.45, 0.112], [1, 0.072]],
    3.0,
    0.055,
    'torso',
    ms,
  );
  seg(I.neck, I.head, [[0, 0.071], [1, 0.06]], [[0, 0.067], [1, 0.058]], 2.4, 0.03, 'torso');

  prims.push(
    ballPrim(
      P[I.hips].clone().add(new Vector3(0, -0.028 * H, -m(0.055))),
      m(0.166),
      m(0.115),
      m(0.1),
      0.05 * S,
      BASIS,
      'torso',
    ),
  );
  for (const sgn of [1, -1]) {
    // Pectoral shelf.
    prims.push(
      ballPrim(
        P[I.upperChest].clone().add(new Vector3(sgn * ms(0.078), -0.018 * H, ms(0.07))),
        ms(0.086),
        m(0.058),
        m(0.05),
        0.035 * S,
        BASIS,
        'torso',
      ),
    );
    // Latissimus: a flat wing running from the armpit down to the waist.
    prims.push(
      ballPrim(
        P[I.chest].clone().add(new Vector3(sgn * ms(0.14), 0.028 * H, -m(0.012))),
        ms(0.05),
        m(0.135),
        m(0.106),
        0.05 * S,
        BASIS,
        'torso',
      ),
    );
    // Trapezius ridge from the neck out to the acromion.
    prims.push(
      ballPrim(
        P[I.neck].clone().add(new Vector3(sgn * ms(0.085), -0.014 * H, -m(0.018))),
        ms(0.112),
        m(0.05),
        m(0.072),
        0.05 * S,
        BASIS,
        'torso',
      ),
    );
    // Deltoid cap: the widest point of the torso, and part of the torso field
    // so the jersey strap can find it.
    const up = sgn > 0 ? I.upperArmL : I.upperArmR;
    prims.push(
      ballPrim(
        P[up].clone().add(new Vector3(-sgn * ms(0.016), 0.012 * H, 0)),
        ms(0.058),
        m(0.082),
        m(0.064),
        0.045 * S,
        BASIS,
        'torso',
      ),
    );
    seg(
      sgn > 0 ? I.clavicleL : I.clavicleR,
      up,
      [[0, 0.05], [1, 0.055]],
      [[0, 0.045], [1, 0.051]],
      2.6,
      0.055,
      'torso',
      ms,
    );
  }

  // --- Head ---------------------------------------------------------------
  const headY = P[I.head].y;
  const cranium = new Vector3(0, lerp(headY, 0.995 * H, 0.46), 0.004 * H);
  prims.push(ballPrim(cranium, 0.079 * H, 0.114 * H, 0.098 * H, 0.03 * S, BASIS, 'head'));
  prims.push(
    ballPrim(
      new Vector3(0, headY + 0.014 * H, 0.026 * H),
      0.063 * H,
      0.043 * H,
      0.071 * H,
      0.035 * S,
      BASIS,
      'head',
    ),
  );
  prims.push(
    ballPrim(
      new Vector3(0, headY + 0.062 * H, 0.074 * H),
      0.047 * H,
      0.015 * H,
      0.022 * H,
      0.022 * S,
      BASIS,
      'head',
    ),
  );
  prims.push(
    ballPrim(
      new Vector3(0, headY + 0.036 * H, 0.084 * H),
      0.013 * H,
      0.022 * H,
      0.026 * H,
      0.012 * S,
      BASIS,
      'head',
    ),
  );
  for (const sgn of [1, -1]) {
    prims.push(
      ballPrim(
        new Vector3(sgn * 0.075 * H, headY + 0.044 * H, -0.004 * H),
        0.012 * H,
        0.03 * H,
        0.021 * H,
        0.012 * S,
        BASIS,
        'head',
      ),
    );
  }

  // --- Arms ---------------------------------------------------------------
  for (const sgn of [1, -1]) {
    const L = sgn > 0;
    const up = L ? I.upperArmL : I.upperArmR;
    const fo = L ? I.foreArmL : I.foreArmR;
    const ha = L ? I.handL : I.handR;

    // Biceps belly a third down, narrowing hard into the elbow.
    seg(
      up,
      fo,
      [[0, 0.052], [0.35, 0.055], [0.72, 0.047], [1, 0.04]],
      [[0, 0.058], [0.33, 0.064], [0.72, 0.051], [1, 0.042]],
      2.3,
      0.03,
      'arm',
    );
    // Forearm: oval at the elbow, swelling at the flexor mass, flattening to a
    // narrow wrist. Independent radii are what make the twist read.
    seg(
      fo,
      ha,
      [[0, 0.044], [0.22, 0.049], [0.72, 0.032], [1, 0.027]],
      [[0, 0.05], [0.22, 0.054], [0.72, 0.03], [1, 0.021]],
      2.4,
      0.028,
      'arm',
    );

    // Hand: palm plane + a mitten of fingers + an opposed thumb. Simplified,
    // but the silhouette and the palm plane have to be right — the hand sits on
    // the ball in almost every frame and the ball is 0.2385 m across.
    const hw = P[ha].clone().sub(P[fo]).normalize();
    const hf = new Vector3();
    const hs = new Vector3();
    frameFor(hw, hf, hs);
    // Palm plane faces medially, as a relaxed arm hangs: the thin axis of the
    // slab is the medial-lateral one and the width axis is anterior-posterior.
    const handBasis = { x: hf, y: hw, z: hs };
    prims.push(
      slabPrim(
        P[ha].clone().addScaledVector(hw, m(0.05)),
        m(0.047),
        m(0.05),
        m(0.017),
        m(0.012),
        0.022 * S,
        handBasis,
        'arm',
      ),
    );
    prims.push(
      slabPrim(
        P[ha].clone().addScaledVector(hw, m(0.132)),
        m(0.045),
        m(0.055),
        m(0.013),
        m(0.011),
        0.02 * S,
        handBasis,
        'arm',
      ),
    );
    prims.push(
      ballPrim(
        P[ha]
          .clone()
          .addScaledVector(hw, m(0.058))
          .addScaledVector(hf, m(0.046))
          .addScaledVector(hs, -sgn * m(0.008)),
        m(0.02),
        m(0.04),
        m(0.017),
        0.02 * S,
        handBasis,
        'arm',
      ),
    );
  }

  // --- Legs ---------------------------------------------------------------
  for (const sgn of [1, -1]) {
    const L = sgn > 0;
    const th = L ? I.thighL : I.thighR;
    const shb = L ? I.shinL : I.shinR;
    const ft = L ? I.footL : I.footR;
    const to = L ? I.toeL : I.toeR;

    seg(
      th,
      shb,
      [[0, 0.09], [0.3, 0.093], [0.78, 0.068], [1, 0.058]],
      [[0, 0.098], [0.3, 0.102], [0.78, 0.07], [1, 0.062]],
      2.6,
      0.04,
      'leg',
    );
    // The calf belly sits HIGH and the achilles is a hard taper; getting that
    // wrong makes a leg read as a sausage.
    seg(
      shb,
      ft,
      [[0, 0.055], [0.22, 0.058], [0.62, 0.041], [0.9, 0.033], [1, 0.032]],
      [[0, 0.058], [0.2, 0.068], [0.62, 0.043], [0.9, 0.031], [1, 0.029]],
      2.4,
      0.032,
      'leg',
    );
    const shw = P[ft].clone().sub(P[shb]).normalize();
    const shf = new Vector3();
    const shs = new Vector3();
    frameFor(shw, shf, shs);
    prims.push(
      ballPrim(
        P[shb]
          .clone()
          .addScaledVector(shw, 0.26 * P[shb].distanceTo(P[ft]))
          .addScaledVector(shf, -m(0.024)),
        m(0.056),
        m(0.076),
        m(0.05),
        0.045 * S,
        { x: shs, y: shw, z: shf },
        'leg',
      ),
    );

    const fw = P[to].clone().sub(P[ft]).normalize();
    const ff = new Vector3();
    const fs = new Vector3();
    frameFor(fw, ff, fs);
    prims.push(
      slabPrim(
        P[ft].clone().addScaledVector(fw, m(0.075)).add(new Vector3(0, -0.012 * H, 0)),
        m(0.042),
        m(0.085),
        m(0.03),
        m(0.018),
        0.03 * S,
        { x: fs, y: fw, z: ff },
        'leg',
      ),
    );
  }

  return prims;
}

// ---------------------------------------------------------------------------
// Polygonisation
// ---------------------------------------------------------------------------

class Field {
  readonly grid: Float32Array;

  constructor(
    readonly prims: Prim[],
    readonly cell: number,
    readonly ox: number,
    readonly oy: number,
    readonly oz: number,
    readonly nx: number,
    readonly ny: number,
    readonly nz: number,
  ) {
    this.grid = new Float32Array(nx * ny * nz).fill(1);
    const { grid } = this;
    for (const p of prims) {
      const i0 = Math.max(0, Math.floor((p.minX - ox) / cell));
      const i1 = Math.min(nx - 1, Math.ceil((p.maxX - ox) / cell));
      const j0 = Math.max(0, Math.floor((p.minY - oy) / cell));
      const j1 = Math.min(ny - 1, Math.ceil((p.maxY - oy) / cell));
      const k0 = Math.max(0, Math.floor((p.minZ - oz) / cell));
      const k1 = Math.min(nz - 1, Math.ceil((p.maxZ - oz) / cell));
      for (let k = k0; k <= k1; k++) {
        const z = oz + k * cell;
        for (let j = j0; j <= j1; j++) {
          const y = oy + j * cell;
          let idx = (k * ny + j) * nx + i0;
          for (let i = i0; i <= i1; i++, idx++) {
            grid[idx] = smin(grid[idx], p.dist(ox + i * cell, y, z), p.k);
          }
        }
      }
    }
  }

  eval(x: number, y: number, z: number): number {
    return evalPrims(this.prims, x, y, z);
  }

  /** Trilinear sample of the rasterised grid — used for ambient occlusion. */
  sampleGrid(x: number, y: number, z: number): number {
    const { nx, ny, nz, cell, ox, oy, oz, grid } = this;
    const fx = (x - ox) / cell;
    const fy = (y - oy) / cell;
    const fz = (z - oz) / cell;
    if (fx < 0 || fy < 0 || fz < 0 || fx > nx - 1 || fy > ny - 1 || fz > nz - 1) return 1;
    const i = Math.floor(fx);
    const j = Math.floor(fy);
    const k = Math.floor(fz);
    const tx = fx - i;
    const ty = fy - j;
    const tz = fz - k;
    const i1 = Math.min(i + 1, nx - 1);
    const j1 = Math.min(j + 1, ny - 1);
    const k1 = Math.min(k + 1, nz - 1);
    const at = (a: number, b: number, c: number): number => grid[(c * ny + b) * nx + a];
    const c00 = lerp(at(i, j, k), at(i1, j, k), tx);
    const c10 = lerp(at(i, j1, k), at(i1, j1, k), tx);
    const c01 = lerp(at(i, j, k1), at(i1, j, k1), tx);
    const c11 = lerp(at(i, j1, k1), at(i1, j1, k1), tx);
    return lerp(lerp(c00, c10, ty), lerp(c01, c11, ty), tz);
  }
}

interface RawMesh {
  positions: number[];
  normals: number[];
  indices: number[];
}

const CORNER: ReadonlyArray<readonly [number, number, number]> = [
  [0, 0, 0],
  [1, 0, 0],
  [0, 1, 0],
  [1, 1, 0],
  [0, 0, 1],
  [1, 0, 1],
  [0, 1, 1],
  [1, 1, 1],
];
const EDGES: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [2, 3],
  [4, 5],
  [6, 7],
  [0, 2],
  [1, 3],
  [4, 6],
  [5, 7],
  [0, 4],
  [1, 5],
  [2, 6],
  [3, 7],
];

/**
 * Naive surface nets: one vertex per sign-changing cell, a quad across every
 * sign-changing edge. Manifold by construction, with none of marching cubes'
 * sliver triangles, and no 4 KB triangle table.
 */
function surfaceNets(f: Field): RawMesh {
  const { nx, ny, nz, cell, ox, oy, oz, grid } = f;
  const cellIndex = new Int32Array((nx - 1) * (ny - 1) * (nz - 1)).fill(-1);
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];
  const v = new Float64Array(8);
  const at = (a: number, b: number, c: number): number => grid[(c * ny + b) * nx + a];

  for (let k = 0; k < nz - 1; k++) {
    for (let j = 0; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        let neg = 0;
        for (let c = 0; c < 8; c++) {
          const val = at(i + CORNER[c][0], j + CORNER[c][1], k + CORNER[c][2]);
          v[c] = val;
          if (val < 0) neg++;
        }
        if (neg === 0 || neg === 8) continue;
        let sx = 0;
        let sy = 0;
        let sz = 0;
        let n = 0;
        for (const [a, b] of EDGES) {
          const va = v[a];
          const vb = v[b];
          if (va < 0 === vb < 0) continue;
          const t = va / (va - vb);
          sx += CORNER[a][0] + (CORNER[b][0] - CORNER[a][0]) * t;
          sy += CORNER[a][1] + (CORNER[b][1] - CORNER[a][1]) * t;
          sz += CORNER[a][2] + (CORNER[b][2] - CORNER[a][2]) * t;
          n++;
        }
        if (n === 0) continue;
        cellIndex[(k * (ny - 1) + j) * (nx - 1) + i] = positions.length / 3;
        positions.push(ox + (i + sx / n) * cell, oy + (j + sy / n) * cell, oz + (k + sz / n) * cell);
        normals.push(0, 1, 0);
      }
    }
  }

  const cellAt = (i: number, j: number, k: number): number =>
    i < 0 || j < 0 || k < 0 || i >= nx - 1 || j >= ny - 1 || k >= nz - 1
      ? -1
      : cellIndex[(k * (ny - 1) + j) * (nx - 1) + i];

  const quad = (a: number, b: number, c: number, d: number, flip: boolean): void => {
    if (a < 0 || b < 0 || c < 0 || d < 0) return;
    if (flip) indices.push(a, c, b, a, d, c);
    else indices.push(a, b, c, a, c, d);
  };

  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const s = at(i, j, k) < 0;
        if (i < nx - 1 && at(i + 1, j, k) < 0 !== s) {
          quad(cellAt(i, j - 1, k - 1), cellAt(i, j, k - 1), cellAt(i, j, k), cellAt(i, j - 1, k), s);
        }
        if (j < ny - 1 && at(i, j + 1, k) < 0 !== s) {
          quad(cellAt(i - 1, j, k - 1), cellAt(i, j, k - 1), cellAt(i, j, k), cellAt(i - 1, j, k), !s);
        }
        if (k < nz - 1 && at(i, j, k + 1) < 0 !== s) {
          quad(cellAt(i - 1, j - 1, k), cellAt(i, j - 1, k), cellAt(i, j, k), cellAt(i - 1, j, k), s);
        }
      }
    }
  }
  return { positions, normals, indices };
}

/** Newton-projects vertices onto the exact isosurface; normals come analytic. */
function refine(f: Field, mesh: RawMesh): void {
  const h = f.cell * 0.35;
  const p = mesh.positions;
  for (let i = 0; i < p.length; i += 3) {
    let x = p[i];
    let y = p[i + 1];
    let z = p[i + 2];
    let gx = 0;
    let gy = 0;
    let gz = 1;
    for (let iter = 0; iter < 3; iter++) {
      const d = f.eval(x, y, z);
      gx = (f.eval(x + h, y, z) - f.eval(x - h, y, z)) / (2 * h);
      gy = (f.eval(x, y + h, z) - f.eval(x, y - h, z)) / (2 * h);
      gz = (f.eval(x, y, z + h) - f.eval(x, y, z - h)) / (2 * h);
      const g2 = gx * gx + gy * gy + gz * gz;
      if (g2 < 1e-8) break;
      const step = d / g2;
      x -= gx * step;
      y -= gy * step;
      z -= gz * step;
      if (Math.abs(d) < 1e-5) break;
    }
    const gl = Math.sqrt(gx * gx + gy * gy + gz * gz) || 1;
    p[i] = x;
    p[i + 1] = y;
    p[i + 2] = z;
    mesh.normals[i] = gx / gl;
    mesh.normals[i + 1] = gy / gl;
    mesh.normals[i + 2] = gz / gl;
  }
}

/**
 * Surface-net edge orientation is easy to get backwards; rather than trust it,
 * compare a sample of triangles' geometric normals against the analytic vertex
 * normals and flip the whole index buffer if they disagree.
 */
function fixWinding(raw: RawMesh): number[] {
  const { positions, normals, indices } = raw;
  let agree = 0;
  let total = 0;
  const step = Math.max(3, Math.floor(indices.length / 900) * 3);
  for (let i = 0; i + 2 < indices.length; i += step) {
    const a = indices[i] * 3;
    const b = indices[i + 1] * 3;
    const c = indices[i + 2] * 3;
    const abx = positions[b] - positions[a];
    const aby = positions[b + 1] - positions[a + 1];
    const abz = positions[b + 2] - positions[a + 2];
    const acx = positions[c] - positions[a];
    const acy = positions[c + 1] - positions[a + 1];
    const acz = positions[c + 2] - positions[a + 2];
    const gx = aby * acz - abz * acy;
    const gy = abz * acx - abx * acz;
    const gz = abx * acy - aby * acx;
    if (gx * normals[a] + gy * normals[a + 1] + gz * normals[a + 2] > 0) agree++;
    total++;
  }
  if (total > 0 && agree < total * 0.5) {
    for (let i = 0; i + 2 < indices.length; i += 3) {
      const t = indices[i + 1];
      indices[i + 1] = indices[i + 2];
      indices[i + 2] = t;
    }
  }
  return indices;
}

// ---------------------------------------------------------------------------
// Skinning
// ---------------------------------------------------------------------------

const _dist = new Float64Array(BONES.length);
const _score = new Float64Array(BONES.length);
const _order = new Int32Array(BONES.length);
const ROOT = BONE_INDEX.root;

/**
 * Weights from *relative* distance to the bone segments: the nearest bone is
 * the reference and every other bone falls off over its own declared blend
 * width. A mid-forearm vertex therefore stays pure while an elbow vertex lands
 * at a clean 50/50, which is what stops the candy-wrapper pinch when a joint
 * folds past 90°.
 */
function skinWeights(
  segs: Segment[],
  x: number,
  y: number,
  z: number,
  outIdx: number[],
  outWt: number[],
): number {
  let dmin = Infinity;
  for (let i = 0; i < segs.length; i++) {
    if (i === ROOT) {
      _dist[i] = Infinity;
      continue;
    }
    const d = segDistance(segs[i], x, y, z);
    _dist[i] = d;
    if (d < dmin) dmin = d;
  }
  let count = 0;
  for (let i = 0; i < segs.length; i++) {
    if (!isFinite(_dist[i])) continue;
    const rel = (_dist[i] - dmin) / segs[i].sigma;
    if (rel > 3.2) continue;
    _score[count] = Math.exp(-rel * rel);
    _order[count] = i;
    count++;
  }
  // Partial selection sort for the top four — cheaper than a full sort and this
  // runs for every vertex of every body.
  const take = Math.min(4, count);
  for (let a = 0; a < take; a++) {
    let best = a;
    for (let b = a + 1; b < count; b++) if (_score[b] > _score[best]) best = b;
    if (best !== a) {
      const ts = _score[a];
      _score[a] = _score[best];
      _score[best] = ts;
      const ti = _order[a];
      _order[a] = _order[best];
      _order[best] = ti;
    }
  }
  let total = 0;
  for (let i = 0; i < 4; i++) {
    outIdx[i] = i < take ? _order[i] : 0;
    outWt[i] = i < take ? _score[i] : 0;
    total += outWt[i];
  }
  if (total <= 1e-6) {
    outIdx[0] = BONE_INDEX.hips;
    outWt[0] = 1;
    outWt[1] = outWt[2] = outWt[3] = 0;
    return BONE_INDEX.hips;
  }
  for (let i = 0; i < 4; i++) outWt[i] /= total;
  return outIdx[0];
}

// ---------------------------------------------------------------------------
// Mesh assembly helpers
// ---------------------------------------------------------------------------

interface LoftVertex {
  x: number;
  y: number;
  z: number;
  nx: number;
  ny: number;
  nz: number;
  u: number;
  v: number;
}

class MeshBuilder {
  pos: number[] = [];
  nrm: number[] = [];
  uv: number[] = [];
  si: number[] = [];
  sw: number[] = [];
  idx: number[] = [];
  private readonly bi: number[] = [0, 0, 0, 0];
  private readonly bw: number[] = [0, 0, 0, 0];

  constructor(private readonly segs: Segment[]) {}

  push(v: LoftVertex): number {
    const id = this.pos.length / 3;
    this.pos.push(v.x, v.y, v.z);
    const nl = Math.hypot(v.nx, v.ny, v.nz) || 1;
    this.nrm.push(v.nx / nl, v.ny / nl, v.nz / nl);
    this.uv.push(v.u, v.v);
    skinWeights(this.segs, v.x, v.y, v.z, this.bi, this.bw);
    this.si.push(this.bi[0], this.bi[1], this.bi[2], this.bi[3]);
    this.sw.push(this.bw[0], this.bw[1], this.bw[2], this.bw[3]);
    return id;
  }

  quad(a: number, b: number, c: number, d: number): void {
    this.idx.push(a, b, c, a, c, d);
  }

  geometry(height: number): BufferGeometry {
    const g = new BufferGeometry();
    g.setAttribute('position', new Float32BufferAttribute(this.pos, 3));
    g.setAttribute('normal', new Float32BufferAttribute(this.nrm, 3));
    g.setAttribute('uv', new Float32BufferAttribute(this.uv, 2));
    g.setAttribute('skinIndex', new Uint16BufferAttribute(this.si, 4));
    g.setAttribute('skinWeight', new Float32BufferAttribute(this.sw, 4));
    g.setIndex(this.idx);
    g.boundingSphere = new Sphere(new Vector3(0, height * 0.5, 0), height * 1.5);
    return g;
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface BodyBuild {
  body: BufferGeometry;
  kit: BufferGeometry;
  shoes: BufferGeometry;
  hair: BufferGeometry | null;
  triangles: number;
}

export type HairStyle = 'bald' | 'fade' | 'crop' | 'afro' | 'headband';

export interface BodyOptions {
  skeleton: BuiltSkeleton;
  shape: BodyShape;
  detail: 0 | 1 | 2;
  hair: HairStyle;
  seed: number;
}

const CELL_FOR_DETAIL = [0.032, 0.024, 0.019];

export function buildBody(opts: BodyOptions): BodyBuild {
  const { skeleton: sk, shape, detail, seed } = opts;
  const H = sk.height;
  const cell = CELL_FOR_DETAIL[detail] * (H / 1.98);
  const prims = buildPrims(sk, shape);
  const segs = buildSegments(sk);

  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (const p of prims) {
    minX = Math.min(minX, p.minX);
    minY = Math.min(minY, p.minY);
    minZ = Math.min(minZ, p.minZ);
    maxX = Math.max(maxX, p.maxX);
    maxY = Math.max(maxY, p.maxY);
    maxZ = Math.max(maxZ, p.maxZ);
  }
  const pad = cell * 2;
  const ox = minX - pad;
  const oy = minY - pad;
  const oz = minZ - pad;
  const nx = Math.ceil((maxX - minX + pad * 2) / cell) + 1;
  const ny = Math.ceil((maxY - minY + pad * 2) / cell) + 1;
  const nz = Math.ceil((maxZ - minZ + pad * 2) / cell) + 1;

  const field = new Field(prims, cell, ox, oy, oz, nx, ny, nz);
  const raw = surfaceNets(field);
  refine(field, raw);

  // The garment shells trace the torso and the legs only — a jersey must not
  // find the arm and grow a sleeve, and the shorts must not find the calf.
  const dressForm = prims.filter((p) => p.group === 'torso');
  const legForm = prims.filter((p) => p.group === 'torso' || p.group === 'leg');

  const body = finishBody(raw, field, segs, sk, seed);
  const kit = buildKit(dressForm, legForm, segs, sk, shape, detail, seed);
  const shoes = buildShoes(segs, sk, shape, detail);
  const hair = opts.hair === 'bald' ? null : buildHair(segs, sk, opts.hair, detail, seed);

  const tri = (g: BufferGeometry | null): number =>
    g ? (g.index ? g.index.count : g.getAttribute('position').count) / 3 : 0;

  return { body, kit, shoes, hair, triangles: tri(body) + tri(kit) + tri(shoes) + tri(hair) };
}

function finishBody(
  raw: RawMesh,
  field: Field,
  segs: Segment[],
  sk: BuiltSkeleton,
  seed: number,
): BufferGeometry {
  const n = raw.positions.length / 3;
  const uv = new Float32Array(n * 2);
  const col = new Float32Array(n * 3);
  const flesh = new Float32Array(n * 2);
  const si = new Uint16Array(n * 4);
  const sw = new Float32Array(n * 4);
  const idx: number[] = [0, 0, 0, 0];
  const wt: number[] = [0, 0, 0, 0];
  const pc: [number, number] = [0, 0];
  const auv: [number, number] = [0, 0];
  const rng = makeRng(seed);
  const freckle = rng() * 40;

  // One shared torso parameterisation so `v` runs pelvis → neck rather than
  // restarting at every vertebra.
  const torsoSeg = makeSegment(
    segs[BONE_INDEX.hips].a,
    segs[BONE_INDEX.neck].a,
    segs[BONE_INDEX.hips].sigma,
  );

  const AO_STEPS = [0.018, 0.042, 0.085, 0.15];

  for (let i = 0; i < n; i++) {
    const x = raw.positions[i * 3];
    const y = raw.positions[i * 3 + 1];
    const z = raw.positions[i * 3 + 2];
    const nxv = raw.normals[i * 3];
    const nyv = raw.normals[i * 3 + 1];
    const nzv = raw.normals[i * 3 + 2];

    const dom = skinWeights(segs, x, y, z, idx, wt);
    for (let k = 0; k < 4; k++) {
      si[i * 4 + k] = idx[k];
      sw[i * 4 + k] = wt[k];
    }

    const name = BONES[dom].name;
    const part = PART_OF_BONE[name] ?? 'torso';
    const ref =
      part === 'torso'
        ? torsoSeg
        : name === 'toeL'
          ? segs[BONE_INDEX.footL]
          : name === 'toeR'
            ? segs[BONE_INDEX.footR]
            : segs[dom];
    partCoords(ref, name.endsWith('R'), x, y, z, pc);
    skinPartUv(part, pc[0], pc[1], auv);
    uv[i * 2] = auv[0];
    uv[i * 2 + 1] = auv[1];

    // --- Ambient occlusion straight out of the field ---------------------
    // Armpit, under the pec, behind the knee and the achilles come out for
    // free, and because it is geometric it tracks the body type.
    let occ = 0;
    let w = 1;
    for (const d of AO_STEPS) {
      const s = field.sampleGrid(x + nxv * d, y + nyv * d, z + nzv * d);
      occ += (w * Math.max(0, d - s)) / d;
      w *= 0.62;
    }
    const ao = clamp01(1 - occ * 0.4);

    const mottle = fbm2(x * 6 + freckle, y * 6, 3, 2.1, 0.5, seed) - 0.5;
    const warm = part === 'hand' ? 0.7 : part === 'foreArm' ? 0.28 : part === 'head' ? 0.3 : 0;
    const shade = 0.6 + 0.4 * ao;
    col[i * 3] = shade * (1 + mottle * 0.06 + warm * 0.05);
    col[i * 3 + 1] = shade * (1 + mottle * 0.04 - warm * 0.015);
    col[i * 3 + 2] = shade * (1 + mottle * 0.03 - warm * 0.032);

    // Flesh data: how readily this patch sweats, and how thin it is — thin
    // parts (ears, fingers, nose) have to glow when backlit.
    const local = segDistance(segs[dom], x, y, z);
    const thin = clamp01(1 - local / (0.05 * sk.height));
    flesh[i * 2] =
      part === 'head' ? 0.95 : part === 'torso' ? 0.85 : part === 'upperArm' ? 0.8 : 0.55;
    flesh[i * 2 + 1] = thin;
  }

  const g = new BufferGeometry();
  g.setAttribute('position', new Float32BufferAttribute(raw.positions, 3));
  g.setAttribute('normal', new Float32BufferAttribute(raw.normals, 3));
  g.setAttribute('uv', new Float32BufferAttribute(uv, 2));
  g.setAttribute('color', new Float32BufferAttribute(col, 3));
  g.setAttribute('aFlesh', new Float32BufferAttribute(flesh, 2));
  g.setAttribute('skinIndex', new Uint16BufferAttribute(si, 4));
  g.setAttribute('skinWeight', new Float32BufferAttribute(sw, 4));
  g.setIndex(fixWinding(raw));
  g.boundingSphere = new Sphere(new Vector3(0, sk.height * 0.5, 0), sk.height * 1.5);
  return g;
}

/** Part-local (u, v): angle around the reference segment, and distance along. */
function partCoords(
  s: Segment,
  mirror: boolean,
  x: number,
  y: number,
  z: number,
  out: [number, number],
): void {
  const dx = x - s.a.x;
  const dy = y - s.a.y;
  const dz = z - s.a.z;
  let t = (dx * s.w.x + dy * s.w.y + dz * s.w.z) / s.len;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const px = dx - s.w.x * t * s.len;
  const py = dy - s.w.y * t * s.len;
  const pz = dz - s.w.z * t * s.len;
  const a = Math.atan2(
    px * s.side.x + py * s.side.y + pz * s.side.z,
    px * s.fwd.x + py * s.fwd.y + pz * s.fwd.z,
  );
  let u = 0.5 + a / (Math.PI * 2);
  // Right-side limbs mirror so anatomy painted once lands on the correct side.
  if (mirror) u = 1 - u;
  if (u < 0) u += 1;
  if (u >= 1) u -= 1;
  out[0] = u;
  out[1] = t;
}

// ---------------------------------------------------------------------------
// Garments
// ---------------------------------------------------------------------------

/** Bisects outward from an axis until the dress form's surface is crossed. */
function surfaceRadius(
  form: readonly Prim[],
  y: number,
  cx: number,
  cz: number,
  dirX: number,
  dirZ: number,
  lo: number,
  hi: number,
): number {
  if (evalPrims(form, cx + dirX * lo, y, cz + dirZ * lo) > 0) return lo;
  if (evalPrims(form, cx + dirX * hi, y, cz + dirZ * hi) < 0) return hi;
  let a = lo;
  let b = hi;
  for (let i = 0; i < 15; i++) {
    const mid = (a + b) * 0.5;
    if (evalPrims(form, cx + dirX * mid, y, cz + dirZ * mid) < 0) a = mid;
    else b = mid;
  }
  return (a + b) * 0.5;
}

/**
 * The jersey is a loft around the torso whose *top edge is a shaped curve* —
 * high over the acromion, scooped at the throat, cut away either side of each
 * armhole — so the armholes and the neckline are geometry rather than an alpha
 * texture. Its radius is the dress form's radius plus a gap that widens toward
 * the hem, which is what makes the cloth hang off the shoulders rather than
 * shrink-wrap the ribs.
 */
function buildKit(
  dressForm: readonly Prim[],
  legForm: readonly Prim[],
  segs: Segment[],
  sk: BuiltSkeleton,
  shape: BodyShape,
  detail: 0 | 1 | 2,
  seed: number,
): BufferGeometry {
  const H = sk.height;
  const S = H / 1.98;
  const mb = new MeshBuilder(segs);
  const around = [22, 30, 38][detail];
  const downJ = [10, 13, 16][detail];
  const hemY = 0.474 * H;

  // Top edge of the jersey, by angle from the chest. The dips at 1.13 and 2.02
  // rad are the front and back of each armhole; between them the cloth rides
  // over the deltoid as a shoulder strap.
  const TOP: ReadonlyArray<readonly [number, number]> = [
    [0, 0.793],
    [0.55, 0.828],
    [1.13, 0.703],
    [1.57, 0.826],
    [2.02, 0.706],
    [2.6, 0.836],
    [Math.PI, 0.842],
  ];
  const topAt = (a: number): number => {
    for (let i = 1; i < TOP.length; i++) {
      if (a <= TOP[i][0]) {
        const p = TOP[i - 1];
        const q = TOP[i];
        const t = (a - p[0]) / (q[0] - p[0]);
        return (p[1] + (q[1] - p[1]) * (t * t * (3 - 2 * t))) * H;
      }
    }
    return TOP[TOP.length - 1][1] * H;
  };

  const ring: number[][] = [];
  for (let vi = 0; vi <= downJ; vi++) {
    const v = vi / downJ;
    const row: number[] = [];
    for (let ui = 0; ui < around; ui++) {
      // u = 0.5 at the chest so the texture's front graphics land on the chest.
      const u = ui / around;
      const theta = (u - 0.5) * Math.PI * 2;
      const dirX = -Math.sin(theta);
      const dirZ = Math.cos(theta);
      const y = lerp(topAt(Math.abs(theta)), hemY, v);
      const rBody = surfaceRadius(dressForm, y, 0, 0, dirX, dirZ, 0.02 * S, 0.34 * S);
      const gap = lerp(0.009, 0.036, Math.pow(v, 0.75)) * S;
      const fold = (fbm2(u * 7.5, v * 3.4, 3, 2.2, 0.55, seed) - 0.5) * lerp(0.004, 0.018, v) * S;
      const r = rBody + gap + fold;
      row.push(
        mb.push({ x: dirX * r, y, z: dirZ * r, nx: dirX, ny: 0.05, nz: dirZ, u, v: KIT_UV.jersey(v) }),
      );
    }
    ring.push(row);
  }
  for (let vi = 0; vi < downJ; vi++) {
    for (let ui = 0; ui < around; ui++) {
      const u2 = (ui + 1) % around;
      mb.quad(ring[vi][ui], ring[vi + 1][ui], ring[vi + 1][u2], ring[vi][u2]);
    }
  }
  rimRow(mb, ring[0], around, 0.006 * S, -0.009 * S, KIT_UV.jersey(0.01));
  rimRow(mb, ring[downJ], around, 0.006 * S, -0.011 * S, KIT_UV.jersey(0.985));

  // --- Shorts -----------------------------------------------------------
  const I = BONE_INDEX;
  const waistY = 0.545 * H;
  const crotchY = 0.468 * H;
  const hipRings = 4;
  const hipRow: number[][] = [];
  for (let vi = 0; vi <= hipRings; vi++) {
    const v = vi / hipRings;
    const y = lerp(waistY, crotchY, v);
    const row: number[] = [];
    for (let ui = 0; ui < around; ui++) {
      const u = ui / around;
      const theta = (u - 0.5) * Math.PI * 2;
      const dirX = -Math.sin(theta);
      const dirZ = Math.cos(theta);
      const rBody = surfaceRadius(legForm, y, 0, 0, dirX, dirZ, 0.02 * S, 0.34 * S);
      const gap = lerp(0.008, 0.028, v) * S;
      const fold =
        (fbm2(u * 4.5, v * 2.2 + 11, 2, 2.2, 0.55, seed + 5) - 0.5) * lerp(0.002, 0.012, v) * S;
      const r = rBody + gap + fold;
      row.push(
        mb.push({
          x: dirX * r,
          y,
          z: dirZ * r,
          nx: dirX,
          ny: 0.1,
          nz: dirZ,
          u,
          v: KIT_UV.shorts(v * 0.28),
        }),
      );
    }
    hipRow.push(row);
  }
  for (let vi = 0; vi < hipRings; vi++) {
    for (let ui = 0; ui < around; ui++) {
      const u2 = (ui + 1) % around;
      mb.quad(hipRow[vi][ui], hipRow[vi + 1][ui], hipRow[vi + 1][u2], hipRow[vi][u2]);
    }
  }
  rimRow(mb, hipRow[0], around, 0.008 * S, 0.013 * S, KIT_UV.shorts(0.02));

  // Leg tubes overlap the hip shell inside the body — invisible, and far
  // better behaved than splitting one tube at the crotch.
  const legRings = [4, 5, 7][detail];
  const hemLegY = 0.278 * H;
  const around2 = Math.max(10, Math.round(around * 0.6));
  for (const sgn of [1, -1]) {
    const thigh = segs[sgn > 0 ? I.thighL : I.thighR];
    const rows: number[][] = [];
    for (let vi = 0; vi <= legRings; vi++) {
      const v = vi / legRings;
      const y = lerp(crotchY + 0.028 * H, hemLegY, v);
      const tAlong = clamp01((thigh.a.y - y) / Math.max(1e-4, thigh.a.y - thigh.b.y));
      const cx = lerp(thigh.a.x, thigh.b.x, tAlong);
      const cz = lerp(thigh.a.z, thigh.b.z, tAlong);
      const row: number[] = [];
      for (let ui = 0; ui < around2; ui++) {
        const u = ui / around2;
        const theta = (u - 0.5) * Math.PI * 2;
        const dirX = -Math.sin(theta);
        const dirZ = Math.cos(theta);
        const rBody = surfaceRadius(legForm, y, cx, cz, dirX, dirZ, 0.02 * S, 0.2 * S);
        // Heavier cloth than the jersey: fewer, larger folds, flaring to the hem.
        const gap = lerp(0.013, 0.05, Math.pow(v, 0.8)) * S * shape.build;
        const fold = Math.sin(u * Math.PI * 2 * 3 + v * 2.1 + sgn) * lerp(0.002, 0.012, v) * S;
        const r = rBody + gap + fold;
        row.push(
          mb.push({
            x: cx + dirX * r,
            y,
            z: cz + dirZ * r,
            nx: dirX,
            ny: 0.12,
            nz: dirZ,
            u,
            v: KIT_UV.shorts(0.3 + v * 0.7),
          }),
        );
      }
      rows.push(row);
    }
    for (let vi = 0; vi < legRings; vi++) {
      for (let ui = 0; ui < around2; ui++) {
        const u2 = (ui + 1) % around2;
        mb.quad(rows[vi][ui], rows[vi + 1][ui], rows[vi + 1][u2], rows[vi][u2]);
      }
    }
    rimRow(mb, rows[legRings], around2, 0.007 * S, -0.013 * S, KIT_UV.shorts(0.985));
  }

  return mb.geometry(H);
}

/** Extrudes a boundary ring inward to give the cloth finite, bound thickness. */
function rimRow(
  mb: MeshBuilder,
  row: number[],
  around: number,
  inward: number,
  drop: number,
  vRow: number,
): void {
  const inner: number[] = [];
  for (let ui = 0; ui < around; ui++) {
    const p = row[ui];
    const px = mb.pos[p * 3];
    const py = mb.pos[p * 3 + 1];
    const pz = mb.pos[p * 3 + 2];
    const len = Math.hypot(px, pz) || 1;
    inner.push(
      mb.push({
        x: px - (px / len) * inward,
        y: py + drop,
        z: pz - (pz / len) * inward,
        nx: -px / len,
        ny: 0,
        nz: -pz / len,
        u: ui / around,
        v: vRow,
      }),
    );
  }
  for (let ui = 0; ui < around; ui++) {
    const u2 = (ui + 1) % around;
    mb.quad(row[ui], inner[ui], inner[u2], row[u2]);
  }
}

// ---------------------------------------------------------------------------
// Shoes
// ---------------------------------------------------------------------------

/** v-centres of the five material bands in the shoe strip texture. */
const BAND = { knit: 0.1, overlay: 0.3, sole: 0.4, sock: 0.7, lace: 0.9 };

/**
 * The last, in the foot's local frame, in metres at a 1.98 m frame:
 * `[z from behind the heel to the toe, half width, top of the upper, sole]`.
 * The sole sits at exactly −0.039 H so the shoe meets the floor when the
 * animator plants the ankle.
 */
const LAST: ReadonlyArray<readonly [number, number, number, number]> = [
  [-0.082, 0.03, -0.03, -0.068],
  [-0.072, 0.042, 0.012, -0.075],
  [-0.052, 0.049, 0.046, -0.077],
  [-0.018, 0.052, 0.056, -0.077],
  [0.018, 0.054, 0.041, -0.077],
  [0.058, 0.055, 0.006, -0.077],
  [0.098, 0.054, -0.014, -0.077],
  [0.138, 0.051, -0.026, -0.076],
  [0.178, 0.045, -0.034, -0.073],
  [0.215, 0.034, -0.042, -0.068],
  [0.242, 0.014, -0.052, -0.057],
];

function sampleLast(t: number): [number, number, number, number] {
  const f = t * (LAST.length - 1);
  const i = Math.min(LAST.length - 2, Math.floor(f));
  const k = f - i;
  const s = k * k * (3 - 2 * k);
  const a = LAST[i];
  const b = LAST[i + 1];
  return [lerp(a[0], b[0], s), lerp(a[1], b[1], s), lerp(a[2], b[2], s), lerp(a[3], b[3], s)];
}

/**
 * A basketball shoe: thick midsole with a bright edge line, high-cut collar,
 * herringbone outsole, laces across the instep, and a ribbed sock above it.
 * These sit at the bottom of a portrait frame in nearly every shot, so a single
 * dark blob is not survivable.
 */
function buildShoes(
  segs: Segment[],
  sk: BuiltSkeleton,
  shape: BodyShape,
  detail: 0 | 1 | 2,
): BufferGeometry {
  const H = sk.height;
  const S = H / 1.98;
  const mb = new MeshBuilder(segs);
  const I = BONE_INDEX;
  const sections = [10, 13, 16][detail];
  const around = [10, 12, 14][detail];

  for (const sgn of [1, -1]) {
    const foot = segs[sgn > 0 ? I.footL : I.footR];
    const toe = segs[sgn > 0 ? I.toeL : I.toeR];
    const origin = foot.a;
    const fz = toe.a.clone().sub(foot.a).setY(0).normalize();
    const fy = new Vector3(0, 1, 0);
    const fx = new Vector3().crossVectors(fy, fz).normalize();
    const wide = S * (1 + (shape.build - 1) * 0.5);

    const rows: number[][] = [];
    for (let si = 0; si <= sections; si++) {
      const t = si / sections;
      const [z, hwRaw, topRaw, soleRaw] = sampleLast(t);
      const hw = hwRaw * wide;
      const topY = topRaw * S;
      const soleY = soleRaw * S;
      const midY = (topY + soleY) * 0.5;
      const halfH = Math.max(1e-4, (topY - soleY) * 0.5);
      const row: number[] = [];
      for (let ui = 0; ui < around; ui++) {
        const phi = (ui / around) * Math.PI * 2;
        const cs = Math.cos(phi);
        const sn = Math.sin(phi);
        // Boxy below (a sole is flat), rounded above (an upper is not).
        const e = sn > 0 ? 2.5 : 6.0;
        const k = Math.pow(Math.pow(Math.abs(cs), e) + Math.pow(Math.abs(sn), e), -1 / e);
        const lx = cs * k * hw;
        const ly = midY + sn * k * halfH;
        const yFrac = (ly - soleY) / (topY - soleY || 1);
        let v: number;
        if (sn < -0.35) v = BAND.sole + 0.12 * 0.2;
        else if (yFrac < 0.62) v = BAND.sole + (0.28 + 0.66 * (yFrac / 0.62)) * 0.2;
        else if (t > 0.26 && t < 0.6 && Math.abs(cs) < 0.6 && sn > 0.4) v = BAND.lace;
        else if (yFrac < 0.82) v = BAND.overlay;
        else v = BAND.knit;
        const p = origin
          .clone()
          .addScaledVector(fx, lx * sgn)
          .addScaledVector(fy, ly)
          .addScaledVector(fz, z * S);
        const nrm = new Vector3()
          .addScaledVector(fx, cs * sgn)
          .addScaledVector(fy, sn * (hw / halfH) * 0.4)
          .addScaledVector(fz, (t - 0.5) * 0.4);
        row.push(
          mb.push({
            x: p.x,
            y: p.y,
            z: p.z,
            nx: nrm.x,
            ny: nrm.y,
            nz: nrm.z,
            u: (ui / around) * 3,
            v,
          }),
        );
      }
      rows.push(row);
    }
    for (let si = 0; si < sections; si++) {
      for (let ui = 0; ui < around; ui++) {
        const u2 = (ui + 1) % around;
        mb.quad(rows[si][ui], rows[si + 1][ui], rows[si + 1][u2], rows[si][u2]);
      }
    }
    capRing(mb, rows[0], around, true);
    capRing(mb, rows[sections], around, false);

    // Sock: a short ribbed tube covering the collar / shin transition.
    const shin = segs[sgn > 0 ? I.shinL : I.shinR];
    const sockRings = 4;
    const sockRows: number[][] = [];
    for (let ri = 0; ri <= sockRings; ri++) {
      const t = ri / sockRings;
      const y = origin.y + lerp(0.004, 0.095, t) * H;
      const tAlong = clamp01((shin.a.y - y) / Math.max(1e-4, shin.a.y - shin.b.y));
      const cx = lerp(shin.a.x, shin.b.x, tAlong);
      const cz = lerp(shin.a.z, shin.b.z, tAlong);
      const r = lerp(0.048, 0.054, t) * S * shape.build;
      const row: number[] = [];
      for (let ui = 0; ui < around; ui++) {
        const phi = (ui / around) * Math.PI * 2;
        const dx = Math.cos(phi);
        const dz = Math.sin(phi);
        row.push(
          mb.push({
            x: cx + dx * r,
            y,
            z: cz + dz * r,
            nx: dx,
            ny: 0.1,
            nz: dz,
            u: (ui / around) * 2,
            v: BAND.sock,
          }),
        );
      }
      sockRows.push(row);
    }
    for (let ri = 0; ri < sockRings; ri++) {
      for (let ui = 0; ui < around; ui++) {
        const u2 = (ui + 1) % around;
        mb.quad(sockRows[ri][ui], sockRows[ri + 1][ui], sockRows[ri + 1][u2], sockRows[ri][u2]);
      }
    }
  }

  return mb.geometry(H);
}

function capRing(mb: MeshBuilder, row: number[], around: number, heel: boolean): void {
  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (const p of row) {
    cx += mb.pos[p * 3];
    cy += mb.pos[p * 3 + 1];
    cz += mb.pos[p * 3 + 2];
  }
  cx /= row.length;
  cy /= row.length;
  cz /= row.length;
  const centre = mb.push({
    x: cx,
    y: cy,
    z: cz,
    nx: 0,
    ny: 0,
    nz: heel ? -1 : 1,
    u: 0.5,
    v: BAND.overlay,
  });
  for (let ui = 0; ui < around; ui++) {
    const u2 = (ui + 1) % around;
    if (heel) mb.idx.push(centre, row[u2], row[ui]);
    else mb.idx.push(centre, row[ui], row[u2]);
  }
}

// ---------------------------------------------------------------------------
// Hair
// ---------------------------------------------------------------------------

interface HairSpec {
  shells: number;
  /** Outermost shell offset, in metres at a 1.98 m frame. */
  depth: number;
  /** How far down the forehead the hairline sits. */
  frontLine: number;
  band: boolean;
}

const HAIR_SPECS: Record<Exclude<HairStyle, 'bald'>, HairSpec> = {
  fade: { shells: 2, depth: 0.01, frontLine: 0.12, band: false },
  crop: { shells: 3, depth: 0.021, frontLine: 0.04, band: false },
  afro: { shells: 4, depth: 0.044, frontLine: 0.06, band: false },
  headband: { shells: 2, depth: 0.012, frontLine: 0.18, band: true },
};

/**
 * Shell hair. Each shell is a spherical patch offset along the scalp normal and
 * cut by a per-shell strand alpha, so the outline frays at a 2–8 px scale
 * instead of reading as a helmet — one of the named tells.
 */
function buildHair(
  segs: Segment[],
  sk: BuiltSkeleton,
  style: Exclude<HairStyle, 'bald'>,
  detail: 0 | 1 | 2,
  seed: number,
): BufferGeometry {
  const H = sk.height;
  const S = H / 1.98;
  const spec = HAIR_SPECS[style];
  const mb = new MeshBuilder(segs);
  const headY = sk.restWorld[BONE_INDEX.head].y;
  const centre = new Vector3(0, lerp(headY, 0.995 * H, 0.46), 0.004 * H);
  const rx = 0.081 * H;
  const ry = 0.116 * H;
  const rz = 0.1 * H;
  const cols = [12, 14, 16][detail];
  const rows = [6, 7, 9][detail];
  const rng = makeRng(seed);

  for (let s = 0; s < spec.shells; s++) {
    const k = (s + 1) / spec.shells;
    const off = spec.depth * S * k;
    const grid: number[][] = [];
    for (let ri = 0; ri <= rows; ri++) {
      const pol = (ri / rows) * 1.65;
      const row: number[] = [];
      for (let ci = 0; ci <= cols; ci++) {
        const az = (ci / cols) * Math.PI * 2;
        const sy = Math.cos(pol);
        const sx = Math.sin(pol) * Math.sin(az);
        const sz = Math.sin(pol) * Math.cos(az);
        // Hairline: hair stops higher at the forehead than at the nape.
        const front = clamp01((sz * 0.5 + 0.5 - spec.frontLine) * 3.2);
        const nape = clamp01((0.55 - sz) * 2 + 0.35);
        const cover = Math.min(1, front * 0.5 + nape * 0.55 + 0.1);
        const grow = off * cover * (0.7 + 0.6 * rng());
        const nl = Math.hypot(sx / rx, sy / ry, sz / rz) || 1;
        row.push(
          mb.push({
            x: centre.x + sx * (rx + grow),
            y: centre.y + sy * (ry + grow),
            z: centre.z + sz * (rz + grow),
            nx: sx / rx / nl,
            ny: sy / ry / nl,
            nz: sz / rz / nl,
            u: (ci / cols) * 3.4,
            // Each shell samples a different density row of the strand mask, so
            // the outer shells are sparser and the silhouette frays.
            v: ((s + (ri / rows) * 0.8) / spec.shells) * 0.94,
          }),
        );
      }
      grid.push(row);
    }
    for (let ri = 0; ri < rows; ri++) {
      for (let ci = 0; ci < cols; ci++) {
        mb.quad(grid[ri][ci], grid[ri + 1][ci], grid[ri + 1][ci + 1], grid[ri][ci + 1]);
      }
    }
  }

  if (spec.band) {
    // Headband: solid, for silhouette variety and a hard specular edge.
    const bandRows = 2;
    const bandCols = 16;
    const grid: number[][] = [];
    for (let ri = 0; ri <= bandRows; ri++) {
      const yy = centre.y + lerp(0.012, 0.05, ri / bandRows) * H;
      const row: number[] = [];
      for (let ci = 0; ci <= bandCols; ci++) {
        const az = (ci / bandCols) * Math.PI * 2;
        const sx = Math.sin(az);
        const sz = Math.cos(az);
        const shrink = Math.sqrt(Math.max(0.05, 1 - ((yy - centre.y) / ry) ** 2));
        row.push(
          mb.push({
            x: centre.x + sx * (rx * shrink + 0.005 * S),
            y: yy,
            z: centre.z + sz * (rz * shrink + 0.005 * S),
            nx: sx,
            ny: 0,
            nz: sz,
            u: (ci / bandCols) * 2,
            // Above the strand rows the mask is solid.
            v: 0.985,
          }),
        );
      }
      grid.push(row);
    }
    for (let ri = 0; ri < bandRows; ri++) {
      for (let ci = 0; ci < bandCols; ci++) {
        mb.quad(grid[ri][ci], grid[ri + 1][ci], grid[ri + 1][ci + 1], grid[ri][ci + 1]);
      }
    }
  }

  return mb.geometry(H);
}
