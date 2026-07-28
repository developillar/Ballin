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
 * The second is **shading model**, and it is not the same thing as triangle
 * count. Round 1 shaded the whole bowl *per vertex* over four-sided
 * axis-aligned prisms and shipped the interpolated colour straight out of the
 * fragment stage. A flat face whose four vertices all carry the same normal
 * resolves to one constant rectangle under any lighting model you care to
 * write, so heads were squares and torsos were boxes at every distance and no
 * amount of extra geometry could have fixed it. The lighting now happens in
 * `CROWD_FRAG` off an interpolated world normal, and the head, torso and hair
 * are eight-sided lofted sections with normals smoothed around the
 * cross-section (see {@link prism}) wherever the figures are big enough to
 * resolve one.
 *
 * The third is **silhouette**. At 12–40 px a face is meaningless but a
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
  BufferAttribute,
  BufferGeometry,
  Color,
  DynamicDrawUsage,
  InstancedBufferAttribute,
  InstancedMesh,
  Matrix4,
  Object3D,
  ShaderMaterial,
  StaticDrawUsage,
  Vector2,
  Vector3,
} from 'three';
import { FACE_SIDES, FACE_TOP, MeshBuilder } from './arenaGeometry';

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
/** A phone held against the chest — the LOD-0/1 stand-in, with no arm to ride. */
const PART_CHESTPHONE = 10;

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
// The shoulder sits *outboard* of the widest torso section (0.194) on purpose.
// A courtside figure whose arms hang inside its own torso silhouette presents a
// single unbroken vertical edge from hip to shoulder — the reviewer measured a
// 48 px straight run on a 120 px figure, 40% of its height — and no amount of
// shading fixes an outline that is literally a line. Hanging the deltoid proud
// of the ribcage is also simply what a body does.
const SHO_X = 0.199;
const SHO_Y = 0.968;
const SHO_Z = 0.185;
const ELB_X = 0.206;
const ELB_Y = 0.745;
const ELB_Z = 0.170;
const HAND_X = 0.172;
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

/** True handset screen, metres. Projected size is clamped, not this. */
const PHONE_W = 0.068;
const PHONE_H = 0.146;
/** §6.3: "2–4 px bright cool-white/blue points". Enforced on the projection. */
const PHONE_MAX_PX = 4.0;
/** Below ~2 px a quad starts dropping out of the raster between frames. */
const PHONE_MIN_PX = 2.2;
/** Local anchor of the chest-height phone, relative to the seat centre. */
const CHEST_PHONE = [0.115, 1.32, 0.06] as const;

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

const _IDENT = /* @__PURE__ */ new Matrix4();

/** One elliptical section of a {@link prism}: `[y, radiusX, radiusZ, centreZ]`. */
type Section = readonly [number, number, number, number];

/**
 * A closed prism lofted through a stack of elliptical sections, with the
 * normals **smoothed around the cross-section**.
 *
 * This is the difference between a head and a die. `MeshBuilder.box` and
 * `MeshBuilder.quad` are flat-shaded by construction — every vertex of a face
 * carries the same normal — so a four-sided box lit by any model at all
 * resolves to at most four constant-colour rectangles. No amount of extra
 * lighting maths changes that: the shading input does not vary across the face.
 * An eight-sided section whose vertex normals follow the ellipse gives the
 * fragment stage something to interpolate, so a skull reads as round and its
 * silhouette stops being two vertical lines and a corner.
 *
 * Eight sides costs 16 triangles per segment. The sections are seeded half a
 * step round so a *face* points down local −Z at the court rather than an edge —
 * an edge-on vertex column puts a hard specular seam down the middle of every
 * face the camera sees.
 */
function prism(
  b: MeshBuilder,
  cx: number,
  sections: readonly Section[],
  sides: number,
  capTop: boolean,
): void {
  const rows = sections.length;
  const pos: number[] = [];
  const nor: number[] = [];
  const idx: number[] = [];
  const cs: number[] = [];
  const sn: number[] = [];
  for (let j = 0; j < sides; j++) {
    const a = ((j + 0.5) / sides) * Math.PI * 2;
    cs.push(Math.cos(a));
    sn.push(Math.sin(a));
  }

  for (let i = 0; i < rows; i++) {
    const [y, rx, rz, cz] = sections[i];
    // Tip the smoothed normal with the taper, or a narrowing skull shades like
    // a barrel and the crown stays as dark as the temple.
    const lo = sections[Math.max(0, i - 1)];
    const hi = sections[Math.min(rows - 1, i + 1)];
    const dy = hi[0] - lo[0];
    const ny = dy > 1e-6 ? -((hi[1] + hi[2]) * 0.5 - (lo[1] + lo[2]) * 0.5) / dy : 0;
    for (let j = 0; j < sides; j++) {
      pos.push(cx + rx * cs[j], y, cz + rz * sn[j]);
      const nx = cs[j] / Math.max(rx, 1e-4);
      const nz = sn[j] / Math.max(rz, 1e-4);
      const l = Math.hypot(nx, ny, nz) || 1;
      nor.push(nx / l, ny / l, nz / l);
    }
  }

  for (let i = 0; i < rows - 1; i++) {
    for (let j = 0; j < sides; j++) {
      const j1 = (j + 1) % sides;
      const a0 = i * sides + j;
      const b0 = i * sides + j1;
      const c1 = (i + 1) * sides + j1;
      const d1 = (i + 1) * sides + j;
      idx.push(a0, d1, c1, a0, c1, b0);
    }
  }
  if (capTop) {
    const top = sections[rows - 1];
    const centre = pos.length / 3;
    pos.push(cx, top[0], top[3]);
    nor.push(0, 1, 0);
    const base = (rows - 1) * sides;
    for (let j = 0; j < sides; j++) idx.push(centre, base + ((j + 1) % sides), base + j);
  }

  const geo = new BufferGeometry();
  geo.setAttribute('position', new BufferAttribute(new Float32Array(pos), 3));
  geo.setAttribute('normal', new BufferAttribute(new Float32Array(nor), 3));
  geo.setIndex(idx);
  b.merge(geo, _IDENT);
  geo.dispose();
}

/**
 * A phone screen. Emitted with both windings and a hair of separation: the pose
 * solve swings the forearm through a wide arc, so which face of a two-triangle
 * quad ends up pointing at the camera is not knowable at build time, and a phone
 * that back-face culls itself away is the §6.3 point light that never appears.
 */
function screenQuad(
  b: MeshBuilder,
  x0: number, y0: number, z0: number,
  x1: number, y1: number, z1: number,
): void {
  b.quad(x0, y0, z0, x1, y0, z0, x1, y1, z1, x0, y1, z1);
  b.quad(x0, y1, z1 - 0.0015, x1, y1, z1 - 0.0015, x1, y0, z0 - 0.0015, x0, y0, z0 - 0.0015);
}

/**
 * One pod: `podSize` seats side by side along local +X, all facing local −Z.
 *
 * Triangle budget per seat: 22 at detail 0 (upper bowl, ~10 px figures), 52 at
 * detail 1 (far lower bowl), ~168 at detail 2 (near lower bowl) and ~216 at
 * detail 3 (courtside, where the figures are 60–120 px and every one of those
 * triangles is doing visible work).
 *
 * The jump at detail 2 buys the rounded head, torso and hair — see {@link prism}.
 * `detail` is chosen per block by the arena's tier plan, so the cost is a
 * `Quality` decision and the far bowl, where a spectator is 15 px tall and a
 * round skull is invisible, keeps the cheap boxes.
 */
function buildPod(podSize: number, pitch: number, detail: CrowdDetail): MeshBuilder {
  const b = new MeshBuilder();
  b.attribute('aPart', 1, [PART_SEAT]);
  b.attribute('aSeat', 1, [0]);
  b.attribute('aSide', 1, [0]);

  // A phone screen is a *light point*, not a prop. §6.3 asks for 2–4 px bright
  // dots. Rounds 3–6 chased that by inflating the *world* size per LOD, which
  // is how the upper deck ended up carrying 185 × 260 mm quads — a 7 × 10 inch
  // tablet — that rendered as 15 × 21 px placards at luminance 200 on a bowl at
  // 30. The size is a *screen-space* criterion, so it is now enforced in screen
  // space: the geometry is built at true handset size and `PHONE_MAX_PX` /
  // `PHONE_MIN_PX` clamp the projected height in the vertex shader (see
  // uPhoneClamp). Count, not area, carries the effect.
  const phoneW = PHONE_W;
  const phoneH = PHONE_H;

  const halfSeat = pitch * 0.5;
  for (let k = 0; k < podSize; k++) {
    const bx = (k - (podSize - 1) * 0.5) * pitch;
    b.set('aSeat', k);
    b.set('aSide', 0);

    // --- the seat itself -----------------------------------------------------
    if (detail === 0) {
      // §6.1 wants the seating rake legible even at 20–45 luminance and §6.3
      // calls a crowd with no visible seating furniture a texture. The upper
      // deck had no seat geometry at all, so it was figures floating on a bare
      // rake — and a whole storey of that is what reads to `analyze.mjs` as a
      // flat field. One quad per seat gives the rows their dark horizontal
      // banding for two triangles.
      b.set('aPart', PART_SEAT);
      const pw = halfSeat * 0.84;
      b.quad(
        bx + pw, 0.44, 0.40,
        bx - pw, 0.44, 0.40,
        bx - pw, 0.90, 0.435,
        bx + pw, 0.90, 0.435,
      );
    } else {
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
    // A stack of lofted sections whose radius is deliberately **non-monotonic**:
    // hip, a waist pinch, the ribcage opening out, a shoulder yoke that is the
    // widest thing on the body, then the trapezius falling away to the neck.
    // That profile is not decoration — §6.3's silhouette test is that no
    // straight vertical edge run exceeds ~15% of the figure's height, and a
    // two-frustum torso is one straight run from hip to shoulder by
    // construction. It is also what stops the top of the body reading as a flat
    // slab, i.e. a bowl full of envelopes.
    //
    // Below detail 2 the torso used to be a `box`, which is flat-shaded by
    // construction: four constant-colour rectangles regardless of the lighting
    // model. The far lower bowl is *most* of the crowd in a RIM framing and it
    // is exactly where the reviewer measured a 25-row constant-value slab, so
    // it gets a six-sided loft too. Six sides and three sections is 30 triangles
    // against the box's 10, and it is the cheapest thing that gives the fragment
    // stage a normal to interpolate.
    b.set('aPart', PART_TORSO);
    if (detail >= 3) {
      prism(b, bx, [
        [0.560, 0.146, 0.100, 0.198],
        [0.665, 0.140, 0.096, 0.196],
        [0.790, 0.166, 0.110, 0.193],
        [0.884, 0.180, 0.115, 0.191],
        [0.938, 0.194, 0.119, 0.193],
        [0.972, 0.140, 0.096, 0.199],
        [0.990, 0.092, 0.074, 0.202],
      ], 8, true);
    } else if (detail >= 2) {
      prism(b, bx, [
        [0.566, 0.146, 0.101, 0.197],
        [0.700, 0.145, 0.099, 0.195],
        [0.860, 0.176, 0.113, 0.192],
        [0.936, 0.190, 0.117, 0.193],
        [0.974, 0.112, 0.086, 0.201],
      ], 8, true);
    } else if (detail >= 1) {
      prism(b, bx, [
        [0.572, 0.148, 0.100, 0.196],
        [0.720, 0.146, 0.099, 0.195],
        [0.900, 0.180, 0.113, 0.193],
        [0.962, 0.116, 0.088, 0.200],
      ], 6, true);
    } else {
      prism(b, bx, [
        [0.585, 0.150, 0.101, 0.195],
        [0.880, 0.176, 0.112, 0.193],
        [0.966, 0.120, 0.090, 0.200],
      ], 5, true);
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
    if (detail >= 3) {
      // Chin, jaw, cheekbone, temple, crown. Five sections and eight sides means
      // the silhouette of a 90 px courtside head is a sixteen-segment curve, not
      // a rectangle, and the value across the face runs from the lit temple to
      // the shadowed cheek instead of sitting at one number.
      prism(b, bx, [
        [1.034, 0.043, 0.050, 0.176],
        [1.072, 0.062, 0.070, 0.177],
        [1.106, 0.070, 0.077, 0.176],
        [1.148, 0.072, 0.079, 0.175],
        [1.192, 0.054, 0.060, 0.173],
      ], 8, true);
    } else if (detail >= 2) {
      prism(b, bx, [
        [1.039, 0.050, 0.058, 0.174],
        [1.090, 0.069, 0.076, 0.176],
        [1.140, 0.071, 0.078, 0.175],
        [1.192, 0.055, 0.061, 0.173],
      ], 8, true);
    } else if (detail >= 1) {
      prism(b, bx, [
        [1.044, 0.055, 0.062, 0.175],
        [1.116, 0.072, 0.079, 0.175],
        [1.190, 0.056, 0.062, 0.173],
      ], 6, true);
    } else {
      b.octa(bx, 1.118, 0.180, 0.080, 0.096, 0.084);
    }

    // --- hair ----------------------------------------------------------------
    if (detail >= 1) {
      b.set('aPart', PART_HAIR);
      if (detail >= 3) {
        prism(b, bx, [
          [HAIR_BASE, 0.081, 0.086, 0.174],
          [HAIR_BASE + 0.050, 0.076, 0.081, 0.172],
          [HAIR_BASE + 0.094, 0.046, 0.054, 0.168],
        ], 8, true);
      } else if (detail >= 2) {
        prism(b, bx, [
          [HAIR_BASE, 0.081, 0.086, 0.174],
          [HAIR_BASE + 0.092, 0.049, 0.057, 0.168],
        ], 8, true);
      } else {
        prism(b, bx, [
          [HAIR_BASE - 0.004, 0.082, 0.087, 0.174],
          [HAIR_BASE + 0.090, 0.050, 0.058, 0.168],
        ], 6, true);
      }
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
      screenQuad(
        b,
        px - phoneW * 0.5, HAND_Y + 0.012, HAND_Z - 0.040,
        px + phoneW * 0.5, HAND_Y + 0.012 + phoneH, HAND_Z - 0.030,
      );
      b.set('aSide', 0);
    } else {
      // §6.3's scattered phone points are the cheapest thing in the whole
      // rubric that sells a dark bowl, and the upper deck — which is most of
      // the bowl area in any court framing — is exactly where the darkness
      // needs selling. There is no arm up here to hang a phone off, so it is
      // four triangles, held up beside the head.
      //
      // Held *up* specifically. A 500 mm riser over a 900 mm tread is a 29°
      // rake, and the sightline from a 2 m camera 45 m away is 15°, so every
      // row hides the one behind it up to roughly head height: a phone at chest
      // height in the far or upper bowl sits below the head in front of it and
      // is never drawn. Measured — round 4 put a 130 x 190 mm screen on the
      // chest at detail 0 and the whole bowl still returned zero pixels above
      // 140. At 1.32 m it clears the row in front, which is also where a phone
      // held up to film the court actually is.
      b.set('aPart', PART_CHESTPHONE);
      screenQuad(
        b,
        bx + 0.115 - phoneW * 0.5, 1.320 - phoneH * 0.5, 0.060,
        bx + 0.115 + phoneW * 0.5, 1.320 + phoneH * 0.5, 0.060,
      );
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
  uniform float uPhoneGain;
  uniform float uStandRate;
  uniform float uHasShin;
  uniform float uStandRise;
  uniform vec3  uSeatCol;
  uniform float uAnimate;
  uniform float uAlbedoCap;
  /** x: viewport height in device px, y: max projected phone px, z: min. */
  uniform vec3  uPhoneClamp;

  varying vec3 vAlbedo;
  varying vec3 vNormal;
  varying vec3 vWorld;
  /** x: specular sheen, y: ambient occlusion, z: row tone, w: screen emission. */
  varying vec4 vShade;

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
      float g = 0.0210 + v * 0.0270;
      c = vec3( g * 0.92, g * 0.96, g * 1.20 );
    } else if ( k < 0.62 ) {
      // Charcoal / denim / olive.
      float g = 0.0400 + v * 0.0260;
      c = mix( vec3( g * 0.88, g * 0.97, g * 1.32 ), vec3( g * 1.16, g * 1.0, g * 0.72 ), step( 0.5, v ) );
    } else if ( k < 0.745 ) {
      c = vec3( 0.024, 0.050, 0.132 ) * ( 0.62 + v * 0.72 );      // home blue
    } else if ( k < 0.830 ) {
      c = vec3( 0.132, 0.036, 0.024 ) * ( 0.62 + v * 0.72 );      // away red
    } else if ( k < 0.878 ) {
      c = vec3( 0.130, 0.086, 0.024 ) * ( 0.62 + v * 0.68 );      // gold
    } else if ( k < 0.906 ) {
      c = vec3( 0.028 + v * 0.034, 0.062 + v * 0.040, 0.044 + v * 0.028 ); // green
    } else if ( k < 0.930 ) {
      c = vec3( 0.100, 0.079, 0.052 ) * ( 0.66 + v * 0.62 );      // tan / camel
    } else {
      float g = 0.055 + v * 0.032;                                // light shirts
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
    bool isHandPhone  = part > 5.5 && part < 6.5;
    bool isChestPhone = part > 9.5;
    bool isPhone = isHandPhone || isChestPhone;
    bool isFore  = ( part > 7.5 && part < 8.5 ) || isHandPhone;
    bool isUpper = part > 4.5 && part < 5.5;
    bool isArm   = isFore || isUpper;
    bool isLeg   = ( part > 0.5 && part < 1.5 ) || ( part > 6.5 && part < 7.5 );
    bool isHead  = part > 2.5 && part < 4.5;

    if ( !isSeat ) {
      float present  = step( kOcc, uOccupancy );
      float hasPhone = step( kPhone, uPhoneRate );
      float keep = present * ( isPhone ? hasPhone : 1.0 );

      vec3 q = vec3( p.x - baseX, p.y, p.z );

      // --- phone screens are sized in screen space --------------------------
      // §6.3 asks for 2–4 px points, which is a *projected* criterion. The quad
      // is modelled at true handset size; here it is scaled about its own anchor
      // so the projection lands inside the band at whatever distance this pod
      // happens to sit. Distance is taken from the pod origin rather than the
      // vertex — a pod is under a metre across and this is sizing a 3 px dot.
      if ( isPhone ) {
        vec3 anchor = isChestPhone
          ? vec3( ${f(CHEST_PHONE[0])}, ${f(CHEST_PHONE[1])}, ${f(CHEST_PHONE[2])} )
          : vec3( ${f(HAND_X)} * 0.92, ${f(HAND_Y + 0.012 + PHONE_H * 0.5)}, ${f(HAND_Z - 0.035)} );
        vec3 podOrigin = ( modelMatrix * instanceMatrix * vec4( 0.0, 0.9, 0.0, 1.0 ) ).xyz;
        float dist = max( distance( cameraPosition, podOrigin ), 0.5 );
        // projectionMatrix[1][1] is 1/tan(fovY/2); half the viewport maps to it.
        float pxPerMetre = projectionMatrix[1][1] * uPhoneClamp.x * 0.5 / dist;
        float projPx = ${f(PHONE_H)} * pxPerMetre;
        float wantPx = clamp( projPx, uPhoneClamp.z, uPhoneClamp.y );
        q = anchor + ( q - anchor ) * ( wantPx / max( projPx, 1e-4 ) );
      }

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
    } else if ( isPhone ) {
      // A dark handset body. The screen light rides in the emissive term with
      // its own colour, so a phone that is *not* lit reads as a phone and not
      // as a scatter of pale blue litter across the bowl — at 0.47 albedo an
      // unlit screen was brighter than most of the clothing around it.
      albedo = vec3( 0.085, 0.095, 0.115 );
      // Sparse on purpose: a bowl where every phone is lit is a string of
      // fairy lights, not an arena. The ones that are lit sit above the §6.3
      // bloom threshold, which is what makes them read as screens.
      emissive = uPhoneGain * step( rnd( id + 31.0 ), uPhoneLit )
               * ( 0.80 + 0.20 * sin( uTime * 3.1 + kPh * 20.0 ) );
    } else if ( part > 8.5 && part < 9.5 ) {
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

    // §6.3's absolute rule — the crowd is never brighter than the near hardwood
    // — is bought **here**, on the albedo, and not with a knee on the shaded
    // result. That distinction is the whole of the round-6 defect: a knee that
    // compresses the final luminance toward an asymptote also compresses every
    // shading gradient that reaches it, so the brightest spectators (the ones
    // the knee was written for) came out as constant-value slabs with their
    // form shaded flat. Albedo is constant over a body, so clamping it moves the
    // whole figure down a stop and leaves its centre-to-silhouette falloff
    // exactly where the lighting model put it.
    if ( !isPhone ) {
      float al = dot( albedo, vec3( 0.2126, 0.7152, 0.0722 ) );
      albedo *= min( 1.0, uAlbedoCap / max( al, 1e-4 ) );
    }

    // --- transform ----------------------------------------------------------
    vec4 world = modelMatrix * instanceMatrix * vec4( p, 1.0 );
    mat3 im = mat3( instanceMatrix );

    // Self-occlusion down the body: rows shadow each other and a seated torso
    // shadows its own lap.
    float ao = mix( 0.20, 1.0, smoothstep( 0.08, 1.05, rel ) );
    if ( isSeat ) ao = 0.42;

    // Everything the fragment stage needs, and nothing it can derive itself.
    // The *lighting* deliberately does not happen here: a face shaded per
    // vertex resolves to one constant colour, because every vertex of a flat
    // face carries the same normal. That is what made the bowl a field of
    // rectangles no matter how many triangles it was given.
    vAlbedo = albedo;
    vNormal = mat3( modelMatrix ) * ( im * n );
    vWorld  = world.xyz;
    vShade  = vec4( sheen, ao, iTone, emissive );

    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const CROWD_FRAG = /* glsl */ `
  #include <common>

  uniform vec3  uKeyCol;
  uniform float uKeyInward;
  uniform vec3  uFillDir;
  uniform vec3  uFillCol;
  uniform vec3  uAmbTop;
  uniform vec3  uAmbBot;
  uniform vec3  uSpillCol;
  uniform vec3  uLedCol;
  uniform float uLedSpill;
  uniform vec3  uRimCol;
  uniform vec3  uHazeCol;
  uniform vec3  uFloorCol;
  uniform vec2  uHazeRange;
  uniform float uHazeAmount;
  uniform float uRim;
  uniform float uSpec;
  uniform float uExposure;
  uniform float uClimbFall;
  uniform float uForm;
  uniform float uToneWeight;
  uniform float uCap;

  varying vec3 vAlbedo;
  varying vec3 vNormal;
  varying vec3 vWorld;
  varying vec4 vShade;

  void main() {
    // The one line that matters: an interpolated, per-pixel normal. On a flat
    // face this is numerically identical to what the vertex stage produced, so
    // the measured bowl-to-court ratio does not move; on the rounded head and
    // torso sections it is what puts a value gradient across a face.
    vec3 N = normalize( vNormal );
    vec3 V = normalize( cameraPosition - vWorld );
    vec3 albedo = vAlbedo;
    float sheen = vShade.x;
    float ao    = vShade.y;
    float tone  = vShade.z;

    // --- lighting -----------------------------------------------------------
    // Every photon that reaches a spectator arrives from over the court: the
    // banks are inboard and high, the hardwood bounce is inboard and low, the
    // ribbon boards are inboard and level. So the rig is stated **relative to
    // the bowl's inward normal** rather than in world axes. A single world-space
    // key direction lights the near arc of a ring and leaves the far arc on
    // ambient alone, and ambient has no angular structure at all — which is how
    // half the bowl ended up as constant-value slabs whatever the geometry did.
    vec3 inward = normalize( vec3( -vWorld.x, 0.001, -vWorld.z ) );
    vec3 keyDir = normalize( inward * uKeyInward + vec3( 0.0, 1.0, 0.0 ) );
    vec3 bounceDir = normalize( inward * 0.85 - vec3( 0.0, 0.55, 0.0 ) );

    float ndK = max( dot( N, keyDir ), 0.0 );
    float ndB = max( dot( N, bounceDir ), 0.0 );
    float ndF = max( dot( N, uFillDir ), 0.0 );
    float hemi = 0.5 + 0.5 * N.y;

    // Form. At 12–120 px the single strongest cue that a spectator is a body
    // rather than a plate is that the shading falls away *symmetrically* toward
    // the silhouette — stronger than any light direction at this scale, because
    // a crowd is packed and every figure is shadowed by its neighbours exactly
    // where it turns away from the open volume. §6.3's test is a smooth ≥ 20
    // sRGB drop centre-to-edge across a courtside torso; this is the term that
    // produces it, and it costs one dot product.
    float form = mix( 1.0, clamp( dot( N, V ), 0.0, 1.0 ), uForm );

    vec3 lit = albedo * (
        uKeyCol * ndK
      + uFillCol * ndF
      + uAmbTop * hemi
      + uAmbBot * ( 1.0 - hemi )
    );

    // Two separate near-floor sources, because they are two different colours
    // and §1.3 asks for both. uSpillCol is the warm hardwood bounce; the LED
    // term is the ribbon and courtside boards, which are team-coloured and fall
    // off much faster with height — hence the squared weight. tone encodes
    // how close the row is to the floor and the boards, and both arrive from
    // below and inboard, so they key off ndB rather than the world fill.
    float near = max( tone - 1.0, 0.0 );
    lit += albedo * uSpillCol * near * ( 0.22 + 0.78 * ndB );
    lit += albedo * uLedCol * ( near * near * uLedSpill ) * ( 0.28 + 0.72 * ndB );
    lit *= ao * form;

    #if CROWD_HQ
    // Nothing in frame has zero specular (§1.4) — a broad, low lobe, stronger
    // on the shell jackets than on knitwear. Off in the far and upper bowl,
    // where a spectator is 15 px and a specular lobe is not resolvable.
    vec3 H = normalize( keyDir + V );
    lit += uKeyCol * pow( max( dot( N, H ), 0.0 ), 20.0 ) * uSpec * sheen * ao;
    #endif

    // Rim. Directional, not a Fresnel halo (§10.4): it only fires where the
    // surface turns up and *away* from the court, so it lands on the tops of
    // shoulders and skulls and cuts each row off the one behind it. The gate is
    // deliberately tight — at smoothstep(-0.05, 0.75) it was still firing on
    // the vertical sides of a torso, which put the brightest pixel of a
    // courtside figure on its own silhouette edge.
    vec3 rimDir = normalize( vec3( 0.0, 0.92, 0.0 ) - inward * 0.40 );
    float fres = pow( 1.0 - clamp( dot( N, V ), 0.0, 1.0 ), 3.6 );
    float back = smoothstep( 0.30, 0.88, dot( N, rimDir ) );
    lit += uRimCol * ( fres * back * uRim ) * ( 0.30 + albedo * 2.0 );

    // The bowl's light comes off the floor and the lower fixtures, so it falls
    // away as the rake climbs. Front row to the top of the upper deck is about
    // a stop and a half, which is what makes the bowl read as a volume rather
    // than as a wall of evenly lit people.
    // The bowl's whole tonal range has to fit inside §1.1's 18–45 band, which
    // is 1.3 stops wide, so the front-row-to-upper-deck gradient is deliberately
    // gentle: present, readable, and about half a stop end to end.
    float climb = 1.0 / ( 1.0 + max( vWorld.y - 0.9, 0.0 ) * uClimbFall );
    lit *= uExposure * mix( 1.0, tone, uToneWeight ) * climb;

    // The §6.3 ceiling is bought on the albedo in the vertex stage (see the
    // note there). What is left here is a *safety* knee for the courtside rows,
    // where the floor bounce and the board spill stack on top of the key: it
    // asymptotes at 1.33 × uCap instead of 2 ×, and because ordinary spectators
    // now sit well under uCap it engages on the brightest few percent only,
    // which is the difference between a limiter and a flattener.
    float lum = dot( lit, vec3( 0.2126, 0.7152, 0.0722 ) );
    if ( lum > uCap ) {
      float over = lum - uCap;
      lit *= ( uCap + over / ( 1.0 + 3.0 * over / uCap ) ) / max( lum, 1e-5 );
    }

    // Phone screens are the one thing in the bowl allowed above that ceiling,
    // and they carry their own cool-white emission rather than modulating the
    // handset's albedo.
    lit += vec3( 0.62, 0.72, 1.0 ) * vShade.w;

    // Depth cueing — far stands lose contrast, which is most of what makes a
    // big room read as big.
    float d = length( cameraPosition - vWorld );
    float haze = smoothstep( uHazeRange.x, uHazeRange.y, d ) * uHazeAmount;
    // Floor the bowl off the black point: crushed regions with no detail are a
    // named tell, and a real building always has some ambient spill.
    vec3 col = max( mix( lit, uHazeCol, haze ), uFloorCol );

    gl_FragColor = vec4( max( col, vec3( 0.0 ) ), 1.0 );
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
  uKeyCol: { value: Color };
  uKeyInward: { value: number };
  uFillDir: { value: Vector3 };
  uFillCol: { value: Color };
  uAmbTop: { value: Color };
  uAmbBot: { value: Color };
  uSpillCol: { value: Color };
  uLedCol: { value: Color };
  uLedSpill: { value: number };
  uRimCol: { value: Color };
  uHazeCol: { value: Color };
  uFloorCol: { value: Color };
  uHazeRange: { value: Vector2 };
  uHazeAmount: { value: number };
  uRim: { value: number };
  uSpec: { value: number };
  uClimbFall: { value: number };
  uForm: { value: number };
  uToneWeight: { value: number };
  uAlbedoCap: { value: number };
  uPhoneClamp: { value: Vector3 };
  uCap: { value: number };
  uAnimate: { value: number };
}

/**
 * The soft knee in the crowd shader asymptotes at **twice** `uCap`, so the cap
 * is a little under a quarter of the hardwood's scene-linear radiance and the
 * brightest possible spectator still lands under the near floor. §6.3 states it
 * flatly and admits no exception: the crowd is never brighter than the hardwood.
 *
 * Driven off `LightingSystem.grade.courtLuminance` rather than hard-coded, so
 * re-exposing the room moves the ceiling with it instead of silently switching
 * the knee off — which is what a fixed 0.185 had done: it sat *above* the
 * brightest spectator in the bowl and never engaged at all.
 */
export function crowdCapFor(courtLuminance: number): number {
  return courtLuminance * 0.26;
}

function makeCrowdMaterial(spec: CrowdBlockSpec, animate: boolean): ShaderMaterial {
  const mat = new ShaderMaterial({
    defines: { CROWD_HQ: spec.detail >= 2 ? 1 : 0 },
    uniforms: {
      uTime: { value: 0 },
      uPodSize: { value: spec.podSize },
      uPitch: { value: spec.pitch },
      uOccupancy: { value: spec.occupancy },
      uPhoneRate: { value: spec.phoneRate },
      // Of the people holding a phone, this share have the screen awake.
      // Sparsity is bought by `phoneRate` (3–5% of spectators are holding one),
      // not here: at 0.20 this was a second gate on top of that and the whole
      // far bowl came out at 0.6% of seats, which measured as *zero* screens
      // above 200 sRGB in three consecutive captures. Forcing it to 1.0 in a
      // diagnostic build put 9 discrete points at 240 into the far bowl, which
      // is what identified the density rather than the geometry as the fault.
      uPhoneLit: { value: 0.70 },
      // §1.1 is the master criterion: with the hardwood at 95–140 the bowl has
      // to land at 18–45, i.e. 2.5–4 stops down. This number is that ratio.
      uExposure: { value: 3.8 },
      // §6.3 wants each phone to carry a faint bloom, and the post stack only
      // grabs what is above `grade.bloomThreshold` (1.1 scene-linear). The
      // screen tint has luminance 0.72, so the gain has to clear ~1.5 before a
      // lit screen is a bloom source rather than a pale sticker. At 2.4 it lands
      // ~235 sRGB with a few pixels of halo.
      uPhoneGain: { value: 2.4 },
      uStandRate: { value: 0.16 },
      uHasShin: { value: spec.detail >= 3 ? 1 : 0 },
      // Blocks with no leg geometry at all cannot stand all the way up without
      // leaving a floating torso, so up there a reaction is a rise in the seat.
      uStandRise: { value: spec.detail >= 1 ? STAND_RISE : 0.12 },
      uSeatCol: { value: new Color(spec.seatColor) },
      uKeyCol: { value: new Color(0.262, 0.278, 0.338) },
      // How far the key tips off vertical toward the court. 0.55 puts the banks
      // at ~61° elevation seen from the seats, which matches `BANK_DIRS`.
      uKeyInward: { value: 0.55 },
      uFillDir: { value: new Vector3(-0.5, 0.42, -0.75).normalize() },
      uFillCol: { value: new Color(0.056, 0.055, 0.070) },
      uAmbTop: { value: new Color(0.062, 0.070, 0.100) },
      uAmbBot: { value: new Color(0.050, 0.041, 0.031) },
      // Hardwood bounce: amber-warm, per §1.3's 3000–3800 K floor kick.
      uSpillCol: { value: new Color(0.24, 0.200, 0.155) },
      // The ribbon and courtside boards, as light rather than as a sticker.
      // §6.2: a bright board that lights nothing is an emissive quad and looks
      // like one. Hue is the dominant panel colour on the strip bake.
      uLedCol: { value: new Color(0.16, 0.30, 0.72) },
      uLedSpill: { value: 0.55 },
      uRimCol: { value: new Color(0.52, 0.60, 0.86) },
      uHazeCol: { value: new Color(0.0150, 0.0182, 0.0262) },
      // §1.1's "deepest arena shadow 6–16, never 0". This used to sit at
      // luminance 0.0128 — a *20 sRGB* per-channel floor that every dark coat in
      // the bowl clamped onto, which is a large flat field wearing a hat.
      uFloorCol: { value: new Color(0.0058, 0.0065, 0.0088) },
      uHazeRange: { value: new Vector2(24, 86) },
      // §6.1 asks for 0.02–0.06 density over 30 m, which compounds to about
      // 0.10–0.25 across the 80 m to the far upper deck. 0.55 was three to five
      // times that: it pulled the far bowl toward one constant colour and was a
      // large part of why `analyze.mjs` found its flattest 159 px window up
      // there at sd 3.77. Haze is meant to cost contrast, not erase it.
      uHazeAmount: { value: 0.24 },
      uRim: { value: 0.048 },
      uSpec: { value: 0.10 },
      uClimbFall: { value: 0.028 },
      // §6.3's centre-to-silhouette falloff. 0.46 puts a courtside torso's edge
      // at ~0.55 of its centre value, i.e. a 25–30 sRGB drop over the half
      // width against the ≥ 20 the rubric asks for.
      uForm: { value: 0.46 },
      // How hard the row's proximity to the floor multiplies its exposure. At
      // 0.72 the front row was running a third of a stop over the fifth row on
      // top of the spill terms, which stacked into the courtside slabs.
      uToneWeight: { value: 0.45 },
      // The §6.3 ceiling, applied per person rather than per pixel.
      uAlbedoCap: { value: 0.058 },
      // x is written from `ArenaSystem.resize`; the other two are §6.3's band.
      uPhoneClamp: { value: new Vector3(1024, PHONE_MAX_PX, PHONE_MIN_PX) },
      // Overwritten from the lighting rig in `ArenaSystem.init`; see crowdCapFor.
      uCap: { value: crowdCapFor(0.19) },
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

  /**
   * Viewport height in device pixels, for the phone-screen size clamp. §6.3
   * states that criterion in *pixels*, so the shader needs to know how many
   * there are; without this the clamp would be a world-space guess again.
   */
  setViewportHeight(px: number): void {
    for (const m of this.materials) {
      (m.uniforms.uPhoneClamp.value as Vector3).x = Math.max(64, px);
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
