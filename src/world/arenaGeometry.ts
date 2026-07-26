/**
 * Geometry toolkit for the arena bowl.
 *
 * Two things live here:
 *
 *  1. `MeshBuilder` — a tiny accumulating builder that emits flat-shaded boxes,
 *     prisms, wedges and pyramids straight into typed arrays with arbitrary
 *     extra vertex attributes. Every piece of arena architecture is built by
 *     appending into one builder and shipping a single merged `BufferGeometry`,
 *     which is how ~40 000 individual seats, steps and rails cost one draw call.
 *
 *  2. The bowl *path*. A real seating bowl is a rounded rectangle offset
 *     outward once per row, and offsetting a rounded rectangle is exact — the
 *     corner radius simply grows by the same amount. So every row, every aisle,
 *     every rail and every ribbon board is sampled off one arc-length
 *     parameterisation and they all line up by construction.
 *
 * Owned by the arena agent.
 */

import { BufferAttribute, BufferGeometry, Matrix4, Vector3 } from 'three';
import { COURT } from '../core/Constants';

// -----------------------------------------------------------------------------
// Layout
// -----------------------------------------------------------------------------

/** Where the hardwood + apron slab ends. */
export const APRON_HX = COURT.halfLength + COURT.apronX;
export const APRON_HZ = COURT.halfWidth + COURT.apronZ;

export const ARENA = {
  /** The event-level deck: hardwood apron, courtside seating, benches, tables. */
  deck: { hx: 21.4, hz: 14.2, r: 7.6 },
  /** Front wall of the lower bowl, carrying the courtside LED boards. */
  fasciaHeight: 1.24,
  lower: {
    rows: 15,
    rise: 0.42,
    run: 0.88,
    /** Path offset of row 0's tread from the deck edge. */
    firstOffset: 1.15,
    seatPitch: 0.54,
  },
  /** Mid-level concourse wall between lower and upper bowls. */
  mid: { top: 9.6, thickness: 1.05 },
  /** Suite / club storey between the two bowls. */
  suiteTop: 12.8,
  upper: {
    rows: 7,
    rise: 0.5,
    run: 0.9,
    firstOffset: 2.6,
    seatPitch: 0.6,
    /** Deck height of upper row 0. */
    baseY: 13.0,
  },
  /** Underside of the truss ceiling. */
  roofY: 23.0,
  /** Height of the catwalk / light-bank plane. */
  riggingY: 17.8,
  /** Number of radial aisles cut through the seating. */
  aisles: 16,
  /** Every Nth aisle also punches a vomitory through the lower rows. */
  vomEvery: 2,
  /** Rows removed above a vomitory mouth. */
  vomRows: 7,
} as const;

export interface Ring {
  hx: number;
  hz: number;
  r: number;
}

export function ringAt(offset: number): Ring {
  return {
    hx: ARENA.deck.hx + offset,
    hz: ARENA.deck.hz + offset,
    r: ARENA.deck.r + offset,
  };
}

export function ringPerimeter(ring: Ring): number {
  return 4 * (ring.hx - ring.r) + 4 * (ring.hz - ring.r) + 2 * Math.PI * ring.r;
}

export interface RingSample {
  x: number;
  z: number;
  /** Outward unit normal in the XZ plane. */
  nx: number;
  nz: number;
  /** Heading for something that faces the court. */
  facing: number;
}

const _s: RingSample = { x: 0, z: 0, nx: 0, nz: 0, facing: 0 };

/**
 * Arc-length sample of the rounded-rectangle ring. `s` wraps. Segment order
 * starts on the +X straight and runs counter-clockwise seen from above.
 */
export function sampleRing(ring: Ring, s: number, out: RingSample = _s): RingSample {
  const { hx, hz, r } = ring;
  const sx = hx - r;
  const sz = hz - r;
  const straightZ = 2 * sz;
  const straightX = 2 * sx;
  const arc = (Math.PI * r) / 2;
  const total = 2 * straightZ + 2 * straightX + 4 * arc;

  let t = s % total;
  if (t < 0) t += total;

  let x = 0;
  let z = 0;
  let nx = 0;
  let nz = 0;

  if (t < straightZ) {
    x = hx;
    z = -sz + t;
    nx = 1;
  } else if ((t -= straightZ) < arc) {
    const a = (t / arc) * (Math.PI / 2);
    nx = Math.cos(a);
    nz = Math.sin(a);
    x = sx + nx * r;
    z = sz + nz * r;
  } else if ((t -= arc) < straightX) {
    x = sx - t;
    z = hz;
    nz = 1;
  } else if ((t -= straightX) < arc) {
    const a = Math.PI / 2 + (t / arc) * (Math.PI / 2);
    nx = Math.cos(a);
    nz = Math.sin(a);
    x = -sx + nx * r;
    z = sz + nz * r;
  } else if ((t -= arc) < straightZ) {
    x = -hx;
    z = sz - t;
    nx = -1;
  } else if ((t -= straightZ) < arc) {
    const a = Math.PI + (t / arc) * (Math.PI / 2);
    nx = Math.cos(a);
    nz = Math.sin(a);
    x = -sx + nx * r;
    z = -sz + nz * r;
  } else if ((t -= arc) < straightX) {
    x = -sx + t;
    z = -hz;
    nz = -1;
  } else {
    t -= straightX;
    const a = Math.PI * 1.5 + (t / arc) * (Math.PI / 2);
    nx = Math.cos(a);
    nz = Math.sin(a);
    x = sx + nx * r;
    z = -sz + nz * r;
  }

  out.x = x;
  out.z = z;
  out.nx = nx;
  out.nz = nz;
  // Face inward, i.e. down the negative normal.
  out.facing = Math.atan2(-nx, -nz);
  return out;
}

/**
 * Aisles are defined in normalised perimeter space so they stay radial as the
 * ring grows. Returns true when the sample sits inside an aisle gap.
 */
export function inAisle(u: number, halfWidthU: number, count = ARENA.aisles, phase = 0.5): boolean {
  const k = (u * count + phase) % 1;
  const d = Math.min(k, 1 - k) / count;
  return d < halfWidthU;
}

/** Index of the nearest aisle, used to decide which ones become vomitories. */
export function aisleIndex(u: number, count = ARENA.aisles, phase = 0.5): number {
  return Math.round(u * count + phase) % count;
}

/**
 * A UV-mapped vertical band following the ring — the LED ribbon boards. `u`
 * runs with real arc length so the scrolling content keeps a constant pixel
 * pitch all the way around, including through the corner radii.
 */
export function buildRingStrip(
  ring: Ring,
  yBottom: number,
  yTop: number,
  uPerMetre: number,
  step = 0.55,
): BufferGeometry {
  const total = ringPerimeter(ring);
  const n = Math.max(24, Math.round(total / step));
  const pos: number[] = [];
  const nor: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];
  const s: RingSample = { x: 0, z: 0, nx: 0, nz: 0, facing: 0 };

  for (let i = 0; i <= n; i++) {
    const arc = (i / n) * total;
    sampleRing(ring, arc, s);
    // Face inward, toward the court.
    pos.push(s.x, yBottom, s.z, s.x, yTop, s.z);
    nor.push(-s.nx, 0, -s.nz, -s.nx, 0, -s.nz);
    const u = arc * uPerMetre;
    uv.push(u, 0, u, 1);
  }
  for (let i = 0; i < n; i++) {
    const a = i * 2;
    idx.push(a, a + 2, a + 3, a, a + 3, a + 1);
  }

  const geo = new BufferGeometry();
  geo.setAttribute('position', new BufferAttribute(new Float32Array(pos), 3));
  geo.setAttribute('normal', new BufferAttribute(new Float32Array(nor), 3));
  geo.setAttribute('uv', new BufferAttribute(new Float32Array(uv), 2));
  geo.setIndex(idx);
  geo.computeBoundingSphere();
  return geo;
}

// -----------------------------------------------------------------------------
// MeshBuilder
// -----------------------------------------------------------------------------

export const FACE_TOP = 1;
export const FACE_BOTTOM = 2;
export const FACE_PX = 4;
export const FACE_NX = 8;
export const FACE_PZ = 16;
export const FACE_NZ = 32;
export const FACE_ALL = 63;
export const FACE_SIDES = FACE_PX | FACE_NX | FACE_PZ | FACE_NZ;
export const FACE_OPEN_BOTTOM = FACE_ALL & ~FACE_BOTTOM;

export interface BoxOptions {
  rotY?: number;
  /** Multiplier on the +Y face's half extents (a taper). */
  topScaleX?: number;
  topScaleZ?: number;
  bottomScaleX?: number;
  bottomScaleZ?: number;
  /** Lateral offset applied to the +Y face — a lean. */
  shearX?: number;
  shearZ?: number;
  faces?: number;
}

const _v = new Vector3();
const _a = new Vector3();
const _b = new Vector3();
const _n = new Vector3();

export class MeshBuilder {
  private px: number[] = [];
  private nrm: number[] = [];
  private idx: number[] = [];
  private extras = new Map<string, { size: number; data: number[]; cur: number[] }>();

  /** Declare an extra float attribute with its default value. */
  attribute(name: string, size: number, initial: number[]): this {
    this.extras.set(name, { size, data: [], cur: initial.slice() });
    return this;
  }

  /** Set the value written for every vertex added from now on. */
  set(name: string, ...values: number[]): this {
    const e = this.extras.get(name);
    if (e) e.cur = values;
    return this;
  }

  get vertexCount(): number {
    return this.px.length / 3;
  }

  get triangleCount(): number {
    return this.idx.length / 3;
  }

  private pushVertex(x: number, y: number, z: number, nx: number, ny: number, nz: number): number {
    const i = this.px.length / 3;
    this.px.push(x, y, z);
    this.nrm.push(nx, ny, nz);
    for (const e of this.extras.values()) {
      for (let k = 0; k < e.size; k++) e.data.push(e.cur[k] ?? 0);
    }
    return i;
  }

  /** Counter-clockwise quad, flat shaded. */
  quad(
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cx: number, cy: number, cz: number,
    dx: number, dy: number, dz: number,
  ): this {
    _a.set(bx - ax, by - ay, bz - az);
    _b.set(dx - ax, dy - ay, dz - az);
    _n.crossVectors(_a, _b);
    if (_n.lengthSq() < 1e-14) return this;
    _n.normalize();
    const i0 = this.pushVertex(ax, ay, az, _n.x, _n.y, _n.z);
    const i1 = this.pushVertex(bx, by, bz, _n.x, _n.y, _n.z);
    const i2 = this.pushVertex(cx, cy, cz, _n.x, _n.y, _n.z);
    const i3 = this.pushVertex(dx, dy, dz, _n.x, _n.y, _n.z);
    this.idx.push(i0, i1, i2, i0, i2, i3);
    return this;
  }

  tri(
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cx: number, cy: number, cz: number,
  ): this {
    _a.set(bx - ax, by - ay, bz - az);
    _b.set(cx - ax, cy - ay, cz - az);
    _n.crossVectors(_a, _b);
    if (_n.lengthSq() < 1e-14) return this;
    _n.normalize();
    const i0 = this.pushVertex(ax, ay, az, _n.x, _n.y, _n.z);
    const i1 = this.pushVertex(bx, by, bz, _n.x, _n.y, _n.z);
    const i2 = this.pushVertex(cx, cy, cz, _n.x, _n.y, _n.z);
    this.idx.push(i0, i1, i2);
    return this;
  }

  /**
   * Axis-aligned-ish box with optional Y rotation, per-face culling, a taper
   * on the top face and a shear. Enough vocabulary to build a tapered torso, a
   * seat back raked 12°, a riser step and a truss chord without a single
   * three.js geometry allocation.
   */
  box(
    cx: number, cy: number, cz: number,
    hx: number, hy: number, hz: number,
    o: BoxOptions = {},
  ): this {
    const faces = o.faces ?? FACE_ALL;
    const rot = o.rotY ?? 0;
    const cr = Math.cos(rot);
    const sr = Math.sin(rot);
    const tsx = o.topScaleX ?? 1;
    const tsz = o.topScaleZ ?? 1;
    const bsx = o.bottomScaleX ?? 1;
    const bsz = o.bottomScaleZ ?? 1;
    const shx = o.shearX ?? 0;
    const shz = o.shearZ ?? 0;

    // corners[ y ][ x ][ z ] → world position
    const p: number[][] = [];
    for (let iy = 0; iy < 2; iy++) {
      const top = iy === 1;
      const sx = top ? tsx : bsx;
      const sz = top ? tsz : bsz;
      const ox = top ? shx : 0;
      const oz = top ? shz : 0;
      for (let ix = 0; ix < 2; ix++) {
        for (let iz = 0; iz < 2; iz++) {
          const lx = (ix ? hx : -hx) * sx + ox;
          const lz = (iz ? hz : -hz) * sz + oz;
          p.push([cx + lx * cr + lz * sr, cy + (top ? hy : -hy), cz - lx * sr + lz * cr]);
        }
      }
    }
    // index helper: (iy*4 + ix*2 + iz)
    const g = (iy: number, ix: number, iz: number): number[] => p[iy * 4 + ix * 2 + iz];
    const q = (a: number[], b: number[], c: number[], d: number[]): void => {
      this.quad(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], d[0], d[1], d[2]);
    };

    if (faces & FACE_TOP) q(g(1, 0, 0), g(1, 1, 0), g(1, 1, 1), g(1, 0, 1));
    if (faces & FACE_BOTTOM) q(g(0, 0, 0), g(0, 0, 1), g(0, 1, 1), g(0, 1, 0));
    if (faces & FACE_PX) q(g(0, 1, 0), g(0, 1, 1), g(1, 1, 1), g(1, 1, 0));
    if (faces & FACE_NX) q(g(0, 0, 1), g(0, 0, 0), g(1, 0, 0), g(1, 0, 1));
    if (faces & FACE_PZ) q(g(0, 0, 1), g(1, 0, 1), g(1, 1, 1), g(0, 1, 1));
    if (faces & FACE_NZ) q(g(1, 0, 0), g(0, 0, 0), g(0, 1, 0), g(1, 1, 0));
    return this;
  }

  /** Eight-triangle diamond — the cheapest thing that still reads as a head. */
  octa(cx: number, cy: number, cz: number, rx: number, ry: number, rz: number, rotY = 0): this {
    const cr = Math.cos(rotY);
    const sr = Math.sin(rotY);
    const put = (lx: number, ly: number, lz: number): number[] => [
      cx + lx * cr + lz * sr,
      cy + ly,
      cz - lx * sr + lz * cr,
    ];
    const top = put(0, ry, 0);
    const bot = put(0, -ry, 0);
    const eq = [put(rx, 0, 0), put(0, 0, rz), put(-rx, 0, 0), put(0, 0, -rz)];
    for (let i = 0; i < 4; i++) {
      const a = eq[i];
      const b = eq[(i + 1) % 4];
      this.tri(top[0], top[1], top[2], a[0], a[1], a[2], b[0], b[1], b[2]);
      this.tri(bot[0], bot[1], bot[2], b[0], b[1], b[2], a[0], a[1], a[2]);
    }
    return this;
  }

  /** Four-sided pyramid, apex up. Used for hair caps and roof cones. */
  pyramid(cx: number, cy: number, cz: number, hx: number, h: number, hz: number, rotY = 0): this {
    const cr = Math.cos(rotY);
    const sr = Math.sin(rotY);
    const put = (lx: number, ly: number, lz: number): number[] => [
      cx + lx * cr + lz * sr,
      cy + ly,
      cz - lx * sr + lz * cr,
    ];
    const apex = put(0, h, 0);
    const base = [put(-hx, 0, -hz), put(hx, 0, -hz), put(hx, 0, hz), put(-hx, 0, hz)];
    for (let i = 0; i < 4; i++) {
      const a = base[i];
      const b = base[(i + 1) % 4];
      this.tri(apex[0], apex[1], apex[2], a[0], a[1], a[2], b[0], b[1], b[2]);
    }
    return this;
  }

  /** Merge an existing geometry (positions + normals only) under a matrix. */
  merge(geo: BufferGeometry, matrix: Matrix4): this {
    const pos = geo.getAttribute('position');
    const nor = geo.getAttribute('normal');
    const index = geo.getIndex();
    const base = this.px.length / 3;
    const nm = new Matrix4().extractRotation(matrix);
    for (let i = 0; i < pos.count; i++) {
      _v.fromBufferAttribute(pos as BufferAttribute, i).applyMatrix4(matrix);
      if (nor) _n.fromBufferAttribute(nor as BufferAttribute, i).applyMatrix4(nm).normalize();
      else _n.set(0, 1, 0);
      this.pushVertex(_v.x, _v.y, _v.z, _n.x, _n.y, _n.z);
    }
    if (index) {
      for (let i = 0; i < index.count; i++) this.idx.push(base + index.getX(i));
    } else {
      for (let i = 0; i < pos.count; i++) this.idx.push(base + i);
    }
    return this;
  }

  /** Total triangles emitted so far — used for the perf report. */
  static tris(geo: BufferGeometry): number {
    const i = geo.getIndex();
    return (i ? i.count : geo.getAttribute('position').count) / 3;
  }

  build(name = 'arena'): BufferGeometry {
    const geo = new BufferGeometry();
    geo.name = name;
    geo.setAttribute('position', new BufferAttribute(new Float32Array(this.px), 3));
    geo.setAttribute('normal', new BufferAttribute(new Float32Array(this.nrm), 3));
    for (const [key, e] of this.extras) {
      geo.setAttribute(key, new BufferAttribute(new Float32Array(e.data), e.size));
    }
    geo.setIndex(this.px.length / 3 > 65000 ? Array.from(this.idx) : this.idx);
    geo.computeBoundingSphere();
    return geo;
  }
}
