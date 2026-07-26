/**
 * The game ball: geometry and a full procedural PBR set.
 *
 * A Wilson NBA game ball is eight panels of full-grain Horween leather with a
 * deep moulded pebble and eight black rubber channels laid to a uniform depth.
 * Nothing about it is a flat orange sphere, and at gameplay distance the thing
 * that identifies it is not the colour — it is the way the pebble stipples the
 * specular band while the channels cut clean, glossy, un-pebbled grooves
 * through it.
 *
 * Three problems have to be solved together:
 *
 * 1. **Parameterisation.** An equirectangular sphere wastes most of its texels
 *    at two poles and leaves a visible meridian seam. Instead the ball is a
 *    *cube sphere* with an equal-angle (tangent-warped) face mapping: texel
 *    density varies by only ~6% across the whole ball, there are no poles, and
 *    the six faces pack into a 3x2 atlas.
 * 2. **Continuity.** Every map is evaluated from the 3D surface direction, not
 *    from UV, so the seam network and the pebble are continuous across cube
 *    edges by construction. Each atlas cell is baked with a padded border whose
 *    content is the *true* continuation of the neighbouring face, so the
 *    finite-difference normals stay correct right up to the face boundary.
 * 3. **Minification.** The mip chain is built by hand, box-filtering strictly
 *    inside each atlas cell so faces can never bleed into one another. While
 *    building it we track how much the pebble normal has averaged away and push
 *    that lost detail back into roughness (a Toksvig-style correction), so the
 *    pebble fades into a roughness constant as the ball recedes instead of
 *    aliasing or vanishing.
 *
 * Owned by the ball agent.
 */

import {
  BufferGeometry,
  CanvasTexture,
  ClampToEdgeWrapping,
  Float32BufferAttribute,
  LinearFilter,
  LinearMipmapLinearFilter,
  SRGBColorSpace,
  Uint16BufferAttribute,
} from 'three';
import { clamp01, makeRng, smoothstep, smootherstep, valueNoise2 } from '../core/MathX';

// ---------------------------------------------------------------------------
// Cube-sphere basis
// ---------------------------------------------------------------------------

/**
 * Face bases in atlas order. `f` is the face centre, `r` the direction of the
 * cell's +u axis and `u` the direction of its +v axis (v runs *down* the
 * canvas; the textures are uploaded with `flipY = false` so canvas rows map
 * straight through).
 */
const FACES: ReadonlyArray<{
  f: readonly [number, number, number];
  r: readonly [number, number, number];
  u: readonly [number, number, number];
}> = [
  { f: [1, 0, 0], r: [0, 0, -1], u: [0, -1, 0] },
  { f: [-1, 0, 0], r: [0, 0, 1], u: [0, -1, 0] },
  { f: [0, 1, 0], r: [1, 0, 0], u: [0, 0, 1] },
  { f: [0, -1, 0], r: [1, 0, 0], u: [0, 0, -1] },
  { f: [0, 0, 1], r: [1, 0, 0], u: [0, -1, 0] },
  { f: [0, 0, -1], r: [-1, 0, 0], u: [0, -1, 0] },
];

const ATLAS_COLS = 3;
const ATLAS_ROWS = 2;
const QUARTER_PI = Math.PI / 4;

/** Equal-angle cube→sphere warp: keeps texel density near-uniform. */
const warp = (t: number): number => Math.tan(t * QUARTER_PI);

/** Direction on the unit sphere for face-local coordinates a, b ∈ [-1, 1]. */
function faceDir(face: number, a: number, b: number, out: Float64Array): void {
  const F = FACES[face];
  const A = warp(a);
  const B = warp(b);
  const x = F.f[0] + A * F.r[0] + B * F.u[0];
  const y = F.f[1] + A * F.r[1] + B * F.u[1];
  const z = F.f[2] + A * F.r[2] + B * F.u[2];
  const inv = 1 / Math.sqrt(x * x + y * y + z * z);
  out[0] = x * inv;
  out[1] = y * inv;
  out[2] = z * inv;
}

/**
 * |∂p/∂a| and |∂p/∂b| for the warped mapping — the metric that turns a
 * height-field gradient in face parameters into one in metres of arc.
 */
function faceScale(a: number, b: number, out: Float64Array): void {
  const A = warp(a);
  const B = warp(b);
  const L2 = 1 + A * A + B * B;
  out[0] = (QUARTER_PI * (1 + A * A) * Math.sqrt(1 + B * B)) / L2;
  out[1] = (QUARTER_PI * (1 + B * B) * Math.sqrt(1 + A * A)) / L2;
}

// ---------------------------------------------------------------------------
// Seam network — eight panels
// ---------------------------------------------------------------------------

/**
 * Two full great circles crossing at right angles, plus a closed curve that
 * wobbles about the great circle perpendicular to them. From any angle that
 * reads as the familiar basketball: a clean perpendicular cross with a pair of
 * curved channels sweeping away either side of it. Together the three curves
 * cut the sphere into eight panels.
 *
 * `WOBBLE` is the peak excursion of the curved channel in radians (~30°); at
 * zero the ball degenerates into a beach ball of three great circles.
 */
const WOBBLE = 0.52;

/**
 * The seam network is rotated off the cube axes so no channel ever runs along a
 * cube-face edge, where the atlas padding and the mip chain are weakest.
 */
const SEAM_ROT = (() => {
  const [ax, ay, az] = [0.62, 0.41, 0.28];
  const cx = Math.cos(ax), sx = Math.sin(ax);
  const cy = Math.cos(ay), sy = Math.sin(ay);
  const cz = Math.cos(az), sz = Math.sin(az);
  // Rz * Ry * Rx, flattened row-major.
  return new Float64Array([
    cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
    sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
    -sy, cy * sx, cy * cx,
  ]);
})();

/**
 * Angular distance (radians) to the nearest channel centreline, plus which of
 * the eight panels the direction falls in. Both come out of the same rotation
 * and the same transcendentals, which matters — this runs a couple of million
 * times per bake.
 *
 * `out[0]` = distance, `out[1]` = panel index 0..7.
 */
function seamSample(nx: number, ny: number, nz: number, out: Float64Array): void {
  const m = SEAM_ROT;
  const x = m[0] * nx + m[1] * ny + m[2] * nz;
  const y = m[3] * nx + m[4] * ny + m[5] * nz;
  const z = m[6] * nx + m[7] * ny + m[8] * nz;

  const d1 = Math.asin(Math.min(1, Math.abs(x)));
  const d2 = Math.asin(Math.min(1, Math.abs(y)));

  const lat = Math.asin(Math.max(-1, Math.min(1, z)));
  const phi = Math.atan2(y, x);
  const dv = lat - WOBBLE * Math.cos(2 * phi);
  // Correct the vertical offset by the curve's local slope so the channel keeps
  // a constant width where it is steepest.
  const slope = (-2 * WOBBLE * Math.sin(2 * phi)) / Math.max(0.28, Math.cos(lat));
  const d3 = Math.abs(dv) / Math.sqrt(1 + slope * slope);

  out[0] = Math.min(d1, Math.min(d2, d3));
  out[1] = (x > 0 ? 1 : 0) | (y > 0 ? 2 : 0) | (dv > 0 ? 4 : 0);
}

// ---------------------------------------------------------------------------
// Band-limited noise on the sphere
// ---------------------------------------------------------------------------

/**
 * A sum of plane waves evaluated on the surface direction. Unlike anything
 * built on UV it is exactly continuous everywhere on the sphere, which is what
 * mottling and wear need — a visible discontinuity in the leather tone at a
 * cube edge would be far worse than the noise being slightly anisotropic.
 */
function sphereNoise(seed: number, terms: number, fLo: number, fHi: number) {
  const rng = makeRng(seed);
  const k = new Float64Array(terms * 4);
  for (let i = 0; i < terms; i++) {
    const z = rng() * 2 - 1;
    const t = rng() * Math.PI * 2;
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    const f = fLo + rng() * (fHi - fLo);
    k[i * 4] = r * Math.cos(t) * f;
    k[i * 4 + 1] = r * Math.sin(t) * f;
    k[i * 4 + 2] = z * f;
    k[i * 4 + 3] = rng() * Math.PI * 2;
  }
  const norm = 1 / (2.55 * Math.sqrt(terms));
  return (x: number, y: number, z: number): number => {
    let s = 0;
    for (let i = 0; i < k.length; i += 4) {
      s += Math.sin(k[i] * x + k[i + 1] * y + k[i + 2] * z + k[i + 3]);
    }
    return clamp01(0.5 + s * norm);
  };
}

// ---------------------------------------------------------------------------
// Bake
// ---------------------------------------------------------------------------

export interface BallMapSet {
  /** Burnt-orange leather, black channels, dirt and polish. sRGB. */
  albedo: CanvasTexture;
  /** Tangent-space normals: pebble domes and recessed channels. */
  normal: CanvasTexture;
  /** R = ambient occlusion, G = roughness, B = metalness (zero). */
  orm: CanvasTexture;
  /** Atlas cell edge in texels. */
  cell: number;
  /** Rough VRAM figure in MB, including the hand-built mip chain. */
  megabytes: number;
  dispose(): void;
}

export interface BallBakeOptions {
  /** Ball radius in metres — every feature size below is physical. */
  radius: number;
  /** Atlas cell edge; must be a power of two. */
  cell?: number;
  anisotropy?: number;
  seed?: number;
}

/** Physical dimensions of the real cover, in metres. */
const CHANNEL_WIDTH = 0.0060;
const CHANNEL_DEPTH = 0.0025;
const PEBBLE_PITCH = 0.0019;
const PEBBLE_HEIGHT = 0.00033;
/** Inflation valve — small, but it is the only asymmetry on the whole cover. */
const VALVE_RADIUS = 0.0042;

function makeCanvas(w: number, h: number): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

/**
 * Box-filters an atlas down one level *within* each cell. Because cell sizes
 * stay even the 2x2 kernel never straddles a cell boundary, so the six faces
 * remain perfectly isolated all the way down the chain.
 */
function halveAtlas(src: ImageData, w: number, h: number): { data: ImageData; w: number; h: number } {
  const nw = w >> 1;
  const nh = h >> 1;
  const out = new ImageData(nw, nh);
  const s = src.data;
  const d = out.data;
  for (let y = 0; y < nh; y++) {
    const r0 = (y * 2) * w;
    const r1 = (y * 2 + 1) * w;
    for (let x = 0; x < nw; x++) {
      const a = (r0 + x * 2) * 4;
      const b = (r0 + x * 2 + 1) * 4;
      const c = (r1 + x * 2) * 4;
      const e = (r1 + x * 2 + 1) * 4;
      const o = (y * nw + x) * 4;
      d[o] = (s[a] + s[b] + s[c] + s[e] + 2) >> 2;
      d[o + 1] = (s[a + 1] + s[b + 1] + s[c + 1] + s[e + 1] + 2) >> 2;
      d[o + 2] = (s[a + 2] + s[b + 2] + s[c + 2] + s[e + 2] + 2) >> 2;
      d[o + 3] = 255;
    }
  }
  return { data: out, w: nw, h: nh };
}

function imageDataToCanvas(img: ImageData): HTMLCanvasElement {
  const c = makeCanvas(img.width, img.height);
  c.getContext('2d')!.putImageData(img, 0, 0);
  return c;
}

function finish(
  levels: HTMLCanvasElement[],
  srgb: boolean,
  anisotropy: number,
): CanvasTexture {
  const t = new CanvasTexture(levels[0]);
  if (srgb) t.colorSpace = SRGBColorSpace;
  t.flipY = false;
  t.wrapS = ClampToEdgeWrapping;
  t.wrapT = ClampToEdgeWrapping;
  t.magFilter = LinearFilter;
  t.minFilter = LinearMipmapLinearFilter;
  t.generateMipmaps = false;
  t.mipmaps = levels;
  t.anisotropy = anisotropy;
  t.needsUpdate = true;
  return t;
}

/**
 * Bakes albedo / normal / ORM for the ball into a 3x2 cube-sphere atlas.
 *
 * Everything is driven off `radius`, so pebble pitch, channel width and channel
 * depth stay physically correct whatever the texture budget: when a texel gets
 * coarser than the pebble the pebble simply band-limits itself away into
 * roughness rather than aliasing.
 */
export function bakeBallMaps(opts: BallBakeOptions): BallMapSet {
  const R = opts.radius;
  const F = Math.max(64, opts.cell ?? 512);
  const PAD = Math.max(4, F >> 5);
  const inner = F - 2 * PAD;
  const W = ATLAS_COLS * F;
  const H = ATLAS_ROWS * F;
  const seed = opts.seed ?? 20260726;

  const albedo = new ImageData(W, H);
  const orm = new ImageData(W, H);
  const height = new Float32Array(W * H);
  const A = albedo.data;
  const O = orm.data;

  // --- angular feature sizes ------------------------------------------------
  const halfChannel = CHANNEL_WIDTH * 0.5 / R;
  // Face parameter units per pebble pitch. The centre of a face runs at
  // π/4 radians per unit `a`, which is where this is calibrated.
  const pitchA = PEBBLE_PITCH / R / QUARTER_PI;
  const texelsPerPitch = (pitchA * 0.5) * inner;
  // If the bake cannot resolve a pebble, open the pitch up rather than alias it
  // into a moiré; the amplitude comes down to match so it never reads as golf
  // ball dimpling.
  const pitchScale = texelsPerPitch < 3.2 ? 3.2 / texelsPerPitch : 1;
  const pebbleK = 1 / (pitchA * pitchScale);
  const pebbleAmp = PEBBLE_HEIGHT / Math.max(1, pitchScale * 0.85);

  // --- surface variation ----------------------------------------------------
  const mottle = sphereNoise(seed + 11, 6, 2.2, 5.5);
  const mottleFine = sphereNoise(seed + 29, 5, 9, 17);
  const wearField = sphereNoise(seed + 47, 5, 2.6, 6.0);
  const grimeField = sphereNoise(seed + 71, 4, 3.5, 8.0);

  // Per-panel manufacturing scatter: hides between −4 and +4 sRGB units.
  const prng = makeRng(seed + 101);
  const panelTone = new Float64Array(8);
  const panelRough = new Float64Array(8);
  for (let i = 0; i < 8; i++) {
    panelTone[i] = (prng() - 0.5) * 8;
    panelRough[i] = (prng() - 0.5) * 0.03;
  }

  const dir = new Float64Array(3);
  const seamOut = new Float64Array(2);
  const SQRT3_2 = 0.8660254;

  // Park the valve well inside a panel — on a real ball it never sits on a rib.
  const valve = (() => {
    const rng = makeRng(seed + 907);
    const probe = new Float64Array(2);
    let best = [0.577, 0.577, 0.577];
    let bestD = -1;
    for (let i = 0; i < 64; i++) {
      const z = rng() * 2 - 1;
      const t = rng() * Math.PI * 2;
      const r = Math.sqrt(Math.max(0, 1 - z * z));
      const c: [number, number, number] = [r * Math.cos(t), r * Math.sin(t), z];
      seamSample(c[0], c[1], c[2], probe);
      if (probe[0] > bestD) {
        bestD = probe[0];
        best = c;
      }
    }
    return best;
  })();
  const valveAng = VALVE_RADIUS / R;
  const valveCosLimit = Math.cos(2.6 * valveAng);

  for (let face = 0; face < 6; face++) {
    const col = face % ATLAS_COLS;
    const row = (face / ATLAS_COLS) | 0;
    const ox = col * F;
    const oy = row * F;

    for (let ty = 0; ty < F; ty++) {
      const b = ((ty + 0.5 - PAD) / inner) * 2 - 1;
      for (let tx = 0; tx < F; tx++) {
        const a = ((tx + 0.5 - PAD) / inner) * 2 - 1;
        faceDir(face, a, b, dir);
        const nx = dir[0];
        const ny = dir[1];
        const nz = dir[2];
        const idx = (oy + ty) * W + (ox + tx);
        const o = idx * 4;

        // --- channel --------------------------------------------------------
        seamSample(nx, ny, nz, seamOut);
        const t = seamOut[0] / halfChannel;
        const pid = seamOut[1];
        // Flat-bottomed groove with steep walls — a moulded rib, not a valley.
        const groove = 1 - smootherstep(clamp01((t - 0.52) / 0.48));
        // A slight bead of leather squeezed up either side of the rib. This is
        // what produces the bright specular lip along the channel.
        const lip =
          Math.exp(-((t - 1.16) * (t - 1.16)) / 0.10) * smoothstep((t - 0.72) / 0.62);
        const pebMask = smoothstep((t - 0.96) / 0.42);

        // --- pebble ---------------------------------------------------------
        // Three cosines 60° apart give a perfect hexagonal dome lattice; a
        // low-frequency domain warp breaks the regularity so it reads as
        // moulded grain rather than as a screen pattern.
        let dome = 0;
        if (pebMask > 0.001) {
          const wx = (valueNoise2(a * 9.0, b * 9.0, seed + 3) - 0.5) * 0.55;
          const wy = (valueNoise2(a * 9.0 + 31, b * 9.0 - 17, seed + 5) - 0.5) * 0.55;
          const p0 = (a + wx * pitchA * 2.2) * pebbleK;
          const q0 = (b + wy * pitchA * 2.2) * pebbleK;
          // 2x2 supersample: the lattice is close to the texel limit and this
          // is what keeps it from beating against the grid.
          let acc = 0;
          for (let s = 0; s < 4; s++) {
            const p = p0 + ((s & 1) - 0.5) * 0.25;
            const q = q0 + (((s >> 1) & 1) - 0.5) * 0.25;
            const c1 = Math.cos(6.2831853 * p);
            const c2 = Math.cos(6.2831853 * (p * 0.5 + q * SQRT3_2));
            const c3 = Math.cos(6.2831853 * (-p * 0.5 + q * SQRT3_2));
            const hx = clamp01(((c1 + c2 + c3) / 3 + 0.5) / 1.5);
            acc += smoothstep((hx - 0.30) / 0.70);
          }
          // Height scatter between neighbouring pebbles.
          const vary = 0.72 + 0.56 * valueNoise2(p0 * 0.85, q0 * 0.85, seed + 13);
          dome = Math.sqrt(acc * 0.25) * vary * pebMask;
        }

        height[idx] = -CHANNEL_DEPTH * groove + CHANNEL_DEPTH * 0.055 * lip + pebbleAmp * dome;

        // --- leather colour ---------------------------------------------------
        const mo = mottle(nx, ny, nz) - 0.5;
        const mf = mottleFine(nx, ny, nz) - 0.5;
        // Warm burnt orange: hue ~21°, saturation ~0.82 in albedo, which lands
        // near 0.65 on screen once the key light and the coat have had their
        // say. Safety-orange is a traffic cone; this is tanned leather.
        let r = 190 + mo * 23 + mf * 9 + panelTone[pid];
        let g = 78 + mo * 12 + mf * 5 + panelTone[pid] * 0.55;
        let bl = 22 + mo * 4 - mf * 3 + panelTone[pid] * 0.3;
        let rough = 0.425 + panelRough[pid] + mo * 0.045 - mf * 0.02;
        let ao = 1;

        // The crown of each pebble catches light and wears smooth; the valleys
        // between them hold dirt and read rougher and a shade darker.
        const crown = dome;
        r += (crown - 0.42) * 15;
        g += (crown - 0.42) * 8;
        bl += (crown - 0.42) * 3;
        rough += (0.5 - crown) * 0.17;
        ao -= (1 - crown) * 0.17 * pebMask;

        // --- polish from use ---------------------------------------------------
        // Hands and hardwood only ever touch the raised leather, so the polish
        // rides the pebble crowns and stops dead at the channels.
        const wear = clamp01((wearField(nx, ny, nz) - 0.46) * 3.1) * pebMask * clamp01(crown * 1.6);
        r += wear * 9;
        g += wear * 7;
        bl += wear * 6;
        rough -= wear * 0.115;

        // --- shoulder of the channel -------------------------------------------
        // The leather immediately outside the rib sits in the groove's own
        // occlusion, and that darkening either side is most of what makes a
        // channel read as cut into the cover rather than drawn onto it.
        const shoulder = 1 - smoothstep((t - 0.92) / 1.9);
        r -= shoulder * 14;
        g -= shoulder * 9;
        bl -= shoulder * 4;
        ao -= shoulder * 0.22;
        // The bead of leather squeezed up against the rib is burnished. It is
        // carried almost entirely in roughness, not albedo — a lightened halo
        // beside every channel reads as a painted outline, which is exactly the
        // thing we are trying not to look like.
        rough -= lip * 0.16;
        r += lip * 4;
        g += lip * 3;
        bl += lip * 2;

        // --- the rib itself ------------------------------------------------------
        if (groove > 0.001) {
          // Black rubber fills only the floor of the groove; the walls stay
          // leather, so the channel has a lit side and a shadowed side instead
          // of being one flat black stripe.
          // The rib fills the groove; only its top edge stays leather. Kept off
          // the floor of the sRGB range on purpose — a channel crushed to 0
          // is a black stripe, and the real thing sits around a quarter of the
          // luminance of the leather beside it, never at nothing.
          const grime = clamp01((grimeField(nx, ny, nz) - 0.40) * 2.2);
          const rr = 42 + grime * 21;
          const rg = 37 + grime * 18;
          const rb = 35 + grime * 14;
          const k = smoothstep((groove - 0.02) / 0.22);
          r += (rr - r) * k;
          g += (rg - g) * k;
          bl += (rb - bl) * k;
          // Moulded rubber is *glossier* than pebbled leather — the single most
          // reliable cue that the channel is a different material.
          rough += (0.27 + grime * 0.22 - rough) * k;
          ao += (0.30 + 0.18 * (1 - groove) - ao) * k;
        }

        // --- valve ---------------------------------------------------------------
        const vdot = nx * valve[0] + ny * valve[1] + nz * valve[2];
        if (vdot > valveCosLimit) {
          const dv2 = Math.acos(Math.min(1, vdot)) / valveAng;
          // A shallow moulded dish with the bore at its centre.
          const dish = 1 - smoothstep((dv2 - 1.0) / 1.5);
          const bore = 1 - smoothstep((dv2 - 0.30) / 0.28);
          height[idx] -= CHANNEL_DEPTH * (0.16 * dish + 0.9 * bore);
          r += (62 - r) * bore * 0.92 - dish * 9;
          g += (55 - g) * bore * 0.92 - dish * 5;
          bl += (52 - bl) * bore * 0.92 - dish * 2;
          rough += (0.33 - rough) * bore * 0.9;
          ao -= dish * 0.12 + bore * 0.4;
        }

        A[o] = clamp01(r / 255) * 255;
        A[o + 1] = clamp01(g / 255) * 255;
        A[o + 2] = clamp01(bl / 255) * 255;
        A[o + 3] = 255;

        O[o] = clamp01(ao) * 255;
        O[o + 1] = clamp01(Math.max(0.18, Math.min(0.78, rough))) * 255;
        O[o + 2] = 0;
        O[o + 3] = 255;
      }
    }
  }

  // --- normals from the height field ---------------------------------------
  const normal = new ImageData(W, H);
  const N = normal.data;
  const sc = new Float64Array(2);
  const daPerTexel = 2 / inner;
  for (let face = 0; face < 6; face++) {
    const col = face % ATLAS_COLS;
    const row = (face / ATLAS_COLS) | 0;
    const ox = col * F;
    const oy = row * F;
    for (let ty = 0; ty < F; ty++) {
      const b = ((ty + 0.5 - PAD) / inner) * 2 - 1;
      const y0 = oy + Math.max(0, ty - 1);
      const y1 = oy + Math.min(F - 1, ty + 1);
      for (let tx = 0; tx < F; tx++) {
        const a = ((tx + 0.5 - PAD) / inner) * 2 - 1;
        const x0 = ox + Math.max(0, tx - 1);
        const x1 = ox + Math.min(F - 1, tx + 1);
        faceScale(a, b, sc);
        // Metres of arc spanned by the central difference.
        const arcU = sc[0] * R * daPerTexel * (x1 - x0);
        const arcV = sc[1] * R * daPerTexel * (y1 - y0);
        const gu = (height[(oy + ty) * W + x1] - height[(oy + ty) * W + x0]) / arcU;
        const gv = (height[y1 * W + ox + tx] - height[y0 * W + ox + tx]) / arcV;
        const inv = 1 / Math.sqrt(gu * gu + gv * gv + 1);
        const o = ((oy + ty) * W + ox + tx) * 4;
        N[o] = (-gu * inv * 0.5 + 0.5) * 255;
        N[o + 1] = (-gv * inv * 0.5 + 0.5) * 255;
        N[o + 2] = (inv * 0.5 + 0.5) * 255;
        N[o + 3] = 255;
      }
    }
  }

  // --- mip chains ----------------------------------------------------------
  // Normals first: each level records how much the pebble has averaged out, and
  // that lost detail is pushed into the matching roughness level so a receding
  // ball loses its grain to a broader specular lobe instead of to aliasing.
  const albedoLevels: HTMLCanvasElement[] = [imageDataToCanvas(albedo)];
  const normalLevels: HTMLCanvasElement[] = [imageDataToCanvas(normal)];
  const ormLevels: HTMLCanvasElement[] = [imageDataToCanvas(orm)];

  let aCur = albedo;
  let nCur = normal;
  let oCur = orm;
  let cw = W;
  let ch = H;
  let cell = F;
  while (cell > 1) {
    const an = halveAtlas(aCur, cw, ch);
    const nn = halveAtlas(nCur, cw, ch);
    const on = halveAtlas(oCur, cw, ch);
    cw = an.w;
    ch = an.h;
    cell >>= 1;

    // Re-normalise the averaged normal and remember its shortened length.
    const nd = nn.data.data;
    const varLevel = new Float32Array(cw * ch);
    for (let i = 0, p = 0; i < nd.length; i += 4, p++) {
      const vx = (nd[i] / 255) * 2 - 1;
      const vy = (nd[i + 1] / 255) * 2 - 1;
      const vz = (nd[i + 2] / 255) * 2 - 1;
      const len = Math.sqrt(vx * vx + vy * vy + vz * vz) || 1;
      varLevel[p] = clamp01(1 - len);
      const inv = 1 / len;
      nd[i] = (vx * inv * 0.5 + 0.5) * 255;
      nd[i + 1] = (vy * inv * 0.5 + 0.5) * 255;
      nd[i + 2] = (vz * inv * 0.5 + 0.5) * 255;
    }
    const od = on.data.data;
    for (let i = 0, p = 0; i < od.length; i += 4, p++) {
      const rr = od[i + 1] / 255;
      od[i + 1] = clamp01(Math.min(0.82, Math.sqrt(rr * rr + 2.1 * varLevel[p]))) * 255;
    }

    albedoLevels.push(imageDataToCanvas(an.data));
    normalLevels.push(imageDataToCanvas(nn.data));
    ormLevels.push(imageDataToCanvas(on.data));
    aCur = an.data;
    nCur = nn.data;
    oCur = on.data;
  }

  const aniso = opts.anisotropy ?? 8;
  const bytes = W * H * 4 * 3 * 1.34;

  const maps: BallMapSet = {
    albedo: finish(albedoLevels, true, aniso),
    normal: finish(normalLevels, false, aniso),
    orm: finish(ormLevels, false, aniso),
    cell: F,
    megabytes: Math.round((bytes / (1024 * 1024)) * 10) / 10,
    dispose() {
      this.albedo.dispose();
      this.normal.dispose();
      this.orm.dispose();
    },
  };
  return maps;
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/**
 * A cube sphere with the same equal-angle warp the bake uses, and UVs that land
 * each face inside its padded atlas cell.
 *
 * Compared with a 64x48 UV sphere at similar cost this has no pole fan, no
 * meridian seam, near-uniform triangle area, and — because every face is its
 * own vertex grid — a clean, continuous tangent frame per face for the normal
 * map. `segments` is the quad count along a face edge; the silhouette gets
 * `4 * segments` chords around the equator.
 */
export function makeBallGeometry(radius: number, segments: number, cell: number): BufferGeometry {
  const n = Math.max(2, segments | 0);
  const PAD = Math.max(4, cell >> 5);
  const inner = cell - 2 * PAD;
  const vertsPerFace = (n + 1) * (n + 1);

  const pos = new Float32Array(6 * vertsPerFace * 3);
  const nor = new Float32Array(6 * vertsPerFace * 3);
  const uv = new Float32Array(6 * vertsPerFace * 2);
  const idx = new Uint16Array(6 * n * n * 6);

  const dir = new Float64Array(3);
  let vi = 0;
  let ii = 0;

  for (let face = 0; face < 6; face++) {
    const base = face * vertsPerFace;
    const col = face % ATLAS_COLS;
    const row = (face / ATLAS_COLS) | 0;

    for (let j = 0; j <= n; j++) {
      const b = (j / n) * 2 - 1;
      // Texel coordinate of this parameter inside the padded cell.
      const vy = row * cell + PAD + ((b + 1) / 2) * inner;
      for (let i = 0; i <= n; i++) {
        const a = (i / n) * 2 - 1;
        const vx = col * cell + PAD + ((a + 1) / 2) * inner;
        faceDir(face, a, b, dir);
        pos[vi * 3] = dir[0] * radius;
        pos[vi * 3 + 1] = dir[1] * radius;
        pos[vi * 3 + 2] = dir[2] * radius;
        nor[vi * 3] = dir[0];
        nor[vi * 3 + 1] = dir[1];
        nor[vi * 3 + 2] = dir[2];
        uv[vi * 2] = vx / (ATLAS_COLS * cell);
        uv[vi * 2 + 1] = vy / (ATLAS_ROWS * cell);
        vi++;
      }
    }

    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const v00 = base + j * (n + 1) + i;
        const v10 = v00 + 1;
        const v01 = v00 + (n + 1);
        const v11 = v01 + 1;
        // Wound so the face normal points outward for this basis.
        idx[ii++] = v00;
        idx[ii++] = v01;
        idx[ii++] = v11;
        idx[ii++] = v00;
        idx[ii++] = v11;
        idx[ii++] = v10;
      }
    }
  }

  const geo = new BufferGeometry();
  geo.setAttribute('position', new Float32BufferAttribute(pos, 3));
  geo.setAttribute('normal', new Float32BufferAttribute(nor, 3));
  geo.setAttribute('uv', new Float32BufferAttribute(uv, 2));
  geo.setIndex(new Uint16BufferAttribute(idx, 1));
  geo.computeBoundingSphere();
  return geo;
}
