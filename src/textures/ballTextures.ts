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
 *    The channels are *not* filtered that way. A 6 mm rib crossing a 6 mm texel
 *    lands split across two of them, and a box filter answers that by making it
 *    both wider and shallower — which is exactly how the round-0 ball lost its
 *    ribs, from 37% of adjacent leather at macro to 61–72% at the distance the
 *    ball spends most of its screen time. So the cover is baked as two layers,
 *    leather and rib coverage, and the rib is rebuilt at every mip level from a
 *    contrast-restored coverage rather than carried down as pixels.
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
import { clamp01, lerp, makeRng, smoothstep, smootherstep, valueNoise2 } from '../core/MathX';

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

/**
 * 3D value noise on the surface direction. `valueNoise2` is face-local, so a
 * field built on it is discontinuous across a cube edge; at the amplitudes the
 * leather grain needs that would show as a six-panel patchwork. This is the
 * same trilinear value noise evaluated on the *direction*, so it is continuous
 * everywhere on the sphere, and it is integer hashing rather than a sum of
 * plane waves, so it stays affordable at wavelengths measured in millimetres.
 */
function hash3(x: number, y: number, z: number, seed: number): number {
  let h = Math.imul(x | 0, 374761393) ^ Math.imul(y | 0, 668265263) ^ Math.imul(z | 0, 2147483647);
  h ^= Math.imul(seed | 0, 2246822519);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function valueNoise3(x: number, y: number, z: number, seed: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const zi = Math.floor(z);
  const u = smootherstep(x - xi);
  const v = smootherstep(y - yi);
  const w = smootherstep(z - zi);
  const c000 = hash3(xi, yi, zi, seed);
  const c100 = hash3(xi + 1, yi, zi, seed);
  const c010 = hash3(xi, yi + 1, zi, seed);
  const c110 = hash3(xi + 1, yi + 1, zi, seed);
  const c001 = hash3(xi, yi, zi + 1, seed);
  const c101 = hash3(xi + 1, yi, zi + 1, seed);
  const c011 = hash3(xi, yi + 1, zi + 1, seed);
  const c111 = hash3(xi + 1, yi + 1, zi + 1, seed);
  return lerp(
    lerp(lerp(c000, c100, u), lerp(c010, c110, u), v),
    lerp(lerp(c001, c101, u), lerp(c011, c111, u), v),
    w,
  );
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
const PEBBLE_HEIGHT = 0.00040;
/**
 * Wavelengths of the two grain-clumping octaves, in metres. The 1.9 mm pebble
 * is sub-pixel at every framing the rubric measures (0.8 px at RIM, 0.44 px at
 * FLOOR), so on its own it band-limits into a roughness constant before it can
 * ever stipple anything. Real moulded pebble is not a perfect lattice: its
 * density and depth clump over several pitches, and *that* band is what carries
 * §4.2's "grainy and irregular" specular to the distances the ball is actually
 * seen at. 4.4 mm is ~1.9 px at RIM and ~1.0 px at FLOOR, which is exactly the
 * decay §4.2 describes.
 */
const GRAIN_COARSE = 0.0044;
const GRAIN_FINE = 0.0024;
/** Inflation valve — small, but it is the only asymmetry on the whole cover. */
const VALVE_RADIUS = 0.0042;

/**
 * Colour of the moulded rubber rib in the *mip chain*, sRGB — the mean of the
 * grime-varied value level 0 uses (44–64).
 *
 * It is the mean and not a darker bias, and that is a measured decision rather
 * than an assumption. §4.3's remaining miss is that the rib reads at 21% of
 * adjacent leather at macro and 68% at RIM, and the reviewer's suggested lever
 * for it was to "bias the channel's albedo/AO so it survives 3–4 mip levels".
 * Tried: dropping this to (26, 23, 21) took macro from 21% to 17% and moved RIM
 * from 68% to 70% — i.e. all of the cost and none of the gain. The rib in the
 * *texture* is already deep enough; what limits it on screen is that a 6 mm
 * channel is 1.7 px at RIM and loses roughly two thirds of its contrast to TAA,
 * the resolve pass and the additive specular floor sitting on top of it. The
 * lever that would work is a smaller specular floor, which §4.2 and §4.4's
 * annulus both need to stay large.
 */
const RIB_RGB: readonly [number, number, number] = [54, 48, 45];
const RIB_ROUGH = 0.37;
const RIB_AO = 0.31;

/**
 * Base leather albedo, sRGB. §4.4 asks for hue 22–30°, saturation 62–78% and
 * value 130–175 *on screen*, and the display chain between here and there is
 * not hue-preserving: ACES plus the grade's highlight desaturation and its warm
 * R − B ceiling both pull a saturated orange toward red as they compress it.
 * This is therefore the pre-image of the wanted screen colour under that chain,
 * solved against a measured frame, not the screen colour itself.
 */
const LEATHER_R = 248;
const LEATHER_G = 140;
const LEATHER_B = 4;

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
  // The two layers the channel is composited from. Keeping them apart is what
  // lets the mip chain rebuild the rib at every level instead of box-filtering
  // it into a grey smear — see the re-composite after the loop.
  const leatherLayer = new ImageData(W, H);
  /** R = rib coverage, G = leather AO, B = leather roughness. */
  const maskLayer = new ImageData(W, H);
  const height = new Float32Array(W * H);
  const A = albedo.data;
  const O = orm.data;
  const LA = leatherLayer.data;
  const MK = maskLayer.data;

  // --- angular feature sizes ------------------------------------------------
  const halfChannel = CHANNEL_WIDTH * 0.5 / R;
  // Face parameter units per pebble pitch. The centre of a face runs at
  // π/4 radians per unit `a`, which is where this is calibrated.
  const pitchA = PEBBLE_PITCH / R / QUARTER_PI;
  const texelsPerPitch = (pitchA * 0.5) * inner;
  // The pitch is *physical at every tier*. §4.2 fixes it at 1.6–2.2 mm and
  // opening it up to fit the texel budget is precisely the golf-ball failure
  // the same section names — at `low` the old guard stretched it to 5.0 mm.
  // What scales with the budget instead is amplitude: under about two texels
  // per pitch the lattice is below Nyquist and has to fade into a roughness
  // constant, which is the same thing the mip chain does as the ball recedes.
  const pebbleK = 1 / pitchA;
  const pebbleFade = clamp01((texelsPerPitch - 1.9) / 1.5);
  const pebbleAmp = PEBBLE_HEIGHT * pebbleFade;
  // Box-filter the lattice over exactly one texel. Near the Nyquist limit that
  // needs more taps, or the supersample itself beats against the grid.
  const SS = texelsPerPitch >= 3.4 ? 2 : 4;
  const ssStep = 1 / (texelsPerPitch * SS);
  const ssBase = -0.5 / texelsPerPitch + ssStep * 0.5;

  // --- surface variation ----------------------------------------------------
  const mottle = sphereNoise(seed + 11, 6, 2.2, 5.5);
  const mottleFine = sphereNoise(seed + 29, 5, 9, 17);
  const wearField = sphereNoise(seed + 47, 5, 2.6, 6.0);
  const grimeField = sphereNoise(seed + 71, 4, 3.5, 8.0);
  // Millimetre-scale grain clumping. Direction-space frequency: one lattice
  // cell spans `R / GRAIN` radians, i.e. exactly one wavelength of arc.
  const grainKa = R / GRAIN_COARSE;
  const grainKb = R / GRAIN_FINE;

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
        // A bead of leather squeezed up either side of the rib. This is what
        // produces the bright specular lip along the channel, and it has to be
        // a real ridge — 0.14 mm of it read as nothing.
        const lip =
          Math.exp(-((t - 1.18) * (t - 1.18)) / 0.24) * smoothstep((t - 0.74) / 0.58);
        const pebMask = smoothstep((t - 0.96) / 0.42);

        // --- pebble ---------------------------------------------------------
        // Three cosines 60° apart give a perfect hexagonal dome lattice; a
        // low-frequency domain warp breaks the regularity so it reads as
        // moulded grain rather than as a screen pattern.
        let dome = 0;
        if (pebMask > 0.001 && pebbleFade > 0.001) {
          // Two warp octaves. The slow one (≈21 mm) shears whole regions of
          // lattice; the fast one (≈5 mm, a couple of pitches) is what stops
          // the neighbourhood of any one pebble from reading as a screen
          // pattern, which one octave on its own does not.
          const wx =
            (valueNoise2(a * 9.0, b * 9.0, seed + 3) - 0.5) * 0.55 +
            (valueNoise2(a * 38.0, b * 38.0, seed + 17) - 0.5) * 0.42;
          const wy =
            (valueNoise2(a * 9.0 + 31, b * 9.0 - 17, seed + 5) - 0.5) * 0.55 +
            (valueNoise2(a * 38.0 - 11, b * 38.0 + 23, seed + 19) - 0.5) * 0.42;
          const p0 = (a + wx * pitchA * 2.2) * pebbleK;
          const q0 = (b + wy * pitchA * 2.2) * pebbleK;
          let acc = 0;
          for (let sy = 0; sy < SS; sy++) {
            const q = q0 + ssBase + sy * ssStep;
            for (let sx = 0; sx < SS; sx++) {
              const p = p0 + ssBase + sx * ssStep;
              const c1 = Math.cos(6.2831853 * p);
              const c2 = Math.cos(6.2831853 * (p * 0.5 + q * SQRT3_2));
              const c3 = Math.cos(6.2831853 * (-p * 0.5 + q * SQRT3_2));
              const hx = clamp01(((c1 + c2 + c3) / 3 + 0.5) / 1.5);
              acc += smoothstep((hx - 0.30) / 0.70);
            }
          }
          // Height scatter between neighbouring pebbles.
          const vary = 0.72 + 0.56 * valueNoise2(p0 * 0.85, q0 * 0.85, seed + 13);
          dome = Math.sqrt(acc / (SS * SS)) * vary * pebMask;
        }

        // --- grain clumping ---------------------------------------------------
        // Two octaves at 4.4 mm and 2.4 mm. The pebble lattice itself is
        // sub-pixel at every framing the rubric measures, so this is the band
        // that actually reaches the screen: it stipples the specular at RIM and
        // has decayed to about a unit by FLOOR, which is §4.2's whole effect.
        const clump =
          (valueNoise3(nx * grainKa, ny * grainKa, nz * grainKa, seed + 211) - 0.5) * 0.66 +
          (valueNoise3(nx * grainKb, ny * grainKb, nz * grainKb, seed + 307) - 0.5) * 0.34;

        height[idx] =
          -CHANNEL_DEPTH * groove +
          CHANNEL_DEPTH * 0.36 * lip +
          pebbleAmp * dome +
          clump * 0.00021 * pebMask;

        // --- leather colour ---------------------------------------------------
        const mo = mottle(nx, ny, nz) - 0.5;
        const mf = mottleFine(nx, ny, nz) - 0.5;
        let r = LEATHER_R + mo * 17 + mf * 7 + panelTone[pid];
        let g = LEATHER_G + mo * 13 + mf * 6 + panelTone[pid] * 0.62;
        let bl = LEATHER_B + mo * 5 - mf * 2 + panelTone[pid] * 0.42;
        let rough = 0.40 + panelRough[pid] + mo * 0.045 - mf * 0.02;
        let ao = 1;

        // The crown of each pebble catches light and wears smooth; the valleys
        // between them hold dirt and read rougher and a shade darker. Fading
        // the *deviation* rather than the field means an under-sampled tier
        // collapses to the lattice's mean instead of to a moiré.
        const crown = 0.42 + (dome - 0.42) * pebbleFade;
        r += (crown - 0.42) * 12;
        g += (crown - 0.42) * 7;
        bl += (crown - 0.42) * 3;
        rough += (0.5 - crown) * 0.22;
        ao -= (1 - crown) * 0.17 * pebMask;

        // Grain clumping rides mostly in roughness — §4.2 is explicit that the
        // albedo stays smooth and the *sheen* is what breaks up.
        rough += clump * 0.70 * pebMask;
        r += clump * 8.5 * pebMask;
        g += clump * 5.0 * pebMask;
        bl += clump * 2.2 * pebMask;

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
        // The burnished bead beside the rib. Round 0 carried it almost entirely
        // in roughness, and with no specular lobe to modulate it measured 1.03x
        // the adjacent leather instead of §4.3's 1.3–1.6x. Roughness still does
        // most of the work — it is what makes the lip swap sides with the key —
        // but enough now sits in albedo and in the ridge's own normal that the
        // rib reads as moulded at any lighting angle.
        rough -= lip * 0.42;
        r += lip * 58;
        g += lip * 34;
        bl += lip * 14;
        ao += lip * 0.12;

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

        const lr = clamp01(r / 255) * 255;
        const lg = clamp01(g / 255) * 255;
        const lb = clamp01(bl / 255) * 255;
        const lrough = clamp01(Math.max(0.14, Math.min(0.78, rough)));
        const lao = clamp01(ao);

        // --- the rib itself ------------------------------------------------------
        // Black rubber fills only the floor of the groove; the walls stay
        // leather, so the channel has a lit side and a shadowed side instead of
        // being one flat black stripe. Kept off the floor of the sRGB range on
        // purpose — a channel crushed to 0 is a stripe, and the real thing sits
        // around a quarter of the luminance of the leather beside it.
        const cov = groove > 0.001 ? smoothstep((groove - 0.02) / 0.22) : 0;
        let fr = lr;
        let fg = lg;
        let fb = lb;
        let frough = lrough;
        let fao = lao;
        if (cov > 0.001) {
          const grime = clamp01((grimeField(nx, ny, nz) - 0.40) * 2.2);
          fr += (44 + grime * 20 - lr) * cov;
          fg += (39 + grime * 17 - lg) * cov;
          fb += (36 + grime * 14 - lb) * cov;
          // Moulded rubber is *glossier* than pebbled leather — the single most
          // reliable cue that the channel is a different material.
          frough += (0.27 + grime * 0.22 - lrough) * cov;
          fao += (0.30 + 0.18 * (1 - groove) - lao) * cov;
        }

        A[o] = fr;
        A[o + 1] = fg;
        A[o + 2] = fb;
        A[o + 3] = 255;

        O[o] = clamp01(fao) * 255;
        O[o + 1] = clamp01(Math.max(0.14, Math.min(0.78, frough))) * 255;
        O[o + 2] = 0;
        O[o + 3] = 255;

        LA[o] = lr;
        LA[o + 1] = lg;
        LA[o + 2] = lb;
        LA[o + 3] = 255;

        MK[o] = cov * 255;
        MK[o + 1] = lao * 255;
        MK[o + 2] = lrough * 255;
        MK[o + 3] = 255;
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
  //
  // Albedo and ORM are *not* box-filtered directly. A 6 mm rib crossing a
  // 6.2 mm texel splits across two texels, and averaging leather into both is
  // what took the channel core from 37% of adjacent leather at macro to 61–72%
  // at gameplay distance — simultaneously too wide and too shallow, which is
  // exactly what a box filter does to a thin dark line. Instead the leather
  // layer and the rib coverage are filtered separately and the rib is
  // *rebuilt* at every level from a contrast-restored coverage. The unsharp
  // term is what puts the depth back without widening the line: it lifts the
  // core, where coverage exceeds its neighbourhood, and trims the skirt, where
  // it does not.
  const albedoLevels: HTMLCanvasElement[] = [imageDataToCanvas(albedo)];
  const normalLevels: HTMLCanvasElement[] = [imageDataToCanvas(normal)];
  const ormLevels: HTMLCanvasElement[] = [imageDataToCanvas(orm)];

  let nCur = normal;
  let lCur = leatherLayer;
  let mCur = maskLayer;
  let cw = W;
  let ch = H;
  let cell = F;
  /** How hard the coverage contrast is restored. 0 reproduces a box filter. */
  const COV_SHARPEN = 3.0;
  while (cell > 1) {
    const nn = halveAtlas(nCur, cw, ch);
    const ln = halveAtlas(lCur, cw, ch);
    const mn = halveAtlas(mCur, cw, ch);
    cw = ln.w;
    ch = ln.h;
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

    // Contrast-restored coverage, blurred strictly inside each atlas cell.
    const md = mn.data.data;
    const sharp = new Float32Array(cw * ch);
    for (let ty = 0; ty < ch; ty++) {
      const cy0 = ty - (ty % cell);
      for (let tx = 0; tx < cw; tx++) {
        const cx0 = tx - (tx % cell);
        let sum = 0;
        let n = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = ty + dy;
          if (yy < cy0 || yy >= cy0 + cell) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = tx + dx;
            if (xx < cx0 || xx >= cx0 + cell) continue;
            sum += md[(yy * cw + xx) * 4];
            n++;
          }
        }
        const p = ty * cw + tx;
        const c = md[p * 4] / 255;
        sharp[p] = clamp01(c + COV_SHARPEN * (c - sum / (n * 255)));
      }
    }

    const lv = new ImageData(cw, ch);
    const ov = new ImageData(cw, ch);
    const ld = ln.data.data;
    const lo = lv.data;
    const oo = ov.data;
    for (let i = 0, p = 0; i < ld.length; i += 4, p++) {
      const k = sharp[p];
      lo[i] = ld[i] + (RIB_RGB[0] - ld[i]) * k;
      lo[i + 1] = ld[i + 1] + (RIB_RGB[1] - ld[i + 1]) * k;
      lo[i + 2] = ld[i + 2] + (RIB_RGB[2] - ld[i + 2]) * k;
      lo[i + 3] = 255;

      const lAo = md[i + 1] / 255;
      const lRough = md[i + 2] / 255;
      oo[i] = clamp01(lAo + (RIB_AO - lAo) * k) * 255;
      // Toksvig: the pebble that averaged out of the normal comes back as a
      // broader lobe. The coefficient and the ceiling were 2.1 and 0.82, which
      // drove a channel wall straight to the cap by mip 1 and left the ball
      // with no specular lobe at all to modulate — the pebble and the channel
      // lip are both downstream of that.
      const rr = lRough + (RIB_ROUGH - lRough) * k;
      oo[i + 1] = clamp01(Math.min(0.68, Math.sqrt(rr * rr + 1.0 * varLevel[p]))) * 255;
      oo[i + 2] = 0;
      oo[i + 3] = 255;
    }

    albedoLevels.push(imageDataToCanvas(lv));
    normalLevels.push(imageDataToCanvas(nn.data));
    ormLevels.push(imageDataToCanvas(ov));
    nCur = nn.data;
    lCur = ln.data;
    mCur = mn.data;
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
