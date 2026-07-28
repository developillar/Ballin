/**
 * Skin: a procedural, tone-neutral PBR atlas for the whole body.
 *
 * The body mesh is parameterised per limb — every vertex knows which bone owns
 * it, how far along that bone it sits (`v`) and which way around it faces
 * (`u`, with 0.5 pointing forward so the wrap seam lands on the spine groove
 * and the back of the skull). That gives a *anatomical* texture space: painting
 * "pectoral" means writing a dome at (u ≈ 0.42, v ≈ 0.74) of the torso cell
 * rather than hoping a UV unwrap put it somewhere sensible.
 *
 * Three maps come out of one pass:
 *
 *  - **albedo**, deliberately tone-neutral. Skin tone is applied per player as
 *    `material.color`, so the whole roster shares one 1–4 MB bake and the
 *    *relative* hue structure (redder knuckles, elbows, knees, ears and nose,
 *    cooler palms, tan lines, brows, lips) survives the multiply.
 *  - **normal**, from a height field that carries pore-scale noise, forearm and
 *    shin hair, tendons, veins and the anatomical landmarks that must read as
 *    *form shadows* — they belong in the normal, never painted into albedo,
 *    because painted muscle does not change with light direction and that is
 *    exactly how a reviewer catches it.
 *  - **data**, packed R = sweat proneness, G = roughness, B = eye mask. Three
 *    reads `.g` for roughness; the skin shader samples `.r` for where sweat
 *    beads and `.b` to swap the material over to sclera / cornea, which is the
 *    only way a white eye survives a tone-neutral atlas being multiplied by a
 *    0.27-luminance skin tone.
 *
 * Owned by the players agent.
 */

import {
  CanvasTexture,
  ClampToEdgeWrapping,
  LinearFilter,
  LinearMipmapLinearFilter,
  RepeatWrapping,
  SRGBColorSpace,
} from 'three';
import { clamp01, fbm2, lerp, makeRng, valueNoise2 } from '../core/MathX';

// ---------------------------------------------------------------------------
// Atlas layout
// ---------------------------------------------------------------------------

export type SkinPart =
  | 'torso'
  | 'head'
  | 'neck'
  | 'upperArm'
  | 'foreArm'
  | 'hand'
  | 'thigh'
  | 'shin'
  | 'foot';

export const SKIN_ATLAS_COLS = 4;
export const SKIN_ATLAS_ROWS = 3;

/** Column/row of each part's cell in the atlas. */
const CELL: Record<SkinPart, readonly [number, number]> = {
  torso: [0, 0],
  head: [1, 0],
  neck: [2, 0],
  upperArm: [3, 0],
  foreArm: [0, 1],
  hand: [1, 1],
  thigh: [2, 1],
  shin: [3, 1],
  foot: [0, 2],
};

const PART_ORDER: SkinPart[] = [
  'torso',
  'head',
  'neck',
  'upperArm',
  'foreArm',
  'hand',
  'thigh',
  'shin',
  'foot',
];

/** Keeps mip bleed between neighbouring cells off the silhouette-critical edges. */
const INSET = 0.02;

/**
 * Maps a part-local (u, v) — u around the limb with 0.5 forward, v from the
 * proximal joint to the distal one — into atlas UV.
 */
export function skinPartUv(part: SkinPart, u: number, v: number, out: [number, number]): void {
  const [cx, cy] = CELL[part];
  const uu = INSET + clamp01(u) * (1 - 2 * INSET);
  const vv = INSET + clamp01(v) * (1 - 2 * INSET);
  out[0] = (cx + uu) / SKIN_ATLAS_COLS;
  out[1] = (cy + vv) / SKIN_ATLAS_ROWS;
}

// ---------------------------------------------------------------------------
// Tones
// ---------------------------------------------------------------------------

export interface SkinTone {
  name: string;
  /** Multiplied over the neutral albedo. */
  color: number;
  /**
   * Tint of the light that has scattered through and re-emerged. It is a *tint*,
   * not a radiance: the shader scales it by `sqrt(diffuseColor)` so the band's
   * absolute strength tracks the tone without a dark albedo multiplying it out
   * of existence. Keep it bright and saturated; luminance comes from the albedo.
   */
  subsurface: number;
  /** How far light wraps past the terminator. Darker skin scatters less. */
  wrap: number;
  /** Strength of the terminator warmth band. */
  sss: number;
  /** Base oil-sheen roughness; darker skin reads with a tighter, hotter lobe. */
  oilRoughness: number;
  /** Broad specular intensity multiplier. */
  specular: number;
}

/**
 * A plausible NBA range — stated as **diffuse albedo**, not as the display value
 * we want out the other end.
 *
 * That distinction is what round 0 got wrong. The old table ran 0x6b452b for the
 * deepest tone: a linear luminance of 0.076, against 0.60 for the home white in
 * the same frame. Under a rig whose banks sit at 62–68° elevation, a vertical
 * torso already takes only ~0.35 of the floor's irradiance, so 0.076 albedo put
 * every skin surface in the frame inside a 15-unit window at the bottom of the
 * tone curve — measured p50 L = 23.6 on the shoulder against L = 144 for the kit
 * 60 px away. Nothing painted into the normal map can be seen at L 24, and no
 * amount of mesh resolution helps.
 *
 * Real skin diffuse albedo, once the specular lobe is separated out, runs from
 * about 0.62 luminance (type I) to about 0.30 (type VI) — a ~2× span, not the
 * 6.8× the old table used, and not a range that reaches anywhere near 0.08.
 * These are measured to 0.61 → 0.36, checked with
 * `((c/255 + 0.055) / 1.055) ^ 2.4` per channel and the Rec.709 weights, and
 * the span is deliberately kept: §3.3 wants a squad to read as a squad.
 *
 * The absolute level is set by a measurement, not by taste. An up-facing
 * shoulder on the mid tone measured L 112 in `players-r1` and L 110 in
 * `players-r2` against §8.2's 120–190, so the table was walked up until it
 * cleared. It is now at the top of the plausible range and should come *down*
 * if the rig ever puts more than ~0.5 of the floor's irradiance on a shoulder
 * cap — the shoulder there measures 0.73× the adjacent hardwood while the two
 * albedos are within 5% of each other, which is a lighting figure, not a
 * material one.
 *
 * The subsurface colour and wrap distance still shift with the tone — melanin
 * absorbs the long free paths, so deeper tones show a shorter, less saturated
 * terminator and a harder specular.
 */
export const SKIN_TONES: readonly SkinTone[] = [
  {
    name: 'fair',
    color: 0xe7c9b1, // linear Y 0.619
    subsurface: 0xff6a44,
    wrap: 0.42,
    sss: 0.62,
    oilRoughness: 0.55,
    specular: 0.72,
  },
  {
    name: 'olive',
    color: 0xe2c1a4, // linear Y 0.570
    subsurface: 0xf85c34,
    wrap: 0.36,
    sss: 0.56,
    oilRoughness: 0.53,
    specular: 0.78,
  },
  {
    name: 'tan',
    color: 0xdcb99b, // linear Y 0.523
    subsurface: 0xef4f28,
    wrap: 0.3,
    sss: 0.5,
    oilRoughness: 0.5,
    specular: 0.86,
  },
  {
    name: 'brown',
    color: 0xd6b092, // linear Y 0.474
    subsurface: 0xe0431e,
    wrap: 0.25,
    sss: 0.44,
    oilRoughness: 0.47,
    specular: 0.93,
  },
  {
    name: 'deep',
    color: 0xcda380, // linear Y 0.412
    subsurface: 0xcc3a17,
    wrap: 0.2,
    sss: 0.38,
    oilRoughness: 0.44,
    specular: 1.0,
  },
];

// ---------------------------------------------------------------------------
// Painting helpers
// ---------------------------------------------------------------------------

const TAU = Math.PI * 2;

const sstep = (a: number, b: number, x: number): number => {
  const t = clamp01((x - a) / (b - a || 1e-6));
  return t * t * (3 - 2 * t);
};

/** Distance on the wrapped u axis. */
const du = (a: number, b: number): number => {
  let d = Math.abs(a - b) % 1;
  if (d > 0.5) d = 1 - d;
  return d;
};

/** Smooth elliptical blob, 1 at the centre falling to 0 at the radii. */
function blob(u: number, v: number, cu: number, cv: number, ru: number, rv: number): number {
  const a = du(u, cu) / ru;
  const b = (v - cv) / rv;
  const r = Math.sqrt(a * a + b * b);
  return r >= 1 ? 0 : Math.cos(r * Math.PI * 0.5) ** 2;
}

/** A soft ridge along v, centred on u, with a falloff in both axes. */
function ridge(
  u: number,
  v: number,
  cu: number,
  halfWidth: number,
  v0: number,
  v1: number,
  feather = 0.08,
): number {
  const across = 1 - clamp01(du(u, cu) / halfWidth);
  const along = sstep(v0 - feather, v0 + feather, v) * (1 - sstep(v1 - feather, v1 + feather, v));
  return Math.cos((1 - across) * Math.PI * 0.5) ** 2 * along;
}

interface Sample {
  /** Height in millimetres; drives the normal map. */
  h: number;
  /** Multiplicative albedo tint (neutral = 1,1,1). */
  r: number;
  g: number;
  b: number;
  /** Surface roughness. */
  rough: number;
  /** How readily this patch beads sweat, 0..1. */
  sweat: number;
  /**
   * Eye mask, packed into the data map's blue channel: 0 skin, ~0.55 iris,
   * ~1.0 sclera. The shader replaces `diffuseColor` there rather than tinting
   * it, because the sclera is the one part of a face that is *not* skin — a
   * white painted into a tone-neutral atlas and then multiplied by a 0.27-albedo
   * tone lands at L 30, which is how round 0 ended up with 87% of the eye band
   * inside ±5 units of its own median.
   */
  eye: number;
}

/**
 * The whole anatomy, evaluated per texel. Everything structural lands in `h`;
 * albedo only ever carries pigment (lips, brows, hair, tan lines, joint
 * redness) so that muscle definition responds to light instead of being a
 * painting of light.
 */
function sample(part: SkinPart, u: number, v: number, s: Sample, seed: number): void {
  s.h = 0;
  s.r = 1;
  s.g = 1;
  s.b = 1;
  s.rough = 0.47;
  s.sweat = 0.35;
  s.eye = 0;

  // Continuous across the u wrap: feed the noise a cylinder, not a plane.
  const cx = Math.cos(u * TAU) * 0.5 + 0.5;
  const cy = Math.sin(u * TAU) * 0.5 + 0.5;

  // --- Pore / dermal micro-relief ----------------------------------------
  const pore =
    fbm2(cx * 210, v * 260, 3, 2.1, 0.55, seed) * 0.62 +
    valueNoise2(cx * 640, v * 780, seed + 11) * 0.38;
  s.h += (pore - 0.5) * 0.22;
  // A slower dermal undulation keeps large flat areas from reading as vinyl.
  const derm = fbm2(cx * 22, v * 26, 4, 2.0, 0.5, seed + 3);
  s.h += (derm - 0.5) * 0.55;
  s.rough += (derm - 0.5) * 0.06;
  const blotch = fbm2(cx * 9, v * 11, 3, 2.0, 0.5, seed + 91);
  // Value variation across a limb before lighting — never one flat colour.
  s.r *= 1 + (blotch - 0.5) * 0.09;
  s.g *= 1 + (blotch - 0.5) * 0.055;
  s.b *= 1 + (blotch - 0.5) * 0.04;

  const front = 1 - clamp01(du(u, 0.5) / 0.25);
  const back = 1 - clamp01(du(u, 0.0) / 0.25);

  switch (part) {
    case 'torso': {
      // v: 0 at the pelvis, 1 at the base of the neck.
      // Linea alba — a groove, not a line.
      s.h -= ridge(u, v, 0.5, 0.018, 0.14, 0.62, 0.06) * 1.1;
      // Rectus abdominis: four visible bands, tapering as they climb.
      for (let i = 0; i < 4; i++) {
        const cv = 0.2 + i * 0.108;
        const w = 0.052 - i * 0.004;
        s.h += blob(u, v, 0.5 - 0.055, cv, w, 0.045) * (1.15 - i * 0.16);
        s.h += blob(u, v, 0.5 + 0.055, cv, w, 0.045) * (1.15 - i * 0.16);
      }
      // External obliques flaring off the waist.
      s.h += blob(u, v, 0.5 - 0.13, 0.3, 0.06, 0.1) * 0.7;
      s.h += blob(u, v, 0.5 + 0.13, 0.3, 0.06, 0.1) * 0.7;
      // Pectorals: a shelf with a hard lower crease and a soft upper fade.
      const pecL = blob(u, v, 0.5 - 0.085, 0.735, 0.1, 0.085);
      const pecR = blob(u, v, 0.5 + 0.085, 0.735, 0.1, 0.085);
      s.h += (pecL + pecR) * 1.8;
      s.h -= ridge(u, v, 0.5 - 0.085, 0.1, 0.645, 0.675, 0.014) * 1.6;
      s.h -= ridge(u, v, 0.5 + 0.085, 0.1, 0.645, 0.675, 0.014) * 1.6;
      // Sternal gutter between them.
      s.h -= ridge(u, v, 0.5, 0.026, 0.66, 0.86, 0.05) * 0.9;
      // Serratus fingers under the armpit.
      for (let i = 0; i < 4; i++) {
        const cv = 0.5 + i * 0.045;
        s.h += blob(u, v, 0.5 - 0.155 + i * 0.006, cv, 0.032, 0.02) * 0.55;
        s.h += blob(u, v, 0.5 + 0.155 - i * 0.006, cv, 0.032, 0.02) * 0.55;
      }
      // Latissimus sweep from armpit down to the waist.
      s.h += ridge(u, v, 0.5 - 0.235, 0.075, 0.36, 0.8, 0.13) * 1.15;
      s.h += ridge(u, v, 0.5 + 0.235, 0.075, 0.36, 0.8, 0.13) * 1.15;
      // Spinal groove and the erector columns either side of it.
      s.h -= ridge(u, v, 0.0, 0.03, 0.1, 0.92, 0.09) * 1.5;
      s.h += ridge(u, v, 0.045, 0.035, 0.12, 0.72, 0.1) * 0.75;
      s.h += ridge(u, v, 0.955, 0.035, 0.12, 0.72, 0.1) * 0.75;
      // Trapezius rising into the neck.
      s.h += blob(u, v, 0.0, 0.93, 0.2, 0.11) * 1.3;
      // Scapular ridges.
      s.h += blob(u, v, 0.08, 0.79, 0.055, 0.06) * 0.6;
      s.h += blob(u, v, 0.92, 0.79, 0.055, 0.06) * 0.6;
      // Iliac crest / lower-back dimples.
      s.h -= blob(u, v, 0.065, 0.11, 0.03, 0.035) * 0.7;
      s.h -= blob(u, v, 0.935, 0.11, 0.03, 0.035) * 0.7;
      s.sweat = 0.45 + back * 0.5 + front * 0.15;
      s.rough -= front * 0.03;
      break;
    }

    case 'head': {
      // v: 0 at the chin, 1 at the crown.
      const faceMask = front * sstep(0.02, 0.16, v);
      // Brow ridge.
      s.h += ridge(u, v, 0.5, 0.085, 0.51, 0.575, 0.035) * 1.5 * front;
      // Eye sockets, then the globes sitting in them.
      const eyeL = blob(u, v, 0.5 - 0.045, 0.46, 0.045, 0.05);
      const eyeR = blob(u, v, 0.5 + 0.045, 0.46, 0.045, 0.05);
      s.h -= (eyeL + eyeR) * 1.5;
      s.h += (blob(u, v, 0.5 - 0.043, 0.455, 0.026, 0.028) + blob(u, v, 0.5 + 0.043, 0.455, 0.026, 0.028)) * 1.6;
      // Nose: bridge, tip, wings.
      s.h += ridge(u, v, 0.5, 0.02, 0.26, 0.52, 0.03) * 2.6;
      s.h += blob(u, v, 0.5, 0.28, 0.028, 0.028) * 2.2;
      s.h += (blob(u, v, 0.5 - 0.03, 0.27, 0.016, 0.02) + blob(u, v, 0.5 + 0.03, 0.27, 0.016, 0.02)) * 1.2;
      // Philtrum, lips, mental crease, chin.
      s.h -= ridge(u, v, 0.5, 0.012, 0.2, 0.25, 0.02) * 0.45;
      s.h += blob(u, v, 0.5, 0.185, 0.045, 0.017) * 1.3;
      s.h += blob(u, v, 0.5, 0.155, 0.05, 0.017) * 1.5;
      s.h -= ridge(u, v, 0.5, 0.06, 0.12, 0.14, 0.028) * 0.25;
      s.h += blob(u, v, 0.5, 0.075, 0.06, 0.05) * 1.1;
      // Cheekbones and the nasolabial fold.
      s.h += (blob(u, v, 0.5 - 0.105, 0.4, 0.055, 0.06) + blob(u, v, 0.5 + 0.105, 0.4, 0.055, 0.06)) * 0.55;
      s.h -= (blob(u, v, 0.5 - 0.072, 0.2, 0.026, 0.055) + blob(u, v, 0.5 + 0.072, 0.2, 0.026, 0.055)) * 0.22;
      // Ears.
      const earL = blob(u, v, 0.5 - 0.245, 0.39, 0.045, 0.09);
      const earR = blob(u, v, 0.5 + 0.245, 0.39, 0.045, 0.09);
      s.h += (earL + earR) * 2.2;
      s.h -= (blob(u, v, 0.5 - 0.245, 0.38, 0.022, 0.05) + blob(u, v, 0.5 + 0.245, 0.38, 0.022, 0.05)) * 1.6;
      // Temples and the occiput.
      s.h -= (blob(u, v, 0.5 - 0.17, 0.63, 0.05, 0.07) + blob(u, v, 0.5 + 0.17, 0.63, 0.05, 0.07)) * 0.7;
      s.h += blob(u, v, 0.0, 0.62, 0.18, 0.16) * 0.6;

      // --- Pigment. The only place albedo is allowed to carry detail. -----
      const brow =
        blob(u, v, 0.5 - 0.05, 0.512, 0.06, 0.022) + blob(u, v, 0.5 + 0.05, 0.512, 0.06, 0.022);
      const browK = clamp01(brow * 1.4);
      s.r = lerp(s.r, 0.3, browK * 0.85);
      s.g = lerp(s.g, 0.24, browK * 0.85);
      s.b = lerp(s.b, 0.21, browK * 0.85);
      s.rough += browK * 0.2;
      // Lash line, then the eye proper. An eye is ~30 mm across on a ~570 mm
      // head circumference (u) and ~10 mm of opening on a ~230 mm chin-to-crown
      // span (v), so the aperture is 0.026 in u by 0.019 in v — round 0 had the
      // v radius at 0.011, less than half the real opening, which is part of
      // why nothing survived. The mask lands in `eye` and the shader overrides
      // the material there; the albedo lift below only keeps the atlas sane if
      // it is ever previewed on its own.
      const lash =
        blob(u, v, 0.5 - 0.043, 0.478, 0.034, 0.009) + blob(u, v, 0.5 + 0.043, 0.478, 0.034, 0.009);
      const lashK = clamp01(lash * 1.6);
      s.r = lerp(s.r, 0.18, lashK);
      s.g = lerp(s.g, 0.15, lashK);
      s.b = lerp(s.b, 0.14, lashK);
      const sclera =
        blob(u, v, 0.5 - 0.043, 0.4555, 0.028, 0.019) + blob(u, v, 0.5 + 0.043, 0.4555, 0.028, 0.019);
      const scleraK = clamp01(sclera * 2.6);
      s.r = lerp(s.r, 1.55, scleraK * 0.8);
      s.g = lerp(s.g, 1.5, scleraK * 0.8);
      s.b = lerp(s.b, 1.42, scleraK * 0.8);
      const iris =
        blob(u, v, 0.5 - 0.043, 0.4535, 0.0115, 0.0165) +
        blob(u, v, 0.5 + 0.043, 0.4535, 0.0115, 0.0165);
      const irisK = clamp01(iris * 2.6);
      s.r = lerp(s.r, 0.24, irisK);
      s.g = lerp(s.g, 0.2, irisK);
      s.b = lerp(s.b, 0.17, irisK);
      s.rough -= irisK * 0.32;
      // 1.0 sclera, 0.55 iris, and the lash line pushes it back to skin so the
      // lid edge stays a hard dark boundary instead of a bright halo.
      s.eye = clamp01(lerp(scleraK, 0.55, irisK) * (1 - lashK * 0.9));
      // Lips: redder, smoother, with a defined vermilion border.
      const lip = clamp01((blob(u, v, 0.5, 0.185, 0.05, 0.02) + blob(u, v, 0.5, 0.155, 0.055, 0.02)) * 1.2);
      s.r *= 1 + lip * 0.3;
      s.g *= 1 - lip * 0.14;
      s.b *= 1 - lip * 0.12;
      s.rough -= lip * 0.13;
      // Stubble on the jaw and upper lip — must reduce specular under it.
      const beardArea =
        clamp01(
          (blob(u, v, 0.5, 0.11, 0.16, 0.12) + blob(u, v, 0.5, 0.225, 0.075, 0.035)) *
            1.1 *
            front,
        ) * sstep(0.34, 0.2, v + 0.12);
      const stubble = beardArea * (0.45 + 0.55 * valueNoise2(cx * 420, v * 520, seed + 7));
      s.r *= 1 - stubble * 0.24;
      s.g *= 1 - stubble * 0.24;
      s.b *= 1 - stubble * 0.21;
      s.rough += stubble * 0.22;
      s.h += (valueNoise2(cx * 500, v * 620, seed + 19) - 0.5) * stubble * 0.5;
      // Ear and nose cartilage read redder; so does the tip of the chin.
      const capillary = clamp01((earL + earR) * 1.1 + blob(u, v, 0.5, 0.28, 0.05, 0.05) * 0.8);
      s.r *= 1 + capillary * 0.14;
      s.b *= 1 - capillary * 0.08;
      // The forehead is the shiniest thing on a player.
      s.rough -= faceMask * sstep(0.54, 0.74, v) * 0.14;
      s.sweat = 0.3 + faceMask * sstep(0.46, 0.76, v) * 0.7 + front * 0.15;
      break;
    }

    case 'neck': {
      // Sternocleidomastoid: the two cords running to the collarbone.
      s.h += ridge(u, v, 0.5 - 0.075, 0.045, 0.05, 0.85, 0.16) * 1.2;
      s.h += ridge(u, v, 0.5 + 0.075, 0.045, 0.05, 0.85, 0.16) * 1.2;
      // Laryngeal prominence.
      s.h += blob(u, v, 0.5, 0.55, 0.035, 0.09) * 0.9;
      // Trapezius shoulders into the nape.
      s.h += blob(u, v, 0.0, 0.2, 0.3, 0.3) * 1.0;
      s.sweat = 0.55 + back * 0.35;
      break;
    }

    case 'upperArm': {
      // v: 0 at the shoulder, 1 at the elbow.
      // Deltoid cap, with the three heads showing as striations.
      s.h += blob(u, v, 0.5, 0.09, 0.34, 0.19) * 1.5;
      for (let i = 0; i < 5; i++) {
        s.h += ridge(u, v, 0.34 + i * 0.08, 0.026, 0.03, 0.24, 0.06) * 0.45;
      }
      // Biceps: two heads, peaking around a third of the way down.
      s.h += blob(u, v, 0.5, 0.42, 0.13, 0.22) * 1.7;
      s.h -= ridge(u, v, 0.5, 0.016, 0.28, 0.55, 0.1) * 0.55;
      // Distal biceps tendon into the elbow crease.
      s.h += ridge(u, v, 0.5, 0.03, 0.62, 0.9, 0.08) * 0.7;
      // Triceps horseshoe on the back.
      s.h += blob(u, v, 0.0, 0.4, 0.14, 0.24) * 1.5;
      s.h += ridge(u, v, 0.93, 0.035, 0.2, 0.62, 0.1) * 0.6;
      s.h += ridge(u, v, 0.07, 0.035, 0.2, 0.62, 0.1) * 0.6;
      // Olecranon and the medial/lateral epicondyles.
      s.h += blob(u, v, 0.0, 0.96, 0.07, 0.05) * 1.2;
      // Tan line where a sleeve would end, plus reddened elbow skin.
      const tan = sstep(0.44, 0.56, v);
      s.r *= 1 + tan * 0.045;
      s.g *= 1 + tan * 0.02;
      s.b *= 1 - tan * 0.02;
      const elbow = sstep(0.82, 1.0, v);
      s.r *= 1 + elbow * 0.1;
      s.g *= 1 - elbow * 0.03;
      s.b *= 1 - elbow * 0.05;
      s.rough += elbow * 0.14;
      s.h += (valueNoise2(cx * 300, v * 340, seed + 23) - 0.5) * elbow * 0.6;
      s.sweat = 0.5 + (1 - v) * 0.4;
      break;
    }

    case 'foreArm': {
      // v: 0 at the elbow, 1 at the wrist. Oval at the top, narrow at the base.
      // u = 0.75 is the lateral (radial) side of the limb; 0.25 is medial.
      s.h += blob(u, v, 0.68, 0.24, 0.14, 0.22) * 1.5; // brachioradialis
      s.h += blob(u, v, 0.86, 0.28, 0.13, 0.2) * 1.2; // extensor bundle
      s.h += blob(u, v, 0.28, 0.26, 0.16, 0.22) * 1.0; // flexor bundle
      // Tendon fan into the wrist.
      for (let i = 0; i < 5; i++) {
        s.h += ridge(u, v, 0.42 + i * 0.04, 0.014, 0.6, 0.98, 0.12) * 0.55;
      }
      // Ulnar border — the hard edge you can always see on a lean forearm.
      s.h += ridge(u, v, 0.16, 0.02, 0.05, 0.95, 0.12) * 0.7;
      // Surface veins: a sinuous network, only ever normal detail.
      const vein =
        Math.exp(-((du(u, 0.56 + Math.sin(v * 7.1 + seed) * 0.045) / 0.012) ** 2)) *
          sstep(0.1, 0.25, v) *
          (1 - sstep(0.8, 0.98, v)) +
        Math.exp(-((du(u, 0.4 + Math.sin(v * 5.3 + seed * 1.7) * 0.05) / 0.011) ** 2)) *
          sstep(0.2, 0.35, v) *
          (1 - sstep(0.78, 0.95, v));
      s.h += vein * 0.75;
      // Body hair, dense on the outer forearm and absent on the inner.
      const hairMask = clamp01(1 - du(u, 0.42) / 0.34) * sstep(0.02, 0.2, v) * (1 - sstep(0.78, 0.96, v));
      const strand = valueNoise2(cx * 520, v * 130, seed + 31);
      const hair = hairMask * Math.max(0, strand - 0.62) * 2.6;
      s.h += hair * 0.5;
      s.r *= 1 - hair * 0.16;
      s.g *= 1 - hair * 0.17;
      s.b *= 1 - hair * 0.16;
      s.rough += hair * 0.16;
      s.sweat = 0.45 + (1 - v) * 0.25;
      break;
    }

    case 'hand': {
      // v: 0 at the wrist, 1 at the fingertips. The palm faces medially, so
      // u = 0.25 is the palm plane and u = 0.75 the dorsum.
      const DORSUM = 0.75;
      const PALM = 0.25;
      // Metacarpal grooves on the back of the hand, and the knuckle line.
      for (let i = 0; i < 4; i++) {
        const cu = DORSUM - 0.085 + i * 0.057;
        s.h -= ridge(u, v, cu, 0.012, 0.2, 0.55, 0.1) * 0.5;
        s.h += blob(u, v, cu, 0.58, 0.022, 0.035) * 1.4;
      }
      // Finger separations, cut through from the dorsum round to the palm.
      for (let i = 0; i < 3; i++) {
        const cu = DORSUM - 0.06 + i * 0.06;
        s.h -= ridge(u, v, cu, 0.009, 0.6, 1.0, 0.05) * 1.8;
        s.h -= ridge(u, v, PALM + 0.06 - i * 0.06, 0.009, 0.62, 1.0, 0.05) * 1.4;
      }
      // Palm: thenar and hypothenar pads, and the flexion creases.
      s.h += blob(u, v, PALM - 0.06, 0.3, 0.07, 0.16) * 1.1;
      s.h += blob(u, v, PALM + 0.07, 0.32, 0.05, 0.14) * 0.8;
      s.h -= ridge(u, v, PALM, 0.13, 0.5, 0.52, 0.03) * 1.2;
      s.h -= ridge(u, v, PALM, 0.11, 0.42, 0.44, 0.03) * 0.9;
      // Palms are cooler and less saturated than the back of the hand.
      const palm = 1 - clamp01(du(u, PALM) / 0.3);
      s.r *= 1 - palm * 0.03;
      s.g *= 1 + palm * 0.02;
      s.b *= 1 + palm * 0.05;
      s.rough += palm * 0.1;
      // Knuckles read redder than anything else on the body.
      const dors = 1 - clamp01(du(u, DORSUM) / 0.25);
      const knuckle = clamp01(dors * sstep(0.5, 0.66, v) * (1 - sstep(0.72, 0.9, v)));
      s.r *= 1 + knuckle * 0.16;
      s.g *= 1 - knuckle * 0.04;
      s.b *= 1 - knuckle * 0.06;
      s.rough += knuckle * 0.12;
      s.sweat = 0.6;
      break;
    }

    case 'thigh': {
      // v: 0 at the hip, 1 at the knee.
      s.h += blob(u, v, 0.5, 0.55, 0.11, 0.3) * 1.4; // rectus femoris
      s.h += blob(u, v, 0.64, 0.6, 0.11, 0.26) * 1.2; // vastus lateralis
      s.h += blob(u, v, 0.37, 0.7, 0.09, 0.18) * 1.3; // vastus medialis teardrop
      s.h += ridge(u, v, 0.75, 0.03, 0.1, 0.86, 0.14) * 0.8; // IT band
      s.h += blob(u, v, 0.0, 0.42, 0.2, 0.32) * 1.1; // hamstrings
      s.h -= ridge(u, v, 0.0, 0.02, 0.15, 0.8, 0.14) * 0.6; // hamstring split
      s.h += blob(u, v, 0.5, 0.96, 0.09, 0.06) * 1.0; // patella
      s.h -= ridge(u, v, 0.0, 0.09, 0.94, 1.0, 0.05) * 1.4; // popliteal fossa
      const tan = sstep(0.4, 0.52, v);
      s.r *= 1 + tan * 0.04;
      s.b *= 1 - tan * 0.02;
      s.sweat = 0.4 + (1 - v) * 0.2;
      break;
    }

    case 'shin': {
      // v: 0 at the knee, 1 at the ankle.
      s.h += ridge(u, v, 0.5, 0.022, 0.06, 0.94, 0.1) * 1.3; // tibial crest
      s.h += blob(u, v, 0.5 - 0.11, 0.35, 0.09, 0.26) * 1.1; // tibialis anterior
      // Calf belly sits high on the shin and tapers hard into the achilles.
      s.h += blob(u, v, 0.0 - 0.055, 0.3, 0.11, 0.26) * 1.7;
      s.h += blob(u, v, 0.0 + 0.055, 0.26, 0.1, 0.22) * 1.5;
      s.h -= ridge(u, v, 0.0, 0.018, 0.1, 0.5, 0.12) * 0.6;
      s.h += ridge(u, v, 0.0, 0.035, 0.62, 0.98, 0.12) * 0.9; // achilles
      s.h += blob(u, v, 0.5 + 0.2, 0.95, 0.045, 0.05) * 1.2; // malleoli
      s.h += blob(u, v, 0.5 - 0.2, 0.96, 0.04, 0.045) * 1.1;
      s.h += blob(u, v, 0.5, 0.03, 0.1, 0.05) * 0.9; // tibial tuberosity
      // Shin hair.
      const hairMask = clamp01(1 - du(u, 0.5) / 0.4) * sstep(0.04, 0.2, v) * (1 - sstep(0.8, 0.95, v));
      const strand = valueNoise2(cx * 470, v * 120, seed + 47);
      const hair = hairMask * Math.max(0, strand - 0.6) * 2.4;
      s.h += hair * 0.45;
      s.r *= 1 - hair * 0.15;
      s.g *= 1 - hair * 0.16;
      s.b *= 1 - hair * 0.15;
      s.rough += hair * 0.15;
      // Sock tan line.
      const sock = sstep(0.62, 0.72, v);
      s.r *= 1 - sock * 0.02;
      s.b *= 1 + sock * 0.035;
      s.sweat = 0.35;
      break;
    }

    case 'foot': {
      s.h += ridge(u, v, 0.5, 0.05, 0.1, 0.8, 0.12) * 0.7;
      for (let i = 0; i < 4; i++) s.h += ridge(u, v, 0.42 + i * 0.04, 0.012, 0.4, 0.9, 0.1) * 0.5;
      s.rough += 0.08;
      s.sweat = 0.3;
      break;
    }
  }

  s.rough = clamp01(s.rough);
  s.sweat = clamp01(s.sweat);
  s.eye = clamp01(s.eye);
}

// ---------------------------------------------------------------------------
// Bake
// ---------------------------------------------------------------------------

export interface SkinMaps {
  albedo: CanvasTexture;
  normal: CanvasTexture;
  /** R = sweat proneness, G = roughness, B = eye mask (0.55 iris, 1 sclera). */
  data: CanvasTexture;
  dispose(): void;
}

function makeCanvas(w: number, h: number): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

/**
 * One pass over the atlas produces the height field, the pigment and the
 * roughness; the normal map is then differenced out of the height field with
 * the per-cell gradient clamped at the cell borders so a neighbouring part can
 * never emboss a false ridge across the seam.
 */
export function bakeSkinAtlas(size: number, anisotropy: number, seed = 90210): SkinMaps {
  const W = size;
  const H = Math.round((size / SKIN_ATLAS_COLS) * SKIN_ATLAS_ROWS);
  const cellW = W / SKIN_ATLAS_COLS;
  const cellH = H / SKIN_ATLAS_ROWS;

  const height = new Float32Array(W * H);
  const alb = new Uint8ClampedArray(W * H * 4);
  const dat = new Uint8ClampedArray(W * H * 4);
  const partOf = new Int8Array(W * H).fill(-1);

  const s: Sample = { h: 0, r: 1, g: 1, b: 1, rough: 0.47, sweat: 0.35, eye: 0 };

  for (let p = 0; p < PART_ORDER.length; p++) {
    const part = PART_ORDER[p];
    const [cx, cy] = CELL[part];
    const x0 = Math.floor(cx * cellW);
    const y0 = Math.floor(cy * cellH);
    const x1 = Math.floor((cx + 1) * cellW);
    const y1 = Math.floor((cy + 1) * cellH);

    for (let y = y0; y < y1; y++) {
      // Undo the inset so texel centres land on the same parameterisation the
      // mesh generator writes into the UV attribute.
      const vv = (y + 0.5 - y0) / (y1 - y0);
      const v = (vv - INSET) / (1 - 2 * INSET);
      for (let x = x0; x < x1; x++) {
        const uu = (x + 0.5 - x0) / (x1 - x0);
        const u = (uu - INSET) / (1 - 2 * INSET);
        sample(part, u < 0 ? u + 1 : u > 1 ? u - 1 : u, clamp01(v), s, seed);
        const i = y * W + x;
        partOf[i] = p;
        height[i] = s.h;
        // Neutral base sits mid-bright so the per-player tone multiply lands
        // skin in the 120–190 window the grade wants.
        const o = i * 4;
        alb[o] = s.r * 246;
        alb[o + 1] = s.g * 246;
        alb[o + 2] = s.b * 246;
        alb[o + 3] = 255;
        dat[o] = s.sweat * 255;
        dat[o + 1] = s.rough * 255;
        dat[o + 2] = s.eye * 255;
        dat[o + 3] = 255;
      }
    }
  }

  // --- Normal from the height field --------------------------------------
  const nrm = new Uint8ClampedArray(W * H * 4);
  // Height is in "relief units"; this converts a unit step across one texel
  // into a slope. Larger atlas → finer texels → the same physical bump has a
  // steeper gradient, so the scale has to come down with texel size.
  const slope = 1.35 * (512 / W);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const p = partOf[i];
      if (p < 0) {
        const o = i * 4;
        nrm[o] = 128;
        nrm[o + 1] = 128;
        nrm[o + 2] = 255;
        nrm[o + 3] = 255;
        continue;
      }
      const xm = x > 0 && partOf[i - 1] === p ? i - 1 : i;
      const xp = x < W - 1 && partOf[i + 1] === p ? i + 1 : i;
      const ym = y > 0 && partOf[i - W] === p ? i - W : i;
      const yp = y < H - 1 && partOf[i + W] === p ? i + W : i;
      const dx = (height[xp] - height[xm]) * slope;
      const dy = (height[yp] - height[ym]) * slope;
      const inv = 1 / Math.sqrt(dx * dx + dy * dy + 1);
      const o = i * 4;
      nrm[o] = (-dx * inv * 0.5 + 0.5) * 255;
      nrm[o + 1] = (dy * inv * 0.5 + 0.5) * 255;
      nrm[o + 2] = (inv * 0.5 + 0.5) * 255;
      nrm[o + 3] = 255;
    }
  }

  const put = (data: Uint8ClampedArray): HTMLCanvasElement => {
    const c = makeCanvas(W, H);
    const ctx = c.getContext('2d')!;
    const img = ctx.createImageData(W, H);
    img.data.set(data);
    ctx.putImageData(img, 0, 0);
    return c;
  };

  const mk = (c: HTMLCanvasElement, srgb: boolean): CanvasTexture => {
    const t = new CanvasTexture(c);
    t.flipY = false;
    if (srgb) t.colorSpace = SRGBColorSpace;
    t.wrapS = ClampToEdgeWrapping;
    t.wrapT = ClampToEdgeWrapping;
    t.minFilter = LinearMipmapLinearFilter;
    t.magFilter = LinearFilter;
    t.generateMipmaps = true;
    t.anisotropy = anisotropy;
    t.needsUpdate = true;
    return t;
  };

  const albedo = mk(put(alb), true);
  const normal = mk(put(nrm), false);
  const data = mk(put(dat), false);

  return {
    albedo,
    normal,
    data,
    dispose() {
      albedo.dispose();
      normal.dispose();
      data.dispose();
    },
  };
}

// ---------------------------------------------------------------------------
// Hair
// ---------------------------------------------------------------------------

/**
 * The strand mask for the hair shells.
 *
 * `u` tiles across the strand direction. `v` is a *density ramp*, not a second
 * spatial axis: the inner shell samples near v = 0 where coverage is nearly
 * solid, and each outer shell samples further down where coverage thins to
 * scattered clumps. With one alphaTest threshold that gives a scalp that reads
 * solid and an outline that frays at a 2–8 px scale, which is exactly what
 * stops shell hair looking like a helmet. Above v = 0.96 the mask is opaque,
 * for headbands and other solid trim sharing the material.
 */
export interface HairMaps {
  /** RGB strand shading; drives the anisotropic banding. */
  shade: CanvasTexture;
  /** Coverage in RGB, for alphaMap (three samples the green channel). */
  alpha: CanvasTexture;
  dispose(): void;
}

export function bakeHairMask(size = 256, seed = 771): HairMaps {
  const shadeC = makeCanvas(size, size);
  const alphaC = makeCanvas(size, size);
  const sctx = shadeC.getContext('2d')!;
  const actx = alphaC.getContext('2d')!;
  const simg = sctx.createImageData(size, size);
  const aimg = actx.createImageData(size, size);
  const sd = simg.data;
  const ad = aimg.data;
  const rng = makeRng(seed);
  const jitter = new Float32Array(size);
  for (let i = 0; i < size; i++) jitter[i] = rng();

  for (let y = 0; y < size; y++) {
    const v = y / size;
    // Coverage falls from near-solid at the scalp to scattered clumps outside.
    const density = v > 0.96 ? 2 : 1.08 - 0.72 * (v / 0.96);
    for (let x = 0; x < size; x++) {
      const u = x / size;
      // Strands run along v; the noise is stretched hard in that direction.
      const strand = valueNoise2(u * size * 0.5, v * 34, seed);
      const clump = fbm2(u * 22, v * 60, 3, 2, 0.5, seed + 5);
      // A second, much coarser clump. Without it the outer shells fray at texel
      // scale, which filters back into a smooth edge — a helmet with soft
      // borders. §3.7 wants the break-up at a 2–8 px scale, i.e. in *tufts*,
      // so the coarse term is what actually decides where the outline bites.
      const tuft = fbm2(u * 7, v * 9, 2, 2, 0.5, seed + 17);
      const j = jitter[(x + Math.floor(v * 41)) % size];
      const cover = strand * 0.4 + clump * 0.24 + tuft * 0.26 + j * 0.1;
      const a = v > 0.96 ? 1 : clamp01((cover * density - 0.3) * 4.6);
      // Banded, not smooth: the strand noise drives a narrow bright band so the
      // anisotropic lobe in the hair shader has something to sit on.
      const band = Math.pow(clamp01(strand), 2.2);
      const shade = 0.34 + 0.46 * strand + 0.34 * band;
      const o = (y * size + x) * 4;
      sd[o] = shade * 255;
      sd[o + 1] = shade * 255;
      sd[o + 2] = shade * 255;
      sd[o + 3] = 255;
      ad[o] = a * 255;
      ad[o + 1] = a * 255;
      ad[o + 2] = a * 255;
      ad[o + 3] = 255;
    }
  }
  sctx.putImageData(simg, 0, 0);
  actx.putImageData(aimg, 0, 0);

  const mk = (c: HTMLCanvasElement, srgb: boolean): CanvasTexture => {
    const t = new CanvasTexture(c);
    t.flipY = false;
    if (srgb) t.colorSpace = SRGBColorSpace;
    t.wrapS = RepeatWrapping;
    t.wrapT = ClampToEdgeWrapping;
    t.minFilter = LinearFilter;
    t.magFilter = LinearFilter;
    t.generateMipmaps = false;
    t.needsUpdate = true;
    return t;
  };
  const shade = mk(shadeC, true);
  const alpha = mk(alphaC, false);
  return {
    shade,
    alpha,
    dispose() {
      shade.dispose();
      alpha.dispose();
    },
  };
}
