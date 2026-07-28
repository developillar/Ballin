/**
 * The grade: a baked 3D LUT, the composite that evaluates it, and the present
 * pass that puts grain and FXAA on at output resolution.
 *
 * ACES is the tone curve and it is already applied (`postACES` in
 * `postPasses.ts` reproduces three's fit exactly, because the scene pass renders
 * into a target and three therefore skipped its own). What sits on top of it is
 * a film grade, and §8.3 is unusually precise about what that means:
 *
 *  - **Cool shadows.** In the darkest quartile of the frame, mean B − R between
 *    +5 and +14.
 *  - **Warm highlights.** In the top 20%, mean R − B between +4 and +12, at
 *    3–8% saturation — i.e. the *highlights desaturate*, per §8.2. This is the
 *    one that our frames fail in the other direction: warm maple under warm
 *    bounce puts the top quintile at R − B ≈ +25 with 15% saturation, so the
 *    highlight end of this grade is a *neutraliser*, not a warmer.
 *  - **Subtle.** "If a reviewer can name the colour of the shadows without
 *    measuring, it is 2–3× too strong."
 *
 * Both ends therefore have two operators rather than one: a *tint*, which moves
 * a neutral value onto the right side of the band, and a *ceiling*, which stops
 * scene content that is already far outside it from dragging the measurement
 * out the other side. The tint alone was round 2's mistake — it can only add,
 * and a blue bowl or an amber floor does not need adding to. `softCeil` is
 * built so a value already inside the band comes back essentially unchanged, so
 * the two never fight.
 *
 * The warm ceiling is deliberately wide through the mid-tones. §8.3's closing
 * bullet — "the hardwood must still read as maple and the ball must still read
 * as leather" — is a constraint on this operator, and on a framing where half
 * the frame is near hardwood the analyser's "top quintile" reaches down to
 * ~100 sRGB and *is* the mid-tone floor. Tightening the ceiling far enough to
 * put that frame's R − B under 12 turns maple into grey plywood; the ceiling
 * stops where the wood stops reading as wood, and the residual is reported
 * rather than bought.
 *
 * Plus the two things the tone curve does not do on its own: a gentle S around a
 * mid-tone pivot (which also *helps* §1.1 — it pushes lit hardwood up and the
 * bowl down, widening the bowl-to-court ratio rather than eating it), a toe lift
 * that puts the black point at 5–14 instead of 0, and a saturation lift confined
 * to the shadows and low mids so the crowd's clothing keeps its eight-plus
 * colour clusters down at 25–35 sRGB where §6.3 needs them readable.
 *
 * It is baked as a LUT rather than evaluated per pixel because a LUT is a single
 * `texture()` in the composite regardless of how many operators the grade grows,
 * and because a LUT is a thing an artist can look at.
 *
 * Owned by the post-processing agent.
 */

import {
  ClampToEdgeWrapping,
  Data3DTexture,
  LinearFilter,
  RGBAFormat,
  UnsignedByteType,
  Vector2,
} from 'three';
import { POST_COMMON, ScreenPass } from './postPasses';
import { FXAA_GLSL } from './postTaa';

/** Cube edge. 32 nodes with trilinear interpolation is well under the grain floor. */
export const LUT_SIZE = 32;

export interface GradeSettings {
  /** S-curve strength around `pivot`; 0 is a straight line. */
  contrast: number;
  pivot: number;
  /** Absolute lift added at black, dying out by ~0.24. Sets the black point. */
  toeLift: number;
  /** Weight on the cool shadow tint. */
  shadowCool: number;
  /** Saturation gain in the shadows and low mids. */
  shadowSat: number;
  /** Pull toward luminance in the highlights (§8.2 highlight desaturation). */
  highlightDesat: number;
  /** Luminance at which the highlight desaturation starts, and where it tops out. */
  highlightOnset: number;
  highlightFull: number;
  /** Hard cap on the pull. Must stay below 1 — see the note in `gradeColour`. */
  highlightDesatMax: number;
  /** Weight on the highlight neutral/cool balance. */
  highlightCool: number;
  /** §8.3 warm ceiling on R − B, in 0..1 units. Mid-tone value and highlight value. */
  warmCeilMid: number;
  warmCeilHigh: number;
  /** Luminance range over which the warm ceiling tightens from mid to high. */
  warmOnset: number;
  warmFull: number;
  /** §8.3 cool ceiling on B − R in the shadows, and where it stops applying. */
  coolCeil: number;
  coolOnset: number;
  coolFull: number;
  /** Gentle global saturation, applied last. */
  saturation: number;
}

export const DEFAULT_GRADE: GradeSettings = {
  contrast: 0.10,
  pivot: 0.42,
  toeLift: 0.024,
  shadowCool: 0.32,
  shadowSat: 0.26,
  // Round 3: the onset was 0.30 and the pull was uncapped at 1.2, which is two
  // separate faults. The onset meant a frame whose top quintile lives at
  // 110–160 sRGB (l = 0.43–0.63) got a pull of only 0.09–0.50, so §8.2's
  // "saturation must fall as values approach white" was not happening where the
  // measurement looks; and a pull above 1.0 pushes a colour *past* neutral into
  // its complement, which is a hue inversion, not a desaturation.
  highlightDesat: 1.30,
  highlightOnset: 0.18,
  highlightFull: 0.74,
  highlightDesatMax: 0.85,
  highlightCool: 1.0,
  warmCeilMid: 0.21,
  warmCeilHigh: 0.022,
  warmOnset: 0.18,
  warmFull: 0.50,
  coolCeil: 0.070,
  coolOnset: 0.34,
  coolFull: 0.04,
  saturation: 1.05,
};

/** Cool, and deliberately nearly neutral in green so shadows do not go teal. */
const SHADOW_TINT: [number, number, number] = [-0.075, -0.012, 0.085];
/** Per-channel weighting on the toe lift. Cool, for the reason above. */
const TOE_TINT: [number, number, number] = [0.86, 0.96, 1.20];
/**
 * The highlight balance, and it is **warm** (§8.3: hue 30–48°, R − B between +4
 * and +12). Round 2 of this grade used a cool balance to drag the frame's warm
 * hardwood highlights down into band, and the flat-field capture immediately
 * showed what that really did: a neutral highlight came out at R − B = −10.8,
 * i.e. the grade was tinting the top of the range blue and calling it a fix.
 * Warm balance here; the *desaturation* does the reducing.
 */
const HIGHLIGHT_TINT: [number, number, number] = [0.014, 0.0, -0.010];

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function luma(c: [number, number, number]): number {
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}

/**
 * Soft ceiling on a single chroma axis: essentially the identity while `v` is
 * well under `ceiling`, asymptotic to `ceiling` well over it, and smooth in
 * between. The fourth power is what buys the sharp knee — a value already inside
 * the band comes out inside the band (0.019 against a 0.022 ceiling is returned
 * at 0.0186, a fifth of a display unit), so this cannot pull a frame that
 * already satisfies §8.3 out through the *bottom* of the band.
 */
function softCeil(v: number, ceiling: number): number {
  const c = Math.max(1e-5, ceiling);
  return v / Math.pow(1 + Math.pow(v / c, 4), 0.25);
}

/**
 * Luma-preserving scale of a colour's chroma. Everything below works this way so
 * an operator that changes a hue relationship cannot also change the exposure —
 * §1.1's court and bowl bands have to survive the grade untouched.
 */
function scaleChroma(c: [number, number, number], f: number): [number, number, number] {
  const l = luma(c);
  return [l + (c[0] - l) * f, l + (c[1] - l) * f, l + (c[2] - l) * f];
}

/** Power-based S around an arbitrary pivot — monotonic, and exact at 0 and 1. */
function sCurve(x: number, pivot: number, k: number): number {
  if (k <= 0) return x;
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  return x < pivot
    ? pivot * Math.pow(x / pivot, 1 + k)
    : 1 - (1 - pivot) * Math.pow((1 - x) / (1 - pivot), 1 + k);
}

/**
 * The grade itself, evaluated in display-referred sRGB. Every stage re-reads
 * luminance from the running value, which is what keeps the operators from
 * fighting each other (a shadow tint applied against the *pre*-S-curve
 * luminance lands in the wrong place once the curve has moved it).
 */
export function gradeColour(input: [number, number, number], s: GradeSettings): [number, number, number] {
  let c: [number, number, number] = [
    sCurve(input[0], s.pivot, s.contrast),
    sCurve(input[1], s.pivot, s.contrast),
    sCurve(input[2], s.pivot, s.contrast),
  ];

  // Toe: the deepest black lands at s.toeLift rather than 0 (§8.2), and it
  // lands *cool*. A multiplicative shadow tint cannot move a value of 12 by
  // more than a unit, so the additive lift is where the deep end of §8.3's
  // B - R actually comes from — which is exactly what the near-black courtside
  // in the RIM framing needed.
  const toeW = 1 - smoothstep(0, 0.24, luma(c));
  c = [
    c[0] + s.toeLift * toeW * TOE_TINT[0],
    c[1] + s.toeLift * toeW * TOE_TINT[1],
    c[2] + s.toeLift * toeW * TOE_TINT[2],
  ];

  // Cool shadows.
  const shW = (1 - smoothstep(0.0, 0.36, luma(c))) * s.shadowCool;
  c = [
    c[0] * (1 + SHADOW_TINT[0] * shW),
    c[1] * (1 + SHADOW_TINT[1] * shW),
    c[2] * (1 + SHADOW_TINT[2] * shW),
  ];

  // Saturation that holds up in the deep bowl.
  const lSat = luma(c);
  const satGain = 1 + s.shadowSat * (1 - smoothstep(0.03, 0.5, lSat));
  c = [
    lSat + (c[0] - lSat) * satGain,
    lSat + (c[1] - lSat) * satGain,
    lSat + (c[2] - lSat) * satGain,
  ];

  // Cool ceiling. The tint above *adds* blue to the shadows, which is right
  // when the shadow is neutral and wrong when it is already blue — an arena's
  // bowl is lit by spill and phone screens and arrives cool on its own. §8.3
  // wants the darkest quartile at B − R of +5 to +14, so the shadow end needs a
  // limiter as much as it needs a tint, or a frame with a lot of dark blue bowl
  // in it (the `arena` framing measured +15.4) sails straight through the top.
  {
    const lCool = luma(c);
    const br = c[2] - c[0];
    if (br > 0) {
      const w = 1 - smoothstep(s.coolFull, s.coolOnset, lCool);
      c = scaleChroma(c, 1 - (1 - softCeil(br, s.coolCeil) / br) * w);
    }
  }

  // Highlight desaturation, then a small neutral balance on top of it.
  const lHi = luma(c);
  const hiW = smoothstep(s.highlightOnset, s.highlightFull, lHi);
  // Capped strictly below 1: at d = 1 the colour is exactly its own luminance,
  // and above 1 it crosses to the far side of neutral and comes back out in the
  // complementary hue. The old 1.2 did that to everything over ~0.83 luma.
  const d = Math.min(s.highlightDesatMax, s.highlightDesat * hiW);
  c = [c[0] + (lHi - c[0]) * d, c[1] + (lHi - c[1]) * d, c[2] + (lHi - c[2]) * d];

  // Warm ceiling. The desaturation above is driven by luminance alone, so it
  // cannot tell a maple floor at 110 sRGB from a grey wall at 110 sRGB; this
  // one is driven by the quantity §8.3 actually measures. The ceiling is wide
  // through the mid-tones — the hardwood must still read as maple and the ball
  // as leather, which is §8.3's own last bullet — and closes to a few display
  // units by the top of the range, so the core of a bright source is neutral
  // and only its skirt keeps the colour.
  {
    const lWarm = luma(c);
    const rb = c[0] - c[2];
    if (rb > 0) {
      const t = smoothstep(s.warmOnset, s.warmFull, lWarm);
      const ceiling = s.warmCeilMid + (s.warmCeilHigh - s.warmCeilMid) * t;
      c = scaleChroma(c, softCeil(rb, ceiling) / rb);
    }
  }

  const hw = hiW * s.highlightCool;
  c = [
    c[0] * (1 + HIGHLIGHT_TINT[0] * hw),
    c[1] * (1 + HIGHLIGHT_TINT[1] * hw),
    c[2] * (1 + HIGHLIGHT_TINT[2] * hw),
  ];

  const lEnd = luma(c);
  c = [
    lEnd + (c[0] - lEnd) * s.saturation,
    lEnd + (c[1] - lEnd) * s.saturation,
    lEnd + (c[2] - lEnd) * s.saturation,
  ];

  return [
    Math.min(1, Math.max(0, c[0])),
    Math.min(1, Math.max(0, c[1])),
    Math.min(1, Math.max(0, c[2])),
  ];
}

export function bakeGradeLut(settings: GradeSettings, size = LUT_SIZE): Data3DTexture {
  const data = new Uint8Array(size * size * size * 4);
  const inv = 1 / (size - 1);
  let p = 0;
  for (let b = 0; b < size; b++) {
    for (let g = 0; g < size; g++) {
      for (let r = 0; r < size; r++) {
        const out = gradeColour([r * inv, g * inv, b * inv], settings);
        data[p++] = Math.round(out[0] * 255);
        data[p++] = Math.round(out[1] * 255);
        data[p++] = Math.round(out[2] * 255);
        data[p++] = 255;
      }
    }
  }
  const tex = new Data3DTexture(data, size, size, size);
  tex.format = RGBAFormat;
  tex.type = UnsignedByteType;
  tex.minFilter = LinearFilter;
  tex.magFilter = LinearFilter;
  tex.wrapS = ClampToEdgeWrapping;
  tex.wrapT = ClampToEdgeWrapping;
  tex.wrapR = ClampToEdgeWrapping;
  tex.unpackAlignment = 1;
  tex.name = 'post.gradeLut';
  tex.needsUpdate = true;
  return tex;
}

// ---------------------------------------------------------------- composite

const COMPOSITE_FRAGMENT = /* glsl */ `
precision highp float;
varying vec2 vUv;

uniform sampler2D tColor;
uniform sampler2D tBloom;
uniform sampler3D tLut;
uniform vec2 uTexel;
uniform float uExposure;
uniform float uBloomIntensity;
uniform float uChromatic;     // half-separation at the corner, in pixels
uniform float uVignette;
uniform float uVignetteStart;
uniform float uVignetteDesat;
uniform float uVignetteDesatStart;
uniform float uVignetteCool;
uniform float uLutScale;
uniform float uLutOffset;
uniform float uLutAmount;

${POST_COMMON}

vec3 sceneAt(vec2 uv) {
  vec3 c = texture2D(tColor, uv).rgb;
#if USE_BLOOM
  c += texture2D(tBloom, uv).rgb * uBloomIntensity;
#endif
  return c;
}

void main() {
  vec2 ndc = (vUv - 0.5) * 2.0;

  vec3 hdr;
#if USE_CHROMATIC
  // Radial, zero at the centre, growing with r^2 the way real lateral
  // dispersion does. R goes outward, B inward.
  vec2 step_ = ndc * dot(ndc, ndc) * 0.5 * uChromatic * uTexel;
  hdr.r = sceneAt(clamp(vUv + step_, vec2(0.0), vec2(1.0))).r;
  hdr.g = sceneAt(vUv).g;
  hdr.b = sceneAt(clamp(vUv - step_, vec2(0.0), vec2(1.0))).b;
#else
  hdr = sceneAt(vUv);
#endif

  vec3 display = postACES(hdr, uExposure);
  vec3 srgb = postLinearToSRGB(display);

  // Vignette BEFORE the grade. A lens darkens light on its way to the sensor,
  // so the corners genuinely are less exposed and the grade should treat them
  // as such — which is also what stops the corner of a bright frame from being
  // graded as a highlight while it is being drawn as a shadow. Elliptical in
  // frame space, which on a 9:19.5 canvas is exactly the shape §8.4 asks for: a
  // circular vignette on a frame this tall bands the top and bottom and reads
  // as a phone filter.
  float r = length(ndc) * 0.70710678;
  float f = smoothstep(uVignetteStart, 1.0, r);
  srgb *= 1.0 - uVignette * f;
  // Real vignetting cools as it darkens. Deliberately tiny: the corner still
  // goes through the highlight desaturation downstream, which multiplies what is
  // left of its chroma by ~0.15, so anything big enough to *see* here is big
  // enough to swing the corner's R − B sign — and a sign flip is what the
  // flat-field capture reads as a 50% edge desaturation (§8.4 wants 3–8%).
  srgb *= vec3(1.0 - 0.8 * uVignetteCool * f, 1.0, 1.0 + uVignetteCool * f);
  srgb = clamp(srgb, 0.0, 1.0);

#if USE_LUT
  vec3 graded = texture(tLut, srgb * uLutScale + uLutOffset).rgb;
  srgb = mix(srgb, graded, uLutAmount);
#endif

  // §8.4's "slight desaturation in the vignette region", and it has to be
  // applied AFTER the grade, unlike the darkening. The grade's highlight end
  // multiplies whatever chroma reaches it by ~0.15 and then adds a warm balance
  // back as a fraction of the *value*, so a desaturation applied upstream is
  // first crushed and then overwritten by a pedestal that scales with
  // brightness — which is why the flat field measured the corner as no less
  // saturated than the centre while the source value said 11%. Downstream it
  // is delivered as written.
  //
  // Its falloff starts earlier than the darkening's on purpose: real optical
  // desaturation from oblique ray bundles is well under way before the falloff
  // is measurable, and §8.4's own figure is quoted at the corner while the
  // analyser samples the edge at r ≈ 0.71.
  float fd = smoothstep(uVignetteDesatStart, 1.0, r);
  float lv = postLuma(srgb);
  srgb = mix(srgb, vec3(lv), uVignetteDesat * fd);

  gl_FragColor = vec4(clamp(srgb, 0.0, 1.0), 1.0);
}
`;

export interface CompositeOptions {
  bloom: boolean;
  chromaticAberration: boolean;
  lut: boolean;
}

export function makeCompositePass(opts: CompositeOptions): ScreenPass {
  return new ScreenPass(
    COMPOSITE_FRAGMENT,
    {
      tColor: { value: null },
      tBloom: { value: null },
      tLut: { value: null },
      uTexel: { value: new Vector2(1, 1) },
      uExposure: { value: 1 },
      uBloomIntensity: { value: 0.55 },
      uChromatic: { value: 1.2 },
      uVignette: { value: 0.16 },
      uVignetteStart: { value: 0.55 },
      uVignetteDesat: { value: 0.070 },
      uVignetteDesatStart: { value: 0.28 },
      uVignetteCool: { value: 0.008 },
      uLutScale: { value: (LUT_SIZE - 1) / LUT_SIZE },
      uLutOffset: { value: 0.5 / LUT_SIZE },
      uLutAmount: { value: 1 },
    },
    {
      USE_BLOOM: opts.bloom ? 1 : 0,
      USE_CHROMATIC: opts.chromaticAberration ? 1 : 0,
      USE_LUT: opts.lut ? 1 : 0,
    },
  );
}

// ------------------------------------------------------------------ present

const PRESENT_FRAGMENT = /* glsl */ `
precision highp float;
varying vec2 vUv;

uniform sampler2D tColor;
uniform vec2 uTexel;
uniform vec2 uOutputSize;   // full-resolution device pixels, NOT render-scaled
uniform float uGrain;
uniform float uFrame;

${POST_COMMON}

#if USE_FXAA
${FXAA_GLSL}
#endif

void main() {
#if USE_FXAA
  vec3 c = fxaa(vUv);
#else
  vec3 c = texture2D(tColor, vUv).rgb;
#endif

#if USE_GRAIN
  // Grain is sized against the *output* buffer, not the render buffer, so the
  // adaptive governor dropping resolution cannot make it chunky (§8.5, tell 43).
  vec2 gp = vUv * uOutputSize;
  float n1 = postHash(gp + vec2(uFrame * 13.37, uFrame * 7.77)) - 0.5;
  float n2 = postHash(gp + vec2(uFrame * 5.11 + 37.0, uFrame * 11.3 + 91.0)) - 0.5;
  float n3 = postHash(gp + vec2(uFrame * 9.73 + 71.0, uFrame * 3.29 + 13.0)) - 0.5;
  // Mostly luminance, slightly chromatic — real film grain is not RGB noise.
  vec3 noise = mix(vec3(n1), vec3(n1, n2, n3), 0.24);

  float l = postLuma(c);
  // Full strength through the mids, eased off in the deepest blacks and shut
  // down in the top of the range so it cannot crawl over a blown LED board.
  float w = (0.70 + 0.30 * smoothstep(0.0, 0.2, l)) * (1.0 - smoothstep(0.80, 0.98, l));
  c += noise * (uGrain * w);
#endif

  gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
`;

export interface PresentOptions {
  fxaa: boolean;
  grain: boolean;
}

export function makePresentPass(opts: PresentOptions): ScreenPass {
  return new ScreenPass(
    PRESENT_FRAGMENT,
    {
      tColor: { value: null },
      uTexel: { value: new Vector2(1, 1) },
      uOutputSize: { value: new Vector2(1080, 2340) },
      uGrain: { value: 0.048 },
      uFrame: { value: 0 },
    },
    {
      USE_FXAA: opts.fxaa ? 1 : 0,
      USE_GRAIN: opts.grain ? 1 : 0,
    },
  );
}
