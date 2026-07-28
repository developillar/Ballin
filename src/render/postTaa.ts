/**
 * Antialiasing. The renderer is constructed with `antialias: false` precisely so
 * this owns it, and before this landed every edge in the game was a staircase.
 *
 * **TAA where `Quality.taa` allows, FXAA everywhere else.**
 *
 * The TAA is the standard modern arrangement and each piece is there for a
 * reason a still capture will show:
 *
 *  - **Halton(2,3) sub-pixel jitter** on the projection matrix, applied to the
 *    scene render only and removed before anything reconstructs a position from
 *    depth. Eight positions at `high`, sixteen at `ultra`.
 *  - **Reprojection through the previous frame's view-projection**, using the
 *    *closest* depth in a 3×3 neighbourhood so a silhouette samples the history
 *    of the object rather than the background sliding behind it.
 *  - **Variance clipping in YCoCg, widened on still pixels.** The history is
 *    clipped — not clamped — toward the current colour against an AABB built
 *    from the neighbourhood's mean and standard deviation. Clamping per axis is
 *    what leaves the coloured ghost trails that make people turn TAA off. The
 *    *width* of that box is the accumulation's real limit: on a static
 *    high-frequency region the 3×3 sigma is large, the history is legitimately
 *    outside a 1.1-sigma box on most frames, and clipping it back every frame
 *    discards exactly the sub-pixel detail the jitter went to gather. Wide when
 *    still, tight when moving.
 *  - **A Catmull-Rom history fetch.** A bilinear resample applied once per
 *    frame at a 0.9 feedback weight is a permanent low-pass, and it is the
 *    quiet reason TAA stacks deliver several times the blur they are configured
 *    for.
 *  - **Tone-mapped accumulation.** Blending raw HDR lets one 1.7-radiance LED
 *    pixel dominate eight frames of history and flicker. Weighting by
 *    1/(1+luma) before the mix and undoing it after is Karis's fix and it is
 *    what keeps the ribbon boards stable.
 *  - **A progressive feedback ramp.** The blend weight is `n/(n+1)` capped at
 *    the steady-state value, and `n` resets when the camera cuts. A hard cut
 *    therefore re-converges as a proper N-sample average instead of dragging the
 *    previous shot through the first half second — which is also what makes the
 *    review harness's 900 ms settle produce a genuinely resolved frame.
 *
 * Owned by the post-processing agent.
 */

import { Matrix4, Vector2 } from 'three';
import { POST_COMMON, ScreenPass } from './postPasses';

/** Radical-inverse Halton, the standard TAA jitter sequence. */
export function haltonSequence(count: number): Array<[number, number]> {
  const radical = (index: number, base: number): number => {
    let f = 1;
    let r = 0;
    let i = index;
    while (i > 0) {
      f /= base;
      r += f * (i % base);
      i = Math.floor(i / base);
    }
    return r;
  };
  const out: Array<[number, number]> = [];
  for (let i = 1; i <= count; i++) out.push([radical(i, 2) - 0.5, radical(i, 3) - 0.5]);
  return out;
}

const TAA_FRAGMENT = /* glsl */ `
precision highp float;
varying vec2 vUv;

uniform sampler2D tCurrent;
uniform sampler2D tHistory;
uniform sampler2D tDepth;
uniform vec2 uTexel;
uniform mat4 uInvViewProj;
uniform mat4 uPrevViewProj;
uniform float uFeedback;
uniform float uClipStill;
uniform float uClipMoving;

${POST_COMMON}

/**
 * Catmull-Rom history fetch, the 5-tap bilinear-assisted form.
 *
 * This is the single largest source of softness in a naive TAA and it is easy
 * to miss because nothing in the code looks like a blur: the history is
 * resampled with a bilinear filter *every frame*, and a bilinear filter applied
 * n times in a row is an n-fold box convolution. At a 0.92 feedback weight that
 * is a permanent low-pass on everything the camera is not moving relative to,
 * and it is what makes a post stack deliver three times the blur it is
 * configured for. Catmull-Rom has negative lobes, so repeated application does
 * not accumulate width.
 */
vec3 sampleHistory(vec2 uv) {
  vec2 texSize = 1.0 / uTexel;
  vec2 samplePos = uv * texSize;
  vec2 texPos1 = floor(samplePos - 0.5) + 0.5;
  vec2 f = samplePos - texPos1;

  vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  vec2 w3 = f * f * (-0.5 + 0.5 * f);

  vec2 w12 = w1 + w2;
  vec2 offset12 = w2 / max(w12, vec2(1e-5));

  vec2 texPos0 = (texPos1 - 1.0) * uTexel;
  vec2 texPos3 = (texPos1 + 2.0) * uTexel;
  vec2 texPos12 = (texPos1 + offset12) * uTexel;

  vec3 result = texture2D(tHistory, vec2(texPos12.x, texPos0.y)).rgb * w12.x * w0.y;
  result += texture2D(tHistory, vec2(texPos0.x, texPos12.y)).rgb * w0.x * w12.y;
  result += texture2D(tHistory, texPos12).rgb * w12.x * w12.y;
  result += texture2D(tHistory, vec2(texPos3.x, texPos12.y)).rgb * w3.x * w12.y;
  result += texture2D(tHistory, vec2(texPos12.x, texPos3.y)).rgb * w12.x * w3.y;
  // The negative lobes can ring below zero on a hard edge; the buffer is HDR
  // and a negative radiance would survive into the next frame's history.
  return max(result, vec3(0.0));
}

vec3 rgbToYCoCg(vec3 c) {
  return vec3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}

vec3 yCoCgToRgb(vec3 c) {
  float t = c.x - c.z;
  return vec3(t + c.y, c.x + c.z, t - c.y);
}

vec3 tonemapWeight(vec3 c) { return c / (1.0 + postLuma(c)); }
vec3 tonemapUnweight(vec3 c) { return c / max(1e-4, 1.0 - postLuma(c)); }

/** Clip q toward the AABB centre. Per-axis clamping leaves colour ghosts. */
vec3 clipToAABB(vec3 lo, vec3 hi, vec3 q) {
  vec3 centre = 0.5 * (hi + lo);
  vec3 extent = 0.5 * (hi - lo) + 1e-5;
  vec3 v = q - centre;
  vec3 a = abs(v / extent);
  float m = max(a.x, max(a.y, a.z));
  return m > 1.0 ? centre + v / m : q;
}

void main() {
  vec3 current = texture2D(tCurrent, vUv).rgb;

  // Closest depth in the neighbourhood drives the reprojection: on a silhouette
  // that picks the foreground, which is the object the history belongs to.
  float bestDepth = 1.0;
  vec2 bestUv = vUv;
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      vec2 uv = vUv + vec2(float(i), float(j)) * uTexel;
      float d = texture2D(tDepth, uv).x;
      if (d < bestDepth) { bestDepth = d; bestUv = uv; }
    }
  }

  vec4 ndc = vec4(bestUv * 2.0 - 1.0, bestDepth * 2.0 - 1.0, 1.0);
  vec4 world = uInvViewProj * ndc;
  world /= world.w;
  vec4 prev = uPrevViewProj * world;
  vec2 prevUv = (prev.xy / prev.w) * 0.5 + 0.5;

  float valid = step(0.0, prevUv.x) * step(prevUv.x, 1.0) * step(0.0, prevUv.y) * step(prevUv.y, 1.0);
  if (valid < 0.5 || uFeedback <= 0.0) {
    gl_FragColor = vec4(current, 1.0);
    return;
  }

  // How far the reprojection moved, in pixels. Drives both the clip width and
  // the feedback weight below — a still pixel wants a long average and a loose
  // clamp, a moving one wants neither.
  float speed = length((vUv - prevUv) / uTexel);
  float moving = clamp(speed / 6.0, 0.0, 1.0);
  float clipWidth = mix(uClipStill, uClipMoving, moving);

  // Neighbourhood statistics of the current frame, in tone-mapped YCoCg.
  vec3 m1 = vec3(0.0);
  vec3 m2 = vec3(0.0);
  vec3 nmin = vec3(1e6);
  vec3 nmax = vec3(-1e6);
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      vec3 s = rgbToYCoCg(tonemapWeight(texture2D(tCurrent, vUv + vec2(float(i), float(j)) * uTexel).rgb));
      m1 += s;
      m2 += s * s;
      nmin = min(nmin, s);
      nmax = max(nmax, s);
    }
  }
  vec3 mean = m1 / 9.0;
  vec3 sigma = sqrt(max(vec3(0.0), m2 / 9.0 - mean * mean));
  // On a still pixel the neighbourhood min/max is itself an aliasing estimate,
  // so intersecting with it re-imposes the tight clamp the width just relaxed.
  // Widen the hard bound with the soft one.
  vec3 slack = (nmax - nmin) * 0.25 * max(0.0, clipWidth - 1.0);
  vec3 lo = max(nmin - slack, mean - sigma * clipWidth);
  vec3 hi = min(nmax + slack, mean + sigma * clipWidth);

  vec3 history = tonemapWeight(sampleHistory(prevUv));
  vec3 clipped = yCoCgToRgb(clipToAABB(lo, hi, rgbToYCoCg(history)));

  // Fast camera motion means reprojection lands further from the truth and the
  // 3x3 clamp has less to work with, so lean on the current frame instead.
  float feedback = uFeedback * mix(1.0, 0.28, clamp(speed / 20.0, 0.0, 1.0));

  vec3 cur = tonemapWeight(current);
  vec3 mixed = mix(cur, clipped, feedback);
  gl_FragColor = vec4(max(vec3(0.0), tonemapUnweight(mixed)), 1.0);
}
`;

/**
 * FXAA — a compact luma-edge variant, run on the graded 8-bit image where its
 * luma estimate is perceptually correct. Used on every tier `Quality.taa` is off
 * at, so the `low` and `medium` frames are still not staircases.
 *
 * Exported as a GLSL fragment rather than a whole pass: the present pass has to
 * sample the graded image anyway, so folding FXAA into it costs no extra
 * full-screen target. It expects `tColor` and `uTexel` to be in scope.
 */
export const FXAA_GLSL = /* glsl */ `
float fxaaLuma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

vec3 fxaa(vec2 uv) {
  vec3 rgbM = texture2D(tColor, uv).rgb;
  vec3 rgbNW = texture2D(tColor, uv + vec2(-1.0, -1.0) * uTexel).rgb;
  vec3 rgbNE = texture2D(tColor, uv + vec2(1.0, -1.0) * uTexel).rgb;
  vec3 rgbSW = texture2D(tColor, uv + vec2(-1.0, 1.0) * uTexel).rgb;
  vec3 rgbSE = texture2D(tColor, uv + vec2(1.0, 1.0) * uTexel).rgb;

  float lM = fxaaLuma(rgbM);
  float lNW = fxaaLuma(rgbNW);
  float lNE = fxaaLuma(rgbNE);
  float lSW = fxaaLuma(rgbSW);
  float lSE = fxaaLuma(rgbSE);

  float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
  float lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
  if (lMax - lMin < max(0.028, lMax * 0.115)) return rgbM;

  vec2 dir = vec2(-((lNW + lNE) - (lSW + lSE)), ((lNW + lSW) - (lNE + lSE)));
  float reduce = max((lNW + lNE + lSW + lSE) * 0.03125, 0.0078125);
  float rcpMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
  dir = clamp(dir * rcpMin, vec2(-8.0), vec2(8.0)) * uTexel;

  vec3 rgbA = 0.5 * (
    texture2D(tColor, uv + dir * (1.0 / 3.0 - 0.5)).rgb +
    texture2D(tColor, uv + dir * (2.0 / 3.0 - 0.5)).rgb);
  vec3 rgbB = rgbA * 0.5 + 0.25 * (
    texture2D(tColor, uv - dir * 0.5).rgb +
    texture2D(tColor, uv + dir * 0.5).rgb);

  float lB = fxaaLuma(rgbB);
  return (lB < lMin || lB > lMax) ? rgbA : rgbB;
}
`;

export function makeTaaPass(): ScreenPass {
  return new ScreenPass(TAA_FRAGMENT, {
    tCurrent: { value: null },
    tHistory: { value: null },
    tDepth: { value: null },
    uTexel: { value: new Vector2(1, 1) },
    uInvViewProj: { value: new Matrix4() },
    uPrevViewProj: { value: new Matrix4() },
    uFeedback: { value: 0 },
    uClipStill: { value: 1.6 },
    uClipMoving: { value: 1.0 },
  });
}
