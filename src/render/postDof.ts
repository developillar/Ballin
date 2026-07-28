/**
 * Depth of field. Gated on `Quality.depthOfField`, so `high` and `ultra` only.
 *
 * Broadcast basketball has moved hard toward shallow glass, but §7.4 is very
 * specific about *how* shallow: 8–22 px circle of confusion on the crowd, 3–8 px
 * on the near hardwood at the bottom of the frame, and the rim and the shooter
 * both acceptably sharp in a RIM framing. That is a long lens with a real
 * aperture, not a portrait-mode cut-out — the crowd's faces dissolve and nothing
 * else changes much.
 *
 * Two things this pass is careful about, both named failure modes:
 *
 *  - **No background bleeding across a silhouette.** A gather kernel naively
 *    averages whatever it lands on, so a sharp player in front of a blurred
 *    crowd picks up a halo of crowd colour — §7.4's "naive single-pass DOF and
 *    very visible". Here a sample only contributes if its *own* circle of
 *    confusion reaches the centre pixel, and a sample that is behind the centre
 *    pixel is additionally weighted by the centre's own CoC. A sharp foreground
 *    pixel therefore gathers essentially only itself. Foreground over background
 *    is left intact, because that is what a real lens does.
 *  - **Sharp stays sharp.** The gathered result is cross-faded in by CoC, so
 *    anything under about a pixel of blur is returned bit-exact and the frame
 *    never goes globally soft.
 *
 * The focus plane tracks the ball with a lag (§7.4 asks for 120–250 ms — a real
 * focus puller is never instant); the tracking itself lives in `PostFX.ts`
 * because it needs the engine's frame time.
 *
 * Owned by the post-processing agent.
 */

import { Vector2 } from 'three';
import { POST_COMMON, ScreenPass } from './postPasses';

const DOF_FRAGMENT = /* glsl */ `
precision highp float;
varying vec2 vUv;

uniform sampler2D tColor;
uniform sampler2D tDepth;
uniform vec2 uTexel;
uniform float uNear;
uniform float uFar;
uniform float uFocus;      // metres
uniform float uLensK;      // px * metres; CoC at infinity is uLensK / uFocus
uniform float uMaxFar;     // px
uniform float uMaxNear;    // px
uniform float uFrame;

${POST_COMMON}

/** Signed circle of confusion in pixels. Positive behind the focus plane. */
float cocPixels(float z) {
  float c = uLensK * (1.0 / uFocus - 1.0 / max(z, 0.05));
  return clamp(c, -uMaxNear, uMaxFar);
}

float depthAt(vec2 uv) {
  return postLinearDepth(texture2D(tDepth, uv).x, uNear, uFar);
}

void main() {
  float zc = depthAt(vUv);
  float cocC = cocPixels(zc);
  float radius = abs(cocC);

  vec3 centre = texture2D(tColor, vUv).rgb;
  // Anything under a full pixel of *radius* — two of diameter — is handed back
  // bit-exact. The focus subject is the thing the player is looking at and §7.4
  // is explicit that it and the rim stay sharp; a gather that starts at three
  // quarters of a pixel was spending a visible amount of edge width on depths
  // that are, by the lens model, in focus.
  if (radius < 1.1) {
    gl_FragColor = vec4(centre, 1.0);
    return;
  }

  // Golden-angle spiral: a uniform disc, so the bokeh is round rather than the
  // hexagon a ring-based kernel produces. Rotated per pixel so the under-sampled
  // interior of the disc dithers instead of banding.
  float rot = postIGN(gl_FragCoord.xy + vec2(uFrame * 1.618034)) * 6.2831853;

  vec3 acc = centre;
  float total = 1.0;

  for (int i = 0; i < DOF_TAPS; i++) {
    float fi = float(i) + 0.5;
    float rr = sqrt(fi / float(DOF_TAPS));
    float ang = fi * 2.39996323 + rot;
    vec2 offPx = vec2(cos(ang), sin(ang)) * rr * radius;
    vec2 uv = vUv + offPx * uTexel;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) continue;

    float zs = depthAt(uv);
    float cocS = cocPixels(zs);
    float dist = length(offPx);

    // The sample only reaches this pixel if its own blur circle is wide enough.
    float spread = clamp(abs(cocS) - dist + 1.0, 0.0, 1.0);
    // ...and if it sits behind us, it may only reach us as far as OUR blur
    // circle allows. This is the whole anti-halo rule.
    float behind = step(zc + 0.02, zs);
    float reach = clamp(radius - dist + 1.0, 0.0, 1.0);
    float w = spread * mix(1.0, reach, behind);

    acc += texture2D(tColor, uv).rgb * w;
    total += w;
  }

  vec3 blurred = acc / max(total, 1e-4);
  gl_FragColor = vec4(mix(centre, blurred, smoothstep(1.1, 3.0, radius)), 1.0);
}
`;

export function makeDofPass(taps: number): ScreenPass {
  return new ScreenPass(
    DOF_FRAGMENT,
    {
      tColor: { value: null },
      tDepth: { value: null },
      uTexel: { value: new Vector2(1, 1) },
      uNear: { value: 0.1 },
      uFar: { value: 260 },
      uFocus: { value: 8 },
      uLensK: { value: 90 },
      uMaxFar: { value: 12 },
      uMaxNear: { value: 5 },
      uFrame: { value: 0 },
    },
    { DOF_TAPS: Math.max(8, taps) },
  );
}
