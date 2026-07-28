/**
 * The resolve pass: joint-bilateral AO upsample and camera motion blur, folded
 * into one full-screen pass because both of them read the HDR colour and the
 * depth buffer and neither needs the other's output.
 *
 * **Motion blur is camera-only, and it is normalised to a 1/120 s shutter.**
 * There is no velocity buffer: writing one means an override material for every
 * skinned, instanced and patched material in the project, all of which belong to
 * other agents. Reconstructing world position from depth and re-projecting it
 * through the previous frame's view-projection gives the camera's contribution
 * exactly, which is the half a broadcast camera actually shows — a long lens
 * panning with the break smears the bowl, not the ball. §4.5's per-object smear
 * on a shot ball is the part this cannot do, and it is called out as a known gap
 * rather than faked.
 *
 * The shutter normalisation matters more than it sounds: the review harness
 * rasterises in software at 5–10 fps, so a per-*frame* velocity would smear a
 * still capture into mush. Scaling by `(1/120) / dt` means the blur describes a
 * real exposure time regardless of what the frame rate is doing.
 *
 * Owned by the post-processing agent.
 */

import { Matrix4, Vector2 } from 'three';
import { POST_COMMON, ScreenPass } from './postPasses';

const RESOLVE_FRAGMENT = /* glsl */ `
precision highp float;
varying vec2 vUv;

uniform sampler2D tColor;
uniform sampler2D tDepth;
uniform sampler2D tAO;
uniform vec2 uTexel;      // 1 / full-res size
uniform vec2 uAOTexel;    // 1 / AO-buffer size
uniform mat4 uInvViewProj;
uniform mat4 uPrevViewProj;
uniform float uNear;
uniform float uFar;
uniform float uShutter;     // 0 disables the blur entirely
uniform float uMaxBlurPx;
uniform float uAOStrength;

${POST_COMMON}

/**
 * Four taps at the AO buffer's texel centres, weighted by how well each one's
 * depth agrees with this pixel's. A plain bilinear fetch here is what puts a
 * grey halo around every silhouette when the AO buffer is at half resolution.
 */
float fetchAO(vec2 uv, float centreZ) {
  float acc = 0.0;
  float total = 0.0;
  for (int j = 0; j < 2; j++) {
    for (int i = 0; i < 2; i++) {
      vec2 off = (vec2(float(i), float(j)) - 0.5) * uAOTexel;
      vec2 s = clamp(uv + off, vec2(0.0), vec2(1.0));
      float sz = postLinearDepth(texture2D(tDepth, s).x, uNear, uFar);
      float w = 1.0 / (0.02 + abs(sz - centreZ));
      acc += texture2D(tAO, s).r * w;
      total += w;
    }
  }
  return acc / max(total, 1e-4);
}

void main() {
  float rawDepth = texture2D(tDepth, vUv).x;
  float z = postLinearDepth(rawDepth, uNear, uFar);
  vec4 base = texture2D(tColor, vUv);
  vec3 colour = base.rgb;

#if MOTION_BLUR
  if (uShutter > 0.0 && rawDepth < 1.0) {
    vec4 ndc = vec4(vUv * 2.0 - 1.0, rawDepth * 2.0 - 1.0, 1.0);
    vec4 world = uInvViewProj * ndc;
    world /= world.w;
    vec4 prev = uPrevViewProj * world;
    vec2 prevUv = (prev.xy / prev.w) * 0.5 + 0.5;

    vec2 velocity = (vUv - prevUv) * uShutter;
    vec2 velPx = velocity / uTexel;
    float len = length(velPx);
    if (len > uMaxBlurPx) velocity *= uMaxBlurPx / len;

    // Under a pixel and a half of span there is nothing to smear and the tap
    // loop is a plain box blur applied for no reason — which on a camera that
    // carries §7.3's micro-handheld noise is *every* frame, including a still
    // capture the operator meant to be sharp.
    if (len > 1.5) {
      float jitter = postIGN(gl_FragCoord.xy) - 0.5;
      vec3 acc = vec3(0.0);
      for (int i = 0; i < MOTION_TAPS; i++) {
        float t = (float(i) + 0.5 + jitter) / float(MOTION_TAPS) - 0.5;
        acc += texture2D(tColor, clamp(vUv + velocity * t, vec2(0.0), vec2(1.0))).rgb;
      }
      colour = acc / float(MOTION_TAPS);
    }
  }
#endif

#if USE_AO
  float ao = fetchAO(vUv, z);
  // Emissive and near-clipping surfaces keep their energy: an LED ribbon
  // dimmed by the wall behind it is a bloom source the stack just threw away.
  float bright = smoothstep(0.55, 1.5, postLuma(colour));
  float amount = uAOStrength * (1.0 - bright);
  colour *= mix(1.0, ao, amount);
#endif

  gl_FragColor = vec4(colour, base.a);
}
`;

export interface ResolveOptions {
  motionBlur: boolean;
  motionTaps: number;
  ambientOcclusion: boolean;
}

export function makeResolvePass(opts: ResolveOptions): ScreenPass {
  return new ScreenPass(
    RESOLVE_FRAGMENT,
    {
      tColor: { value: null },
      tDepth: { value: null },
      tAO: { value: null },
      uTexel: { value: new Vector2(1, 1) },
      uAOTexel: { value: new Vector2(1, 1) },
      uInvViewProj: { value: new Matrix4() },
      uPrevViewProj: { value: new Matrix4() },
      uNear: { value: 0.1 },
      uFar: { value: 260 },
      uShutter: { value: 0 },
      uMaxBlurPx: { value: 18 },
      uAOStrength: { value: 0.72 },
    },
    {
      MOTION_BLUR: opts.motionBlur ? 1 : 0,
      MOTION_TAPS: Math.max(3, opts.motionTaps),
      USE_AO: opts.ambientOcclusion ? 1 : 0,
    },
  );
}
