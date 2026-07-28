/**
 * Screen-space ambient occlusion, and the depth-aware blur that makes it
 * usable.
 *
 * The job here is narrow and specific: **tighten contact**. Under a planted
 * sole, in the diamonds of the net, where the stanchion base meets the apron,
 * in the armpit and under the pec. §9.1 asks for the floor immediately under a
 * shoe to sit at 35–55% of its unoccluded luminance rising to 70–85% at 150 mm,
 * and that gradient is exactly a 0.3 m-radius occlusion term. It is *not* a
 * global dirt layer — every unit of AO on an unoccluded surface is a unit off
 * the court-to-bowl ratio in §1.1, which is the criterion the whole frame is
 * judged on.
 *
 * Three decisions worth stating:
 *
 *  - **Normal-oriented hemisphere sampling, not a depth-difference blur.** The
 *    normal comes from the *closest* of the four neighbouring depth taps in each
 *    axis, which is what keeps a silhouette edge from generating a bogus normal
 *    and therefore a bogus dark halo around every player.
 *  - **Half resolution, joint-bilateral upsampled.** AO at a 0.3 m radius has
 *    essentially no detail above the half-res Nyquist, and the pass is the most
 *    expensive one in the stack. The halo that half-res AO is famous for comes
 *    from a bilinear upsample across a depth discontinuity; the resolve pass
 *    weights its four taps by depth similarity instead, so the term stops dead
 *    at a silhouette.
 *  - **A range check on every sample.** Without it, a wall 8 m behind a player
 *    occludes the player, which is the classic SSAO black outline.
 *
 * Owned by the post-processing agent.
 */

import { Vector2, Vector3 } from 'three';
import { POST_COMMON, ScreenPass } from './postPasses';

/**
 * Cosine-ish hemisphere kernel, deterministically generated so a frame is
 * reproducible between runs. Samples are pulled toward the origin so the
 * majority of the taps sit in the near field where contact actually lives.
 */
export function aoKernel(count: number): Vector3[] {
  const out: Vector3[] = [];
  let seed = 0x9e3779b9;
  const rnd = (): number => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  for (let i = 0; i < count; i++) {
    const v = new Vector3(rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 0.92 + 0.08);
    v.normalize();
    // Quadratic bias toward the centre of the hemisphere.
    const t = (i + 0.5) / count;
    v.multiplyScalar(0.18 + 0.82 * t * t);
    out.push(v);
  }
  return out;
}

const AO_FRAGMENT = /* glsl */ `
precision highp float;
varying vec2 vUv;

uniform sampler2D tDepth;
uniform vec2 uTexel;      // 1 / AO-buffer size
uniform vec2 uTanHalf;
uniform float uNear;
uniform float uFar;
uniform float uRadius;    // world metres
uniform float uBias;      // world metres
uniform float uIntensity;
uniform float uMaxScreenRadius; // uv units, stops close-up taps thrashing cache
uniform float uFrame;
uniform vec3 uKernel[AO_SAMPLES];

${POST_COMMON}

float depthAt(vec2 uv) {
  return postLinearDepth(texture2D(tDepth, uv).x, uNear, uFar);
}

vec3 viewAt(vec2 uv) {
  return postViewPos(uv, depthAt(uv), uTanHalf);
}

void main() {
  float z = depthAt(vUv);

  // Nothing behind the far plane, and nothing in the sky, gets occluded.
  if (z >= uFar * 0.985) {
    gl_FragColor = vec4(1.0);
    return;
  }

  vec3 p = postViewPos(vUv, z, uTanHalf);

  // Silhouette-safe normal: take the nearer of the two neighbours on each axis
  // so the derivative never straddles a depth discontinuity.
  vec3 l = viewAt(vUv - vec2(uTexel.x, 0.0));
  vec3 r = viewAt(vUv + vec2(uTexel.x, 0.0));
  vec3 d = viewAt(vUv - vec2(0.0, uTexel.y));
  vec3 u = viewAt(vUv + vec2(0.0, uTexel.y));
  vec3 dx = abs(l.z - p.z) < abs(r.z - p.z) ? (p - l) : (r - p);
  vec3 dy = abs(d.z - p.z) < abs(u.z - p.z) ? (p - d) : (u - p);
  vec3 n = normalize(cross(dx, dy));
  if (n.z < 0.0) n = -n;

  // Per-pixel rotation, animated so TAA resolves the sampling noise away.
  float angle = postIGN(gl_FragCoord.xy + vec2(uFrame * 5.588238, uFrame * 3.141593)) * 6.2831853;
  vec3 rv = vec3(cos(angle), sin(angle), 0.0);
  vec3 t = normalize(rv - n * dot(rv, n));
  vec3 b = cross(n, t);
  mat3 tbn = mat3(t, b, n);

  float occlusion = 0.0;
  float weight = 0.0;

  for (int i = 0; i < AO_SAMPLES; i++) {
    vec3 sv = p + (tbn * uKernel[i]) * uRadius;
    if (sv.z > -uNear) continue;

    vec2 suv = postProjectView(sv, uTanHalf);
    vec2 delta = suv - vUv;
    float dl = length(delta);
    if (dl > uMaxScreenRadius) {
      suv = vUv + delta * (uMaxScreenRadius / max(dl, 1e-5));
    }
    if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) continue;

    float sampleZ = depthAt(suv);
    float sampleViewZ = -sampleZ;

    // A surface closer to the camera than the sample point occludes it, but
    // only while it is within the sampling radius — otherwise the far stands
    // would occlude everything in front of them.
    float range = smoothstep(0.0, 1.0, uRadius / max(1e-4, abs(p.z - sampleViewZ)));
    float occluded = step(sv.z + uBias, sampleViewZ);
    occlusion += occluded * range;
    weight += 1.0;
  }

  float ao = 1.0 - uIntensity * (occlusion / max(1.0, weight));
  gl_FragColor = vec4(clamp(ao, 0.0, 1.0));
}
`;

/**
 * Separable cross-bilateral blur. The depth term is what stops the occlusion
 * under a shoe from smearing out onto the hardwood as a soft grey puddle — the
 * "haloing" failure mode named in the task.
 */
const AO_BLUR_FRAGMENT = /* glsl */ `
precision highp float;
varying vec2 vUv;

uniform sampler2D tAO;
uniform sampler2D tDepth;
uniform vec2 uDirection;  // texel-sized step
uniform float uNear;
uniform float uFar;
uniform float uDepthSharpness;

${POST_COMMON}

void main() {
  float centre = postLinearDepth(texture2D(tDepth, vUv).x, uNear, uFar);
  const float w0 = 0.2270270270;
  const float w1 = 0.1945945946;
  const float w2 = 0.1216216216;
  const float w3 = 0.0540540541;
  float weights[4];
  weights[0] = w0; weights[1] = w1; weights[2] = w2; weights[3] = w3;

  float sum = texture2D(tAO, vUv).r * w0;
  float total = w0;

  for (int i = 1; i < 4; i++) {
    for (int s = 0; s < 2; s++) {
      vec2 off = uDirection * float(i) * (s == 0 ? 1.0 : -1.0);
      vec2 uv = vUv + off;
      float dz = postLinearDepth(texture2D(tDepth, uv).x, uNear, uFar);
      float dw = exp(-abs(dz - centre) * uDepthSharpness);
      float w = weights[i] * dw;
      sum += texture2D(tAO, uv).r * w;
      total += w;
    }
  }

  gl_FragColor = vec4(sum / max(total, 1e-4), 0.0, 0.0, 1.0);
}
`;

export function makeAOPass(samples: number): ScreenPass {
  const kernel = aoKernel(samples);
  return new ScreenPass(
    AO_FRAGMENT,
    {
      tDepth: { value: null },
      uTexel: { value: new Vector2(1, 1) },
      uTanHalf: { value: new Vector2(1, 1) },
      uNear: { value: 0.1 },
      uFar: { value: 260 },
      uRadius: { value: 0.34 },
      uBias: { value: 0.022 },
      uIntensity: { value: 0.85 },
      uMaxScreenRadius: { value: 0.08 },
      uFrame: { value: 0 },
      uKernel: { value: kernel },
    },
    { AO_SAMPLES: Math.max(4, samples) },
  );
}

export function makeAOBlurPass(): ScreenPass {
  return new ScreenPass(AO_BLUR_FRAGMENT, {
    tAO: { value: null },
    tDepth: { value: null },
    uDirection: { value: new Vector2(1, 0) },
    uNear: { value: 0.1 },
    uFar: { value: 260 },
    uDepthSharpness: { value: 22 },
  });
}
