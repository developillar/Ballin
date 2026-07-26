/**
 * Bloom — progressive down-sample / up-sample, `Quality.bloomMips` levels deep.
 *
 * §8.1 and tell #40 between them rule out the thing most WebGL projects ship: a
 * single-radius gaussian applied to a low threshold, which lifts the whole frame
 * into a haze and eats the bowl-to-court ratio. What is required instead is a
 * *shape* — "wide and faint, not tight and bright", the largest mip reaching
 * 12–25% of frame height at 2–6% intensity and the tightest 6–20 px at 20–40%.
 * A COD-style chain produces that shape for free: each up-sample step adds a
 * tent-filtered copy of the smaller mip on top of what the down-sample already
 * left in the larger one, so energy falls off roughly geometrically with radius.
 *
 * **The threshold is a real number, not a guess.** `LightingSystem.grade`
 * publishes the rig's scene-linear levels for exactly this: lit hardwood sits at
 * ~0.19, the hottest varnish streak and sweat specular around 0.9, and the LED
 * ribbon and fixture pods at 1.7. Thresholding at the published 1.1 therefore
 * makes the bloom sources countable — boards, jumbotron, fixture reflections in
 * the glass and chrome, camera flashes — and leaves lit hardwood completely
 * alone, which is the explicit test in §8.1.
 *
 * Owned by the post-processing agent.
 */

import { AdditiveBlending, Vector2, type WebGLRenderTarget, type WebGLRenderer } from 'three';
import { POST_COMMON, PostQuad, ScreenPass, disposeTarget, makeTarget } from './postPasses';

const PREFILTER_FRAGMENT = /* glsl */ `
precision highp float;
varying vec2 vUv;

uniform sampler2D tColor;
uniform vec2 uTexel;
uniform float uThreshold;
uniform float uKnee;
uniform float uClamp;

${POST_COMMON}

void main() {
  // A 4-tap box at the source resolution: halves the firefly rate for one extra
  // fetch, which matters because a single hot specular pixel becomes a visible
  // pulsing dot once it has been spread across five mips.
  vec3 c = texture2D(tColor, vUv + vec2(-0.5, -0.5) * uTexel).rgb;
  c += texture2D(tColor, vUv + vec2(0.5, -0.5) * uTexel).rgb;
  c += texture2D(tColor, vUv + vec2(-0.5, 0.5) * uTexel).rgb;
  c += texture2D(tColor, vUv + vec2(0.5, 0.5) * uTexel).rgb;
  c *= 0.25;

  float l = max(postLuma(c), 1e-5);
  // Quadratic knee, so a surface drifting past the threshold fades in rather
  // than switching on and producing a crawling edge.
  float soft = clamp(l - uThreshold + uKnee, 0.0, 2.0 * uKnee);
  soft = soft * soft / (4.0 * uKnee + 1e-5);
  float contribution = max(soft, l - uThreshold) / l;

  c *= contribution;
  c = min(c, vec3(uClamp));
  gl_FragColor = vec4(max(c, vec3(0.0)), 1.0);
}
`;

/** Jimenez's 13-tap down-sample: a partial Karis average with no wobble. */
const DOWNSAMPLE_FRAGMENT = /* glsl */ `
precision highp float;
varying vec2 vUv;

uniform sampler2D tColor;
uniform vec2 uTexel;   // texel size of the SOURCE

void main() {
  vec2 t = uTexel;
  vec3 a = texture2D(tColor, vUv + vec2(-2.0, 2.0) * t).rgb;
  vec3 b = texture2D(tColor, vUv + vec2(0.0, 2.0) * t).rgb;
  vec3 c = texture2D(tColor, vUv + vec2(2.0, 2.0) * t).rgb;
  vec3 d = texture2D(tColor, vUv + vec2(-2.0, 0.0) * t).rgb;
  vec3 e = texture2D(tColor, vUv).rgb;
  vec3 f = texture2D(tColor, vUv + vec2(2.0, 0.0) * t).rgb;
  vec3 g = texture2D(tColor, vUv + vec2(-2.0, -2.0) * t).rgb;
  vec3 h = texture2D(tColor, vUv + vec2(0.0, -2.0) * t).rgb;
  vec3 i = texture2D(tColor, vUv + vec2(2.0, -2.0) * t).rgb;
  vec3 j = texture2D(tColor, vUv + vec2(-1.0, 1.0) * t).rgb;
  vec3 k = texture2D(tColor, vUv + vec2(1.0, 1.0) * t).rgb;
  vec3 l = texture2D(tColor, vUv + vec2(-1.0, -1.0) * t).rgb;
  vec3 m = texture2D(tColor, vUv + vec2(1.0, -1.0) * t).rgb;

  vec3 result = e * 0.125;
  result += (a + c + g + i) * 0.03125;
  result += (b + d + f + h) * 0.0625;
  result += (j + k + l + m) * 0.125;
  gl_FragColor = vec4(result, 1.0);
}
`;

/** 9-tap tent, blended additively into the larger mip. */
const UPSAMPLE_FRAGMENT = /* glsl */ `
precision highp float;
varying vec2 vUv;

uniform sampler2D tColor;
uniform vec2 uTexel;    // texel size of the SOURCE (smaller mip)
uniform float uRadius;
uniform float uScatter;

void main() {
  vec2 t = uTexel * uRadius;
  vec3 result = texture2D(tColor, vUv + vec2(-1.0, 1.0) * t).rgb * 1.0;
  result += texture2D(tColor, vUv + vec2(0.0, 1.0) * t).rgb * 2.0;
  result += texture2D(tColor, vUv + vec2(1.0, 1.0) * t).rgb * 1.0;
  result += texture2D(tColor, vUv + vec2(-1.0, 0.0) * t).rgb * 2.0;
  result += texture2D(tColor, vUv).rgb * 4.0;
  result += texture2D(tColor, vUv + vec2(1.0, 0.0) * t).rgb * 2.0;
  result += texture2D(tColor, vUv + vec2(-1.0, -1.0) * t).rgb * 1.0;
  result += texture2D(tColor, vUv + vec2(0.0, -1.0) * t).rgb * 2.0;
  result += texture2D(tColor, vUv + vec2(1.0, -1.0) * t).rgb * 1.0;
  gl_FragColor = vec4(result * (uScatter / 16.0), 1.0);
}
`;

export class BloomChain {
  private mips: WebGLRenderTarget[] = [];
  private readonly prefilter: ScreenPass;
  private readonly downsample: ScreenPass;
  private readonly upsample: ScreenPass;

  constructor(private levels: number) {
    this.prefilter = new ScreenPass(PREFILTER_FRAGMENT, {
      tColor: { value: null },
      uTexel: { value: new Vector2() },
      uThreshold: { value: 1.1 },
      uKnee: { value: 0.55 },
      uClamp: { value: 24 },
    });
    this.downsample = new ScreenPass(DOWNSAMPLE_FRAGMENT, {
      tColor: { value: null },
      uTexel: { value: new Vector2() },
    });
    this.upsample = new ScreenPass(
      UPSAMPLE_FRAGMENT,
      {
        tColor: { value: null },
        uTexel: { value: new Vector2() },
        uRadius: { value: 1.0 },
        uScatter: { value: 1.0 },
      },
      {},
      AdditiveBlending,
    );
  }

  /** The half-resolution mip the composite adds back. */
  get texture(): WebGLRenderTarget | null {
    return this.mips[0] ?? null;
  }

  setThreshold(threshold: number): void {
    this.prefilter.set('uThreshold', threshold);
    // A knee half the threshold wide: the fade-in spans 0.55–1.1× the LED
    // level, which is short enough that hardwood never enters it.
    this.prefilter.set('uKnee', Math.max(0.05, threshold * 0.5));
  }

  resize(width: number, height: number): void {
    this.dispose();
    this.mips = [];
    let w = Math.max(1, Math.floor(width / 2));
    let h = Math.max(1, Math.floor(height / 2));
    for (let i = 0; i < this.levels; i++) {
      this.mips.push(makeTarget(w, h, { name: `bloom.mip${i}` }));
      w = Math.max(1, Math.floor(w / 2));
      h = Math.max(1, Math.floor(h / 2));
      if (w <= 2 || h <= 2) break;
    }
  }

  render(renderer: WebGLRenderer, quad: PostQuad, source: WebGLRenderTarget, sourceScale: number): void {
    if (this.mips.length === 0) return;

    this.prefilter.set('tColor', source.texture);
    (this.prefilter.uniforms.uTexel.value as Vector2).set(1 / source.width, 1 / source.height);
    quad.draw(renderer, this.prefilter, this.mips[0]);

    for (let i = 1; i < this.mips.length; i++) {
      const src = this.mips[i - 1];
      this.downsample.set('tColor', src.texture);
      (this.downsample.uniforms.uTexel.value as Vector2).set(1 / src.width, 1 / src.height);
      quad.draw(renderer, this.downsample, this.mips[i]);
    }

    // Up-sample additively. `uScatter` under 1 is what keeps the widest mip at
    // the few-percent intensity §8.1 asks for while the tight mip stays strong.
    for (let i = this.mips.length - 1; i > 0; i--) {
      const src = this.mips[i];
      this.upsample.set('tColor', src.texture);
      (this.upsample.uniforms.uTexel.value as Vector2).set(1 / src.width, 1 / src.height);
      this.upsample.set('uRadius', 1.0 + sourceScale * 0.25);
      this.upsample.set('uScatter', 0.72);
      quad.draw(renderer, this.upsample, this.mips[i - 1]);
    }
  }

  dispose(): void {
    for (const m of this.mips) disposeTarget(m);
    this.mips = [];
  }

  disposePasses(): void {
    this.dispose();
    this.prefilter.dispose();
    this.downsample.dispose();
    this.upsample.dispose();
  }
}
