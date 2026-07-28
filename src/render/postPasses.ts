/**
 * Full-screen pass plumbing for the post stack.
 *
 * Everything in `post*.ts` is a fragment shader run over a single clip-space
 * triangle. This file owns the three things they all share: the triangle, the
 * render-target helpers, and the GLSL preamble (depth linearisation, view-space
 * reconstruction, luma, noise) that four of the passes would otherwise each
 * define their own slightly-different copy of.
 *
 * Two facts about three that the whole stack is built on, stated here because
 * they are load-bearing and non-obvious:
 *
 *  1. **Rendering into a render target disables tone mapping.** `WebGLPrograms`
 *     picks `NoToneMapping` whenever `_currentRenderTarget !== null`, so the
 *     scene pass lands in the HDR buffer as honest scene-linear radiance and the
 *     bloom threshold can be stated in the same units the light rig publishes.
 *     Exposure lives inside three's tone-mapping function, so it is *not*
 *     applied either — the composite applies `renderer.toneMappingExposure`
 *     itself, which keeps `LightingSystem` the single owner of that number.
 *  2. **A non-raw `ShaderMaterial` is always compiled as `#version 300 es`**,
 *     with `varying`, `texture2D` and `gl_FragColor` defined back to their ES
 *     1.00 spellings. So these shaders can use ES 3.0 features — `sampler3D`
 *     for the grade LUT above all — while still reading like GLSL 1.
 *
 * Owned by the post-processing agent.
 */

import {
  BufferGeometry,
  ClampToEdgeWrapping,
  DepthFormat,
  DepthTexture,
  Float32BufferAttribute,
  HalfFloatType,
  LinearFilter,
  LinearSRGBColorSpace,
  Mesh,
  NearestFilter,
  NoBlending,
  OrthographicCamera,
  RGBAFormat,
  Scene,
  ShaderMaterial,
  UnsignedByteType,
  UnsignedIntType,
  WebGLRenderTarget,
  type Blending,
  type IUniform,
  type TextureDataType,
  type WebGLRenderer,
} from 'three';

/** The rubric's reference frame height. Pixel figures scale off this. */
export const REFERENCE_HEIGHT = 2340;

/**
 * One clip-space triangle rather than a quad: no diagonal seam, one fewer
 * vertex, and — the reason that actually matters — no pixels shaded twice along
 * the diagonal, which on a stack this deep is a real cost.
 */
function screenTriangle(): BufferGeometry {
  const g = new BufferGeometry();
  g.setAttribute('position', new Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
  g.setAttribute('uv', new Float32BufferAttribute([0, 0, 2, 0, 0, 2], 2));
  return g;
}

export const SCREEN_VERTEX = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

/**
 * Shared GLSL. `uNear`/`uFar`/`uTanHalf` are declared by whichever pass needs
 * them, so this block is functions only.
 */
export const POST_COMMON = /* glsl */ `
float postLuma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

/** Window-space depth [0,1] -> positive view-space distance. */
float postLinearDepth(float d, float near, float far) {
  float z = d * 2.0 - 1.0;
  return (2.0 * near * far) / (far + near - z * (far - near));
}

/** View-space position from uv + linear distance. tanHalf = (tanFovY*aspect, tanFovY). */
vec3 postViewPos(vec2 uv, float linearZ, vec2 tanHalf) {
  return vec3((uv * 2.0 - 1.0) * tanHalf, -1.0) * linearZ;
}

/** View-space point -> uv. The analytic inverse of postViewPos. */
vec2 postProjectView(vec3 v, vec2 tanHalf) {
  return 0.5 + 0.5 * (v.xy / max(1e-5, -v.z)) / tanHalf;
}

/** Interleaved gradient noise — the cheapest well-distributed per-pixel dither. */
float postIGN(vec2 p) {
  return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

float postHash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

vec3 postLinearToSRGB(vec3 c) {
  c = clamp(c, 0.0, 1.0);
  return mix(c * 12.92, 1.055 * pow(c, vec3(0.41666)) - 0.055, step(0.0031308, c));
}

vec3 postSRGBToLinear(vec3 c) {
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}

/**
 * three's ACES filmic fit, reproduced exactly. The scene pass renders into a
 * render target, so three skipped its own tone mapping; matching the fit here
 * means switching the post stack on or off does not change the tone curve, only
 * what is layered on top of it.
 */
vec3 postACES(vec3 color, float exposure) {
  const mat3 ACESInputMat = mat3(
    vec3(0.59719, 0.07600, 0.02840),
    vec3(0.35458, 0.90834, 0.13383),
    vec3(0.04823, 0.01566, 0.83777)
  );
  const mat3 ACESOutputMat = mat3(
    vec3(1.60475, -0.10208, -0.00327),
    vec3(-0.53108, 1.10813, -0.07276),
    vec3(-0.07367, -0.00605, 1.07602)
  );
  color *= exposure / 0.6;
  color = ACESInputMat * color;
  vec3 a = color * (color + 0.0245786) - 0.000090537;
  vec3 b = color * (0.983729 * color + 0.4329510) + 0.238081;
  color = a / b;
  color = ACESOutputMat * color;
  return clamp(color, 0.0, 1.0);
}
`;

/** A fragment shader plus its uniforms, ready to be drawn by {@link PostQuad}. */
export class ScreenPass {
  readonly material: ShaderMaterial;

  constructor(
    fragmentShader: string,
    uniforms: Record<string, IUniform>,
    defines: Record<string, string | number> = {},
    blending: Blending = NoBlending,
  ) {
    this.material = new ShaderMaterial({
      vertexShader: SCREEN_VERTEX,
      fragmentShader,
      uniforms,
      defines,
      depthTest: false,
      depthWrite: false,
      blending,
      transparent: blending !== NoBlending,
    });
  }

  get uniforms(): Record<string, IUniform> {
    return this.material.uniforms;
  }

  set(name: string, value: unknown): void {
    const u = this.material.uniforms[name];
    if (u) u.value = value;
  }

  /** Recompiles the program. Only for defines that change with the tier. */
  invalidate(): void {
    this.material.needsUpdate = true;
  }

  dispose(): void {
    this.material.dispose();
  }
}

/** Draws {@link ScreenPass}es. One geometry and one camera for the whole stack. */
export class PostQuad {
  private readonly scene = new Scene();
  private readonly camera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly mesh: Mesh;

  constructor() {
    this.mesh = new Mesh(screenTriangle(), new ShaderMaterial());
    this.mesh.frustumCulled = false;
    this.mesh.matrixAutoUpdate = false;
    this.scene.add(this.mesh);
  }

  draw(renderer: WebGLRenderer, pass: ScreenPass, target: WebGLRenderTarget | null): void {
    this.mesh.material = pass.material;
    const autoClear = renderer.autoClear;
    // Never clear: the triangle covers every pixel, and the bloom up-sample
    // deliberately blends into what the down-sample already wrote there.
    renderer.autoClear = false;
    renderer.setRenderTarget(target);
    renderer.render(this.scene, this.camera);
    renderer.autoClear = autoClear;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
  }
}

export interface TargetOptions {
  type?: TextureDataType;
  depth?: boolean;
  filter?: typeof LinearFilter | typeof NearestFilter;
  name?: string;
}

export function makeTarget(width: number, height: number, opts: TargetOptions = {}): WebGLRenderTarget {
  const filter = opts.filter ?? LinearFilter;
  const rt = new WebGLRenderTarget(Math.max(1, width), Math.max(1, height), {
    type: opts.type ?? HalfFloatType,
    format: RGBAFormat,
    minFilter: filter,
    magFilter: filter,
    wrapS: ClampToEdgeWrapping,
    wrapT: ClampToEdgeWrapping,
    depthBuffer: opts.depth === true,
    stencilBuffer: false,
    generateMipmaps: false,
  });
  rt.texture.colorSpace = LinearSRGBColorSpace;
  rt.texture.name = opts.name ?? 'post';
  if (opts.depth) {
    const depth = new DepthTexture(Math.max(1, width), Math.max(1, height));
    depth.type = UnsignedIntType;
    depth.format = DepthFormat;
    depth.minFilter = NearestFilter;
    depth.magFilter = NearestFilter;
    rt.depthTexture = depth;
  }
  return rt;
}

/** An 8-bit target for anything already through the tone curve. */
export function makeLdrTarget(width: number, height: number, name: string): WebGLRenderTarget {
  return makeTarget(width, height, { type: UnsignedByteType, name });
}

export function disposeTarget(rt: WebGLRenderTarget | null): void {
  if (!rt) return;
  rt.depthTexture?.dispose();
  rt.dispose();
}
