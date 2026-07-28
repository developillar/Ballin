/**
 * Device tiering. Everything expensive (shadow resolution, post stack, crowd
 * density, net segment count) reads its budget from the active tier so the game
 * degrades gracefully from a desktop GPU down to a mid-range phone without any
 * subsystem having to sniff the hardware itself.
 */

export type TierName = 'low' | 'medium' | 'high' | 'ultra';

export interface QualityBudget {
  tier: TierName;
  /** Upper bound on devicePixelRatio. */
  maxPixelRatio: number;
  /** Cascaded shadow map resolution. */
  shadowMapSize: number;
  shadowCascades: number;
  softShadows: boolean;
  /** Screen-space ambient occlusion. */
  ssao: boolean;
  ssaoSamples: number;
  /** Screen-space reflections on the hardwood. */
  floorReflections: boolean;
  reflectionResolution: number;
  bloom: boolean;
  bloomMips: number;
  motionBlur: boolean;
  depthOfField: boolean;
  filmGrain: boolean;
  chromaticAberration: boolean;
  /** Temporal antialiasing; falls back to FXAA when off. */
  taa: boolean;
  /** Number of individually simulated crowd members. */
  crowdCount: number;
  crowdAnimated: boolean;
  /** Net verlet iterations per step. */
  netIterations: number;
  /** Anisotropic filtering cap. */
  anisotropy: number;
  /** Texture atlas edge length for procedural bakes. */
  textureSize: number;
  /** Skinned-mesh subdivision for players. */
  playerDetail: 0 | 1 | 2;
  particles: boolean;
  volumetricLight: boolean;
}

const TIERS: Record<TierName, QualityBudget> = {
  low: {
    tier: 'low',
    maxPixelRatio: 1.0,
    shadowMapSize: 1024,
    shadowCascades: 1,
    softShadows: false,
    ssao: false,
    ssaoSamples: 0,
    floorReflections: false,
    reflectionResolution: 0,
    bloom: true,
    bloomMips: 3,
    motionBlur: false,
    depthOfField: false,
    filmGrain: true,
    chromaticAberration: false,
    taa: false,
    crowdCount: 260,
    crowdAnimated: false,
    netIterations: 3,
    anisotropy: 2,
    textureSize: 512,
    playerDetail: 0,
    particles: false,
    volumetricLight: false,
  },
  medium: {
    tier: 'medium',
    maxPixelRatio: 1.5,
    shadowMapSize: 1536,
    shadowCascades: 2,
    softShadows: true,
    ssao: true,
    ssaoSamples: 8,
    floorReflections: true,
    reflectionResolution: 256,
    bloom: true,
    bloomMips: 4,
    motionBlur: false,
    depthOfField: false,
    filmGrain: true,
    chromaticAberration: true,
    taa: false,
    crowdCount: 720,
    crowdAnimated: true,
    netIterations: 5,
    anisotropy: 4,
    textureSize: 1024,
    playerDetail: 1,
    particles: true,
    volumetricLight: false,
  },
  high: {
    tier: 'high',
    maxPixelRatio: 2.0,
    shadowMapSize: 2048,
    shadowCascades: 3,
    softShadows: true,
    ssao: true,
    ssaoSamples: 12,
    floorReflections: true,
    reflectionResolution: 512,
    bloom: true,
    bloomMips: 5,
    motionBlur: true,
    depthOfField: true,
    filmGrain: true,
    chromaticAberration: true,
    taa: true,
    crowdCount: 1600,
    crowdAnimated: true,
    netIterations: 7,
    anisotropy: 8,
    textureSize: 2048,
    playerDetail: 2,
    particles: true,
    volumetricLight: true,
  },
  ultra: {
    tier: 'ultra',
    maxPixelRatio: 2.5,
    shadowMapSize: 4096,
    shadowCascades: 4,
    softShadows: true,
    ssao: true,
    ssaoSamples: 16,
    floorReflections: true,
    reflectionResolution: 1024,
    bloom: true,
    bloomMips: 6,
    motionBlur: true,
    depthOfField: true,
    filmGrain: true,
    chromaticAberration: true,
    taa: true,
    crowdCount: 2600,
    crowdAnimated: true,
    netIterations: 10,
    anisotropy: 16,
    textureSize: 2048,
    playerDetail: 2,
    particles: true,
    volumetricLight: true,
  },
};

export function tierBudget(name: TierName): QualityBudget {
  return { ...TIERS[name] };
}

/** Heuristic first guess; the adaptive governor refines it from real frame times. */
export function detectTier(gl: WebGL2RenderingContext | WebGLRenderingContext): TierName {
  const forced = new URLSearchParams(location.search).get('quality');
  if (forced && forced in TIERS) return forced as TierName;

  const ua = navigator.userAgent;
  const mobile = /Android|iPhone|iPad|iPod|Mobile/i.test(ua);
  const cores = navigator.hardwareConcurrency ?? 4;
  const mem = (navigator as unknown as { deviceMemory?: number }).deviceMemory ?? (mobile ? 4 : 8);

  let renderer = '';
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  if (dbg) renderer = String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) ?? '');

  let score = 0;
  score += Math.min(cores, 12) * 1.0;
  score += Math.min(mem, 16) * 0.9;
  score += mobile ? -4 : 6;

  // Known-fast mobile silicon.
  if (/Apple\s*(A1[5-9]|A2\d|M[1-9])/i.test(renderer)) score += 9;
  if (/Adreno\s*(7[0-9]{2}|8[0-9]{2})/i.test(renderer)) score += 7;
  if (/Mali-G(7[1-9]|[89]\d)/i.test(renderer)) score += 4;
  if (/RTX|Radeon RX|Arc A/i.test(renderer)) score += 12;
  if (/SwiftShader|llvmpipe|Software/i.test(renderer)) score -= 30;

  if (score >= 26) return 'ultra';
  if (score >= 16) return 'high';
  if (score >= 8) return 'medium';
  return 'low';
}

/**
 * Closed-loop resolution governor. Rather than dropping whole feature tiers mid
 * play — which reads as a visual glitch — we first trade pixels, and only fall
 * back a tier when the render scale bottoms out and frames are still late.
 */
export class AdaptiveGovernor {
  renderScale = 1;
  private history: number[] = [];
  private cooldown = 0;
  private readonly targetMs: number;

  constructor(
    targetFps = 60,
    private readonly minScale = 0.62,
    private readonly maxScale = 1,
  ) {
    this.targetMs = 1000 / targetFps;
  }

  /** @returns true when the render scale changed and buffers must be resized. */
  update(frameMs: number, dt: number): boolean {
    this.history.push(frameMs);
    if (this.history.length > 45) this.history.shift();
    this.cooldown -= dt;
    if (this.history.length < 45 || this.cooldown > 0) return false;

    const sorted = [...this.history].sort((a, b) => a - b);
    const p90 = sorted[Math.floor(sorted.length * 0.9)];
    const median = sorted[Math.floor(sorted.length * 0.5)];
    const before = this.renderScale;

    if (p90 > this.targetMs * 1.22) {
      this.renderScale = Math.max(this.minScale, this.renderScale - 0.08);
      this.cooldown = 1.4;
    } else if (median < this.targetMs * 0.72) {
      this.renderScale = Math.min(this.maxScale, this.renderScale + 0.04);
      this.cooldown = 2.2;
    }

    if (this.renderScale !== before) {
      this.history.length = 0;
      return true;
    }
    return false;
  }

  /** True when we have bottomed out on resolution and still cannot hold frame. */
  get starving(): boolean {
    return this.renderScale <= this.minScale + 1e-3;
  }
}
