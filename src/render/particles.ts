/**
 * A GPU-instanced particle pool.
 *
 * One pool is one draw call. Instances are simulated on the CPU — the counts
 * here top out in the low thousands, which is nothing next to the cost of the
 * readback or the compute setup a GPU simulation would need on a phone — and
 * only the live prefix of each attribute is uploaded per frame.
 *
 * Quads rather than point sprites, for three reasons: point size is capped by
 * the driver at values low enough to matter on a tall portrait buffer, points
 * cannot rotate, and `gl_PointSize` attenuation has to be hand-rolled anyway.
 * A four-vertex instanced quad costs no more and behaves.
 */

import {
  AdditiveBlending,
  Blending,
  BufferAttribute,
  ClampToEdgeWrapping,
  DataTexture,
  DynamicDrawUsage,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  LinearFilter,
  LinearMipmapLinearFilter,
  Mesh,
  NormalBlending,
  RGBAFormat,
  ShaderMaterial,
  Vector3,
} from 'three';
import { makeRng } from '../core/MathX';

/** Sprite tiles in the generated 2×2 atlas. */
export const SPRITE = {
  /** Soft gaussian dot — sweat, glints, motes. */
  dot: 0,
  /** Noisy soft-edged blob — floor dust and haze. */
  puff: 1,
  /** Elongated soft streak — fast-moving droplets. */
  streak: 2,
  /** Hard-edged rounded rectangle — confetti and paper. */
  chip: 3,
} as const;

export type SpriteName = keyof typeof SPRITE;

const TILE = 64;
const ATLAS = TILE * 2;

/**
 * Bakes the sprite atlas. Everything in this project is generated in code, and
 * a handful of soft falloffs is exactly the kind of thing that has no business
 * being a downloaded PNG.
 */
export function makeParticleAtlas(): DataTexture {
  const data = new Uint8Array(ATLAS * ATLAS * 4);
  const rng = makeRng(0x5eed_1234);

  // Value noise for the puff tile, so dust does not read as a clean airbrush dot.
  const noise = new Float32Array(16 * 16);
  for (let i = 0; i < noise.length; i++) noise[i] = rng();
  const sampleNoise = (u: number, v: number): number => {
    const x = u * 15;
    const y = v * 15;
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const fx = x - x0;
    const fy = y - y0;
    const sx = fx * fx * (3 - 2 * fx);
    const sy = fy * fy * (3 - 2 * fy);
    const at = (px: number, py: number) => noise[(py & 15) * 16 + (px & 15)];
    const a = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * sx;
    const b = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * sx;
    return a + (b - a) * sy;
  };

  const put = (tile: number, x: number, y: number, a: number) => {
    const ox = (tile % 2) * TILE;
    const oy = Math.floor(tile / 2) * TILE;
    const p = ((oy + y) * ATLAS + ox + x) * 4;
    data[p] = 255;
    data[p + 1] = 255;
    data[p + 2] = 255;
    data[p + 3] = Math.max(0, Math.min(255, Math.round(a * 255)));
  };

  for (let y = 0; y < TILE; y++) {
    for (let x = 0; x < TILE; x++) {
      const u = (x + 0.5) / TILE;
      const v = (y + 0.5) / TILE;
      const dx = u * 2 - 1;
      const dy = v * 2 - 1;
      const r = Math.sqrt(dx * dx + dy * dy);

      // dot — gaussian, with a slightly hotter core than a pure bell so it
      // still reads as a highlight once bloom gets hold of it.
      const g = Math.exp(-r * r * 5.5);
      put(SPRITE.dot, x, y, Math.min(1, g * 1.15));

      // puff — soft disc broken up by noise, fading hard at the rim
      const edge = Math.max(0, 1 - r);
      const n = 0.55 + 0.75 * sampleNoise(u * 1.7, v * 1.7);
      put(SPRITE.puff, x, y, Math.pow(edge, 1.7) * n);

      // streak — the dot squashed along x, so a droplet stretches with motion
      const sr = Math.sqrt(dx * dx * 0.16 + dy * dy);
      put(SPRITE.streak, x, y, Math.exp(-sr * sr * 6.5));

      // chip — rounded rectangle with a 1px feather
      const box = Math.max(Math.abs(dx) / 0.78, Math.abs(dy) / 0.42);
      put(SPRITE.chip, x, y, Math.min(1, Math.max(0, (1.02 - box) * 24)));
    }
  }

  // Left in the default (no) colour space deliberately: only the alpha channel
  // is ever sampled, and alpha is not colour-managed.
  const tex = new DataTexture(data, ATLAS, ATLAS, RGBAFormat);
  tex.magFilter = LinearFilter;
  tex.minFilter = LinearMipmapLinearFilter;
  tex.wrapS = ClampToEdgeWrapping;
  tex.wrapT = ClampToEdgeWrapping;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return tex;
}

const VERT = /* glsl */ `
  attribute vec3 iPos;
  attribute vec4 iColor;      // rgb + alpha
  attribute vec4 iParams;     // size, rotation, tile index, stretch
  varying vec2 vUv;
  varying vec4 vColor;
  varying float vTile;

  void main() {
    float size = iParams.x;
    float rot = iParams.y;
    float stretch = iParams.w;

    vec4 mv = modelViewMatrix * vec4(iPos, 1.0);

    // Billboard in view space: the quad always faces the camera, then spins
    // about the view axis. Stretch elongates it along its local x, which is how
    // a droplet reads as moving rather than hovering.
    float c = cos(rot);
    float s = sin(rot);
    vec2 corner = position.xy * vec2(size * stretch, size);
    vec2 spun = vec2(corner.x * c - corner.y * s, corner.x * s + corner.y * c);

    mv.xy += spun;
    gl_Position = projectionMatrix * mv;

    vUv = uv;
    vColor = iColor;
    vTile = iParams.z;
  }
`;

const FRAG = /* glsl */ `
  uniform sampler2D uAtlas;
  varying vec2 vUv;
  varying vec4 vColor;
  varying float vTile;

  void main() {
    // Half-texel inset so bilinear filtering cannot bleed one atlas tile into
    // the next along a shared edge.
    vec2 tileOrigin = vec2(mod(vTile, 2.0), floor(vTile * 0.5)) * 0.5;
    vec2 uv = tileOrigin + clamp(vUv, 0.004, 0.996) * 0.5;
    float a = texture2D(uAtlas, uv).a * vColor.a;
    if (a < 0.004) discard;
    gl_FragColor = vec4(vColor.rgb, a);
    #include <colorspace_fragment>
  }
`;

/** The quad every instanced sprite in this module is drawn from. */
function makeQuadGeometry(): InstancedBufferGeometry {
  const geo = new InstancedBufferGeometry();
  geo.setAttribute(
    'position',
    new BufferAttribute(new Float32Array([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0]), 3),
  );
  geo.setAttribute('uv', new BufferAttribute(new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]), 2));
  geo.setIndex([0, 1, 2, 0, 2, 3]);
  geo.instanceCount = 0;
  return geo;
}

function makeSpriteMaterial(atlas: DataTexture, blending: Blending, depthWrite: boolean): ShaderMaterial {
  return new ShaderMaterial({
    uniforms: { uAtlas: { value: atlas } },
    vertexShader: VERT,
    fragmentShader: FRAG,
    transparent: true,
    depthTest: true,
    depthWrite,
    blending,
  });
}

/** Allocates and attaches the three per-instance attribute buffers. */
function attachInstanceAttributes(
  geo: InstancedBufferGeometry,
  capacity: number,
): { iPos: InstancedBufferAttribute; iColor: InstancedBufferAttribute; iParams: InstancedBufferAttribute } {
  const iPos = new InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
  const iColor = new InstancedBufferAttribute(new Float32Array(capacity * 4), 4);
  const iParams = new InstancedBufferAttribute(new Float32Array(capacity * 4), 4);
  for (const a of [iPos, iColor, iParams]) a.setUsage(DynamicDrawUsage);
  geo.setAttribute('iPos', iPos);
  geo.setAttribute('iColor', iColor);
  geo.setAttribute('iParams', iParams);
  return { iPos, iColor, iParams };
}

interface Particle {
  pos: Vector3;
  vel: Vector3;
  color: [number, number, number];
  /** Seconds remaining, and the value it started at. */
  life: number;
  maxLife: number;
  size: number;
  /** Size multiplier applied over the particle's life. */
  grow: number;
  rot: number;
  spin: number;
  tile: number;
  /** Per-second velocity retention; 1 is frictionless. */
  drag: number;
  gravity: number;
  /** Peak opacity; the fade curve scales this. */
  alpha: number;
  /** Elongation along travel, for streaks. */
  stretch: number;
  /** Bounces off y = floor instead of passing through it. */
  floor: number | null;
}

export interface EmitOptions {
  count: number;
  position: Vector3;
  /** Base velocity; `spread` is added on top in a random cone. */
  velocity?: Vector3;
  spread?: number;
  speed?: number;
  life?: [number, number];
  size?: [number, number];
  color?: [number, number, number];
  /** Per-particle random tint applied to `color`, 0..1. */
  colorJitter?: number;
  grow?: number;
  gravity?: number;
  drag?: number;
  alpha?: number;
  spin?: number;
  stretch?: number;
  sprite?: SpriteName;
  floor?: number | null;
}

/**
 * A fixed-capacity pool. Emitting past capacity recycles the oldest particles
 * rather than growing the buffers or dropping the request silently — a burst
 * that matters (a dunk, a swish) should always be visible, and the particles it
 * displaces are by definition the ones closest to dying anyway.
 */
export class ParticlePool {
  readonly mesh: Mesh;

  private readonly capacity: number;
  private readonly live: Particle[] = [];
  private readonly free: Particle[] = [];
  private readonly iPos: InstancedBufferAttribute;
  private readonly iColor: InstancedBufferAttribute;
  private readonly iParams: InstancedBufferAttribute;
  private readonly rng = makeRng(0xbadc0de);

  constructor(capacity: number, atlas: DataTexture, blending: Blending = NormalBlending, depthWrite = false) {
    this.capacity = capacity;

    const geo = makeQuadGeometry();
    const attrs = attachInstanceAttributes(geo, capacity);
    this.iPos = attrs.iPos;
    this.iColor = attrs.iColor;
    this.iParams = attrs.iParams;

    this.mesh = new Mesh(geo, makeSpriteMaterial(atlas, blending, depthWrite));
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = blending === AdditiveBlending ? 12 : 11;

    for (let i = 0; i < capacity; i++) {
      this.free.push({
        pos: new Vector3(),
        vel: new Vector3(),
        color: [1, 1, 1],
        life: 0,
        maxLife: 1,
        size: 0.1,
        grow: 1,
        rot: 0,
        spin: 0,
        tile: 0,
        drag: 1,
        gravity: 0,
        alpha: 1,
        stretch: 1,
        floor: null,
      });
    }
  }

  get liveCount(): number {
    return this.live.length;
  }

  emit(o: EmitOptions): void {
    const rng = this.rng;
    const count = Math.min(o.count, this.capacity);
    for (let i = 0; i < count; i++) {
      let p = this.free.pop();
      if (!p) {
        // Recycle whichever live particle has the least life left.
        let oldest = 0;
        for (let k = 1; k < this.live.length; k++) if (this.live[k].life < this.live[oldest].life) oldest = k;
        p = this.live.splice(oldest, 1)[0];
        if (!p) return;
      }

      p.pos.copy(o.position);
      const speed = o.speed ?? 1;
      const spread = o.spread ?? 1;
      // Uniform direction on the sphere, then blended toward the base velocity.
      const z = rng() * 2 - 1;
      const t = rng() * Math.PI * 2;
      const r = Math.sqrt(Math.max(0, 1 - z * z));
      p.vel.set(Math.cos(t) * r, z, Math.sin(t) * r).multiplyScalar(speed * spread * (0.35 + rng() * 0.65));
      if (o.velocity) p.vel.add(o.velocity);

      const [l0, l1] = o.life ?? [0.4, 0.9];
      p.maxLife = l0 + rng() * (l1 - l0);
      p.life = p.maxLife;

      const [s0, s1] = o.size ?? [0.02, 0.05];
      p.size = s0 + rng() * (s1 - s0);

      const base = o.color ?? [1, 1, 1];
      const j = o.colorJitter ?? 0;
      for (let c = 0; c < 3; c++) p.color[c] = Math.max(0, base[c] * (1 - j * 0.5 + rng() * j));

      p.grow = o.grow ?? 1;
      p.gravity = o.gravity ?? 0;
      p.drag = o.drag ?? 1;
      p.alpha = o.alpha ?? 1;
      p.rot = rng() * Math.PI * 2;
      p.spin = (o.spin ?? 0) * (rng() * 2 - 1);
      p.stretch = o.stretch ?? 1;
      p.tile = SPRITE[o.sprite ?? 'dot'];
      p.floor = o.floor ?? null;

      this.live.push(p);
    }
  }

  update(dt: number): void {
    const step = Math.min(dt, 0.05);
    for (let i = this.live.length - 1; i >= 0; i--) {
      const p = this.live[i];
      p.life -= step;
      if (p.life <= 0) {
        this.live.splice(i, 1);
        this.free.push(p);
        continue;
      }
      p.vel.y -= p.gravity * step;
      if (p.drag !== 1) p.vel.multiplyScalar(Math.pow(p.drag, step));
      p.pos.addScaledVector(p.vel, step);
      p.rot += p.spin * step;

      if (p.floor !== null && p.pos.y < p.floor) {
        p.pos.y = p.floor;
        p.vel.y = Math.abs(p.vel.y) * 0.25;
        p.vel.x *= 0.6;
        p.vel.z *= 0.6;
      }
    }
    this.upload();
  }

  private upload(): void {
    const pos = this.iPos.array as Float32Array;
    const col = this.iColor.array as Float32Array;
    const par = this.iParams.array as Float32Array;

    for (let i = 0; i < this.live.length; i++) {
      const p = this.live[i];
      const t = 1 - p.life / p.maxLife; // 0 at birth, 1 at death

      pos[i * 3] = p.pos.x;
      pos[i * 3 + 1] = p.pos.y;
      pos[i * 3 + 2] = p.pos.z;

      // Fast rise, slow fall. A linear fade reads as a light being switched
      // off; this reads as something dispersing.
      const fade = t < 0.12 ? t / 0.12 : Math.pow(1 - (t - 0.12) / 0.88, 1.6);
      col[i * 4] = p.color[0];
      col[i * 4 + 1] = p.color[1];
      col[i * 4 + 2] = p.color[2];
      col[i * 4 + 3] = p.alpha * fade;

      par[i * 4] = p.size * (1 + (p.grow - 1) * t);
      par[i * 4 + 1] = p.rot;
      par[i * 4 + 2] = p.tile;
      par[i * 4 + 3] = p.stretch;
    }

    (this.mesh.geometry as InstancedBufferGeometry).instanceCount = this.live.length;
    this.iPos.needsUpdate = true;
    this.iColor.needsUpdate = true;
    this.iParams.needsUpdate = true;
  }

  clear(): void {
    while (this.live.length) this.free.push(this.live.pop()!);
    (this.mesh.geometry as InstancedBufferGeometry).instanceCount = 0;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as ShaderMaterial).dispose();
  }
}

export interface HazeOptions {
  count: number;
  /** Half-extents of the box the motes occupy, centred on the origin in X and Z. */
  extent: Vector3;
  /** Floor of the volume; motes fill from here up to `extent.y`. */
  baseY: number;
  color: [number, number, number];
  /** Peak opacity of a mote at the bottom of the volume, at full twinkle. */
  alpha: number;
  size: number;
  seed?: number;
}

/**
 * Motes of dust suspended in the overhead light.
 *
 * Separate from `ParticlePool` because none of what a pool does applies: these
 * never spawn, never die, and are not simulated. Each one orbits a fixed home
 * point on a slow sinusoid, which costs a few trig calls and buys the single
 * strongest piece of atmosphere in the renderer. An arena interior with
 * perfectly clear air reads as a CAD viewport, and no amount of work on the
 * surfaces fixes that.
 */
export class HazeField {
  readonly mesh: Mesh;

  private readonly count: number;
  private readonly home: Float32Array;
  /** Per-mote phase, drift rate and amplitude scale. */
  private readonly seeds: Float32Array;
  private readonly opts: HazeOptions;
  private readonly iPos: InstancedBufferAttribute;
  private readonly iColor: InstancedBufferAttribute;
  private readonly iParams: InstancedBufferAttribute;

  constructor(atlas: DataTexture, opts: HazeOptions) {
    this.opts = opts;
    this.count = opts.count;

    const geo = makeQuadGeometry();
    const attrs = attachInstanceAttributes(geo, this.count);
    this.iPos = attrs.iPos;
    this.iColor = attrs.iColor;
    this.iParams = attrs.iParams;
    geo.instanceCount = this.count;

    this.mesh = new Mesh(geo, makeSpriteMaterial(atlas, AdditiveBlending, false));
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 10;

    const rng = makeRng(opts.seed ?? 0xda57);
    this.home = new Float32Array(this.count * 3);
    this.seeds = new Float32Array(this.count * 3);
    for (let i = 0; i < this.count; i++) {
      this.home[i * 3] = (rng() * 2 - 1) * opts.extent.x;
      this.home[i * 3 + 1] = opts.baseY + rng() * opts.extent.y;
      this.home[i * 3 + 2] = (rng() * 2 - 1) * opts.extent.z;
      this.seeds[i * 3] = rng() * Math.PI * 2;
      this.seeds[i * 3 + 1] = 0.1 + rng() * 0.35;
      this.seeds[i * 3 + 2] = 0.4 + rng() * 0.9;
    }

    // Colour, sprite and rotation never change, so write them once.
    const col = this.iColor.array as Float32Array;
    const par = this.iParams.array as Float32Array;
    for (let i = 0; i < this.count; i++) {
      col[i * 4] = opts.color[0];
      col[i * 4 + 1] = opts.color[1];
      col[i * 4 + 2] = opts.color[2];
      par[i * 4] = opts.size * this.seeds[i * 3 + 2];
      par[i * 4 + 1] = 0;
      par[i * 4 + 2] = SPRITE.dot;
      par[i * 4 + 3] = 1;
    }
    this.iParams.needsUpdate = true;
  }

  /** `time` is absolute, not a delta — the motion is a function of it, not integrated. */
  update(time: number): void {
    const pos = this.iPos.array as Float32Array;
    const col = this.iColor.array as Float32Array;
    const { extent, baseY, alpha } = this.opts;

    for (let i = 0; i < this.count; i++) {
      const phase = this.seeds[i * 3];
      const rate = this.seeds[i * 3 + 1];
      const scale = this.seeds[i * 3 + 2];
      const t = time * rate + phase;

      const y = this.home[i * 3 + 1] + Math.sin(t * 0.61 + 1.3) * 0.22 * scale;
      pos[i * 3] = this.home[i * 3] + Math.sin(t) * 0.5 * scale;
      pos[i * 3 + 1] = y;
      pos[i * 3 + 2] = this.home[i * 3 + 2] + Math.cos(t * 0.83) * 0.5 * scale;

      // Brightest low down where the beams are tightest, and twinkling as each
      // mote turns through them. The cube keeps most motes dim most of the time
      // so the few that catch the light actually read as catching it.
      const height = 1 - Math.min(1, Math.max(0, (y - baseY) / Math.max(extent.y, 1e-3)));
      const twinkle = 0.45 + 0.55 * Math.pow(Math.abs(Math.sin(t * 2.1 + phase)), 3);
      col[i * 4 + 3] = alpha * (0.3 + 0.7 * height) * twinkle;
    }

    this.iPos.needsUpdate = true;
    this.iColor.needsUpdate = true;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as ShaderMaterial).dispose();
  }
}
