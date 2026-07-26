/**
 * Arena lighting rig and image-based lighting.
 *
 * The look we are chasing is a modern NBA broadcast: a bright, slightly cool
 * key from the catwalk over the near sideline, a warmer fill from the opposite
 * side, hard rim light from the far corner to separate players from the dark
 * bowl, and an environment map that is mostly dark seating with two bright
 * banks of ceiling fixtures — so chrome, glass and sweaty skin all pick up the
 * signature twin highlight streaks.
 */

import {
  AmbientLight,
  DataTexture,
  DirectionalLight,
  EquirectangularReflectionMapping,
  FloatType,
  Group,
  HalfFloatType,
  LinearFilter,
  PMREMGenerator,
  PointLight,
  RGBAFormat,
  Scene,
  SpotLight,
  Texture,
  Vector3,
} from 'three';
import type { Engine, System } from '../core/Engine';
import { COURT, HOOP } from '../core/Constants';
import { clamp01, fbm2, smoothstep } from '../core/MathX';

/**
 * Procedurally bakes an equirectangular HDR of the arena interior. Generating
 * it in code (rather than shipping an .hdr) keeps the download tiny and lets
 * the environment respond to the arena's own colour scheme.
 */
export function bakeArenaEnvironment(width = 512): DataTexture {
  const height = width >> 1;
  const data = new Float32Array(width * height * 4);

  // Two catwalk banks, mirrored across the court's long axis.
  const banks = [
    { az: Math.PI * 0.5, el: 0.86, spread: 0.42, power: 26, tint: [1.0, 0.97, 0.92] },
    { az: -Math.PI * 0.5, el: 0.86, spread: 0.42, power: 22, tint: [0.94, 0.96, 1.0] },
    { az: 0, el: 1.02, spread: 0.3, power: 12, tint: [1.0, 0.98, 0.95] },
    { az: Math.PI, el: 1.02, spread: 0.3, power: 12, tint: [1.0, 0.98, 0.95] },
  ];

  for (let y = 0; y < height; y++) {
    const v = (y + 0.5) / height;
    // theta: 0 at zenith → PI at nadir.
    const theta = v * Math.PI;
    const sinT = Math.sin(theta);
    const cosT = Math.cos(theta);

    for (let x = 0; x < width; x++) {
      const u = (x + 0.5) / width;
      const phi = (u * 2 - 1) * Math.PI;
      const i = (y * width + x) * 4;

      let r = 0;
      let g = 0;
      let b = 0;

      if (cosT > 0.06) {
        // Ceiling: dark structure with exposed truss noise.
        const truss = fbm2(u * 34, v * 18, 3, 2, 0.5, 7);
        const base = 0.05 + truss * 0.05;
        r += base * 0.9;
        g += base * 0.92;
        b += base * 1.0;
      } else if (cosT < -0.12) {
        // Floor bounce: warm hardwood filling the lower hemisphere.
        const k = clamp01((-cosT - 0.12) / 0.7);
        r += 0.14 * k;
        g += 0.098 * k;
        b += 0.055 * k;
      } else {
        // Seating bowl: dark, with a faint band of concourse light.
        const seats = fbm2(u * 90, v * 40, 4, 2, 0.5, 19);
        const band = Math.exp(-Math.pow((cosT + 0.02) / 0.05, 2));
        const s = 0.028 + seats * 0.03 + band * 0.09;
        r += s * 1.0;
        g += s * 0.92;
        b += s * 0.86;
      }

      // Light banks.
      for (const bank of banks) {
        let dAz = phi - bank.az;
        while (dAz > Math.PI) dAz -= Math.PI * 2;
        while (dAz < -Math.PI) dAz += Math.PI * 2;
        const dEl = theta - (Math.PI * 0.5 - bank.el);
        // Anisotropic: the banks are long strips, wide in azimuth, thin in elevation.
        const d = Math.pow(dAz / bank.spread, 2) * 0.34 + Math.pow(dEl / 0.075, 2);
        const fall = Math.exp(-d);
        // Individual fixtures inside the strip.
        const cells = 0.55 + 0.45 * Math.pow(Math.abs(Math.cos(dAz * 26)), 6);
        const e = fall * bank.power * cells;
        r += e * bank.tint[0];
        g += e * bank.tint[1];
        b += e * bank.tint[2];
      }

      // Jumbotron glow above centre court.
      const jumboEl = theta - 0.5;
      const jumbo = Math.exp(-(Math.pow(jumboEl / 0.16, 2) + Math.pow(Math.sin(phi) / 0.9, 2))) * 0.5;
      r += jumbo * 0.9;
      g += jumbo * 0.95;
      b += jumbo * 1.25;

      // A hint of the sinT term keeps the poles from banding.
      const pole = 0.5 + 0.5 * sinT;
      data[i] = r * pole;
      data[i + 1] = g * pole;
      data[i + 2] = b * pole;
      data[i + 3] = 1;
    }
  }

  const tex = new DataTexture(data, width, height, RGBAFormat, FloatType);
  tex.mapping = EquirectangularReflectionMapping;
  tex.magFilter = LinearFilter;
  tex.minFilter = LinearFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

export class LightingSystem implements System {
  readonly name = 'lighting';
  readonly order = 5;

  group = new Group();
  key!: DirectionalLight;
  fill!: DirectionalLight;
  rim!: DirectionalLight;
  ambient!: AmbientLight;
  spots: SpotLight[] = [];
  envTexture: Texture | null = null;

  private pmrem: PMREMGenerator | null = null;

  init(engine: Engine): void {
    const { scene, renderer, quality } = engine;
    this.group.name = 'lighting';
    scene.add(this.group);

    this.pmrem = new PMREMGenerator(renderer);
    this.pmrem.compileEquirectangularShader();
    const raw = bakeArenaEnvironment(quality.tier === 'low' ? 256 : 512);
    const rt = this.pmrem.fromEquirectangular(raw);
    this.envTexture = rt.texture;
    scene.environment = rt.texture;
    scene.environmentIntensity = 1.0;
    raw.dispose();

    // Key: catwalk over the near sideline, slightly cool, casts the hero shadow.
    this.key = new DirectionalLight(0xf4f7ff, 3.2);
    this.key.position.set(6.5, 13.5, 9.5);
    this.key.castShadow = true;
    this.configureShadow(this.key, engine);
    this.group.add(this.key, this.key.target);

    // Fill: opposite catwalk, warmer, no shadow.
    this.fill = new DirectionalLight(0xffe9cf, 1.15);
    this.fill.position.set(-8.5, 11, -8.5);
    this.group.add(this.fill, this.fill.target);

    // Rim: low and behind, separates silhouettes from the dark bowl.
    this.rim = new DirectionalLight(0xbfd6ff, 1.65);
    this.rim.position.set(-4, 6.2, -16);
    this.group.add(this.rim, this.rim.target);

    this.ambient = new AmbientLight(0x2c3648, 0.34);
    this.group.add(this.ambient);

    // Practical fixtures over each basket give the glass its specular streaks.
    for (const side of [1, -1] as const) {
      const x = side * (COURT.halfLength - COURT.basketFromBaseline);
      const spot = new SpotLight(0xfff2dd, 42, 22, 0.72, 0.55, 1.6);
      spot.position.set(x * 0.82, 9.4, 0);
      spot.target.position.set(x, HOOP.rimHeight, 0);
      spot.castShadow = false;
      this.spots.push(spot);
      this.group.add(spot, spot.target);
    }

    // Bounce from the hardwood; keeps under-chins from going pure black.
    const bounce = new PointLight(0xffb877, 6, 26, 2);
    bounce.position.set(0, 0.45, 0);
    this.group.add(bounce);
  }

  private configureShadow(light: DirectionalLight, engine: Engine): void {
    const size = engine.quality.shadowMapSize;
    light.shadow.mapSize.set(size, size);
    const extent = 16;
    light.shadow.camera.left = -extent;
    light.shadow.camera.right = extent;
    light.shadow.camera.top = extent;
    light.shadow.camera.bottom = -extent;
    light.shadow.camera.near = 1;
    light.shadow.camera.far = 46;
    light.shadow.bias = -0.00042;
    light.shadow.normalBias = 0.021;
    light.shadow.radius = engine.quality.softShadows ? 2.4 : 1;
    light.shadow.camera.updateProjectionMatrix();
  }

  /** Keeps the shadow frustum tight around the action for crisp contact shadows. */
  focusShadowOn(target: Vector3): void {
    this.key.target.position.copy(target);
    this.key.position.set(target.x + 6.5, 13.5, target.z + 9.5);
    this.key.target.updateMatrixWorld();
    this.key.updateMatrixWorld();
  }

  update(_dt: number, _alpha: number, engine: Engine): void {
    const ball = engine.get<{ focusPoint?: Vector3 }>('ball');
    if (ball?.focusPoint) this.focusShadowOn(ball.focusPoint);
  }

  dispose(): void {
    this.pmrem?.dispose();
    this.envTexture?.dispose();
  }
}
