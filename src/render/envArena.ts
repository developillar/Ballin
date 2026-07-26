/**
 * Procedural HDR environment for the arena interior.
 *
 * There is no .hdr to download, so the whole room is written into an
 * equirectangular float buffer and handed to PMREM. What it has to deliver, in
 * order of how much it shows up in the frame:
 *
 *  1. **Recognisable fixture shapes.** Chrome and glass must reflect a *grid of
 *     bright rectangles*, not a smooth gradient. Every lens in `lightRig` is
 *     painted as a small anisotropic quad with a soft housing halo around it.
 *  2. **A dark bowl.** The seating band is 2.5–4 stops under the floor and has
 *     architecture in it — seat rows, vomitory mouths, phone screens — so a
 *     rough metal picks up structure rather than a flat grey.
 *  3. **A warm floor hemisphere.** The hardwood throws a lot of amber light back
 *     up; everything below the horizon is warm and carries a smeared, vertically
 *     stretched reflection of the banks, which is what the varnish actually does.
 *  4. **LED features.** The ribbon board and the jumbotron are emissive bands so
 *     the apron and the stanchion pick up team colour.
 *
 * Orientation matters and is easy to get backwards: three samples an equirect
 * with `v = asin(dir.y)/PI + 0.5`, and a `DataTexture` has `flipY = false`, so
 * **row 0 of the buffer is straight down** and the last row is the zenith.
 */

import {
  ClampToEdgeWrapping,
  DataTexture,
  EquirectangularReflectionMapping,
  FloatType,
  LinearFilter,
  RGBAFormat,
  RepeatWrapping,
  Vector3,
} from 'three';
import { clamp01, fbm2, hash2, lerp, smoothstep } from '../core/MathX';
import { LIGHT_RIG } from '../core/Constants';
import { rigBanks } from './lightRig';

export interface ArenaEnvOptions {
  /** Equirect width in texels; height is half. PMREM uses width/4 per cube face. */
  width?: number;
  /** Scene-linear radiance of a fixture lens. The brightest thing in the room. */
  fixtureLuminance?: number;
  /** Scene-linear radiance of the LED ribbon. */
  ledLuminance?: number;
  ledColor?: readonly [number, number, number];
  /** Scene-linear radiance of lit hardwood — drives the warm bounce hemisphere. */
  floorLuminance?: number;
  /** Scene-linear radiance of the seating bowl. */
  bowlLuminance?: number;
  seed?: number;
}

const LUMA = [0.2126, 0.7152, 0.0722] as const;

/** Hardwood bounce, ~3300 K. */
const WARM = [1.0, 0.755, 0.472] as const;
/** Broadcast banks, ~5600 K. */
const BANK = [0.965, 0.978, 1.0] as const;
/** Bowl spill: phone screens and concourse fluorescents read cool. */
const BOWL = [0.80, 0.87, 1.0] as const;

function scaleFor(tint: readonly number[], luminance: number): number {
  const y = tint[0] * LUMA[0] + tint[1] * LUMA[1] + tint[2] * LUMA[2];
  return luminance / Math.max(1e-5, y);
}

/**
 * Bakes the arena interior. Returns a float equirect ready for
 * `PMREMGenerator.fromEquirectangular`.
 */
export function bakeArenaEnvironment(opts: ArenaEnvOptions = {}): DataTexture {
  const width = Math.max(64, opts.width ?? 1024);
  const height = width >> 1;
  const fixtureL = opts.fixtureLuminance ?? 1.7;
  const ledL = opts.ledLuminance ?? 1.7;
  const led = opts.ledColor ?? ([0.16, 0.42, 1.0] as const);
  const floorL = opts.floorLuminance ?? 0.145;
  const bowlL = opts.bowlLuminance ?? 0.0135;
  const seed = opts.seed ?? 1337;

  const data = new Float32Array(width * height * 4);

  const warmS = scaleFor(WARM, 1);
  const bowlS = scaleFor(BOWL, 1);

  // ---------------------------------------------------------------- base pass
  for (let py = 0; py < height; py++) {
    const v = (py + 0.5) / height;
    const e = (v - 0.5) * Math.PI; // elevation: -PI/2 nadir, +PI/2 zenith
    const cosE = Math.cos(e);

    for (let px = 0; px < width; px++) {
      const u = (px + 0.5) / width;
      const i = (py * width + px) * 4;
      let r = 0;
      let g = 0;
      let b = 0;

      if (e < -0.03) {
        // ---- Lower hemisphere: varnished hardwood, apron, courtside.
        // Steeply down is the court itself; near the horizon we are looking
        // across the apron and the courtside furniture, which is much darker.
        const k = smoothstep(clamp01((-e - 0.03) / 0.62));
        const lum = lerp(floorL * 0.52, floorL, k);
        // Board-to-board tone scatter survives PMREM as a faint break-up.
        const grain = 1 + (fbm2(u * 26, v * 52, 3, 2, 0.5, seed) - 0.5) * 0.22;
        const s = lum * grain * warmS;
        r = WARM[0] * s;
        g = WARM[1] * s;
        b = WARM[2] * s;
      } else if (e < 0.035) {
        // ---- Courtside: apron, scorer's table, photographers. Warm-neutral,
        // and the closest thing in the bowl to the floor's brightness.
        const t = smoothstep(clamp01((e + 0.03) / 0.065));
        const lum = lerp(floorL * 0.48, floorL * 0.17, t);
        const s = lum * warmS;
        r = WARM[0] * s * 0.94;
        g = WARM[1] * s * 1.02;
        b = WARM[2] * s * 1.2;
      } else {
        // ---- The bowl and everything above it.
        // Seat rake: fine horizontal banding that fades out with height.
        const rake = 0.5 + 0.5 * Math.sin(e * 168 + Math.sin(u * 31) * 0.9);
        const rakeAmt = 0.42 * (1 - smoothstep(clamp01((e - 0.06) / 0.30)));
        // Vomitory mouths: dark rectangles punched through the seating.
        const vom = smoothstep(clamp01((Math.abs(Math.sin(u * Math.PI * 9)) - 0.965) * 46));
        // Crowd texture.
        const crowd = fbm2(u * 210, v * 96, 4, 2.1, 0.52, seed + 7);

        let lum: number;
        if (e < 0.34) {
          // Lower bowl. Brightest at the bottom where floor spill reaches it.
          const drop = 1 - smoothstep(clamp01((e - 0.04) / 0.30)) * 0.45;
          lum = bowlL * drop * (0.55 + crowd * 0.9) * (1 - rakeAmt * rake) * (1 - vom * 0.8);
        } else if (e < 0.62) {
          // Suite band / upper deck front: a faint lit strip of glass.
          const suite = Math.exp(-Math.pow((e - 0.455) / 0.030, 2));
          lum = bowlL * 0.42 * (0.6 + crowd * 0.7) + suite * bowlL * 1.9;
        } else if (e < 1.02) {
          // Upper darkness between the bowl lip and the rafters. Never zero —
          // exit signs and catwalk service lighting live here.
          lum = bowlL * (0.20 + crowd * 0.22);
        } else {
          // Rafters: exposed truss, catwalks, hanging speaker arrays.
          const truss = fbm2(u * 40, v * 26, 3, 2, 0.55, seed + 21);
          const beam = 0.5 + 0.5 * Math.sin(u * Math.PI * 24);
          lum = bowlL * (0.5 + truss * 1.1 + beam * 0.5);
        }

        const s = lum * bowlS;
        r = BOWL[0] * s;
        g = BOWL[1] * s;
        b = BOWL[2] * s;

        // ---- LED ribbon around the lip of the lower bowl.
        const ribbon = Math.exp(-Math.pow((e - 0.132) / 0.0165, 2));
        if (ribbon > 0.004) {
          // Content, not a flat bar: sponsor blocks and a scrolling score strip.
          const seg = Math.floor(u * 46);
          const block = 0.45 + 0.55 * hash2(seg, 3, seed + 55);
          const scan = 0.82 + 0.18 * Math.sin(e * 900);
          const hue = hash2(seg, 9, seed + 91);
          const eR = lerp(led[0], 1.0, hue * 0.5);
          const eG = lerp(led[1], 0.35, hue * 0.35);
          const eB = lerp(led[2], 0.62, hue * 0.2);
          const s2 = ribbon * ledL * block * scan;
          r += eR * s2;
          g += eG * s2;
          b += eB * s2;
        }

        // ---- Scorer's table LED face, low and only along the sidelines.
        const table = Math.exp(-Math.pow((e - 0.055) / 0.014, 2)) *
          Math.pow(Math.abs(Math.sin(u * Math.PI * 2)), 6);
        if (table > 0.004) {
          const s2 = table * ledL * 0.5;
          r += led[0] * s2;
          g += led[1] * s2;
          b += led[2] * s2;
        }

        // ---- Phone screens scattered through the dark bowl.
        if (e > 0.05 && e < 0.60) {
          const cellU = Math.floor(u * 260);
          const cellV = Math.floor(v * 130);
          const h = hash2(cellU, cellV, seed + 313);
          if (h > 0.9965) {
            const s2 = 0.42;
            r += 0.72 * s2;
            g += 0.82 * s2;
            b += 1.0 * s2;
          }
        }

        // ---- Jumbotron, hung over centre court: a large soft cool rectangle
        // near the zenith with visible pixel-grid structure.
        if (e > 0.98) {
          const face = Math.exp(-Math.pow((e - 1.14) / 0.115, 2));
          const grid = 0.86 + 0.14 * Math.sin(u * width * 0.09) * Math.sin(e * 620);
          const content = 0.4 + 0.6 * fbm2(u * 18, v * 30, 2, 2, 0.5, seed + 401);
          const s2 = face * 1.05 * grid * content;
          r += 0.80 * s2;
          g += 0.88 * s2;
          b += 1.0 * s2;
        }
      }

      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
      data[i + 3] = 1;
    }
  }

  // ------------------------------------------------------------- fixture pass
  // Painted after the base so each lens is additive, and only over the small
  // window of texels it actually subtends. This is what puts rectangular
  // highlight streaks on the backboard glass and the stanchion chrome.
  const dir = new Vector3();
  const t1 = new Vector3();
  const t2 = new Vector3();
  const up = new Vector3(0, 1, 0);
  const xAxis = new Vector3(1, 0, 0);
  const zAxis = new Vector3(0, 0, 1);
  const tmp = new Vector3();

  const paintQuad = (
    world: Vector3,
    halfW: number,
    halfD: number,
    lum: number,
    tint: readonly number[],
    spread: number,
    stretchV: number,
    mirror: boolean,
  ): void => {
    dir.copy(world);
    if (mirror) dir.y = -dir.y;
    const dist = dir.length();
    if (dist < 1e-3) return;
    dir.multiplyScalar(1 / dist);

    // Tangent frame: t1 runs around the azimuth, t2 up the elevation.
    t1.crossVectors(up, dir);
    if (t1.lengthSq() < 1e-6) t1.set(1, 0, 0);
    t1.normalize();
    t2.crossVectors(dir, t1).normalize();

    // Angular half-extents of the (horizontal) lens rectangle in that frame.
    const a1 = (Math.abs(xAxis.dot(t1)) * halfW + Math.abs(zAxis.dot(t1)) * halfD) / dist;
    const a2 =
      ((Math.abs(xAxis.dot(t2)) * halfW + Math.abs(zAxis.dot(t2)) * halfD) / dist) * stretchV;
    const ra = Math.max(a1, a2) * spread + 0.004;

    const eC = Math.asin(Math.max(-1, Math.min(1, dir.y)));
    const phiC = Math.atan2(dir.z, dir.x);
    const vC = eC / Math.PI + 0.5;
    const uC = phiC / (Math.PI * 2) + 0.5;

    const dv = (a2 * spread + 0.012) / Math.PI;
    const du = (a1 * spread + 0.012) / (Math.PI * 2 * Math.max(0.08, Math.cos(eC)));

    const y0 = Math.max(0, Math.floor((vC - dv) * height));
    const y1 = Math.min(height - 1, Math.ceil((vC + dv) * height));
    const x0 = Math.floor((uC - du) * width);
    const x1 = Math.ceil((uC + du) * width);

    const sc = scaleFor(tint, lum);

    for (let py = y0; py <= y1; py++) {
      const vv = (py + 0.5) / height;
      const ee = (vv - 0.5) * Math.PI;
      const ce = Math.cos(ee);
      const se = Math.sin(ee);
      for (let xx = x0; xx <= x1; xx++) {
        const px = ((xx % width) + width) % width;
        const uu = (px + 0.5) / width;
        const ph = (uu - 0.5) * Math.PI * 2;
        tmp.set(Math.cos(ph) * ce, se, Math.sin(ph) * ce);
        const cosang = tmp.dot(dir);
        if (cosang <= 0) continue;
        // Offset within the tangent frame, in radians.
        tmp.addScaledVector(dir, -cosang);
        const o1 = Math.abs(tmp.dot(t1));
        const o2 = Math.abs(tmp.dot(t2));
        if (o1 > ra || o2 > ra) continue;
        // Rectangular core with a soft shoulder — a real fixture has a hard
        // lens edge and a glow off the reflector housing around it.
        const f1 = 1 - smoothstep(clamp01((o1 - a1) / Math.max(1e-5, a1 * (spread - 1) + 1e-4)));
        const f2 = 1 - smoothstep(clamp01((o2 - a2) / Math.max(1e-5, a2 * (spread - 1) + 1e-4)));
        const f = f1 * f2;
        if (f <= 0.0005) continue;
        const i = (py * width + px) * 4;
        data[i] += tint[0] * sc * f;
        data[i + 1] += tint[1] * sc * f;
        data[i + 2] += tint[2] * sc * f;
      }
    }
  };

  const halfW = LIGHT_RIG.pod.width * 0.5;
  const halfD = LIGHT_RIG.pod.depth * 0.5;
  const podTint: [number, number, number] = [0, 0, 0];
  for (const bank of rigBanks()) {
    podTint[0] = BANK[0] * bank.tint[0];
    podTint[1] = BANK[1] * bank.tint[1];
    podTint[2] = BANK[2] * bank.tint[2];
    const L = fixtureL * bank.gain;
    // Spacing between pods in this run — used to bridge them into a continuous
    // strip. A run painted as one quad would be wrong: 44 m of catwalk subtends
    // more than a radian and the small-angle tangent frame falls apart.
    const step =
      bank.pods.length > 1 ? bank.pods[0].distanceTo(bank.pods[1]) : LIGHT_RIG.pod.width;
    const along = new Vector3();
    if (bank.pods.length > 1) along.subVectors(bank.pods[1], bank.pods[0]).normalize();
    const alongX = Math.abs(along.x) >= Math.abs(along.z);

    for (const p of bank.pods) {
      // The lens: small, hard-edged, very bright. This is the 1–3 small hard
      // speculars at 200–255 on the backboard glass.
      paintQuad(p, halfW, halfD, L, podTint, 1.25, 1, false);
      // The reflector housing glow: broad, an order dimmer, still rectangular.
      paintQuad(p, halfW * 2.8, halfD * 2.8, L * 0.045, podTint, 1.8, 1, false);
      // The catwalk run bridged into a continuous strip. This is what turns the
      // reflection in glass and chrome into recognisable *bars* rather than a
      // scatter of dots, and it is the single strongest environment cue.
      const bw = alongX ? step * 0.62 : halfD * 1.3;
      const bd = alongX ? halfD * 1.3 : step * 0.62;
      paintQuad(p, bw, bd, L * 0.085, podTint, 1.5, 1, false);
      // And the same pod smeared back off the varnish, stretched vertically, so
      // chrome carries a bright warm floor band with streak structure in it.
      paintQuad(p, halfW * 2.0, halfD * 2.0, L * 0.055, WARM, 2.6, 4.5, true);
    }
  }

  const tex = new DataTexture(data, width, height, RGBAFormat, FloatType);
  tex.mapping = EquirectangularReflectionMapping;
  tex.magFilter = LinearFilter;
  tex.minFilter = LinearFilter;
  tex.wrapS = RepeatWrapping;
  tex.wrapT = ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}
