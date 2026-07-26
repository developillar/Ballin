/**
 * Whole-court bake: the signals that must *not* repeat.
 *
 * Three outputs, all procedural:
 *
 *  • **albedo** — maple base tone, panel-to-panel stain scatter, the darker
 *    stained apron, the centre logo, and a full wear pass (traffic-weighted
 *    scuffs, drag arcs, sweat spots, bench haze, edge dust).
 *  • **mask** — R: signed distance to the painted lane, G: signed distance to
 *    the line work, B: coat roughness field, A: ambient occlusion (panel grid
 *    plus the apron falloff). The two distance fields are what let 2 in lines
 *    stay a crisp 1-pixel edge from a 2.6-texel-wide bake.
 *  • **detail** — the tiling maple grain/normal from `courtWood`.
 *
 * Paint is *not* composited into the albedo. It is mixed in the shader from the
 * distance fields, modulated by the wood's own luminance, so the grain and the
 * scuffs telegraph through the paint the way they do under real polyurethane —
 * and so a specular streak crossing a sideline never changes shape.
 */

import { COURT, FT } from '../core/Constants';
import { clamp01, fbm2, lerp, makeRng, smoothstep } from '../core/MathX';
import { drawLines, drawPaintedAreas, makeLayout, type CourtLayout } from './courtPaint';
import { bakeMapleDetail, type MapleDetail } from './courtWood';

/** Distance fields are stored as 0.5 + d / SDF_RANGE, in texels. */
export const SDF_RANGE = 24;
/** Shader maps the roughness channel as base + B * range. */
export const ROUGH_BASE = 0.09;
export const ROUGH_RANGE = 0.32;
/** Centre logo radius, metres. Matched analytically in the shader. */
export const LOGO_RADIUS = 2.06;

export interface CourtBake {
  albedo: HTMLCanvasElement;
  mask: HTMLCanvasElement;
  detail: MapleDetail;
  layout: CourtLayout;
}

/**
 * Where the game is actually played. Drives both the coat-roughness field and
 * the scuff distribution, so wear follows traffic instead of noise: densest in
 * the restricted area, high at the elbows and on the wings, thin in the
 * corners, near zero behind the baseline.
 */
export function trafficAt(x: number, z: number): number {
  const inCourt = Math.abs(x) <= COURT.halfLength && Math.abs(z) <= COURT.halfWidth;
  let t = inCourt ? 0.3 : 0.02;
  const gauss = (d: number, s: number): number => Math.exp(-(d * d) / (s * s));

  for (const side of [1, -1] as const) {
    const bx = side * (COURT.halfLength - COURT.basketFromBaseline);
    const ftX = side * (COURT.halfLength - COURT.key.length);

    // Restricted area / low block: the busiest square metres on the floor.
    t = Math.max(t, 0.99 * gauss(Math.hypot(x - bx, z), 2.5));

    // The lane itself.
    const dx = Math.max(0, Math.abs(x - (bx + ftX) * 0.5) - COURT.key.length * 0.45);
    const dz = Math.max(0, Math.abs(z) - COURT.key.width * 0.42);
    t = Math.max(t, 0.86 * gauss(Math.hypot(dx, dz), 1.1));

    // Elbows and the top of the key.
    t = Math.max(t, 0.8 * gauss(Math.hypot(x - (ftX - side * 0.7), z), 2.5));

    // Wings, out on the arc.
    for (const s2 of [1, -1] as const) {
      const a = s2 * 0.74;
      const wx = bx - side * Math.cos(a) * COURT.threePoint.radius;
      const wz = Math.sin(a) * COURT.threePoint.radius;
      t = Math.max(t, 0.68 * gauss(Math.hypot(x - wx, z - wz), 2.4));
    }
  }

  // The trip up the floor plus the jump-ball circle.
  if (inCourt) {
    t = Math.max(t, 0.52 * gauss(z, 7.5) * (0.55 + 0.45 * gauss(x, 12)));
    t = Math.max(t, 0.6 * gauss(Math.hypot(x, z), 2.2));
  }
  return clamp01(t);
}

/** Squared euclidean distance transform (Felzenszwalb), one dimension. */
function dt1d(f: Float64Array, n: number, d: Float64Array, v: Int32Array, zz: Float64Array): void {
  let k = 0;
  v[0] = 0;
  zz[0] = -Infinity;
  zz[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= zz[k]) {
      k--;
      s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    }
    k++;
    v[k] = q;
    zz[k] = s;
    zz[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (zz[k + 1] < q) k++;
    const dq = q - v[k];
    d[q] = dq * dq + f[v[k]];
  }
}

function edt2d(f: Float64Array, W: number, H: number): void {
  const n = Math.max(W, H);
  const buf = new Float64Array(n);
  const out = new Float64Array(n);
  const v = new Int32Array(n + 1);
  const zz = new Float64Array(n + 2);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) buf[x] = f[y * W + x];
    dt1d(buf, W, out, v, zz);
    for (let x = 0; x < W; x++) f[y * W + x] = out[x];
  }
  for (let x = 0; x < W; x++) {
    for (let y = 0; y < H; y++) buf[y] = f[y * W + x];
    dt1d(buf, H, out, v, zz);
    for (let y = 0; y < H; y++) f[y * W + x] = out[y];
  }
}

/** Signed distance in texels; positive inside the mask. */
function signedField(cov: Uint8ClampedArray, W: number, H: number, stride: number): Float32Array {
  const N = W * H;
  const INF = 1e18;
  const a = new Float64Array(N);
  const b = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const inside = cov[i * stride] > 127;
    a[i] = inside ? 0 : INF;
    b[i] = inside ? INF : 0;
  }
  edt2d(a, W, H);
  edt2d(b, W, H);
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    out[i] = cov[i * stride] > 127 ? Math.sqrt(b[i]) - 0.5 : -(Math.sqrt(a[i]) - 0.5);
  }
  return out;
}

function scratch(W: number, H: number): CanvasRenderingContext2D {
  const c = document.createElement('canvas');
  c.width = W;
  c.height = H;
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, W, H);
  return ctx;
}

/**
 * @param height texel height of the whole-court maps; width follows from the
 *   real aspect so texels stay square (the distance fields depend on it).
 * @param detailSize edge length of the tiling grain texture.
 */
export function bakeCourt(height: number, detailSize: number): CourtBake {
  const L = makeLayout(height);
  const { W, H, ppm } = L;
  const rng = makeRng(0x8ba11);

  // ---- albedo ------------------------------------------------------------
  const acvs = document.createElement('canvas');
  acvs.width = W;
  acvs.height = H;
  const actx = acvs.getContext('2d', { willReadFrequently: true })!;
  const aimg = actx.createImageData(W, H);
  const ap = aimg.data;

  // Panel grid: portable floors come apart into ~4 × 8 ft panels, and the
  // panels were sanded and sealed as a batch, so each carries its own tone.
  const panelX = 8 * FT;
  const panelZ = 4 * FT;

  for (let j = 0; j < H; j++) {
    const z = (j + 0.5) / ppm - L.totalH * 0.5;
    for (let i = 0; i < W; i++) {
      const x = (i + 0.5) / ppm - L.totalW * 0.5;
      const o = (j * W + i) * 4;

      // Hard maple under amber polyurethane. Kept deliberately below the
      // "pale laminate" range so the lit floor lands near 110–135 sRGB.
      let r = 0.53;
      let g = 0.383;
      let b = 0.212;

      // Slow stain drift over metres.
      const drift = fbm2(x * 0.055, z * 0.075, 3, 2, 0.5, 11) - 0.5;
      const k = 1 + drift * 0.15;

      // Per-panel batch tone.
      const pi = Math.floor((x + L.totalW * 0.5) / panelX);
      const pj = Math.floor((z + L.totalH * 0.5) / panelZ);
      const pt = (fbm2(pi * 3.7 + 0.5, pj * 5.3 + 0.5, 1, 2, 0.5, 91) - 0.5) * 0.036;

      let mul = k + pt;

      const inCourt = Math.abs(x) <= COURT.halfLength && Math.abs(z) <= COURT.halfWidth;
      if (!inCourt) {
        // The apron is the same wood, stained a good deal darker. It is not a
        // different surface and it must never simply stop at the line.
        const d = Math.max(
          Math.abs(x) - COURT.halfLength,
          Math.abs(z) - COURT.halfWidth,
        );
        const s = smoothstep(d / 0.55);
        mul *= lerp(1, 0.5, s);
        r *= lerp(1, 0.93, s);
        b *= lerp(1, 0.86, s);
      }

      // Dust and grime creep in at the very edge of the floor.
      const edge = Math.max(
        clamp01((Math.abs(x) - (L.totalW * 0.5 - 0.8)) / 0.8),
        clamp01((Math.abs(z) - (L.totalH * 0.5 - 0.7)) / 0.7),
      );
      mul *= 1 - edge * 0.09;

      ap[o] = Math.round(clamp01(r * mul) * 255);
      ap[o + 1] = Math.round(clamp01(g * mul) * 255);
      ap[o + 2] = Math.round(clamp01(b * mul) * 255);
      ap[o + 3] = 255;
    }
  }
  actx.putImageData(aimg, 0, 0);

  drawCentreLogo(actx, L, rng);
  drawWear(actx, L, rng);

  // ---- distance fields ---------------------------------------------------
  const keyCtx = scratch(W, H);
  drawPaintedAreas(keyCtx, L);
  const keySdf = signedField(keyCtx.getImageData(0, 0, W, H).data, W, H, 4);

  const lineCtx = scratch(W, H);
  drawLines(lineCtx, L);
  const lineSdf = signedField(lineCtx.getImageData(0, 0, W, H).data, W, H, 4);

  // ---- mask (SDFs + roughness + AO) --------------------------------------
  const mcvs = document.createElement('canvas');
  mcvs.width = W;
  mcvs.height = H;
  const mctx = mcvs.getContext('2d', { willReadFrequently: true })!;
  const mimg = mctx.createImageData(W, H);
  const mp = mimg.data;

  // Traffic is smooth over metres, so it is evaluated on a coarse lattice and
  // interpolated rather than paying fourteen exponentials per texel.
  const step = 4;
  const gw = Math.ceil(W / step) + 1;
  const gh = Math.ceil(H / step) + 1;
  const grid = new Float32Array(gw * gh);
  for (let gj = 0; gj < gh; gj++) {
    const z = (gj * step) / ppm - L.totalH * 0.5;
    for (let gi = 0; gi < gw; gi++) {
      grid[gj * gw + gi] = trafficAt((gi * step) / ppm - L.totalW * 0.5, z);
    }
  }
  const sampleTraffic = (i: number, j: number): number => {
    const fx = i / step;
    const fy = j / step;
    const x0 = Math.min(gw - 2, Math.floor(fx));
    const y0 = Math.min(gh - 2, Math.floor(fy));
    const tx = fx - x0;
    const ty = fy - y0;
    const a = grid[y0 * gw + x0];
    const b = grid[y0 * gw + x0 + 1];
    const c = grid[(y0 + 1) * gw + x0];
    const d = grid[(y0 + 1) * gw + x0 + 1];
    return lerp(lerp(a, b, tx), lerp(c, d, tx), ty);
  };

  // Sweat spots and towel smears: a handful per half, wetter and glossier.
  const wet: { x: number; z: number; r: number }[] = [];
  for (let i = 0; i < 9; i++) {
    const side = i % 2 === 0 ? 1 : -1;
    const bx = side * (COURT.halfLength - COURT.basketFromBaseline);
    wet.push({
      x: bx - side * (1 + rng() * 7),
      z: (rng() - 0.5) * 9,
      r: 0.045 + rng() * 0.08,
    });
  }

  for (let j = 0; j < H; j++) {
    const z = (j + 0.5) / ppm - L.totalH * 0.5;
    for (let i = 0; i < W; i++) {
      const x = (i + 0.5) / ppm - L.totalW * 0.5;
      const o = (j * W + i) * 4;
      const idx = j * W + i;

      // --- coat roughness ---
      const traffic = sampleTraffic(i, j);
      // Buffed lanes are micro-scuffed and therefore *duller*; the corners and
      // the dead ground behind the basket keep the fresh mirror finish.
      let rough = 0.11 + traffic * 0.09;
      const swirl = fbm2(x * 0.42, z * 0.55, 3, 2, 0.55, 23) - 0.5;
      rough += swirl * 0.05;
      const micro = fbm2(x * 6.5, z * 8.5, 2, 2, 0.5, 77) - 0.5;
      rough += micro * 0.014;

      const inCourt = Math.abs(x) <= COURT.halfLength && Math.abs(z) <= COURT.halfWidth;
      if (!inCourt) rough += 0.075; // apron is walked on in street shoes

      // Dust against the very boundary of the wood.
      const edge = Math.max(
        clamp01((Math.abs(x) - (L.totalW * 0.5 - 0.9)) / 0.9),
        clamp01((Math.abs(z) - (L.totalH * 0.5 - 0.8)) / 0.8),
      );
      rough += edge * 0.05;

      // Bench haze — resin, spray and towel lint in front of the seats.
      const bench = Math.exp(-Math.pow((z - (COURT.halfWidth + 0.9)) / 1.1, 2)) *
        Math.exp(-Math.pow(x / 9, 4));
      rough += bench * 0.05;

      // Sole-print sheen: discrete patches where shoes have abraded the coat.
      // Albedo-invisible; they only show as highlight modulation.
      rough +=
        clamp01(fbm2(x * 1.7, z * 2.1, 2, 2, 0.5, 131) - 0.62) * 0.17 * traffic;

      for (const w of wet) {
        const d = Math.hypot(x - w.x, z - w.z);
        if (d < w.r * 3.2) {
          const c = 1 - smoothstep(d / (w.r * 3.2));
          rough -= 0.055 * c * c;
        }
      }

      // --- ambient occlusion ---
      let ao = 1;
      const ux = (x + L.totalW * 0.5) % panelX;
      const uz = (z + L.totalH * 0.5) % panelZ;
      const pw = 1.6 / ppm;
      const panelSeam = Math.max(
        1 - clamp01(Math.min(ux, panelX - ux) / pw),
        1 - clamp01(Math.min(uz, panelZ - uz) / pw),
      );
      // The panel joint holds a hair more coat, so the grid reads under a
      // raking reflection even where the occlusion term is invisible.
      rough += panelSeam * 0.014;
      ao -= panelSeam * 0.05;
      // The stands and the scorer's table shade the outer apron.
      ao -= edge * 0.3;
      ao -= bench * 0.05;

      mp[o] = Math.round(clamp01(0.5 + keySdf[idx] / SDF_RANGE) * 255);
      mp[o + 1] = Math.round(clamp01(0.5 + lineSdf[idx] / SDF_RANGE) * 255);
      mp[o + 2] = Math.round(clamp01((rough - ROUGH_BASE) / ROUGH_RANGE) * 255);
      mp[o + 3] = Math.round(clamp01(ao) * 255);
    }
  }
  mctx.putImageData(mimg, 0, 0);

  const detail = bakeMapleDetail(detailSize, detailSize);
  return { albedo: acvs, mask: mcvs, detail, layout: L };
}

/**
 * Centre logo. Painted on sanded wood before the coat went down, which means
 * it is slightly desaturated, warm-shifted by the varnish, worn where the
 * jump ball happens, and — because the shader mixes it against the wood's own
 * luminance — carries the grain through it.
 */
function drawCentreLogo(ctx: CanvasRenderingContext2D, L: CourtLayout, rng: () => number): void {
  const R = L.m(LOGO_RADIUS);
  ctx.save();
  ctx.translate(L.cx(0), L.cz(0));

  // Painted colours stay under ~72% HSV saturation: pigment under amber gloss.
  const navy = '#2b415c';
  const gold = '#a8823f';
  const cream = '#ddd2bc';

  ctx.globalAlpha = 0.88;
  ctx.fillStyle = navy;
  ctx.beginPath();
  ctx.arc(0, 0, R, 0, Math.PI * 2);
  ctx.fill();

  ctx.globalAlpha = 0.9;
  ctx.strokeStyle = gold;
  ctx.lineWidth = L.m(0.075);
  ctx.beginPath();
  ctx.arc(0, 0, R * 0.88, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(0, 0, R * 0.8, 0, Math.PI * 2);
  ctx.stroke();

  // Wordmark, tracked out the way a court graphic is.
  ctx.globalAlpha = 0.92;
  ctx.fillStyle = cream;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const fs = Math.round(R * 0.34);
  ctx.font = `900 ${fs}px ui-sans-serif, system-ui, sans-serif`;
  const word = 'BALLIN';
  const track = fs * 0.2;
  let total = 0;
  for (const ch of word) total += ctx.measureText(ch).width + track;
  total -= track;
  let cx = -total / 2;
  for (const ch of word) {
    const w = ctx.measureText(ch).width;
    ctx.fillText(ch, cx + w / 2, 0);
    cx += w + track;
  }

  // Ball glyph under the wordmark.
  ctx.globalAlpha = 0.8;
  ctx.strokeStyle = gold;
  ctx.lineWidth = L.m(0.045);
  const br = R * 0.2;
  const by = R * 0.44;
  ctx.beginPath();
  ctx.arc(0, by, br, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(-br, by);
  ctx.lineTo(br, by);
  ctx.moveTo(0, by - br);
  ctx.lineTo(0, by + br);
  ctx.stroke();

  // Wear: centre court is a jump-ball circle and a lot of feet. Scrub the
  // paint back with soft wood-coloured strokes rather than lowering opacity,
  // so the wear has structure.
  ctx.globalCompositeOperation = 'destination-out';
  for (let i = 0; i < 260; i++) {
    const a = rng() * Math.PI * 2;
    const rr = Math.sqrt(rng()) * R;
    const len = L.m(0.08 + rng() * 0.45);
    const ang = a + (rng() - 0.5) * 1.6;
    ctx.globalAlpha = 0.05 + rng() * 0.14;
    ctx.strokeStyle = '#000';
    ctx.lineWidth = L.m(0.01 + rng() * 0.035);
    ctx.beginPath();
    const sx = Math.cos(a) * rr;
    const sy = Math.sin(a) * rr;
    ctx.moveTo(sx, sy);
    ctx.quadraticCurveTo(
      sx + Math.cos(ang) * len * 0.5,
      sy + Math.sin(ang) * len * 0.5 - len * 0.14,
      sx + Math.cos(ang) * len,
      sy + Math.sin(ang) * len,
    );
    ctx.stroke();
  }
  ctx.restore();
}

/**
 * The wear pass. Scuff density is driven by `trafficAt`, so the paint and the
 * arc are black with rubber and the corners are nearly clean — a uniform noise
 * overlay is explicitly worse than none.
 */
function drawWear(ctx: CanvasRenderingContext2D, L: CourtLayout, rng: () => number): void {
  ctx.save();

  // --- long drag marks: a body hitting the floor, a chair being pulled ---
  ctx.globalCompositeOperation = 'multiply';
  for (let i = 0; i < 46; i++) {
    const x = (rng() - 0.5) * COURT.length * 0.94;
    const z = (rng() - 0.5) * COURT.width * 0.9;
    if (rng() > trafficAt(x, z) * 0.85 + 0.1) continue;
    const len = 0.5 + rng() * 2.4;
    const a = (rng() - 0.5) * 1.1 + (rng() < 0.5 ? 0 : Math.PI);
    ctx.strokeStyle = `rgba(96,74,48,${0.05 + rng() * 0.06})`;
    ctx.lineWidth = L.m(0.012 + rng() * 0.03);
    ctx.beginPath();
    ctx.moveTo(L.cx(x), L.cz(z));
    ctx.quadraticCurveTo(
      L.cx(x + Math.cos(a) * len * 0.5),
      L.cz(z + Math.sin(a) * len * 0.5 + (rng() - 0.5) * 0.4),
      L.cx(x + Math.cos(a) * len),
      L.cz(z + Math.sin(a) * len),
    );
    ctx.stroke();
  }

  // --- rubber scuffs -----------------------------------------------------
  // Short arcing streaks, 60–350 mm, clustered on the direction of play.
  let placed = 0;
  for (let i = 0; i < 120000 && placed < 26000; i++) {
    const x = (rng() - 0.5) * L.totalW;
    const z = (rng() - 0.5) * L.totalH;
    const t = trafficAt(x, z);
    if (rng() > t * t) continue;
    placed++;
    const len = 0.06 + rng() * 0.29;
    // Play runs up and down the floor, so scuffs bias toward the long axis.
    const a = (rng() - 0.5) * 1.5 + (rng() < 0.5 ? 0 : Math.PI);
    const bend = (rng() - 0.5) * 0.5;
    const dark = (0.03 + rng() * 0.075) * (0.35 + 0.65 * t);
    ctx.strokeStyle = `rgba(58,44,30,${dark})`;
    ctx.lineWidth = L.m(0.004 + rng() * 0.011);
    ctx.lineCap = 'round';
    ctx.beginPath();
    const x0 = L.cx(x);
    const y0 = L.cz(z);
    const x1 = L.cx(x + Math.cos(a) * len);
    const y1 = L.cz(z + Math.sin(a) * len);
    ctx.moveTo(x0, y0);
    ctx.quadraticCurveTo(
      (x0 + x1) * 0.5 - Math.sin(a) * L.m(len * bend),
      (y0 + y1) * 0.5 + Math.cos(a) * L.m(len * bend),
      x1,
      y1,
    );
    ctx.stroke();
  }

  // --- sweat spots: darker, wetter discs with a towel smear ---------------
  ctx.globalCompositeOperation = 'multiply';
  for (let i = 0; i < 9; i++) {
    const side = i % 2 === 0 ? 1 : -1;
    const bx = side * (COURT.halfLength - COURT.basketFromBaseline);
    const x = bx - side * (1 + rng() * 7);
    const z = (rng() - 0.5) * 9;
    const r = L.m(0.045 + rng() * 0.08);
    const grad = ctx.createRadialGradient(L.cx(x), L.cz(z), 0, L.cx(x), L.cz(z), r * 2.6);
    grad.addColorStop(0, 'rgba(150,128,102,0.55)');
    grad.addColorStop(0.5, 'rgba(196,176,150,0.3)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(L.cx(x), L.cz(z), r * 2.6, 0, Math.PI * 2);
    ctx.fill();
  }

  // --- bench haze --------------------------------------------------------
  ctx.globalCompositeOperation = 'source-over';
  const hz = ctx.createLinearGradient(0, L.cz(COURT.halfWidth - 1.4), 0, L.cz(L.totalH * 0.5));
  hz.addColorStop(0, 'rgba(226,214,192,0)');
  hz.addColorStop(1, 'rgba(226,214,192,0.075)');
  ctx.fillStyle = hz;
  ctx.fillRect(L.cx(-11), L.cz(COURT.halfWidth - 1.4), L.m(22), L.m(3.6));

  ctx.restore();
}
