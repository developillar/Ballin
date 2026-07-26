/**
 * Whole-court bake: the signals that must *not* repeat.
 *
 * Three outputs, all procedural:
 *
 *  • **albedo** — maple base tone, panel-to-panel stain scatter, the darker
 *    stained apron, the centre logo, and a full wear pass (traffic-weighted
 *    scuffs, drag arcs, sweat spots, bench haze, edge dust).
 *  • **mask** — R: signed distance to the painted lane, G: signed distance to
 *    the line work, B: coat roughness field, A: ambient occlusion (the apron
 *    falloff and the bench haze). The two distance fields are what let 2 in
 *    lines stay a crisp 1-pixel edge from a 2.6-texel-wide bake.
 *  • **detail** — the tiling maple grain/normal from `courtWood`.
 *
 * The joinery — strip seams, butt joints, the panel grid — is *not* here and is
 * not in the tile either. At 52 texels/m a 2 mm groove is a tenth of a texel;
 * it is drawn analytically in `Court.ts` off the world position, which is the
 * only way it stays a real 1 px line at the near floor and fades honestly to
 * sub-pixel by mid-court.
 *
 * Paint is *not* composited into the albedo. It is mixed in the shader from the
 * distance fields, modulated by the wood's own luminance, so the grain and the
 * scuffs telegraph through the paint the way they do under real polyurethane —
 * and so a specular streak crossing a sideline never changes shape.
 */

import { COURT } from '../core/Constants';
import { clamp01, fbm2, lerp, makeRng, smoothstep } from '../core/MathX';
import { drawLines, drawPaintedAreas, makeLayout, type CourtLayout } from './courtPaint';
import { bakeMapleDetail, PANEL_LENGTH, PANEL_WIDTH, type MapleDetail } from './courtWood';

/** Distance fields are stored as 0.5 + d / SDF_RANGE, in texels. */
export const SDF_RANGE = 24;
/**
 * Shader maps the roughness channel as base + B * range.
 *
 * §2.3 puts a poly-finished floor at 0.06–0.14 in the tight (across-grain) axis.
 * three.js widens the other axis for us — `alphaT = mix(roughness², 1, anisotropy²)`
 * — so the *material* roughness is the tight axis and the base has to sit at the
 * bottom of that band, not in the middle of it. The old 0.09–0.41 field put the
 * broad axis at 0.39–0.45 against §2.3's 0.30 ceiling, which is why the
 * highlight was a wide dull smear with no core. Note three floors
 * `material.roughness` at 0.0525, so anything under that is wasted.
 */
export const ROUGH_BASE = 0.055;
export const ROUGH_RANGE = 0.16;
/** Centre logo radius, metres. Matched analytically in the shader. */
export const LOGO_RADIUS = 2.06;

export interface CourtBake {
  albedo: HTMLCanvasElement;
  mask: HTMLCanvasElement;
  /**
   * Centre logo, RGBA, on its own square texture covering exactly
   * 2 × `LOGO_RADIUS` metres. It is *not* composited into the albedo: a
   * whole-court bake gives the centre circle barely 50 texels per metre, which
   * magnifies to mush at the FLOOR framing, and compositing would also throw
   * away the coverage channel that lets worn paint show bare wood through.
   * RGB is pigment; A is paint coverage after wear.
   */
  logo: HTMLCanvasElement;
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
 * @param logoSize edge length of the dedicated centre-logo texture.
 */
export function bakeCourt(height: number, detailSize: number, logoSize = 512): CourtBake {
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

  // Panel grid: portable floors come apart into ~4 × 7 ft panels, and the
  // panels were sanded and sealed as a batch, so each carries its own tone.
  // The cross-court pitch is a whole number of boards (`PANEL_WIDTH`) so the
  // panel joint lands on a milled seam, which is where it is on a real deck.
  const panelX = PANEL_LENGTH;
  const panelZ = PANEL_WIDTH;

  for (let j = 0; j < H; j++) {
    const z = (j + 0.5) / ppm - L.totalH * 0.5;
    for (let i = 0; i < W; i++) {
      const x = (i + 0.5) / ppm - L.totalW * 0.5;
      const o = (j * W + i) * 4;

      // Hard maple under amber polyurethane.
      //
      // §2 opens by saying maple is chosen partly *because it is light and
      // reflects light back into the arena*. The previous base — HSV 32/66/58 —
      // was carried dark and saturated on the theory that the coat's achromatic
      // specular would bleach it back, and that is exactly what the frame did:
      // wherever a bank streak landed the floor went to a near-white sheet
      // (178/175/174 measured), and wherever one did not, the near hardwood fell
      // to 56 — a third of §1.1's 95–140 court band, and dark walnut rather than
      // maple. The base is now a photographed finished-maple swatch, HSV
      // ~34° / 44% / 78°, i.e. 0.38 linear reflectance instead of 0.17. The
      // specular is held down to match (`uSpec` in Court.ts) so the two are not
      // fighting for the same stops.
      let r = 0.768;
      let g = 0.648;
      let b = 0.515;

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
      // Everything here is a *tight-axis* roughness and therefore lives inside
      // §2.3's 0.06–0.14 band, with only the apron, the dust line and the
      // heaviest sole prints allowed past it. §2.3's traffic-lane figure is
      // "locally rougher by 0.04–0.08", which is the whole width of the band —
      // so the clean coat has to start at the bottom of it.
      const traffic = sampleTraffic(i, j);
      // Buffed lanes are micro-scuffed and therefore *duller*; the corners and
      // the dead ground behind the basket keep the fresh mirror finish.
      let rough = 0.062 + traffic * 0.052;
      const swirl = fbm2(x * 0.42, z * 0.55, 3, 2, 0.55, 23) - 0.5;
      rough += swirl * 0.026;
      const micro = fbm2(x * 6.5, z * 8.5, 2, 2, 0.5, 77) - 0.5;
      rough += micro * 0.008;

      const inCourt = Math.abs(x) <= COURT.halfLength && Math.abs(z) <= COURT.halfWidth;
      if (!inCourt) rough += 0.045; // apron is walked on in street shoes

      // Dust against the very boundary of the wood.
      const edge = Math.max(
        clamp01((Math.abs(x) - (L.totalW * 0.5 - 0.9)) / 0.9),
        clamp01((Math.abs(z) - (L.totalH * 0.5 - 0.8)) / 0.8),
      );
      rough += edge * 0.032;

      // Bench haze — resin, spray and towel lint in front of the seats.
      const bench = Math.exp(-Math.pow((z - (COURT.halfWidth + 0.9)) / 1.1, 2)) *
        Math.exp(-Math.pow(x / 9, 4));
      rough += bench * 0.03;

      // Sole-print sheen: discrete patches where shoes have abraded the coat.
      // Albedo-invisible; they only show as highlight modulation.
      rough +=
        clamp01(fbm2(x * 1.7, z * 2.1, 2, 2, 0.5, 131) - 0.62) * 0.1 * traffic;

      for (const w of wet) {
        const d = Math.hypot(x - w.x, z - w.z);
        if (d < w.r * 3.2) {
          const c = 1 - smoothstep(d / (w.r * 3.2));
          rough -= 0.026 * c * c;
        }
      }

      // --- ambient occlusion ---
      // The panel joint itself is *not* baked here any more. At 52 texels/m a
      // 1.6-texel seam is a 31 mm smear that the near floor then magnifies
      // tenfold, which is why it was not discernible at a grazing angle in any
      // frame; it is drawn analytically in the floor shader alongside the strip
      // seams and the butt joints, where it stays a real 1 px line at any
      // distance. What is left here is what genuinely varies over metres.
      let ao = 1;
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
  const logo = bakeCentreLogo(logoSize, makeRng(0x10c0));
  return { albedo: acvs, mask: mcvs, logo, detail, layout: L };
}

/**
 * Centre logo, on its own texture.
 *
 * A court graphic is stencilled pigment on sanded maple and then buried under
 * the same polyurethane as everything else, so three things have to be true of
 * it and all three were wrong when it was composited into the whole-court
 * albedo at 50 texels/m:
 *
 *  - it is *confidently coloured*. Under-saturating a court logo to "look
 *    aged" is the wrong instinct; the varnish yellows and mutes it a few
 *    percent, it does not turn it grey. The pigments here sit at 60–70% HSV
 *    saturation, warm-shifted, which is what stencil paint under amber gloss
 *    actually measures. §2.5's ceiling is 72%.
 *  - it is *worn*, and the wear has structure: the jump-ball circle is scrubbed
 *    hardest, pivot arcs run through it, and in the worst patches the pigment
 *    is gone rather than merely faded. That is why coverage lives in alpha —
 *    the shader can then show bare wood through it instead of blending toward
 *    a lighter flat colour.
 *  - its *edges* are as crisp as the line work, which needs resolution the
 *    whole-court bake does not have.
 *
 * Everything else that makes it read as under the coat — the grain telegraph,
 * the shared specular streak, the roughness fill — is applied in the shader.
 *
 * @param S edge length in texels; the texture spans exactly 2 × LOGO_RADIUS m.
 */
function bakeCentreLogo(S: number, rng: () => number): HTMLCanvasElement {
  const cvs = document.createElement('canvas');
  cvs.width = S;
  cvs.height = S;
  const ctx = cvs.getContext('2d', { willReadFrequently: true })!;
  const R = S * 0.5;
  /** Metres → texels on this canvas. */
  const m = (v: number) => (v / (LOGO_RADIUS * 2)) * S;
  ctx.translate(R, R);

  // Pigment under amber gloss. §2.5's ceiling is 72% HSV saturation and it is
  // a ceiling on the *frame*, not on the swatch. Two things push a court blue
  // up on the way out: §8.3's shadow split lifts blue and drops red across the
  // darkest quartile of the frame, and the vignette cools as it darkens — and
  // the centre mark sits low and outboard in a FLOOR framing, which is exactly
  // where both are strongest. A navy mixed at 68% measured 74.9% in the review
  // capture and 86% once the mark stopped being washed out by a specular
  // sheet. So the blues are mixed lighter as well as flatter: at 44–46%
  // saturation and half a stop up they land in the low 60s post-grade with
  // room for the grade to move underneath them.
  const navy = '#4d6786';
  const navyDeep = '#43597a';
  const gold = '#b28a4c';
  const brick = '#a86647';
  const cream = '#eadfc6';

  // --- field ---------------------------------------------------------------
  // Slightly deeper toward the rim: stencil paint pools at the mask edge.
  const field = ctx.createRadialGradient(0, -R * 0.15, R * 0.08, 0, 0, R);
  field.addColorStop(0, navy);
  field.addColorStop(0.72, navy);
  field.addColorStop(1, navyDeep);
  ctx.fillStyle = field;
  ctx.beginPath();
  ctx.arc(0, 0, R * 0.995, 0, Math.PI * 2);
  ctx.fill();

  // --- rings ---------------------------------------------------------------
  // A bare-wood gap between the field and the inner rule, so the wood reads
  // *through* the mark and not only around it.
  ctx.globalCompositeOperation = 'destination-out';
  ctx.strokeStyle = '#000';
  ctx.lineWidth = m(0.075);
  ctx.beginPath();
  ctx.arc(0, 0, R * 0.9, 0, Math.PI * 2);
  ctx.stroke();
  ctx.globalCompositeOperation = 'source-over';

  ctx.strokeStyle = gold;
  ctx.lineWidth = m(0.105);
  ctx.beginPath();
  ctx.arc(0, 0, R * 0.835, 0, Math.PI * 2);
  ctx.stroke();

  ctx.strokeStyle = brick;
  ctx.lineWidth = m(0.036);
  ctx.beginPath();
  ctx.arc(0, 0, R * 0.775, 0, Math.PI * 2);
  ctx.stroke();

  // --- wordmark ------------------------------------------------------------
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  const fs = Math.round(R * 0.26);
  ctx.font = `900 ${fs}px ui-sans-serif, system-ui, sans-serif`;
  const word = 'BALLIN';
  const track = fs * 0.16;
  let total = -track;
  for (const ch of word) total += ctx.measureText(ch).width + track;

  const drawWord = (dy: number, fill: string) => {
    ctx.fillStyle = fill;
    let cx = -total / 2;
    for (const ch of word) {
      const w = ctx.measureText(ch).width;
      ctx.fillText(ch, cx + w / 2, dy);
      cx += w + track;
    }
  };
  // Offset drop colour, the way a two-plate court graphic is screened.
  drawWord(fs * 0.36 + m(0.055), brick);
  drawWord(fs * 0.36, cream);

  // --- ball glyph, above the wordmark -------------------------------------
  const br = R * 0.23;
  const by = -R * 0.42;
  ctx.fillStyle = gold;
  ctx.beginPath();
  ctx.arc(0, by, br, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalCompositeOperation = 'destination-out';
  ctx.strokeStyle = '#000';
  ctx.lineWidth = m(0.05);
  ctx.beginPath();
  ctx.moveTo(-br, by);
  ctx.lineTo(br, by);
  ctx.moveTo(0, by - br);
  ctx.lineTo(0, by + br);
  ctx.stroke();
  // The two curved channels of an eight-panel ball.
  for (const s of [-1, 1] as const) {
    ctx.beginPath();
    ctx.moveTo(s * br * 0.98, by - br * 0.2);
    ctx.quadraticCurveTo(s * br * 0.12, by, s * br * 0.98, by + br * 0.2);
    ctx.stroke();
  }
  ctx.globalCompositeOperation = 'source-over';

  // --- underline rule ------------------------------------------------------
  ctx.fillStyle = gold;
  ctx.fillRect(-total * 0.39, R * 0.5, total * 0.78, m(0.05));

  // --- wear ---------------------------------------------------------------
  // Two passes, because worn court paint does two different things. Abrasion
  // first *bleaches* the pigment (source-atop, so it stays inside the mark),
  // and only where the traffic is heaviest does it take the paint off down to
  // the wood (destination-out). Both are weighted toward the jump-ball circle
  // and run as short curved pivot arcs, not as noise.
  const arc = (
    op: GlobalCompositeOperation,
    n: number,
    bias: number,
    aLo: number,
    aHi: number,
    style: string,
  ) => {
    ctx.globalCompositeOperation = op;
    ctx.lineCap = 'round';
    for (let i = 0; i < n; i++) {
      const a = rng() * Math.PI * 2;
      // `bias` < 1 pulls the distribution in toward the centre circle.
      const rr = Math.pow(rng(), bias) * R * 0.99;
      const sx = Math.cos(a) * rr;
      const sy = Math.sin(a) * rr;
      // Pivots and jump-ball scrambles arc around the centre, so the stroke
      // direction follows the tangent more often than not.
      const tang = a + Math.PI * 0.5 + (rng() - 0.5) * 1.5;
      const len = m(0.1 + rng() * rng() * 0.62);
      const bend = (rng() - 0.5) * 0.9;
      ctx.globalAlpha = aLo + rng() * (aHi - aLo);
      ctx.strokeStyle = style;
      ctx.lineWidth = m(0.008 + rng() * rng() * 0.05);
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.quadraticCurveTo(
        sx + Math.cos(tang) * len * 0.5 - Math.sin(tang) * len * bend,
        sy + Math.sin(tang) * len * 0.5 + Math.cos(tang) * len * bend,
        sx + Math.cos(tang) * len,
        sy + Math.sin(tang) * len,
      );
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
  };

  // Bleached pigment — a warm chalky grey, which is what scuffed paint under
  // yellowed varnish goes.
  //
  // Carried at roughly 1.6× the alpha it used to be. §2.5 wants the wear
  // *legible*, and at 0.04–0.17 over a thousand strokes the mark measured 3.26
  // high-frequency RMS against 2.64 on the bare wood beside it — i.e. the whole
  // pass was sitting under the film grain and the logo read as a flat disc.
  arc('source-atop', 520, 0.62, 0.07, 0.28, '#b8ac93');
  // Rubber transferred off soles: darker, tighter, dead centre.
  arc('source-atop', 230, 0.45, 0.06, 0.21, '#2a2320');
  // Bare wood, only in the worst of it.
  arc('destination-out', 210, 0.4, 0.09, 0.42, '#000');

  // A couple of long drag scars right across the mark.
  ctx.globalCompositeOperation = 'destination-out';
  for (let i = 0; i < 3; i++) {
    const a = rng() * Math.PI * 2;
    ctx.globalAlpha = 0.1 + rng() * 0.14;
    ctx.strokeStyle = '#000';
    ctx.lineWidth = m(0.014 + rng() * 0.02);
    ctx.beginPath();
    ctx.moveTo(Math.cos(a) * R, Math.sin(a) * R);
    ctx.quadraticCurveTo(
      (rng() - 0.5) * R * 0.7,
      (rng() - 0.5) * R * 0.7,
      Math.cos(a + 2.2 + rng()) * R,
      Math.sin(a + 2.2 + rng()) * R,
    );
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';

  // --- edge --------------------------------------------------------------
  // Kill anything that crept outside the stencil, and hold a 1-texel edge.
  ctx.globalCompositeOperation = 'destination-in';
  const edge = ctx.createRadialGradient(0, 0, R * 0.985, 0, 0, R * 0.999);
  edge.addColorStop(0, '#fff');
  edge.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = edge;
  ctx.fillRect(-R, -R, S, S);
  ctx.globalCompositeOperation = 'source-over';

  return cvs;
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
