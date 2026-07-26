/**
 * Procedural bakes for the hoop: braided nylon net cord, the rim's chipped
 * powder coat, the backboard's painted glass and the vinyl padding wraps.
 *
 * Everything is drawn into a 2D canvas at load time — there are no external
 * assets anywhere in this project, so the "art" is arithmetic.
 *
 * Owned by the hoop agent.
 */

import {
  CanvasTexture,
  ClampToEdgeWrapping,
  LinearMipmapLinearFilter,
  RepeatWrapping,
  SRGBColorSpace,
  type Wrapping,
} from 'three';
import { clamp01, fbm2, hash2, makeRng, ridged2, smoothstep } from '../core/MathX';

function surface(w: number, h: number): CanvasRenderingContext2D {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d')!;
  ctx.imageSmoothingEnabled = true;
  return ctx;
}

interface TexOpts {
  srgb?: boolean;
  wrapS?: Wrapping;
  wrapT?: Wrapping;
  anisotropy?: number;
}

function finish(ctx: CanvasRenderingContext2D, o: TexOpts = {}): CanvasTexture {
  const t = new CanvasTexture(ctx.canvas);
  if (o.srgb) t.colorSpace = SRGBColorSpace;
  t.wrapS = o.wrapS ?? RepeatWrapping;
  t.wrapT = o.wrapT ?? RepeatWrapping;
  t.anisotropy = o.anisotropy ?? 8;
  t.minFilter = LinearMipmapLinearFilter;
  t.generateMipmaps = true;
  t.needsUpdate = true;
  return t;
}

/** Shortest signed distance between two wrapped [0,1) coordinates. */
function wrapDist(a: number, b: number): number {
  let d = a - b;
  d -= Math.floor(d + 0.5);
  return d;
}

// ---------------------------------------------------------------------------
// Net cord
// ---------------------------------------------------------------------------

/**
 * A tiling patch of 120-count braided nylon. `u` runs around the circumference
 * of the cord, `v` along it; three plies spiral so that repeating the tile down
 * a strand produces a continuous twist. Tiles seamlessly on both axes.
 */
export function bakeNetCord(size = 64): { map: CanvasTexture; rough: CanvasTexture } {
  const W = size;
  const H = size;
  const ac = surface(W, H);
  const rc = surface(W, H);
  const ai = ac.createImageData(W, H);
  const ri = rc.createImageData(W, H);
  const PLIES = 3;

  for (let y = 0; y < H; y++) {
    const v = y / H;
    for (let x = 0; x < W; x++) {
      const u = x / W;
      // Diagonal ply ridges. u * PLIES + v is integer-periodic on both axes, so
      // the tile wraps cleanly however many times it repeats along the strand.
      const phase = (u * PLIES + v) % 1;
      const ridge = 0.5 + 0.5 * Math.cos(phase * Math.PI * 2);
      const ply = Math.pow(ridge, 0.55);

      // Individual filaments running along the ply, plus a little fuzz.
      const fil = ridged2((u * PLIES + v) * 26, (u * 6 - v * 2) * 5, 2, 17);
      const fuzz = fbm2(x * 0.55, y * 0.55, 3, 2, 0.5, 91);

      const lum = 206 + 46 * ply + 11 * (fil - 0.5) - 13 * (fuzz - 0.5);
      const o = (y * W + x) * 4;
      ai.data[o] = clamp01(lum / 255) * 255;
      ai.data[o + 1] = clamp01((lum * 0.994) / 255) * 255;
      ai.data[o + 2] = clamp01((lum * 0.952) / 255) * 255;
      ai.data[o + 3] = 255;

      // The valleys between plies trap light and read rougher than the crowns.
      const rough = clamp01(0.40 + 0.30 * (1 - ply) + 0.09 * (fuzz - 0.5));
      const g = rough * 255;
      ri.data[o] = g;
      ri.data[o + 1] = g;
      ri.data[o + 2] = g;
      ri.data[o + 3] = 255;
    }
  }

  ac.putImageData(ai, 0, 0);
  rc.putImageData(ri, 0, 0);
  return {
    map: finish(ac, { srgb: true, anisotropy: 4 }),
    rough: finish(rc, { anisotropy: 4 }),
  };
}

// ---------------------------------------------------------------------------
// Rim
// ---------------------------------------------------------------------------

/**
 * Powder-coated ring after a season of abuse.
 *
 * Torus UVs: `u` runs around the ring, `v` around the tube — with the ring laid
 * flat, v ≈ 0.75 is the top of the bar and v ≈ 0.25 the underside. `frontU` is
 * where the ball arrives, and that third of the ring loses its paint first.
 */
export function bakeRimMaps(
  size = 1024,
  frontU = 0.5,
  hooks = 12,
): { map: CanvasTexture; orm: CanvasTexture } {
  const W = size;
  const H = size >> 2;
  const ac = surface(W, H);
  const oc = surface(W, H);
  const ai = ac.createImageData(W, H);
  const oi = oc.createImageData(W, H);
  const rng = makeRng(913377);

  // A handful of deep gouges — the ring gets hit by rings, hands and elbows.
  const gouges: { u: number; v: number; len: number; ang: number; w: number }[] = [];
  for (let i = 0; i < 26; i++) {
    const bias = rng();
    gouges.push({
      u: (frontU + (bias * bias - 0.5) * 0.9 + 1) % 1,
      v: 0.55 + rng() * 0.42,
      len: 0.008 + rng() * 0.035,
      ang: (rng() - 0.5) * 1.5,
      w: 0.0018 + rng() * 0.004,
    });
  }

  for (let y = 0; y < H; y++) {
    const v = y / H;
    for (let x = 0; x < W; x++) {
      const u = x / W;
      const o = (y * W + x) * 4;

      // --- powder coat base ------------------------------------------------
      // Orange peel: powder coat is sprayed, so it has a fine dimpled texture.
      const peel = fbm2(u * 220, v * 34, 3, 2.1, 0.55, 5);
      let r = 184 + (peel - 0.5) * 20;
      let g = 58 + (peel - 0.5) * 12;
      let b = 22 + (peel - 0.5) * 7;
      // The underside stays cleaner and reads a shade deeper.
      const underside = smoothstep(clamp01((0.42 - Math.abs(v - 0.25)) / 0.3));
      r -= underside * 16;
      g -= underside * 7;
      b -= underside * 2;

      let rough = 0.33 + (peel - 0.5) * 0.09;
      let metal = 0.06;

      // --- bare metal on the strike face -----------------------------------
      // Balls land on the top-front of the bar; that arc polishes to steel.
      const du = Math.abs(wrapDist(u, frontU));
      const topness = clamp01(1 - Math.abs(wrapDist(v, 0.78)) / 0.30);
      const arc = clamp01(1 - du / 0.17);
      const wearNoise = fbm2(u * 96, v * 26, 4, 2.2, 0.5, 71);
      let wear = clamp01(Math.pow(arc, 0.9) * Math.pow(topness, 1.1) * 1.5 - 0.30);
      // Chipped paint has a ragged edge, never a soft gradient.
      wear = clamp01((wear - 0.50 + (wearNoise - 0.5) * 0.62) * 6);

      // Chipping along the outer equator where the net and hands scrape.
      const equator = clamp01(1 - Math.abs(wrapDist(v, 0.0)) / 0.09);
      const chip = clamp01((fbm2(u * 320, v * 60, 3, 2, 0.5, 23) - 0.62) * 9) * equator * 0.85;
      wear = Math.max(wear, chip);

      if (wear > 0) {
        const scratch = ridged2(u * 700, v * 40, 2, 3);
        const steel = 124 + scratch * 30;
        r += (steel - r) * wear;
        g += (steel * 1.005 - g) * wear;
        b += (steel * 1.03 - b) * wear;
        rough += (0.20 + scratch * 0.10 - rough) * wear;
        metal += (0.94 - metal) * wear;
      }

      // --- gouges ----------------------------------------------------------
      for (const gg of gouges) {
        const dx = wrapDist(u, gg.u);
        const dy = v - gg.v;
        const ca = Math.cos(gg.ang);
        const sa = Math.sin(gg.ang);
        const lx = dx * ca + dy * sa;
        const ly = -dx * sa + dy * ca;
        if (Math.abs(lx) > gg.len || Math.abs(ly) > gg.w) continue;
        const k = clamp01(1 - Math.abs(ly) / gg.w) * clamp01(1 - Math.abs(lx) / gg.len);
        r += (196 - r) * k * 0.9;
        g += (200 - g) * k * 0.9;
        b += (206 - b) * k * 0.9;
        rough += (0.16 - rough) * k;
        metal += (0.96 - metal) * k;
      }

      // --- net burn at the twelve hooks ------------------------------------
      // Nylon saws a dark, matte crescent into the paint at every attachment.
      let burn = 0;
      for (let k = 0; k < hooks; k++) {
        const hu = k / hooks;
        const d = Math.hypot(wrapDist(u, hu) * 2.2, (v - 0.27) * 0.9);
        burn = Math.max(burn, clamp01(1 - d / 0.052));
      }
      burn *= 0.8 * clamp01(0.4 + fbm2(u * 180, v * 40, 2, 2, 0.5, 33));
      if (burn > 0) {
        r += (58 - r) * burn;
        g += (36 - g) * burn;
        b += (28 - b) * burn;
        rough += (0.68 - rough) * burn;
        metal += (0.04 - metal) * burn;
      }

      // --- grime in the crevices -------------------------------------------
      const grime = clamp01(fbm2(u * 40, v * 12, 4, 2, 0.5, 61) - 0.44) * 0.9;
      r -= grime * 22;
      g -= grime * 16;
      b -= grime * 10;

      ai.data[o] = clamp01(r / 255) * 255;
      ai.data[o + 1] = clamp01(g / 255) * 255;
      ai.data[o + 2] = clamp01(b / 255) * 255;
      ai.data[o + 3] = 255;

      // Cheap baked cavity term in R; three only samples G and B here but the
      // channel is free and keeps the map useful if AO is wired up later.
      oi.data[o] = clamp01(0.82 + 0.18 * (1 - burn)) * 255;
      oi.data[o + 1] = clamp01(rough) * 255;
      oi.data[o + 2] = clamp01(metal) * 255;
      oi.data[o + 3] = 255;
    }
  }

  ac.putImageData(ai, 0, 0);
  oc.putImageData(oi, 0, 0);
  return { map: finish(ac, { srgb: true }), orm: finish(oc) };
}

// ---------------------------------------------------------------------------
// Backboard glass
// ---------------------------------------------------------------------------

export interface GlassMaps {
  /** Base colour: the painted markings, glass tint elsewhere. */
  paint: CanvasTexture;
  /**
   * Coverage. Near-transparent through the glass, opaque under the paint, and
   * creeping up toward the perimeter where you are looking through more of it.
   */
  alpha: CanvasTexture;
  /** Paint is matte-ish; the glass is optically smooth. */
  rough: CanvasTexture;
}

/**
 * The painted front face of the glass: perimeter border, shooter's square, a
 * small manufacturer mark and the wear that lives on both. The paint rides in
 * the coverage map so it is genuinely opaque while the rest of the board is
 * genuinely not — no decal plane, no sort order, and the markings pick up the
 * same specular as the glass around them.
 */
export function bakeBackboardMaps(
  boardW: number,
  boardH: number,
  square: { width: number; height: number; borderWidth: number },
  rimHeightAboveBoardBottom: number,
  size = 1024,
): GlassMaps {
  const W = size;
  const H = Math.round((size * boardH) / boardW);
  const pc = surface(W, H);
  const tc = surface(W, H);
  const rc = surface(W, H);
  const px = W / boardW; // pixels per metre

  // --- unpainted glass ------------------------------------------------------
  pc.fillStyle = '#cfe6e0';
  pc.fillRect(0, 0, W, H);
  tc.fillStyle = '#000000';
  tc.fillRect(0, 0, W, H);
  rc.fillStyle = '#050505';
  rc.fillRect(0, 0, W, H);

  // Paint is laid down on all three maps at once.
  const paintOn = (fn: (c: CanvasRenderingContext2D) => void) => {
    pc.save();
    pc.fillStyle = '#e6e4da';
    pc.strokeStyle = '#e6e4da';
    fn(pc);
    pc.restore();
    tc.save();
    tc.fillStyle = '#ffffff';
    tc.strokeStyle = '#ffffff';
    fn(tc);
    tc.restore();
    rc.save();
    rc.fillStyle = '#4e4e4e';
    rc.strokeStyle = '#4e4e4e';
    fn(rc);
    rc.restore();
  };

  const lw = square.borderWidth * px;

  // Perimeter border, inset a hair from the glass edge.
  const inset = 0.026 * px;
  paintOn((c) => {
    c.lineWidth = lw;
    c.strokeRect(inset + lw / 2, inset + lw / 2, W - 2 * inset - lw, H - 2 * inset - lw);
  });

  // Shooter's square, sitting on the rim line.
  const sqW = square.width * px;
  const sqH = square.height * px;
  const sqX = (W - sqW) / 2;
  // Canvas y is measured down from the top of the board.
  const rimY = H - rimHeightAboveBoardBottom * px;
  const sqY = rimY - sqH + lw * 0.5;
  paintOn((c) => {
    c.lineWidth = lw;
    c.strokeRect(sqX + lw / 2, sqY + lw / 2, sqW - lw, sqH - lw);
  });

  // Manufacturer mark, bottom-left, and a size legend bottom-right. Small,
  // low-contrast, the kind of thing you only notice once you look for it.
  pc.save();
  pc.globalAlpha = 0.5;
  pc.fillStyle = '#eceadf';
  pc.font = `600 ${Math.round(0.022 * px)}px sans-serif`;
  pc.textBaseline = 'alphabetic';
  pc.fillText('BALLIN  PRO GLASS', inset + lw * 2.0, H - inset - lw * 1.6);
  pc.textAlign = 'right';
  pc.fillText('72 x 42 IN', W - inset - lw * 2.0, H - inset - lw * 1.6);
  pc.restore();
  tc.save();
  tc.globalAlpha = 0.4;
  tc.fillStyle = '#fff';
  tc.font = `600 ${Math.round(0.022 * px)}px sans-serif`;
  tc.fillText('BALLIN  PRO GLASS', inset + lw * 2.0, H - inset - lw * 1.6);
  tc.textAlign = 'right';
  tc.fillText('72 x 42 IN', W - inset - lw * 2.0, H - inset - lw * 1.6);
  tc.restore();

  // --- wear -----------------------------------------------------------------
  // Everything here is deliberately tiny in amplitude. Paint wear on a board
  // that gets wiped at every timeout is a hint, not a texture.
  const pi = pc.getImageData(0, 0, W, H);
  const ti = tc.getImageData(0, 0, W, H);
  const ri = rc.getImageData(0, 0, W, H);
  const cxq = sqX + sqW / 2;
  const cyq = sqY + sqH / 2;
  for (let y = 0; y < H; y++) {
    const ny = (y / H) * 2 - 1;
    for (let x = 0; x < W; x++) {
      const o = (y * W + x) * 4;
      const painted = ti.data[o] > 128;
      const film = fbm2(x * 0.012, y * 0.012, 4, 2, 0.5, 401);
      if (painted) {
        // Thinning and a few chips, worst inside the square where the ball hits.
        const near = clamp01(
          1 - Math.hypot((x - cxq) / (sqW * 0.75), (y - cyq) / (sqH * 0.75)),
        );
        const n = fbm2(x * 0.12, y * 0.12, 4, 2.3, 0.5, 77);
        const chip = clamp01((n - 0.865 + near * 0.04) * 11);
        if (chip > 0) {
          ti.data[o] = ti.data[o + 1] = ti.data[o + 2] = 255 - chip * 105;
          ri.data[o] = ri.data[o + 1] = ri.data[o + 2] = 78 - chip * 60;
        }
        const dirt = (film - 0.5) * 7;
        pi.data[o] = clamp01((pi.data[o] + dirt) / 255) * 255;
        pi.data[o + 1] = clamp01((pi.data[o + 1] + dirt) / 255) * 255;
        pi.data[o + 2] = clamp01((pi.data[o + 2] + dirt * 1.4) / 255) * 255;
      } else {
        // The glass itself: a cleaning-cloth swirl that only shows in the
        // specular, and the tint deepening toward the perimeter where the sight
        // line runs through more glass.
        const nx = (x / W) * 2 - 1;
        const rim = clamp01((Math.max(Math.abs(nx), Math.abs(ny)) - 0.62) / 0.38);
        const smear = clamp01((ridged2(x * 0.02, y * 0.05, 3, 12) - 0.62) * 1.6);
        const grease = clamp01((film - 0.62) * 3);
        ri.data[o] = ri.data[o + 1] = ri.data[o + 2] = 5 + smear * 16 + grease * 12;
        const cov = 30 + rim * rim * 46 + grease * 9 + smear * 5;
        ti.data[o] = ti.data[o + 1] = ti.data[o + 2] = cov;
        // Green builds with path length, so the edges of the pane are greener.
        pi.data[o] = clamp01((pi.data[o] - rim * 34) / 255) * 255;
        pi.data[o + 2] = clamp01((pi.data[o + 2] - rim * 12) / 255) * 255;
      }
    }
  }
  pc.putImageData(pi, 0, 0);
  tc.putImageData(ti, 0, 0);
  rc.putImageData(ri, 0, 0);

  return {
    paint: finish(pc, { srgb: true, wrapS: ClampToEdgeWrapping, wrapT: ClampToEdgeWrapping }),
    alpha: finish(tc, { wrapS: ClampToEdgeWrapping, wrapT: ClampToEdgeWrapping }),
    rough: finish(rc, { wrapS: ClampToEdgeWrapping, wrapT: ClampToEdgeWrapping }),
  };
}

/**
 * The reflection the glass carries.
 *
 * A vertical pane can only ever mirror what sits opposite it, and opposite this
 * one is a dark bowl — so left to geometry alone the board renders as a black
 * rectangle. Real broadcast glass reads bright because it picks up the ceiling
 * banks at a grazing angle, and that is what this bakes: a soft, blurred array
 * of bank quads plus two small hard speculars from the nearest fixtures. It is
 * added over the glass and slid with the camera so it parallaxes like a
 * reflection rather than sitting there like a decal.
 *
 * Kept inside a black margin so the camera slide never drags content off-board.
 */
export function bakeGlassReflection(boardW: number, boardH: number, size = 512): CanvasTexture {
  const W = size;
  const H = Math.round((size * boardH) / boardW);
  const c = surface(W, H);
  c.fillStyle = '#000000';
  c.fillRect(0, 0, W, H);

  // Broad, low sheen: the whole ceiling plane smeared across the upper board.
  const sheen = c.createRadialGradient(W * 0.38, H * 0.20, 0, W * 0.38, H * 0.22, W * 0.70);
  sheen.addColorStop(0, 'rgba(128,144,158,0.34)');
  sheen.addColorStop(0.45, 'rgba(80,94,110,0.17)');
  sheen.addColorStop(1, 'rgba(0,0,0,0)');
  c.fillStyle = sheen;
  c.fillRect(0, 0, W, H);

  // The bank array itself: rows of long linear fixtures converging with
  // distance, blurred hard. Individually dim — the population wants to sit at
  // 60–120 on screen, not at white.
  c.save();
  c.globalCompositeOperation = 'lighter';
  const rows = [
    { y: 0.125, h: 0.038, n: 5, x0: 0.09, x1: 0.90, w: 0.135, a: 0.26, blur: 30 },
    { y: 0.245, h: 0.030, n: 4, x0: 0.14, x1: 0.83, w: 0.110, a: 0.17, blur: 38 },
    { y: 0.360, h: 0.024, n: 3, x0: 0.21, x1: 0.74, w: 0.088, a: 0.10, blur: 44 },
  ];
  for (const r of rows) {
    for (let i = 0; i < r.n; i++) {
      const t = r.n === 1 ? 0.5 : i / (r.n - 1);
      // Fixtures are not evenly spaced in a real house rig.
      const seed = Math.round(r.y * 100);
      const jx = (hash2(i, seed, 7) - 0.5) * 0.045;
      const jy = (hash2(i, seed, 19) - 0.5) * 0.030;
      const js = 0.78 + hash2(i, seed, 31) * 0.5;
      const ja = 0.7 + hash2(i, seed, 43) * 0.6;
      const cxq = (r.x0 + (r.x1 - r.x0) * t + jx) * W;
      const cyq = (r.y + jy) * H;
      c.shadowColor = `rgba(190,208,226,${r.a * ja})`;
      c.shadowBlur = r.blur;
      c.shadowOffsetX = 0;
      c.shadowOffsetY = 0;
      c.fillStyle = `rgba(190,208,226,${r.a * ja * 0.7})`;
      c.beginPath();
      c.ellipse(cxq, cyq, (r.w * W * js) / 2, (r.h * H * js) / 2, (jy - 0.5) * 0.12, 0, Math.PI * 2);
      c.fill();
    }
  }
  c.restore();

  // Two small hard speculars from the nearest fixtures. These are the only
  // things on the board allowed anywhere near clipping, and the only ones that
  // should bloom.
  c.save();
  c.globalCompositeOperation = 'lighter';
  for (const [hx, hy, hr] of [
    [0.268, 0.122, 0.013],
    [0.598, 0.124, 0.0095],
  ]) {
    const g = c.createRadialGradient(hx * W, hy * H, 0, hx * W, hy * H, hr * W);
    g.addColorStop(0, 'rgba(255,252,244,1)');
    g.addColorStop(0.28, 'rgba(226,236,248,0.55)');
    g.addColorStop(1, 'rgba(0,0,0,0)');
    c.fillStyle = g;
    c.fillRect(0, 0, W, H);
  }
  c.restore();

  // Fade to black at the border so sliding the map never smears anything in.
  const img = c.getImageData(0, 0, W, H);
  for (let y = 0; y < H; y++) {
    const fy = clamp01(Math.min(y, H - 1 - y) / (H * 0.085));
    for (let x = 0; x < W; x++) {
      const fx = clamp01(Math.min(x, W - 1 - x) / (W * 0.085));
      const k = smoothstep(Math.min(fx, fy));
      const o = (y * W + x) * 4;
      img.data[o] *= k;
      img.data[o + 1] *= k;
      img.data[o + 2] *= k;
      img.data[o + 3] = 255;
    }
  }
  c.putImageData(img, 0, 0);

  return finish(c, { srgb: true, wrapS: ClampToEdgeWrapping, wrapT: ClampToEdgeWrapping });
}

// ---------------------------------------------------------------------------
// Padding
// ---------------------------------------------------------------------------

export interface PadOpts {
  label?: string;
  /** Base vinyl colour. */
  base?: [number, number, number];
  /** Accent stripe colour for the branded wrap. */
  accent?: [number, number, number];
  /** Seams per tile along U. */
  panels?: number;
  stripe?: boolean;
}

/**
 * Stitched vinyl padding — the stanchion wrap and the strip under the glass.
 * Panels are welded together with a raised seam and a run of saddle stitching;
 * the vinyl itself has a low, broad sheen that varies where it has creased.
 */
export function bakeVinylPad(
  width = 1024,
  height = 256,
  opts: PadOpts = {},
): { map: CanvasTexture; rough: CanvasTexture } {
  const W = width;
  const H = height;
  const label = opts.label ?? '';
  const base = opts.base ?? [17, 19, 26];
  const accent = opts.accent ?? [186, 58, 32];
  const panels = opts.panels ?? 4;

  const mc = surface(W, H);
  const rc = surface(W, H);
  const mi = mc.createImageData(W, H);
  const ri = rc.createImageData(W, H);

  for (let y = 0; y < H; y++) {
    const v = y / H;
    for (let x = 0; x < W; x++) {
      const u = x / W;
      const o = (y * W + x) * 4;

      // Pebbled vinyl grain.
      const grain = fbm2(x * 0.42, y * 0.42, 4, 2.1, 0.52, 5);
      // Slow creasing from being kicked, leaned on and stacked in a truck.
      const crease = ridged2(u * 7, v * 3.4, 3, 88);
      const soft = fbm2(u * 5, v * 2.5, 3, 2, 0.5, 140);

      let r = base[0] + (grain - 0.5) * 12 + (soft - 0.5) * 9;
      let g = base[1] + (grain - 0.5) * 12 + (soft - 0.5) * 9;
      let b = base[2] + (grain - 0.5) * 13 + (soft - 0.5) * 11;

      // A brand stripe across the middle of the wrap.
      if (opts.stripe !== false) {
        const band = clamp01(1 - Math.abs(v - 0.5) / 0.155);
        const edge = smoothstep(clamp01(band * 6));
        r += (accent[0] - r) * edge;
        g += (accent[1] - g) * edge;
        b += (accent[2] - b) * edge;
      }

      let rough = 0.62 + (grain - 0.5) * 0.16 + crease * 0.09;

      // Welded panel seams with a raised bead and stitch marks either side.
      const seam = Math.abs((u * panels) % 1 - 0.5) * 2; // 0 at seam
      const seamK = clamp01(1 - seam * W / (panels * 5));
      if (seamK > 0) {
        r -= seamK * 9;
        g -= seamK * 9;
        b -= seamK * 9;
        rough -= seamK * 0.22;
      }
      // Saddle stitching: short dashes flanking each seam.
      const stitchDist = Math.abs(seam * (W / (panels * 2)) - 7);
      if (stitchDist < 1.6) {
        const dash = (y % 11) < 6 ? 1 : 0;
        if (dash) {
          r += 26;
          g += 24;
          b += 22;
          rough -= 0.14;
        }
      }
      // Horizontal top and bottom binding tape.
      const bind = Math.max(clamp01(1 - v / 0.045), clamp01(1 - (1 - v) / 0.045));
      r += bind * 14;
      g += bind * 13;
      b += bind * 12;
      rough -= bind * 0.12;

      // Scuffs low down where shoes and chairs hit it.
      const scuff = clamp01((fbm2(x * 0.06, y * 0.16, 4, 2, 0.5, 303) - 0.55) * 5) * clamp01((v - 0.55) / 0.4);
      r += scuff * 34;
      g += scuff * 33;
      b += scuff * 31;
      rough += scuff * 0.16;

      mi.data[o] = clamp01(r / 255) * 255;
      mi.data[o + 1] = clamp01(g / 255) * 255;
      mi.data[o + 2] = clamp01(b / 255) * 255;
      mi.data[o + 3] = 255;
      const rr = clamp01(rough) * 255;
      ri.data[o] = rr;
      ri.data[o + 1] = rr;
      ri.data[o + 2] = rr;
      ri.data[o + 3] = 255;
    }
  }
  mc.putImageData(mi, 0, 0);
  rc.putImageData(ri, 0, 0);

  if (label) {
    mc.save();
    mc.translate(W / 2, H / 2);
    mc.textAlign = 'center';
    mc.textBaseline = 'middle';
    mc.font = `800 ${Math.round(H * 0.17)}px sans-serif`;
    mc.fillStyle = 'rgba(238,238,232,0.93)';
    mc.letterSpacing = `${Math.round(H * 0.03)}px`;
    mc.fillText(label, 0, 0);
    mc.restore();
    // The print is smoother than the vinyl under it.
    rc.save();
    rc.translate(W / 2, H / 2);
    rc.textAlign = 'center';
    rc.textBaseline = 'middle';
    rc.font = `800 ${Math.round(H * 0.17)}px sans-serif`;
    rc.fillStyle = 'rgba(88,88,88,0.85)';
    rc.letterSpacing = `${Math.round(H * 0.03)}px`;
    rc.fillText(label, 0, 0);
    rc.restore();
  }

  void hash2;
  return {
    map: finish(mc, { srgb: true, wrapT: ClampToEdgeWrapping }),
    rough: finish(rc, { wrapT: ClampToEdgeWrapping }),
  };
}
