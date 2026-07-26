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
      //
      // Base hue 17.5°, saturation 83%, value 84% — §5.2's 14–24° / 70–85%
      // window. It has to be pitched this high in albedo because the ring hangs
      // in a bowl that is deliberately 3 stops down and ACES pulls saturated
      // reds toward black: a "correct-looking" swatch in the texture viewer
      // lands as dark maroon on screen, which is exactly what round 1 caught.
      const peel = fbm2(u * 220, v * 34, 3, 2.1, 0.55, 5);
      let r = 214 + (peel - 0.5) * 22;
      let g = 88 + (peel - 0.5) * 14;
      let b = 36 + (peel - 0.5) * 8;
      // The underside stays cleaner and reads a shade deeper.
      const underside = smoothstep(clamp01((0.42 - Math.abs(v - 0.25)) / 0.3));
      r -= underside * 15;
      g -= underside * 8;
      b -= underside * 3;

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
      r -= grime * 18;
      g -= grime * 13;
      b -= grime * 8;

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
 *
 * The single most important number in this bake is the *base colour of the
 * unpainted glass*. Alpha compositing is `src·a + dst·(1−a)`, so a pale base
 * adds light to whatever is behind the pane: the crowd comes out brighter
 * through the board than beside it and the whole thing reads as a lit grey
 * slab. Real glass subtracts. The base here is therefore a very dark green-
 * teal, which darkens and greens the bowl exactly the way tempered float glass
 * does, and every bright thing on the board — the paint, the edge band, the
 * bank reflections — is added back deliberately on top.
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
  // Dark, faintly green. See the note above: this must subtract, not add.
  const GLASS_R = 13;
  const GLASS_G = 31;
  const GLASS_B = 25;
  pc.fillStyle = `rgb(${GLASS_R},${GLASS_G},${GLASS_B})`;
  pc.fillRect(0, 0, W, H);
  tc.fillStyle = '#000000';
  tc.fillRect(0, 0, W, H);
  rc.fillStyle = '#040404';
  rc.fillRect(0, 0, W, H);

  // Paint is laid down on all three maps at once.
  const paintOn = (
    colour: string,
    rough: string,
    fn: (c: CanvasRenderingContext2D) => void,
  ) => {
    pc.save();
    pc.fillStyle = colour;
    pc.strokeStyle = colour;
    fn(pc);
    pc.restore();
    tc.save();
    tc.fillStyle = '#ffffff';
    tc.strokeStyle = '#ffffff';
    fn(tc);
    tc.restore();
    rc.save();
    rc.fillStyle = rough;
    rc.strokeStyle = rough;
    fn(rc);
    rc.restore();
  };

  const lw = square.borderWidth * px;

  // Perimeter border: white enamel, the brightest paint on the board.
  const inset = 0.026 * px;
  paintOn('#e9eae5', '#4b4b4b', (c) => {
    c.lineWidth = lw;
    c.strokeRect(inset + lw / 2, inset + lw / 2, W - 2 * inset - lw, H - 2 * inset - lw);
  });

  // Shooter's square, sitting on the rim line. Struck in the same NBA orange as
  // the ring so the two read as one piece of hardware from the shooting angle.
  const sqW = square.width * px;
  const sqH = square.height * px;
  const sqX = (W - sqW) / 2;
  // Canvas y is measured down from the top of the board.
  const rimY = H - rimHeightAboveBoardBottom * px;
  const sqY = rimY - sqH + lw * 0.5;
  // 21° hue, 72% saturation — bright enough to read as NBA orange against a
  // three-stops-down bowl, restrained enough not to read as a vinyl sticker.
  paintOn('#c76a38', '#565656', (c) => {
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

  // Rear-surface ghost of the square (§5.1). The paint is on the front face and
  // the back face is 38 mm behind it, so at any oblique angle the square is
  // faintly doubled. It lives in *this* map rather than in the reflection map
  // because the reflection map is slid against the camera to parallax, which
  // would drag the ghost right off the paint it is meant to be doubling — it
  // stops reading as a doubled edge and starts reading as a stray bright line.
  const ghostDX = lw * 0.62;
  const ghostDY = -lw * 0.40;
  /** Distance to a rectangle's outline; negative inside. */
  const frameDist = (x: number, y: number, x0: number, y0: number, x1: number, y1: number) => {
    const ox = Math.max(x0 - x, 0, x - x1);
    const oy = Math.max(y0 - y, 0, y - y1);
    const out = Math.hypot(ox, oy);
    if (out > 0) return out;
    return -Math.min(x - x0, x1 - x, y - y0, y1 - y);
  };
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
        // The glass itself. Three things are happening in this branch and they
        // are all path-length effects:
        //
        //  · coverage climbs toward the perimeter, because a sight line near the
        //    frame crosses more glass than one through the middle;
        //  · the tint goes greener with it (iron oxide in float glass), so the
        //    crowd seen through the corners is measurably greener than the crowd
        //    seen through the centre;
        //  · a cleaning-cloth swirl that lives only in the roughness, so it is
        //    invisible until a bank reflection crosses it.
        const nx = (x / W) * 2 - 1;
        const rim = clamp01((Math.max(Math.abs(nx), Math.abs(ny)) - 0.54) / 0.46);
        const smear = clamp01((ridged2(x * 0.02, y * 0.05, 3, 12) - 0.62) * 1.6);
        const grease = clamp01((film - 0.62) * 3);
        // Float glass is drawn on a tin bath and keeps a very long, very shallow
        // waviness along one axis; it is what makes a reflected bank ripple.
        const wave = Math.sin(y * 0.055 + fbm2(x * 0.006, y * 0.02, 2, 2, 0.5, 17) * 6.0);
        ri.data[o] = ri.data[o + 1] = ri.data[o + 2] =
          4 + smear * 15 + grease * 11 + (wave + 1) * 1.6;
        let cov = 66 + rim * rim * 92 + grease * 11 + smear * 7;
        let gr = GLASS_R * (1 - rim * 0.45);
        let gg = GLASS_G * (1 + rim * 0.42);
        let gb = GLASS_B * (1 - rim * 0.10);

        const gd = Math.abs(
          frameDist(x - ghostDX, y - ghostDY, sqX, sqY, sqX + sqW, sqY + sqH),
        );
        const ghost = 1 - smoothstep(clamp01(gd / (lw * 0.62)));
        if (ghost > 0) {
          // The back face reflects the orange in front of it, at a few percent.
          const k = ghost * 0.55;
          gr += (168 - gr) * k;
          gg += (94 - gg) * k;
          gb += (58 - gb) * k;
          cov += ghost * 26;
        }

        ti.data[o] = ti.data[o + 1] = ti.data[o + 2] = cov;
        pi.data[o] = clamp01(gr / 255) * 255;
        pi.data[o + 1] = clamp01(gg / 255) * 255;
        pi.data[o + 2] = clamp01(gb / 255) * 255;
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
export function bakeGlassReflection(
  boardW: number,
  boardH: number,
  size = 512,
): CanvasTexture {
  const W = size;
  const H = Math.round((size * boardH) / boardW);
  const c = surface(W, H);
  c.fillStyle = '#000000';
  c.fillRect(0, 0, W, H);

  // Broad, low sheen. Deliberately weak and confined to the top third: a wide
  // even wash over the whole pane is what makes a board read as a grey slab,
  // and round 1 caught exactly that.
  const sheen = c.createRadialGradient(W * 0.40, H * 0.11, 0, W * 0.40, H * 0.13, W * 0.52);
  sheen.addColorStop(0, 'rgba(112,128,146,0.15)');
  sheen.addColorStop(0.45, 'rgba(64,78,94,0.07)');
  sheen.addColorStop(1, 'rgba(0,0,0,0)');
  c.fillStyle = sheen;
  c.fillRect(0, 0, W, H);

  // Population (a) of §1.4: the bank array. Rows of long linear fixtures
  // converging with distance, blurred hard, individually dim — this population
  // wants to sit around 60–120 on screen, never near white.
  c.save();
  c.globalCompositeOperation = 'lighter';
  const rows = [
    { y: 0.108, h: 0.030, n: 5, x0: 0.10, x1: 0.91, w: 0.115, a: 0.200, blur: 26 },
    { y: 0.222, h: 0.025, n: 4, x0: 0.15, x1: 0.84, w: 0.098, a: 0.145, blur: 34 },
    { y: 0.352, h: 0.021, n: 4, x0: 0.13, x1: 0.86, w: 0.082, a: 0.098, blur: 40 },
    { y: 0.505, h: 0.017, n: 3, x0: 0.20, x1: 0.78, w: 0.068, a: 0.062, blur: 46 },
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
      c.shadowColor = `rgba(186,204,224,${r.a * ja})`;
      c.shadowBlur = r.blur;
      c.shadowOffsetX = 0;
      c.shadowOffsetY = 0;
      c.fillStyle = `rgba(186,204,224,${r.a * ja * 0.7})`;
      c.beginPath();
      c.ellipse(cxq, cyq, (r.w * W * js) / 2, (r.h * H * js) / 2, (jy - 0.5) * 0.12, 0, Math.PI * 2);
      c.fill();
    }
  }
  c.restore();

  // --- the streaks ----------------------------------------------------------
  // Population (b), and the thing that actually says "glass". A single catwalk
  // run reflects off a 1.8 m pane as one long, *hard* bar with a hot core, not
  // as a soft wash — and because the glass has two surfaces 38 mm apart, it
  // reflects twice: a second, dimmer, blurrier copy sits below and behind the
  // first. Computed per pixel rather than drawn, because a canvas gradient
  // cannot give a sharp cross-section and a tapered length at the same time.
  // A reflected catwalk run is a metre-wide strip of fixtures seen in a mirror,
  // so it lands on the pane as a *broad* bar with a defined core. Drawn any
  // narrower than this and it stops being a reflection and starts being an
  // anamorphic lens flare, which §8.7 rules out by name.
  const streaks = [
    // Primary: the near sideline catwalk.
    { cx: 0.500, cy: 0.268, ang: -0.185, half: 0.520, w: 0.046, amp: 148, seg: 4.6, tight: 1.55 },
    // Rear-surface second reflection: same run, one glass thickness down and
    // blurrier, because it has been through the pane twice.
    { cx: 0.548, cy: 0.378, ang: -0.150, half: 0.470, w: 0.086, amp: 34, seg: 3.0, tight: 0.75 },
    // Far cross bank, catching the pane at a shallower angle.
    { cx: 0.400, cy: 0.560, ang: 0.115, half: 0.360, w: 0.056, amp: 22, seg: 2.2, tight: 0.95 },
  ];

  // Three small hard speculars from the nearest fixtures. These are the only
  // things on the board allowed to clip, and the only ones that should bloom.
  const hots = [
    { x: 0.262, y: 0.148, r: 0.0128, a: 250 },
    { x: 0.618, y: 0.196, r: 0.0094, a: 216 },
    { x: 0.418, y: 0.108, r: 0.0068, a: 170 },
  ];

  const img = c.getImageData(0, 0, W, H);
  const d = img.data;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const o = (y * W + x) * 4;
      let add = 0;

      for (const s of streaks) {
        const rx = x - s.cx * W;
        const ry = y - s.cy * H;
        const ca = Math.cos(s.ang);
        const sa = Math.sin(s.ang);
        const along = rx * ca + ry * sa;
        const half = s.half * W;
        if (Math.abs(along) > half) continue;
        const perp = -rx * sa + ry * ca;
        const q = perp / (s.w * H);
        // A tight core over a wide skirt: that combination is what makes a
        // specular read as hard rather than as a blur.
        const core = Math.exp(-q * q * s.tight * 3.2);
        const skirt = Math.exp(-q * q * 0.30) * 0.26;
        const taper = Math.pow(Math.cos((along / half) * Math.PI * 0.5), 0.85);
        // Individual fixtures inside the run, so the bar is not a plain bar.
        const seg = 0.70 + 0.30 * Math.pow(
          0.5 + 0.5 * Math.cos((along / W) * s.seg * Math.PI * 2),
          0.55,
        );
        add += (core + skirt) * taper * seg * s.amp;
      }

      for (const h of hots) {
        const dx = (x - h.x * W) / (h.r * W);
        const dy = (y - h.y * H) / (h.r * W);
        const dd = Math.hypot(dx, dy);
        if (dd >= 1) continue;
        add += Math.pow(1 - dd, 2.1) * h.a;
      }

      // The rear surface also reflects the paint on the front face, so the
      // border and the square carry a faint doubled ghost, displaced by twice
      // the glass thickness. §5.1 asks for it and almost nothing has it.
      if (add <= 0) continue;
      d[o] = Math.min(255, d[o] + add * 0.96);
      d[o + 1] = Math.min(255, d[o + 1] + add * 0.985);
      d[o + 2] = Math.min(255, d[o + 2] + add);
    }
  }

  // Fade to black at the border so sliding the map never smears anything in.
  for (let y = 0; y < H; y++) {
    const fy = clamp01(Math.min(y, H - 1 - y) / (H * 0.085));
    for (let x = 0; x < W; x++) {
      const fx = clamp01(Math.min(x, W - 1 - x) / (W * 0.085));
      const k = smoothstep(Math.min(fx, fy));
      const o = (y * W + x) * 4;
      d[o] *= k;
      d[o + 1] *= k;
      d[o + 2] *= k;
      d[o + 3] = 255;
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
