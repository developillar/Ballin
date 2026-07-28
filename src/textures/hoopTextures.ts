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

      // §5.3 puts a used net at 200–235 sRGB on the top rings once it is lit,
      // and the cord is lit by nothing but the overhead banks three stops above
      // a dark bowl, so the albedo has to start high. The vertex-colour soil
      // ramp in Hoop.ts takes it down from here, not up.
      const lum = 221 + 39 * ply + 11 * (fil - 0.5) - 13 * (fuzz - 0.5);
      const o = (y * W + x) * 4;
      ai.data[o] = clamp01(lum / 255) * 255;
      ai.data[o + 1] = clamp01((lum * 0.994) / 255) * 255;
      ai.data[o + 2] = clamp01((lum * 0.952) / 255) * 255;
      ai.data[o + 3] = 255;

      // The valleys between plies trap light and read rougher than the crowns.
      // Smoother than round 1: a tighter lobe puts more light on the crown of
      // the cord — the part facing the camera — where §5.3 measures it, rather
      // than smearing it round to the silhouette where it widens the strand.
      const rough = clamp01(0.27 + 0.26 * (1 - ply) + 0.09 * (fuzz - 0.5));
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
      // §5.2 wants the ring measured *in the frame* at hue 16–20°, saturation
      // 70–85%. This bake is therefore deliberately NOT at that hue: the frame
      // is graded with a warm-highlight split that measured +23 R−B, and that
      // plus an ACES shoulder rotates a saturated orange about 9° toward red on
      // its way to the screen. Round 1 baked a "correct" 17.5° swatch and the
      // ring rendered at 8.7° — rose, not NBA orange. So the bake is pitched
      // yellow of the target (hue ≈ 28°, sat 83%) and lands inside the window
      // once the pipeline has taken its 9° back. Re-measure on the ring in the
      // frame, never in the texture viewer.
      const peel = fbm2(u * 220, v * 34, 3, 2.1, 0.55, 5);
      let r = 214 + (peel - 0.5) * 22;
      let g = 126 + (peel - 0.5) * 16;
      let b = 36 + (peel - 0.5) * 8;
      // The *inner* face stays cleaner and reads a shade deeper. This used to
      // darken v ≈ 0.25, which is the underside — and the underside is the one
      // face a portrait camera below rim height actually sees, so it was
      // dulling the only visible part of the bar. v = 0.5 is the inside of the
      // ring, which is genuinely in its own shadow.
      const inner = smoothstep(clamp01((0.34 - Math.abs(wrapDist(v, 0.5))) / 0.3));
      r -= inner * 16;
      g -= inner * 10;
      b -= inner * 3;

      let rough = 0.33 + (peel - 0.5) * 0.09;
      let metal = 0.06;

      // --- bare metal on the strike face -----------------------------------
      // Balls land on the front of the bar and grind the coat off it. Where
      // that shows matters more than that it exists: v = 0 is the outer
      // equator, 0.25 the underside, 0.5 the inside of the ring and 0.75 the
      // top face. Every portrait framing in this game puts the camera 0.5–1.0 m
      // *below* rim height, so the top face is never on screen — a wear band
      // centred there (which is what round 1 had, at v = 0.78) is invisible and
      // the ring reads pristine, which is §10 tell #21.
      //
      // So the band now runs across the outer/lower/front faces the camera
      // actually sees, wrapping v ≈ 0.85 → 0.45 with its peak on the outer
      // shoulder, and the top keeps a weaker patch so the ring is not wrong
      // from above either.
      const du = Math.abs(wrapDist(u, frontU));
      const faceLower = clamp01(1 - Math.abs(wrapDist(v, 0.13)) / 0.33);
      const faceTop = clamp01(1 - Math.abs(wrapDist(v, 0.78)) / 0.24) * 0.62;
      const facing = Math.max(faceLower, faceTop);
      const arc = clamp01(1 - du / 0.18);
      const wearNoise = fbm2(u * 96, v * 26, 4, 2.2, 0.5, 71);
      let wear = clamp01(Math.pow(arc, 0.9) * Math.pow(facing, 1.05) * 1.55 - 0.28);
      // Chipped paint has a ragged edge, never a soft gradient.
      wear = clamp01((wear - 0.48 + (wearNoise - 0.5) * 0.62) * 6);

      // Chipping along the outer equator where the net and hands scrape. Wider
      // than round 1 and carried the whole way round the ring, because the
      // outer equator is the silhouette edge in every below-rim framing.
      const equator = clamp01(1 - Math.abs(wrapDist(v, 0.0)) / 0.17);
      const chip =
        clamp01((fbm2(u * 320, v * 60, 3, 2, 0.5, 23) - 0.55) * 8) * equator * (0.55 + 0.45 * arc);
      wear = Math.max(wear, chip);

      if (wear > 0) {
        // Ground-back steel: neutral, a little brighter than the coat it
        // replaces (coat luma ≈ 134) so §5.2's "desaturated grey-silver with
        // high specular" reads as a lighter patch and not a dirty one. Kept
        // off full metalness — a mirror in a three-stops-down bowl just goes
        // black, and the whole point of the wear is that you can see it.
        const scratch = ridged2(u * 700, v * 40, 2, 3);
        const steel = 152 + scratch * 34;
        r += (steel - r) * wear;
        g += (steel * 1.005 - g) * wear;
        b += (steel * 1.035 - b) * wear;
        rough += (0.19 + scratch * 0.10 - rough) * wear;
        metal += (0.74 - metal) * wear;
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
  const GLASS_R = 10;
  const GLASS_G = 34;
  const GLASS_B = 27;
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
  // Pre-compensated the same way the ring is (see bakeRimMaps): baked at hue
  // 28° so it lands near 19° once the warm-highlight grade and the ACES
  // shoulder have rotated it, which keeps the square and the ring reading as
  // the same piece of hardware from the shooting angle.
  paintOn('#e08c3e', '#565656', (c) => {
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
        // Coverage is the whole ball game for §5.1's transmission figure.
        // Alpha compositing is `src·a + dst·(1−a)`, so measured transmission is
        // exactly `1 − a` and measured contrast retention is `1 − a` as well.
        // Round 1 sat at a = 0.26 mid-pane and 0.62 at the perimeter, i.e. a
        // ceiling of 0.74 transmission falling to 0.38 — the crowd behind the
        // pane came out 36% dark and three times flatter than the crowd beside
        // it, and no amount of lighting recovers that. §5.1 asks for 0.88–0.94,
        // so the base is a = 0.067 (cov 17/255) climbing to a ≈ 0.13 at the
        // perimeter where the sight line crosses more glass. The board's
        // *darkness* has to come from the bowl behind it being dark, which it
        // is; the pane's job is only to tint and very slightly dim.
        let cov = 17 + rim * rim * 16 + grease * 3 + smear * 2;
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
          gg += (108 - gg) * k;
          gb += (52 - gb) * k;
          cov += ghost * 13;
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

  // --- population (a): the bank array --------------------------------------
  // §1.4(a) is explicit: "a blurred grid of 4–12 bright quads, luminance
  // 60–120, occupying 20–50% of the glass area." Round 1 had this population
  // an order of magnitude too dim — 82.9% of the pane measured under 40 and
  // only 5.5% landed in the band, so the board squinted as a black rectangle
  // and everything bright on it came from one full-width bar.
  //
  // Twelve quads in four converging rows. Each is a soft-edged rectangle — a
  // plateau out to 62% of its half-extent and then a wide blurred skirt —
  // because that is what a 2 m linear fixture looks like in a pane with a
  // little surface waviness. Amplitudes are chosen so the plateau itself sits
  // in the 60–120 band: nothing in this population is allowed anywhere near
  // white, that is what the hard speculars below are for.
  const banks: { x: number; y: number; rx: number; ry: number; amp: number; rot: number }[] = [];
  // Widths are set so the 60-luminance footprints do NOT merge — §1.4 asks for
  // *discrete* quads and three touching ones read as one grey wash. The lower
  // two rows keep off the shooter's square (x 0.33–0.67, y 0.43–0.86; a bank
  // quad landing on it drowns the paint) and off the pane's left margin, which
  // is where §5.1's transmission is measured: a reflection sitting on the
  // sample area reads as glass that transmits more than 100%.
  const rows = [
    { y: 0.118, xs: [0.190, 0.500, 0.810], rx: 0.126, ry: 0.078, amp: 112 },
    { y: 0.312, xs: [0.225, 0.520, 0.808], rx: 0.122, ry: 0.070, amp: 102 },
    { y: 0.545, xs: [0.268, 0.822], rx: 0.112, ry: 0.062, amp: 86 },
    { y: 0.748, xs: [0.258, 0.832], rx: 0.104, ry: 0.055, amp: 72 },
  ];
  for (let ri = 0; ri < rows.length; ri++) {
    const r = rows[ri];
    for (let i = 0; i < r.xs.length; i++) {
      // A house rig is not a spreadsheet: jitter position, size and output so
      // the array does not read as a stamped pattern.
      const jx = (hash2(i, ri * 7 + 1, 7) - 0.5) * 0.030;
      const jy = (hash2(i, ri * 7 + 1, 19) - 0.5) * 0.026;
      const js = 0.90 + hash2(i, ri * 7 + 1, 31) * 0.22;
      const ja = 0.86 + hash2(i, ri * 7 + 1, 43) * 0.26;
      banks.push({
        x: r.xs[i] + jx,
        y: r.y + jy,
        rx: r.rx * js,
        ry: r.ry * (2 - js),
        amp: r.amp * ja,
        rot: (hash2(i, ri * 7 + 1, 57) - 0.5) * 0.16,
      });
    }
  }

  // --- population (b): the catwalk runs ------------------------------------
  // Round 1 drew this as ONE bar spanning the whole 1.8 m pane at a constant
  // 17–21 px cross-section with a hot core — which is not a reflected catwalk,
  // it is the anamorphic streak §8.7 bans by name. A real house has several
  // short runs at different angles, each subtending a fraction of the board,
  // and each soft enough to belong to the 60–120 population rather than to sit
  // on top of it. None of these spans more than 0.27 of the board width.
  const streaks = [
    { cx: 0.300, cy: 0.212, ang: -0.30, half: 0.132, w: 0.058, amp: 78, seg: 1.5, tight: 0.70 },
    { cx: 0.648, cy: 0.158, ang: 0.21, half: 0.116, w: 0.050, amp: 72, seg: 1.3, tight: 0.80 },
    { cx: 0.232, cy: 0.640, ang: -0.44, half: 0.126, w: 0.072, amp: 52, seg: 1.2, tight: 0.50 },
  ];

  // Three small hard speculars from the nearest fixtures. §1.4(b) wants 1–3 at
  // 200–255, each 6–20 px across. These are the only things on the board
  // allowed to clip, and the only ones that should bloom; the core is kept
  // flat (low exponent) so the plateau, not just the centre pixel, renders
  // hot, and the radii put them at 8–17 px on screen at the RIM framing.
  const hots = [
    { x: 0.283, y: 0.152, r: 0.0090, a: 252, p: 1.35 },
    { x: 0.658, y: 0.216, r: 0.0082, a: 246, p: 1.45 },
    { x: 0.452, y: 0.104, r: 0.0074, a: 238, p: 1.6 },
  ];

  const img = c.getImageData(0, 0, W, H);
  const d = img.data;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const o = (y * W + x) * 4;
      let add = 0;

      for (const q of banks) {
        const rx = x - q.x * W;
        const ry = y - q.y * H;
        const ca = Math.cos(q.rot);
        const sa = Math.sin(q.rot);
        const ax = (rx * ca + ry * sa) / (q.rx * W);
        const ay = (-rx * sa + ry * ca) / (q.ry * H);
        // Chebyshev distance keeps the shape a quad rather than a blob.
        const dq = Math.max(Math.abs(ax), Math.abs(ay));
        if (dq >= 1) continue;
        add += (1 - smoothstep(clamp01((dq - 0.62) / 0.38))) * q.amp;
      }

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
        add += Math.pow(1 - dd, h.p) * h.a;
      }

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
  /** Padded rolls across V — 1 for a single tube, 2+ for a tall wrap. */
  rolls?: number;
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
  const rolls = opts.rolls ?? 1;

  const mc = surface(W, H);
  const rc = surface(W, H);
  const mi = mc.createImageData(W, H);
  const ri = rc.createImageData(W, H);

  for (let y = 0; y < H; y++) {
    const v = y / H;
    for (let x = 0; x < W; x++) {
      const u = x / W;
      const o = (y * W + x) * 4;

      // Everything here modulates the base *multiplicatively*. Round 1 added
      // fixed sRGB offsets, which meant that lifting the base out of the black
      // it was measured at (mean 10.4, sd 2.0 — an extruded box, the one thing
      // §5.1 says padding must not be) diluted every detail in proportion. As
      // ratios the seams, bead, stitching and creases hold their contrast at
      // any exposure the lighting rig ends up at.
      const grain = fbm2(x * 0.42, y * 0.42, 4, 2.1, 0.52, 5);
      const crease = ridged2(u * 7, v * 3.4, 3, 88);
      const soft = fbm2(u * 5, v * 2.5, 3, 2, 0.5, 140);

      // The wrap is a padded tube, not a plank: it turns away from the light
      // top and bottom. This single term is most of what stops the pad reading
      // as a flat extruded box, and it is what §5.1 means by "a slightly
      // compressed/creased profile".
      const roll = 0.5 + 0.5 * Math.cos((v * rolls - 0.26) * Math.PI * 2);
      let shade = 0.62 + 0.62 * roll;
      shade *= 1 + (grain - 0.5) * 0.46 + (soft - 0.5) * 0.34 + (crease - 0.5) * 0.40;

      let rough = 0.62 + (grain - 0.5) * 0.16 + crease * 0.09 - roll * 0.06;

      // Welded panel seams: a shaded valley with a lit bead on its far lip, so
      // the seam survives the mip chain as a light/dark pair rather than
      // averaging away the way a lone dark line does.
      const seam = Math.abs((u * panels) % 1 - 0.5) * 2; // 0 at seam
      const seamPx = seam * (W / (panels * 2));
      const seamK = clamp01(1 - seamPx / 5);
      shade *= 1 - seamK * 0.42;
      rough -= seamK * 0.24;
      const bead = clamp01(1 - Math.abs(seamPx - 6.5) / 3.0);
      shade *= 1 + bead * 0.40;
      rough -= bead * 0.10;
      // Saddle stitching: a run of dashes flanking each seam. The dash period
      // is deliberately coarse — this texture is minified 2–3x on screen and an
      // 11 px dash simply mipped away in round 1.
      const stitchDist = Math.abs(seamPx - 11);
      if (stitchDist < 2.4) {
        const k = (1 - stitchDist / 2.4) * ((y % 24) < 14 ? 1 : 0);
        shade *= 1 + 0.95 * k;
        rough -= 0.16 * k;
      }
      // Binding tape along the top and bottom edges of the wrap.
      const bind = Math.max(clamp01(1 - v / 0.05), clamp01(1 - (1 - v) / 0.05));
      shade *= 1 + bind * 0.45;
      rough -= bind * 0.14;
      // Scuffs low down where shoes and chairs hit it.
      const scuff =
        clamp01((fbm2(x * 0.06, y * 0.16, 4, 2, 0.5, 303) - 0.52) * 5) *
        clamp01((v - 0.5) / 0.4);
      shade *= 1 + scuff * 0.75;
      rough += scuff * 0.16;

      let r = base[0] * shade;
      let g = base[1] * shade;
      let b = base[2] * shade;

      // A brand stripe across the middle of the wrap.
      if (opts.stripe !== false) {
        const band = clamp01(1 - Math.abs(v - 0.5) / 0.155);
        const edge = smoothstep(clamp01(band * 6));
        r += (accent[0] * shade - r) * edge;
        g += (accent[1] * shade - g) * edge;
        b += (accent[2] * shade - b) * edge;
      }

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

  return {
    map: finish(mc, { srgb: true, wrapT: ClampToEdgeWrapping }),
    rough: finish(rc, { wrapT: ClampToEdgeWrapping }),
  };
}
