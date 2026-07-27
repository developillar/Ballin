/**
 * Uniforms: jersey, shorts, shoes and socks, baked procedurally.
 *
 * The jersey and shorts are *surfaces of their own*, lofted a few millimetres
 * off the body and skinned separately, so this file only has to answer "what is
 * this cloth made of". Three things decide whether a kit reads as polyester
 * rather than as a coloured shell:
 *
 *  1. **Weave at the right scale.** Hole pitch on a game mesh is 1.2–2.2 mm.
 *     Across a 1.1 m torso that is ~600 holes, i.e. well under a pixel at
 *     gameplay distance — so the weave has to be *roughness and normal*
 *     micro-detail that breaks up the sheen, never a visible grid. It is baked
 *     at that pitch and allowed to mip away into a roughness constant.
 *  2. **Appliqué, not decals.** Numbers and names are tackle twill: 1–3 mm
 *     proud of the base cloth, smoother than the mesh, with an embroidered
 *     stitch perimeter and their own contact shadow along the lower edge. All
 *     three land in the height field, so the number lights like a raised patch.
 *  3. **Piping and bound trim.** The side seam, the armhole and the neck all
 *     carry a bound edge of a different colour and a different roughness.
 *
 * Owned by the players agent.
 */

import {
  CanvasTexture,
  ClampToEdgeWrapping,
  LinearFilter,
  LinearMipmapLinearFilter,
  RepeatWrapping,
  SRGBColorSpace,
  type Wrapping,
} from 'three';
import { clamp01, fbm2, valueNoise2 } from '../core/MathX';

export interface TeamKit {
  name: string;
  /** Short display name on the back of the jersey. */
  city: string;
  /**
   * Base cloth, as **diffuse albedo** — not as the display value we want.
   *
   * §3.4 puts a rendered home white at 210–238 and round 0 read that as an
   * instruction for this field, setting 205/203/197. That is 0.60 linear, well
   * under the ~0.88 of real white polyester, and once the light was applied the
   * chest measured 144. An albedo is what the surface does to light, not what
   * the frame is supposed to show; the display figure is the *output* of this
   * number, the tone curve and the rig.
   */
  base: [number, number, number];
  /** Number / name fill. */
  ink: [number, number, number];
  /** Outline and piping. */
  trim: [number, number, number];
  /** Accent used on the shorts side panel and the waistband. */
  accent: [number, number, number];
  /** Shoe body colour. */
  shoe: [number, number, number];
  /** Sock colour. */
  sock: [number, number, number];
}

export const TEAM_KITS: readonly [TeamKit, TeamKit] = [
  {
    name: 'home',
    city: 'BALLIN',
    base: [243, 241, 235], // 0.891 linear — white polyester knit
    ink: [26, 38, 72],
    trim: [176, 138, 66],
    accent: [26, 38, 72],
    shoe: [240, 238, 233],
    sock: [246, 245, 241],
  },
  {
    name: 'away',
    city: 'RIVALS',
    base: [58, 74, 138],
    ink: [230, 230, 234],
    trim: [186, 148, 72],
    accent: [158, 50, 58],
    shoe: [48, 51, 62],
    sock: [52, 60, 92],
  },
];

export const HOME_NUMBERS = [3, 7, 11, 24, 34];
export const AWAY_NUMBERS = [1, 8, 13, 21, 45];
export const HOME_NAMES = ['REEVES', 'AKANNO', 'DUVAL', 'MARSH', 'OKONKWO'];
export const AWAY_NAMES = ['HALE', 'SANTOS', 'BRIGGS', 'NWOSU', 'VELEZ'];

/**
 * Where the kit atlas splits: rows above this fraction are the jersey, rows
 * below are the shorts, so one texture and one draw call carry both. The mesh
 * generator maps its garment `v` through `KIT_UV`, which leaves a small unused
 * band at the seam so mip filtering cannot bleed the waistband into the hem.
 */
const SPLIT = 0.62;

export const KIT_UV = {
  jersey: (v: number): number => 0.006 + v * 0.6,
  shorts: (v: number): number => 0.634 + v * 0.358,
} as const;

export interface ClothMaps {
  albedo: CanvasTexture;
  normal: CanvasTexture;
  /** G = roughness. R = sheen mask. */
  data: CanvasTexture;
  dispose(): void;
}

function makeCanvas(w: number, h: number): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

function texture(
  c: HTMLCanvasElement,
  srgb: boolean,
  aniso: number,
  wrap: Wrapping = ClampToEdgeWrapping,
): CanvasTexture {
  const t = new CanvasTexture(c);
  t.flipY = false;
  if (srgb) t.colorSpace = SRGBColorSpace;
  t.wrapS = wrap;
  t.wrapT = wrap;
  t.minFilter = LinearMipmapLinearFilter;
  t.magFilter = LinearFilter;
  t.anisotropy = aniso;
  t.needsUpdate = true;
  return t;
}

function put(data: Uint8ClampedArray, W: number, H: number): HTMLCanvasElement {
  const c = makeCanvas(W, H);
  const ctx = c.getContext('2d')!;
  const img = ctx.createImageData(W, H);
  img.data.set(data);
  ctx.putImageData(img, 0, 0);
  return c;
}

/** Height field → tangent-space normal, with a global slope scale. */
function heightToNormal(height: Float32Array, W: number, H: number, slope: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      // u wraps around the garment, so the horizontal difference wraps too.
      const xm = (x - 1 + W) % W;
      const xp = (x + 1) % W;
      const ym = Math.max(0, y - 1);
      const yp = Math.min(H - 1, y + 1);
      const dx = (height[y * W + xp] - height[y * W + xm]) * slope;
      const dy = (height[yp * W + x] - height[ym * W + x]) * slope;
      const inv = 1 / Math.sqrt(dx * dx + dy * dy + 1);
      const o = i * 4;
      out[o] = (-dx * inv * 0.5 + 0.5) * 255;
      out[o + 1] = (dy * inv * 0.5 + 0.5) * 255;
      out[o + 2] = (inv * 0.5 + 0.5) * 255;
      out[o + 3] = 255;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Jersey
// ---------------------------------------------------------------------------

/**
 * Cached per-kit fabric. The weave and the slub noise are the expensive part of
 * a bake and they are identical for every player in a kit, so they are computed
 * once and the per-player pass only composites the appliqué on top.
 */
export class KitBaker {
  private readonly W: number;
  private readonly H: number;
  private readonly weaveH: Float32Array;
  private readonly weaveR: Float32Array;
  private readonly slub: Float32Array;

  constructor(
    readonly kit: TeamKit,
    size: number,
    private readonly aniso: number,
    seed = 4471,
  ) {
    this.W = size;
    this.H = size;
    const W = this.W;
    const H = this.H;
    this.weaveH = new Float32Array(W * H);
    this.weaveR = new Float32Array(W * H);
    this.slub = new Float32Array(W * H);

    // Torso circumference ≈ 1.15 m across the full u range. 1.6 mm hole pitch
    // → ~720 holes around; the warp runs at twice that count.
    const holesU = 720;
    const holesV = 300;
    for (let y = 0; y < H; y++) {
      const v = y / H;
      for (let x = 0; x < W; x++) {
        const u = x / W;
        // Two interlocking sinusoid families give the knit its diamond eyelet.
        const a = Math.sin(u * holesU * Math.PI * 2 + v * 3.1);
        const b = Math.sin(v * holesV * Math.PI * 2);
        const c = Math.sin((u * holesU + v * holesV) * Math.PI * 2 * 0.5);
        const eyelet = a * b * 0.5 + c * 0.28;
        const i = y * W + x;
        this.weaveH[i] = eyelet;
        // The hole floor is rougher than the thread crown, which is what makes
        // a knit's sheen break up instead of sitting as one smooth lobe.
        this.weaveR[i] = eyelet * 0.5;
        this.slub[i] = fbm2(u * 90, v * 90, 3, 2.1, 0.55, seed) - 0.5;
      }
    }
  }

  /**
   * Bakes one player's kit into a single atlas so the jersey and the shorts
   * share one draw call: rows above `SPLIT` are the jersey, rows below are the
   * shorts. `u` runs around the garment with 0.5 at the front and 0.0 at the
   * spine / outer seam; `v` runs from the shoulder (or waistband) to the hem.
   */
  bake(number: number, name: string): ClothMaps {
    const { W, H, kit } = this;
    const num = String(number);

    // --- Appliqué masks, drawn with canvas text -------------------------
    const mc = makeCanvas(W, H);
    const g = mc.getContext('2d')!;
    g.fillStyle = '#000';
    g.fillRect(0, 0, W, H);
    g.fillStyle = '#fff';
    g.textAlign = 'center';
    g.textBaseline = 'middle';

    // Back number: large, centred on the spine (u = 0 wraps, so it straddles
    // the texture edge — draw it twice).
    const backSize = H * 0.3;
    g.font = `bold ${backSize}px "Arial Narrow", Helvetica, Arial, sans-serif`;
    const backY = H * 0.44;
    g.fillText(num, 0, backY);
    g.fillText(num, W, backY);
    // Player name on an arc above it.
    const nameSize = H * 0.075;
    g.font = `bold ${nameSize}px Helvetica, Arial, sans-serif`;
    this.arcText(g, name, 0, H * 0.2, W * 0.16);
    this.arcText(g, name, W, H * 0.2, W * 0.16);
    // Front number: smaller, high on the right chest as the modern cut has it.
    const frontSize = H * 0.155;
    g.font = `bold ${frontSize}px "Arial Narrow", Helvetica, Arial, sans-serif`;
    g.fillText(num, W * 0.5 + W * 0.085, H * 0.3);
    // Front wordmark.
    g.font = `bold ${H * 0.06}px Helvetica, Arial, sans-serif`;
    g.fillText(kit.city, W * 0.5 - W * 0.03, H * 0.29);

    const mask = g.getImageData(0, 0, W, H).data;

    // --- Composite -------------------------------------------------------
    const alb = new Uint8ClampedArray(W * H * 4);
    const dat = new Uint8ClampedArray(W * H * 4);
    const height = new Float32Array(W * H);
    const [br, bg, bb] = kit.base;
    const [ir, ig, ib] = kit.ink;
    const [tr, tg, tb] = kit.trim;

    const splitRow = Math.floor(H * SPLIT);
    const [ar, ag, ab] = kit.accent;

    for (let y = 0; y < splitRow; y++) {
      const v = y / splitRow;
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const u = x / W;
        const m = mask[i * 4] / 255;

        // Outline: the ring just outside the fill, from a cheap dilation.
        let dil = m;
        for (let k = 1; k <= 3; k++) {
          const xm = (x - k + W) % W;
          const xp = (x + k) % W;
          const ym = Math.max(0, y - k) * W;
          const yp = Math.min(H - 1, y + k) * W;
          dil = Math.max(
            dil,
            mask[(y * W + xm) * 4] / 255,
            mask[(y * W + xp) * 4] / 255,
            mask[(ym + x) * 4] / 255,
            mask[(yp + x) * 4] / 255,
          );
        }
        const outline = clamp01(dil - m);

        // Bound trim: a band along the neck / armhole edge and the two side
        // seams, so cloth boundaries read as finished edges.
        const topTrim = 1 - clamp01((v - 0.012) / 0.026);
        const hemTrim = clamp01((v - 0.955) / 0.03);
        const seamL = 1 - clamp01(Math.abs(u - 0.25) / 0.008);
        const seamR = 1 - clamp01(Math.abs(u - 0.75) / 0.008);
        const piping = clamp01(Math.max(seamL, seamR) * 0.9 + Math.max(topTrim, hemTrim));

        const slub = this.slub[i];
        let r = br * (1 + slub * 0.05);
        let gg = bg * (1 + slub * 0.05);
        let b = bb * (1 + slub * 0.05);
        let rough = 0.66 + this.weaveR[i] * 0.09 + slub * 0.05;
        let h = this.weaveH[i] * 0.45 + slub * 0.5;

        if (piping > 0) {
          r = r + (tr - r) * piping;
          gg = gg + (tg - gg) * piping;
          b = b + (tb - b) * piping;
          rough = rough + (0.42 - rough) * piping;
          h += piping * 2.4;
        }
        if (outline > 0) {
          r = r + (tr - r) * outline;
          gg = gg + (tg - gg) * outline;
          b = b + (tb - b) * outline;
          rough = rough + (0.38 - rough) * outline;
          h += outline * 2.6;
        }
        if (m > 0) {
          r = r + (ir - r) * m;
          gg = gg + (ig - gg) * m;
          b = b + (ib - b) * m;
          // Tackle twill is markedly glossier than the mesh around it.
          rough = rough + (0.36 - rough) * m;
          h += m * 3.4;
          // Embroidery: a stitch bead running the perimeter of the patch.
          const edge = clamp01((dil - m) * 2.2);
          const stitch = Math.max(0, Math.sin((x + y) * 1.9) ) * edge;
          h += stitch * 1.4;
          rough += stitch * 0.12;
        }

        // Contact shadow under the appliqué's lower and right edges.
        const yUp = Math.max(0, y - 4) * W + x;
        const xLeft = y * W + ((x - 4 + W) % W);
        const shade = clamp01(Math.max(mask[yUp * 4], mask[xLeft * 4]) / 255 - m) * 0.32;
        r *= 1 - shade;
        gg *= 1 - shade;
        b *= 1 - shade;

        const o = i * 4;
        alb[o] = r;
        alb[o + 1] = gg;
        alb[o + 2] = b;
        alb[o + 3] = 255;
        dat[o] = 255 - clamp01(rough) * 200; // sheen mask: smoother = hotter
        dat[o + 1] = clamp01(rough) * 255;
        dat[o + 2] = 0;
        dat[o + 3] = 255;
        height[i] = h;
      }
    }
    // --- Shorts, in the lower band of the same atlas --------------------
    for (let y = splitRow; y < H; y++) {
      const v = (y - splitRow) / (H - splitRow);
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const u = x / W;
        const slub = this.slub[i];
        // Shorts are heavier cloth: coarser weave, duller sheen, bigger folds.
        let h = this.weaveH[i] * 0.26 + slub * 0.9;
        let rough = 0.72 + this.weaveR[i] * 0.05 + slub * 0.06;
        let r = br * (1 + slub * 0.06);
        let g2 = bg * (1 + slub * 0.06);
        let b = bb * (1 + slub * 0.06);

        // Waistband: 1.5–3× the cloth thickness, elastic ribbing, own colour.
        // The garment maps v = 0..0.28 of this band to the waist→crotch run of
        // 152 mm, so 0.082 of atlas v is a 45 mm band — a real waistband, with
        // a hard lower edge rather than the one-ring fade round 0 produced.
        const band = 1 - clamp01((v - 0.082) / 0.008);
        if (band > 0) {
          const rib = Math.sin(u * W * 0.35) * 0.5 + 0.5;
          r += (ar - r) * band;
          g2 += (ag - g2) * band;
          b += (ab - b) * band;
          h += band * (3.4 + rib * 1.2);
          rough += band * 0.06;
        }
        // Side panel and its piping, running down the outer seam.
        const seam = Math.min(Math.abs(u - 0.25), Math.abs(u - 0.75));
        // A narrow colour panel down the outer seam. Wide enough to read as a
        // panel at gameplay distance, not so wide it becomes a dark blob on the
        // thigh of a home white.
        const panel = (1 - clamp01((seam - 0.008) / 0.028)) * clamp01((v - 0.12) / 0.08);
        if (panel > 0) {
          r += (ar - r) * panel * 0.55;
          g2 += (ag - g2) * panel * 0.55;
          b += (ab - b) * panel * 0.55;
        }
        const pipe = (1 - clamp01(Math.abs(seam - 0.04) / 0.006)) * clamp01((v - 0.1) / 0.06);
        if (pipe > 0) {
          r += (tr - r) * pipe;
          g2 += (tg - g2) * pipe;
          b += (tb - b) * pipe;
          h += pipe * 2.4;
          rough += (0.4 - rough) * pipe;
        }
        // Hem binding.
        const hem = clamp01((v - 0.93) / 0.03);
        if (hem > 0) {
          r += (tr - r) * hem * 0.75;
          g2 += (tg - g2) * hem * 0.75;
          b += (tb - b) * hem * 0.75;
          h += hem * 2.2;
        }

        const o = i * 4;
        alb[o] = r;
        alb[o + 1] = g2;
        alb[o + 2] = b;
        alb[o + 3] = 255;
        dat[o] = 255 - clamp01(rough) * 200;
        dat[o + 1] = clamp01(rough) * 255;
        dat[o + 2] = 0;
        dat[o + 3] = 255;
        height[i] = h;
      }
    }

    const nrm = heightToNormal(height, W, H, 0.5 * (512 / W));
    const albedo = texture(put(alb, W, H), true, this.aniso, RepeatWrapping);
    const normal = texture(put(nrm, W, H), false, this.aniso, RepeatWrapping);
    const data = texture(put(dat, W, H), false, this.aniso, RepeatWrapping);
    return {
      albedo,
      normal,
      data,
      dispose() {
        albedo.dispose();
        normal.dispose();
        data.dispose();
      },
    };
  }

  /** Curved name text — real jerseys arc the name over the number. */
  private arcText(
    g: CanvasRenderingContext2D,
    text: string,
    cx: number,
    cy: number,
    radius: number,
  ): void {
    const chars = [...text];
    const spread = Math.min(0.85, chars.length * 0.11);
    for (let i = 0; i < chars.length; i++) {
      const t = chars.length === 1 ? 0 : i / (chars.length - 1) - 0.5;
      const a = t * spread;
      g.save();
      g.translate(cx + Math.sin(a) * radius, cy + (1 - Math.cos(a)) * radius * 0.85);
      g.rotate(a * 0.75);
      g.fillText(chars[i], 0, 0);
      g.restore();
    }
  }
}

// ---------------------------------------------------------------------------
// Footwear
// ---------------------------------------------------------------------------

/**
 * A four-band material strip for the shoe. The shoe mesh writes u = the centre
 * of the band its face belongs to, which lets one draw call carry the four
 * materials a basketball shoe needs: knit upper, synthetic overlay, rubber
 * outsole with tread relief, and the lace/sock band.
 *
 * Bands, in v: 0.0–0.2 knit upper, 0.2–0.4 synthetic overlay, 0.4–0.6 sole
 * (herringbone tread at the bottom of the band rising through the midsole foam
 * to the bright edge line at the top), 0.6–0.8 ribbed sock, 0.8–1.0 laces.
 */
export function bakeShoeStrip(kit: TeamKit, size: number, aniso: number, seed = 66): ClothMaps {
  const W = size;
  const H = size;
  const alb = new Uint8ClampedArray(W * H * 4);
  const dat = new Uint8ClampedArray(W * H * 4);
  const height = new Float32Array(W * H);
  const [sr, sg, sb] = kit.shoe;
  const [kr, kg, kb] = kit.sock;
  const [tr, tg, tb] = kit.trim;

  for (let y = 0; y < H; y++) {
    const v = y / H;
    const band = Math.min(4, Math.floor(v * 5));
    const t = v * 5 - band;
    for (let x = 0; x < W; x++) {
      const u = x / W;
      const i = y * W + x;
      let r = sr;
      let g = sg;
      let b = sb;
      let rough = 0.5;
      let h = 0;

      if (band === 0) {
        // Engineered knit: fibrous, matte, with a directional weave.
        const knit =
          Math.sin(u * 240 * Math.PI * 2) * Math.sin(t * 240 * Math.PI * 2) * 0.5 +
          (valueNoise2(u * 240, t * 240, seed) - 0.5);
        h = knit * 0.8;
        rough = 0.78 + knit * 0.06;
        const fade = 0.92 + 0.12 * fbm2(u * 12, t * 12, 3, 2, 0.5, seed + 1);
        r *= fade;
        g *= fade;
        b *= fade;
      } else if (band === 1) {
        // Synthetic overlay / heel counter: glossier, subtly grained.
        const grain = fbm2(u * 120, t * 120, 3, 2.1, 0.5, seed + 2) - 0.5;
        h = grain * 0.9;
        rough = 0.31 + grain * 0.07;
        r = r * 0.84 + tr * 0.16;
        g = g * 0.84 + tg * 0.16;
        b = b * 0.84 + tb * 0.16;
      } else if (band === 2) {
        // Sole. t = 0 is the ground-facing tread, t = 1 the top of the midsole.
        if (t < 0.3) {
          const skew = u * 40 + t * 90;
          const herring = Math.sin(skew * Math.PI * 2) * Math.sin((u * 40 - t * 90) * Math.PI * 2);
          h = herring * 2.8;
          rough = 0.52 + herring * 0.05;
          r = 62;
          g = 63;
          b = 66;
        } else if (t < 0.9) {
          // Midsole foam: pale, slightly speckled, low relief.
          const foam = fbm2(u * 70, t * 70, 3, 2, 0.5, seed + 3) - 0.5;
          h = foam * 0.6;
          rough = 0.44 + foam * 0.06;
          r = 226 + foam * 22;
          g = 224 + foam * 22;
          b = 218 + foam * 22;
        } else {
          // The bright outsole edge line — the single detail that stops a shoe
          // reading as one blob — with its shadow groove above it.
          const edge = 1 - clamp01(Math.abs(t - 0.935) / 0.03);
          const groove = clamp01((t - 0.965) / 0.035);
          h = edge * 1.6 - groove * 2.2;
          rough = 0.34;
          r = 246 - groove * 130;
          g = 244 - groove * 128;
          b = 238 - groove * 124;
        }
      } else if (band === 3) {
        // Ribbed sock cuff. The shoe mesh spans u = 0..2 around the ankle, so
        // this frequency is ribs-per-half-turn: 26 gives ~52 ribs on a 310 mm
        // circumference, i.e. a 6 mm rib. Round 0 ran 80 — a 2 mm rib, which is
        // 0.26 px at FLOOR framing and therefore aliasing, not ribbing.
        const rib = Math.sin(u * 26 * Math.PI * 2);
        h = rib * 1.5;
        rough = 0.82;
        const cuff = 1 - clamp01(Math.abs(t - 0.86) / 0.12);
        r = kr * (1 - cuff * 0.12);
        g = kg * (1 - cuff * 0.12);
        b = kb * (1 - cuff * 0.12);
        h += cuff * 1.2;
      } else {
        // Laces: two families of crossing cords over a dark tongue.
        r = sr * 0.55;
        g = sg * 0.55;
        b = sb * 0.55;
        rough = 0.72;
        h = (fbm2(u * 60, t * 60, 3, 2, 0.5, seed + 6) - 0.5) * 0.8;
        // Four crossings over the ~60 mm instep the lace band covers → 2–5 px
        // strokes at FLOOR framing, which is what §3.6 asks to be able to see.
        const cordA = 1 - clamp01(Math.abs(((u * 4.2 + t * 3.2) % 1) - 0.5) / 0.11);
        const cordB = 1 - clamp01(Math.abs(((u * 4.2 - t * 3.2 + 1) % 1) - 0.5) / 0.11);
        const lace = Math.max(cordA, cordB);
        if (lace > 0) {
          const twist = Math.sin((u * 60 + t * 40) * Math.PI * 2) * 0.5 + 0.5;
          h += lace * (2.2 + twist * 1.2);
          const pale = 236 - twist * 40;
          r += (pale - r) * lace;
          g += (pale - g) * lace;
          b += (pale * 0.97 - b) * lace;
          rough += (0.6 - rough) * lace;
        }
      }

      const o = i * 4;
      alb[o] = r;
      alb[o + 1] = g;
      alb[o + 2] = b;
      alb[o + 3] = 255;
      dat[o] = 255 - clamp01(rough) * 200;
      dat[o + 1] = clamp01(rough) * 255;
      dat[o + 2] = 0;
      dat[o + 3] = 255;
      height[i] = h;
    }
  }

  const nrm = heightToNormal(height, W, H, 0.7 * (512 / W));
  const albedo = texture(put(alb, W, H), true, aniso, RepeatWrapping);
  const normal = texture(put(nrm, W, H), false, aniso, RepeatWrapping);
  const data = texture(put(dat, W, H), false, aniso, RepeatWrapping);
  return {
    albedo,
    normal,
    data,
    dispose() {
      albedo.dispose();
      normal.dispose();
      data.dispose();
    },
  };
}
