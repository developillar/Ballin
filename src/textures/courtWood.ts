/**
 * Milled hard-maple detail tile.
 *
 * A regulation floor is ~2-1/4 in tongue-and-groove strip maple, so the thing
 * the eye actually reads at gameplay distance — grain pitch, the tight dark
 * seam between strips, the bevel that catches the overhead banks, staggered
 * butt joints — lives at a spatial frequency far finer than a single
 * whole-court bake can hold. So the court is split in two:
 *
 *   • this tile, repeated across the floor, carries every high-frequency cue
 *     (grain in height *and* tone, knots, medullary rays, and the *relief* of
 *     the seams and butt joints);
 *   • `courtBake` carries everything that must not repeat (lines, logo, wear,
 *     traffic gloss);
 *   • the joinery's dark line — strip seams, staggered butt joints and the
 *     panel grid — is drawn analytically in `Court.ts` off the world position.
 *     It has to be: a 2.2 mm groove is two of the sixty-four texels this tile
 *     spends across a board and about half a texel along it, and the near floor
 *     minifies the tile roughly two to one, so a trilinear tap erased the seams
 *     and the butt joints never survived the bake at all.
 *
 * The texel budget is spent anisotropically on purpose: grain lines run *along*
 * the board, so all of the detail that matters varies across the board (V), and
 * the tile is sized so V resolves ~0.9 mm while U — where features are metres
 * long — gets away with ~4.8 mm.
 *
 * The RGB channels hold a tangent-space normal derived from a real height
 * field; the alpha channel holds the tone modulation (0.5 = neutral) so the
 * same grain that bends the light also darkens the late wood and can be read
 * back for roughness. One texture, four signals.
 */

import { FT, IN } from '../core/Constants';
import { clamp01, fbm2, makeRng, smootherstep, valueNoise2 } from '../core/MathX';

/**
 * 2-1/8 in face width = 53.98 mm. The milled norm is 2 to 2-1/4 in; sitting in
 * the middle of it puts a board at ~38 px at the FLOOR framing's near edge
 * (~700 px/m at the rubric's reference frame), dead centre of the 36–42 px
 * band, and puts 27–30 boards across the visible near floor rather than 24.
 */
export const BOARD_WIDTH = 2.125 * IN;
/** Boards per tile. Keeps the tile's V wrap exactly on a seam. */
export const TILE_BOARDS = 16;
/** 16 ft along the grain: long enough for 2–3 butt joints per strip. */
export const TILE_LENGTH = 16 * FT;
export const TILE_WIDTH = BOARD_WIDTH * TILE_BOARDS;

/**
 * Half the width of a milled groove, in metres — the tongue-and-groove joint
 * between two strips. 2.2 mm total, which is 1.5 px at the FLOOR framing's near
 * edge and sub-pixel by mid-court, matching §2.1's 1–2 px seam.
 *
 * Shared with the floor shader, which draws the seam analytically off the world
 * coordinate. The relief in this tile has to sit on the same line or the bright
 * milled lip lands beside the dark groove instead of on its shoulder.
 */
export const SEAM_HALF = 0.0011;

/**
 * Portable-floor panel module. §2.1 calls for ~4 × 7 ft panels; a panel edge is
 * a strip edge on a real deck, so the cross-court pitch is stated in whole
 * boards — 22 × 53.98 mm = 1.187 m = 3 ft 10.7 in — and the panel joint then
 * falls exactly on a milled seam instead of wandering across the boards.
 */
export const PANEL_BOARDS = 22;
export const PANEL_LENGTH = 7 * FT;
export const PANEL_WIDTH = PANEL_BOARDS * BOARD_WIDTH;

interface BoardSpec {
  seed: number;
  /** Cathedral figuring shows on roughly one board in eight. */
  cathedral: boolean;
  cathCentre: number;
  cathPhase: number;
  /** Growth-ring pitch for this board, metres. */
  pitch: number;
  /** Butt-joint positions along the strip, metres. */
  joints: number[];
  /** Mineral streak: offset within the board and its strength. */
  streakAt: number;
  streakK: number;
  wander: number;
}

interface Knot {
  x: number;
  z: number;
  r: number;
  k: number;
}

export interface MapleDetail {
  canvas: HTMLCanvasElement;
  /** Metres covered by a single repeat. */
  tile: { x: number; z: number };
}

/**
 * @param W texels along the grain, @param H texels across it.
 */
export function bakeMapleDetail(W: number, H: number, seed = 20260726): MapleDetail {
  const rng = makeRng(seed);
  const mx = TILE_LENGTH / W;
  const mz = TILE_WIDTH / H;

  // Hard maple wants 15–25 rings across a 57 mm face, i.e. a 2.3–3.8 mm pitch.
  // Anything finer sits on Nyquist and mips straight into mud, so the pitch is
  // also floored at three texels for the low tiers.
  const minPitch = mz * 3;
  /** Ray-fleck frequency capped at one noise cell per ~3.2 texels along U. */
  const rayFx = 1 / (mx * 3.2);

  const boards: BoardSpec[] = [];
  for (let b = 0; b < TILE_BOARDS; b++) {
    const joints: number[] = [];
    let jx = 0.18 + rng() * 1.1;
    while (jx < TILE_LENGTH - 0.18) {
      joints.push(jx);
      jx += 1.25 + rng() * 1.5;
    }
    boards.push({
      seed: Math.floor(rng() * 65536),
      cathedral: rng() < 0.13,
      cathCentre: 0.3 + rng() * 0.4,
      cathPhase: rng() * 6.28,
      // 16–21 growth rings across a 54 mm face. Hard maple is tight-grained;
      // anything wider reads as oak and instantly as a stock texture.
      pitch: Math.max(minPitch, 0.0021 + rng() * 0.0012),
      joints,
      streakAt: rng(),
      streakK: rng() < 0.3 ? 0.5 + rng() * 0.5 : 0,
      wander: 0.6 + rng() * 1.6,
    });
  }

  // Two pin knots per tile. Maple is graded tight, and because the tile
  // repeats ~150 times a generous knot count would read as a lattice — the
  // named tiling artefact — so they stay small and low contrast.
  const knots: Knot[] = [];
  for (let i = 0; i < 2; i++) {
    knots.push({
      x: 0.2 + rng() * (TILE_LENGTH - 0.4),
      z: (1.2 + rng() * (TILE_BOARDS - 2.4)) * BOARD_WIDTH,
      r: 0.0025 + rng() * 0.0035,
      k: 0.3 + rng() * 0.35,
    });
  }

  const N = W * H;
  const height = new Float32Array(N);
  const tone = new Float32Array(N);

  for (let j = 0; j < H; j++) {
    const z = (j + 0.5) * mz;
    const bi = Math.min(TILE_BOARDS - 1, Math.floor(z / BOARD_WIDTH));
    const spec = boards[bi];
    const p = z / BOARD_WIDTH - bi;
    // Distance to the nearest strip edge, in metres.
    const dEdge = Math.min(p, 1 - p) * BOARD_WIDTH;

    for (let i = 0; i < W; i++) {
      const x = (i + 0.5) * mx;
      const o = j * W + i;

      // --- grain coordinate -------------------------------------------------
      // Gentle 2–6° wander so the rings are not machine-straight.
      let zw = (p - 0.5) * BOARD_WIDTH;
      zw += (fbm2(x * 0.55, spec.seed * 0.017, 2, 2, 0.5, spec.seed) - 0.5) * 0.006 * spec.wander;

      let gcoord: number;
      if (spec.cathedral) {
        // Nested arches: contours of |z - c| / s(x) with a slowly bulging s.
        const bulge =
          0.22 +
          0.95 *
            fbm2(x * 0.8 + spec.cathPhase, spec.seed * 0.011, 3, 2, 0.55, spec.seed + 5);
        gcoord = Math.abs(zw - (spec.cathCentre - 0.5) * BOARD_WIDTH) / bulge;
      } else {
        gcoord = zw;
      }

      const t = gcoord / spec.pitch;
      const g =
        valueNoise2(t, x * 0.45, spec.seed) * 0.55 +
        valueNoise2(t * 2.31 + 7.7, x * 1.15, spec.seed + 313) * 0.25 +
        valueNoise2(t * 0.37, x * 0.19, spec.seed + 77) * 0.2;
      // Hard maple is tight and light: thin late-wood lines on a pale field.
      let dark = Math.pow(clamp01((g - 0.46) / 0.55), 1.6);

      // Mineral streak — a long darker vein a few boards in ten carry.
      if (spec.streakK > 0) {
        const dS = Math.abs(p - spec.streakAt);
        const sN = valueNoise2(x * 2.4, spec.seed * 0.03, spec.seed + 991);
        dark += spec.streakK * 0.5 * Math.exp(-Math.pow(dS / 0.055, 2)) * (0.4 + sN * 0.6);
      }

      // Medullary rays: short pale flecks running across the ring lines.
      //
      // The tile is deliberately anisotropic — ~4.8 mm per texel along the
      // grain against ~0.85 mm across it — so a ray frequency chosen in metres
      // sits far above Nyquist on the U axis and beats against the sample grid.
      // That produced an evenly spaced bright stipple that latched onto the
      // strip seams and read as machine stitching down every board. The
      // frequency is therefore pinned to the texel pitch, not to the world.
      const ray = valueNoise2(x * rayFx, z * 150, spec.seed + 61);
      const rayK = Math.pow(clamp01((ray - 0.76) / 0.24), 2) * 0.34;

      // Tone amplitude up from 0.18. The near floor measured 1.80 sRGB of
      // hp(7 px) RMS and the closeup 2.79 against a film-grain floor of
      // 3.65–4.52 (§8.5), i.e. the maple's own figure was sitting *under* the
      // post chain's noise. The height field is left where it was: the relief
      // is already at the depth a sanded board has, and deepening it makes the
      // milled lip outrun the tonal seam, which inverts the joint into a bright
      // line.
      let h = -dark * 0.00012 + rayK * 0.00002;
      let ton = 0.5 - dark * 0.22 + rayK * 0.045;

      // --- knots ------------------------------------------------------------
      for (const k of knots) {
        const dx = x - k.x;
        const dz = z - k.z;
        const d = Math.sqrt(dx * dx + dz * dz);
        if (d > k.r * 5) continue;
        if (d < k.r) {
          const c = 1 - smootherstep(d / k.r);
          ton -= 0.2 * k.k * c;
          h -= 0.00022 * k.k * c;
        } else {
          // Grain sweeps around the knot rather than through it.
          const ring = 1 - smootherstep((d - k.r) / (k.r * 4));
          const sw = valueNoise2(d / (spec.pitch * 1.6), Math.atan2(dz, dx) * 2.2, spec.seed + 17);
          ton -= 0.09 * k.k * ring * clamp01((sw - 0.45) / 0.55);
        }
      }

      // --- strip seam -------------------------------------------------------
      // A tight channel with a shallow eased shoulder either side. The shoulder
      // throws a bright lip when a bank rakes across the floor — but the coat
      // is glossy enough that an over-deep bevel makes that lip *outrun* the
      // tonal darkening, and the seam inverts into a bright line down the
      // board, which is the opposite of what §2.1 asks for. The relief is
      // therefore shallow.
      //
      // The *tone* of the seam is deliberately light here and the dark line is
      // drawn analytically in the floor shader instead. A 1.5 mm groove is two
      // texels out of the sixty-four this tile spends on a board, and the near
      // floor of a portrait frame minifies the tile about two to one, so a
      // trilinear tap averaged the seam away exactly where the camera is
      // closest — the frame came back with no countable boards at all. What
      // stays here is the milled relief, which is what gives the seam its
      // specular break; the tone is carried by a coordinate that cannot be
      // filtered out.
      if (dEdge < 0.0034) {
        const shoulder = 1 - smootherstep(dEdge / 0.0034);
        h -= 0.00013 * shoulder * shoulder;
        ton -= 0.05 * shoulder;
      }
      if (dEdge < 0.0011) {
        const core = 1 - dEdge / 0.0011;
        h -= 0.0003 * core;
        ton -= 0.13 * core * core;
      }

      // --- butt joints ------------------------------------------------------
      // Same split, and worse: a joint is a feature *along* U, where this tile
      // spends 4.8 mm on a texel, so a 2 mm joint was sub-texel and never
      // survived the bake at all. The relief stays; the line is analytic.
      for (const jx of spec.joints) {
        const d = Math.abs(x - jx);
        if (d > 0.004) continue;
        const shoulder = 1 - smootherstep(d / 0.004);
        h -= 0.0001 * shoulder * shoulder;
        ton -= 0.04 * shoulder;
        if (d < 0.0011) {
          const core = 1 - d / 0.0011;
          h -= 0.00025 * core;
          ton -= 0.12 * core * core;
        }
      }

      height[o] = h;
      tone[o] = ton;
    }
  }

  // --- height → tangent-space normal ---------------------------------------
  const cvs = document.createElement('canvas');
  cvs.width = W;
  cvs.height = H;
  const ctx = cvs.getContext('2d', { willReadFrequently: true })!;
  const img = ctx.createImageData(W, H);
  const px = img.data;

  for (let j = 0; j < H; j++) {
    const jm = (j - 1 + H) % H;
    const jp = (j + 1) % H;
    for (let i = 0; i < W; i++) {
      const im = (i - 1 + W) % W;
      const ip = (i + 1) % W;
      // Central differences in metres — the tile is anisotropic, so the two
      // axes must not share a scale.
      const dhdx = (height[j * W + ip] - height[j * W + im]) / (2 * mx);
      const dhdz = (height[jp * W + i] - height[jm * W + i]) / (2 * mz);
      let nx = -dhdx;
      let ny = -dhdz;
      const nz = 1;
      const inv = 1 / Math.sqrt(nx * nx + ny * ny + nz * nz);
      nx *= inv;
      ny *= inv;

      const o = (j * W + i) * 4;
      px[o] = Math.round((nx * 0.5 + 0.5) * 255);
      // Canvas rows run +z downward while the tangent frame's V runs the other
      // way once flipY lands, hence the sign.
      px[o + 1] = Math.round((-ny * 0.5 + 0.5) * 255);
      px[o + 2] = Math.round((nz * inv * 0.5 + 0.5) * 255);
      px[o + 3] = Math.round(clamp01(tone[j * W + i]) * 255);
    }
  }
  ctx.putImageData(img, 0, 0);

  return { canvas: cvs, tile: { x: TILE_LENGTH, z: TILE_WIDTH } };
}
