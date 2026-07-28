/**
 * Regulation court markings, drawn as coverage masks.
 *
 * Nothing here knows about colour. Both passes rasterise pure white on black
 * so `courtBake` can turn them into signed distance fields — line work is only
 * ~2.6 texels wide in a whole-court bake, which would read as a grey smear
 * under magnification, whereas an SDF stays a 1-pixel edge at any distance and
 * still filters cleanly when the far baseline is six texels tall.
 *
 * Geometry is taken from `COURT` and cross-checked against the NBA rule book:
 * 94 × 50 ft boundary, 16 ft lane, free-throw circle dashed on the lane side,
 * 23 ft 9 in arc breaking to a 22 ft corner straight 3 ft off the sideline,
 * restricted arc tied back to the backboard plane, lane space marks, 28 ft
 * sideline hashes and the substitution box at the scorer's table.
 */

import { COURT, FT, IN } from '../core/Constants';

export interface CourtLayout {
  /** Canvas size, in texels. Texels are square by construction. */
  W: number;
  H: number;
  /** Metres covered, apron included. */
  totalW: number;
  totalH: number;
  /** Texels per metre (isotropic). */
  ppm: number;
  /** Court metres → canvas texels. */
  cx(x: number): number;
  cz(z: number): number;
  m(v: number): number;
}

export function makeLayout(height: number): CourtLayout {
  const totalW = COURT.length + COURT.apronX * 2;
  const totalH = COURT.width + COURT.apronZ * 2;
  const ppm = height / totalH;
  const W = Math.round((totalW * ppm) / 8) * 8;
  return {
    W,
    H: height,
    totalW,
    totalH,
    ppm,
    cx: (x) => W * 0.5 + x * ppm,
    cz: (z) => height * 0.5 + z * ppm,
    m: (v) => v * ppm,
  };
}

/** Distance from the baseline to the front face of the backboard. */
const BOARD_FACE = 4 * FT;
/** Lane space marks, measured from the baseline. */
const BLOCK_START = 7 * FT;
const BLOCK_LENGTH = 1 * FT;
const LANE_MARKS = [11 * FT, 14 * FT, 17 * FT];
const LANE_MARK_DEPTH = 8 * IN;
const HASH_FROM_BASELINE = 28 * FT;

/** The painted lane. Everything else on an NBA floor is line work, not fill. */
export function drawPaintedAreas(ctx: CanvasRenderingContext2D, L: CourtLayout): void {
  ctx.fillStyle = '#fff';
  for (const side of [1, -1] as const) {
    const baseX = side * COURT.halfLength;
    const ftX = side * (COURT.halfLength - COURT.key.length);
    const x0 = Math.min(L.cx(baseX), L.cx(ftX));
    const x1 = Math.max(L.cx(baseX), L.cx(ftX));
    ctx.fillRect(x0, L.cz(-COURT.key.width / 2), x1 - x0, L.m(COURT.key.width));
  }
}

/** Every regulation line, stroked at `COURT.lineWidth`. */
export function drawLines(ctx: CanvasRenderingContext2D, L: CourtLayout): void {
  const lw = L.m(COURT.lineWidth);
  ctx.strokeStyle = '#fff';
  ctx.fillStyle = '#fff';
  ctx.lineWidth = lw;
  ctx.lineCap = 'butt';
  ctx.lineJoin = 'miter';

  const line = (x0: number, z0: number, x1: number, z1: number) => {
    ctx.beginPath();
    ctx.moveTo(L.cx(x0), L.cz(z0));
    ctx.lineTo(L.cx(x1), L.cz(z1));
    ctx.stroke();
  };

  // --- boundary --------------------------------------------------------
  // Court dimensions are measured to the *inside* of the boundary, so the
  // stroke sits half a width inboard of 94 × 50.
  const bx = COURT.halfLength - COURT.lineWidth * 0.5;
  const bz = COURT.halfWidth - COURT.lineWidth * 0.5;
  ctx.strokeRect(L.cx(-bx), L.cz(-bz), L.m(bx * 2), L.m(bz * 2));

  // --- centre ----------------------------------------------------------
  line(0, -COURT.halfWidth, 0, COURT.halfWidth);
  for (const r of [COURT.centreCircleRadius, 2 * FT]) {
    ctx.beginPath();
    ctx.arc(L.cx(0), L.cz(0), L.m(r), 0, Math.PI * 2);
    ctx.stroke();
  }

  for (const side of [1, -1] as const) {
    const baseX = side * COURT.halfLength;
    const ftX = side * (COURT.halfLength - COURT.key.length);
    const basket = side * (COURT.halfLength - COURT.basketFromBaseline);
    const half = COURT.key.width / 2;

    // --- lane ----------------------------------------------------------
    // The 16 ft lane is measured to the outside of the lane lines, so the
    // strokes sit half a width inboard of the painted edge.
    const laneZ = half - COURT.lineWidth * 0.5;
    line(baseX, -laneZ, ftX, -laneZ);
    line(baseX, laneZ, ftX, laneZ);
    line(ftX, -laneZ, ftX, laneZ); // free-throw line

    // --- free-throw circle ---------------------------------------------
    // Solid on the far side, dashed where it crosses the lane toward the
    // basket. Canvas angle 0 points along +x.
    const r = COURT.key.circleRadius;
    const toBasket = side === 1 ? 0 : Math.PI;
    ctx.beginPath();
    ctx.arc(L.cx(ftX), L.cz(0), L.m(r), toBasket + Math.PI * 0.5, toBasket + Math.PI * 1.5);
    ctx.stroke();

    ctx.save();
    ctx.setLineDash([L.m(0.36), L.m(0.3)]);
    ctx.beginPath();
    ctx.arc(L.cx(ftX), L.cz(0), L.m(r), toBasket - Math.PI * 0.5, toBasket + Math.PI * 0.5);
    ctx.stroke();
    ctx.restore();

    // --- restricted area ------------------------------------------------
    const rr = COURT.restrictedRadius;
    const faceX = side * (COURT.halfLength - BOARD_FACE);
    ctx.beginPath();
    ctx.arc(
      L.cx(basket),
      L.cz(0),
      L.m(rr),
      toBasket + Math.PI * 0.5,
      toBasket + Math.PI * 1.5,
    );
    ctx.stroke();
    line(basket, rr, faceX, rr);
    line(basket, -rr, faceX, -rr);

    // --- three-point line -----------------------------------------------
    const cornerZ = COURT.halfWidth - COURT.threePoint.cornerInsetFromSideline;
    const R = COURT.threePoint.radius;
    const dx = Math.sqrt(Math.max(0, R * R - cornerZ * cornerZ));
    const joinX = basket - side * dx;
    for (const z of [cornerZ, -cornerZ]) line(baseX, z, joinX, z);
    ctx.beginPath();
    const a0 = Math.atan2(cornerZ, joinX - basket);
    const a1 = Math.atan2(-cornerZ, joinX - basket);
    // Sweep the long way round, so the arc bulges toward half court rather
    // than folding back over the baseline.
    ctx.arc(L.cx(basket), L.cz(0), L.m(R), a0, a1, side === -1);
    ctx.stroke();

    // --- lane space marks ------------------------------------------------
    for (const sgn of [1, -1] as const) {
      const zEdge = sgn * half;
      const zOut = sgn * (half + LANE_MARK_DEPTH);
      // Neutral-zone block.
      const bx0 = baseX - side * BLOCK_START;
      const bx1 = baseX - side * (BLOCK_START + BLOCK_LENGTH);
      ctx.fillRect(
        Math.min(L.cx(bx0), L.cx(bx1)),
        Math.min(L.cz(zEdge), L.cz(zOut)),
        Math.abs(L.cx(bx1) - L.cx(bx0)),
        L.m(LANE_MARK_DEPTH),
      );
      for (const d of LANE_MARKS) {
        const mx = baseX - side * d;
        ctx.fillRect(
          L.cx(mx) - lw * 0.5,
          Math.min(L.cz(zEdge), L.cz(zOut)),
          lw,
          L.m(LANE_MARK_DEPTH),
        );
      }
    }

    // --- 28 ft sideline hashes -------------------------------------------
    const hx = side * (COURT.halfLength - HASH_FROM_BASELINE);
    for (const sgn of [1, -1] as const) {
      const zs = sgn * COURT.halfWidth;
      line(hx, zs, hx, zs - sgn * 3 * FT);
    }
    // Coaching box boundary, marked out on the apron at the same station.
    line(hx, COURT.halfWidth, hx, COURT.halfWidth + 0.92);
  }

  // --- substitution box at the scorer's table ----------------------------
  const subOut = COURT.halfWidth + 1.22;
  for (const s of [1, -1] as const) {
    line(s * 4 * FT, COURT.halfWidth, s * 4 * FT, subOut);
  }
  line(-4 * FT, subOut, 4 * FT, subOut);
}
