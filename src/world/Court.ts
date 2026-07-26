/**
 * The hardwood: floor slab, procedurally milled maple with parquet run
 * direction, painted keys and lines, centre logo, and the glossy clear-coat
 * that gives the court its signature reflections.
 *
 * Owned by the court agent — everything from the albedo bake to the varnish
 * roughness map lives here.
 */

import {
  CanvasTexture,
  Group,
  LinearMipmapLinearFilter,
  Mesh,
  MeshPhysicalMaterial,
  PlaneGeometry,
  RepeatWrapping,
  SRGBColorSpace,
  Vector3,
} from 'three';
import type { Engine, System } from '../core/Engine';
import { COURT } from '../core/Constants';
import { clamp01, fbm2, makeRng, ridged2 } from '../core/MathX';

export class CourtSystem implements System {
  readonly name = 'court';
  readonly order = 10;

  group = new Group();
  floor!: Mesh;

  init(engine: Engine): void {
    this.group.name = 'court';
    engine.scene.add(this.group);

    const size = engine.quality.textureSize;
    const albedo = this.bakeAlbedo(size);
    const rough = this.bakeRoughness(size >> 1);

    albedo.colorSpace = SRGBColorSpace;
    for (const t of [albedo, rough]) {
      t.wrapS = t.wrapT = RepeatWrapping;
      t.anisotropy = engine.anisotropy;
      t.minFilter = LinearMipmapLinearFilter;
      t.generateMipmaps = true;
      t.needsUpdate = true;
    }

    const mat = new MeshPhysicalMaterial({
      map: albedo,
      roughnessMap: rough,
      roughness: 1,
      metalness: 0,
      clearcoat: 0.86,
      clearcoatRoughness: 0.09,
      reflectivity: 0.42,
      envMapIntensity: 0.72,
    });

    const w = COURT.length + COURT.apronX * 2;
    const d = COURT.width + COURT.apronZ * 2;
    const geo = new PlaneGeometry(w, d, 1, 1);
    geo.rotateX(-Math.PI / 2);
    this.floor = new Mesh(geo, mat);
    this.floor.receiveShadow = true;
    this.floor.name = 'hardwood';
    this.group.add(this.floor);
  }

  /**
   * Bakes the full court into one texture: board-by-board maple, the painted
   * keys, every regulation line, and the wear that a real floor accumulates.
   * The bake is UV-mapped 1:1 over the apron-inclusive plane.
   */
  private bakeAlbedo(size: number): CanvasTexture {
    const W = size * 2;
    const H = size;
    const cvs = document.createElement('canvas');
    cvs.width = W;
    cvs.height = H;
    const ctx = cvs.getContext('2d')!;
    const rng = makeRng(20260726);

    const totalW = COURT.length + COURT.apronX * 2;
    const totalH = COURT.width + COURT.apronZ * 2;
    const pxPerM = W / totalW;
    const m2px = (m: number) => m * pxPerM;
    /** Court space (metres, origin centre) → canvas pixels. */
    const cx = (x: number) => W * 0.5 + m2px(x);
    const cz = (z: number) => H * 0.5 + (z / totalH) * H;

    // ---- Maple substrate -------------------------------------------------
    ctx.fillStyle = '#b7803f';
    ctx.fillRect(0, 0, W, H);

    const boardWidthM = 0.0635; // 2.5 in strip flooring
    const boardPx = m2px(boardWidthM);
    const boards = Math.ceil(H / boardPx) + 1;
    for (let i = 0; i < boards; i++) {
      const y0 = i * boardPx;
      // Each board gets its own hue: maple runs from pale straw to amber.
      const t = rng();
      const l = 52 + t * 13;
      const s = 38 + rng() * 12;
      const h = 30 + rng() * 7;
      ctx.fillStyle = `hsl(${h} ${s}% ${l}%)`;
      ctx.fillRect(0, y0, W, boardPx + 1);

      // Grain: long ridged streaks along the board.
      const img = ctx.getImageData(0, Math.max(0, Math.floor(y0)), W, Math.ceil(boardPx) + 1);
      const px = img.data;
      const rows = img.height;
      const seed = Math.floor(rng() * 9999);
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < W; x++) {
          const n =
            ridged2(x * 0.0125, (y0 + y) * 0.26, 4, seed) * 0.62 +
            fbm2(x * 0.055, (y0 + y) * 0.9, 3, 2, 0.5, seed + 41) * 0.38;
          const k = (n - 0.5) * 46;
          const o = (y * W + x) * 4;
          px[o] = clamp01((px[o] + k * 1.1) / 255) * 255;
          px[o + 1] = clamp01((px[o + 1] + k * 0.86) / 255) * 255;
          px[o + 2] = clamp01((px[o + 2] + k * 0.6) / 255) * 255;
        }
      }
      ctx.putImageData(img, 0, Math.max(0, Math.floor(y0)));

      // Board seam.
      ctx.strokeStyle = 'rgba(50,28,10,0.34)';
      ctx.lineWidth = Math.max(1, pxPerM * 0.0016);
      ctx.beginPath();
      ctx.moveTo(0, y0 + 0.5);
      ctx.lineTo(W, y0 + 0.5);
      ctx.stroke();

      // Butt joints where planks end.
      let x = rng() * m2px(2);
      while (x < W) {
        ctx.beginPath();
        ctx.moveTo(x, y0);
        ctx.lineTo(x, y0 + boardPx);
        ctx.stroke();
        x += m2px(1.5 + rng() * 2.6);
      }
    }

    // ---- Apron shading ---------------------------------------------------
    // Out-of-bounds surround is stained darker on a broadcast floor.
    ctx.save();
    ctx.globalCompositeOperation = 'multiply';
    ctx.fillStyle = 'rgba(96,58,26,0.62)';
    ctx.fillRect(0, 0, W, cz(-COURT.halfWidth));
    ctx.fillRect(0, cz(COURT.halfWidth), W, H - cz(COURT.halfWidth));
    ctx.fillRect(0, 0, cx(-COURT.halfLength), H);
    ctx.fillRect(cx(COURT.halfLength), 0, W - cx(COURT.halfLength), H);
    ctx.restore();

    // ---- Painted keys ----------------------------------------------------
    const lw = m2px(COURT.lineWidth);
    const paint = '#1b4d8f';
    for (const side of [1, -1] as const) {
      const baseX = side * COURT.halfLength;
      const ftX = side * (COURT.halfLength - COURT.key.length);
      ctx.fillStyle = paint;
      ctx.fillRect(
        Math.min(cx(baseX), cx(ftX)),
        cz(-COURT.key.width / 2),
        Math.abs(cx(ftX) - cx(baseX)),
        m2px(COURT.key.width),
      );
      // Free-throw semicircle, also painted.
      ctx.beginPath();
      ctx.arc(cx(ftX), cz(0), m2px(COURT.key.circleRadius), 0, Math.PI * 2);
      ctx.fill();
    }

    // ---- Regulation lines ------------------------------------------------
    ctx.strokeStyle = '#f6f2ea';
    ctx.fillStyle = '#f6f2ea';
    ctx.lineWidth = lw;
    ctx.lineCap = 'butt';

    const rect = (x0: number, z0: number, x1: number, z1: number) => {
      ctx.strokeRect(cx(x0), cz(z0), cx(x1) - cx(x0), cz(z1) - cz(z0));
    };

    // Boundary.
    rect(-COURT.halfLength, -COURT.halfWidth, COURT.halfLength, COURT.halfWidth);
    // Half-court line.
    ctx.beginPath();
    ctx.moveTo(cx(0), cz(-COURT.halfWidth));
    ctx.lineTo(cx(0), cz(COURT.halfWidth));
    ctx.stroke();
    // Centre circles.
    for (const r of [COURT.centreCircleRadius, 2 * 0.3048]) {
      ctx.beginPath();
      ctx.arc(cx(0), cz(0), m2px(r), 0, Math.PI * 2);
      ctx.stroke();
    }

    for (const side of [1, -1] as const) {
      const baseX = side * COURT.halfLength;
      const ftX = side * (COURT.halfLength - COURT.key.length);
      const basket = side * (COURT.halfLength - COURT.basketFromBaseline);

      // Lane boundary.
      rect(baseX, -COURT.key.width / 2, ftX, COURT.key.width / 2);

      // Free-throw circle: solid toward the basket, dashed away from it.
      ctx.beginPath();
      ctx.arc(cx(ftX), cz(0), m2px(COURT.key.circleRadius), 0, Math.PI * 2);
      ctx.stroke();

      // Restricted-area arc under the rim.
      ctx.beginPath();
      const a0 = side === 1 ? Math.PI * 0.5 : -Math.PI * 0.5;
      ctx.arc(cx(basket), cz(0), m2px(COURT.restrictedRadius), a0, a0 + Math.PI, side !== 1);
      ctx.stroke();

      // Three-point line: corner straights + arc.
      const cornerZ = COURT.halfWidth - COURT.threePoint.cornerInsetFromSideline;
      const R = COURT.threePoint.radius;
      // X where the arc meets the corner straight.
      const dx = Math.sqrt(Math.max(0, R * R - cornerZ * cornerZ));
      const joinX = basket - side * dx;

      for (const z of [cornerZ, -cornerZ]) {
        ctx.beginPath();
        ctx.moveTo(cx(baseX), cz(z));
        ctx.lineTo(cx(joinX), cz(z));
        ctx.stroke();
      }
      ctx.beginPath();
      const startA = Math.atan2(cornerZ, joinX - basket);
      const endA = Math.atan2(-cornerZ, joinX - basket);
      ctx.arc(cx(basket), cz(0), m2px(R), startA, endA, side === 1);
      ctx.stroke();

      // Lane blocks / hash marks along the key.
      const blocks = [0.9144, 0.9144 + 0.9144, 0.9144 + 0.9144 + 0.8636];
      for (const d of blocks) {
        for (const sgn of [1, -1] as const) {
          const bx = baseX - side * d;
          ctx.fillRect(
            cx(bx) - lw * 0.5,
            cz(sgn * (COURT.key.width / 2)) - (sgn > 0 ? 0 : m2px(0.2)),
            lw,
            m2px(0.2),
          );
        }
      }
    }

    // ---- Centre logo -----------------------------------------------------
    ctx.save();
    ctx.translate(cx(0), cz(0));
    ctx.globalAlpha = 0.92;
    const R = m2px(2.1);
    const grad = ctx.createRadialGradient(0, 0, 0, 0, 0, R);
    grad.addColorStop(0, 'rgba(226,110,32,0.95)');
    grad.addColorStop(0.72, 'rgba(160,62,18,0.9)');
    grad.addColorStop(1, 'rgba(90,34,10,0.85)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(0, 0, R, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#f7f3ec';
    ctx.font = `900 ${Math.round(R * 0.52)}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('BALLIN', 0, 0);
    ctx.restore();

    // ---- Wear, scuffs and polish streaks ---------------------------------
    ctx.save();
    ctx.globalCompositeOperation = 'multiply';
    for (let i = 0; i < 900; i++) {
      const x = rng() * W;
      const y = rng() * H;
      // Traffic concentrates in the paint and along the arc.
      const mx = (x / W - 0.5) * COURT.length;
      const mz = (y / H - 0.5) * COURT.width;
      const nearKey = Math.min(
        Math.abs(Math.abs(mx) - (COURT.halfLength - COURT.key.length * 0.5)),
        6,
      );
      const w = clamp01(1 - nearKey / 6) * 0.7 + 0.3;
      if (rng() > w) continue;
      const len = 6 + rng() * 34;
      const a = rng() * Math.PI;
      ctx.strokeStyle = `rgba(70,48,26,${0.02 + rng() * 0.05})`;
      ctx.lineWidth = 1 + rng() * 2.4;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + Math.cos(a) * len, y + Math.sin(a) * len * 0.4);
      ctx.stroke();
      void mz;
    }
    ctx.restore();

    const tex = new CanvasTexture(cvs);
    tex.colorSpace = SRGBColorSpace;
    return tex;
  }

  /** Varnish gloss variation — buffed lanes read glossier than the corners. */
  private bakeRoughness(size: number): CanvasTexture {
    const W = size * 2;
    const H = size;
    const cvs = document.createElement('canvas');
    cvs.width = W;
    cvs.height = H;
    const ctx = cvs.getContext('2d')!;
    const img = ctx.createImageData(W, H);
    const px = img.data;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const u = x / W;
        const v = y / H;
        const buff = fbm2(u * 9, v * 5, 4, 2, 0.55, 3);
        const micro = fbm2(u * 180, v * 90, 2, 2, 0.5, 88);
        // 0.16 (mirror-buffed) → 0.42 (dull corner)
        const r = 0.17 + buff * 0.17 + micro * 0.07;
        const o = (y * W + x) * 4;
        const c = Math.round(clamp01(r) * 255);
        px[o] = px[o + 1] = px[o + 2] = c;
        px[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    return new CanvasTexture(cvs);
  }

  /** World-space query used by physics and AI for the floor plane. */
  heightAt(_p: Vector3): number {
    return 0;
  }
}
