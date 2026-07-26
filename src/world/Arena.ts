/**
 * The bowl.
 *
 * A broadcast NBA frame is a bright island of hardwood inside a dark, *built*
 * room. Getting that to read is three separate jobs and this system does all
 * three:
 *
 *  1. **Architecture.** An event-level deck, a padded fascia carrying the
 *     courtside LED boards, fifteen rows of lower bowl with real 420 mm risers
 *     and 880 mm treads, sixteen radial aisles with their own stepped nosings,
 *     eight vomitory tunnels punched through the lower rows, a rail at the lip,
 *     a mid concourse wall with the ribbon boards, a suite band and an upper
 *     deck. All of it is one rounded-rectangle path offset outward once per row,
 *     which is how the rows, the aisles, the rails and the boards line up.
 *
 *  2. **Population.** `Crowd` — instanced pods of seated bodies with per-person
 *     variation resolved in the vertex shader. See that file; it is the thing
 *     that stops the bowl reading as confetti.
 *
 *  3. **Exposure.** The single most important number in the frame is the ratio
 *     between the hardwood and the seats. The crowd shader carries its own
 *     lighting model — not the scene's — precisely so that ratio is a constant
 *     we set, not an accident of how many lights reach the twentieth row.
 *
 * Owned by the arena agent.
 */

import {
  BufferGeometry,
  CylinderGeometry,
  Group,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  PlaneGeometry,
  PointLight,
  ShaderMaterial,
  Texture,
  Vector3,
} from 'three';
import type { Engine, System } from '../core/Engine';
import { basketX } from '../core/Constants';
import { makeRng } from '../core/MathX';
import {
  APRON_HX,
  APRON_HZ,
  ARENA,
  FACE_ALL,
  FACE_TOP,
  MeshBuilder,
  buildRingStrip,
  ringAt,
  ringPerimeter,
  sampleRing,
  type RingSample,
} from './arenaGeometry';
import {
  bakeCourtsideStrip,
  bakeJumbotronFace,
  bakeRibbonStrip,
  makeArenaPalette,
  makeHazeMaterial,
  makeLedMaterial,
  makeShaftMaterial,
  rgb,
  type ArenaPalette,
} from './arenaMaterials';
import { Crowd, type CrowdDetail, type PodPlacement } from './Crowd';
import { buildCourtside } from './arenaProps';
import { LIGHT_BANKS, buildRafters } from './arenaRafters';

/** Seat plastic. Two tones in bands is how real bowls are laid out. */
const SEAT_A = 0x0f1830;
const SEAT_B = 0x131f3c;

/** Half-width of a normal aisle and of a vomitory mouth, in metres. */
const AISLE_HALF = 0.62;
const VOM_HALF = 1.5;

type Builders = {
  matte: MeshBuilder;
  satin: MeshBuilder;
  steel: MeshBuilder;
  emissive: MeshBuilder;
};

interface TierPlan {
  lowerRows: number;
  pitch: number;
  nearDetail: CrowdDetail;
  farDetail: CrowdDetail;
  upperRows: number;
  upperPitch: number;
  nearRows: number;
  occupancy: number;
  phoneRate: number;
}

function planFor(tier: string): TierPlan {
  switch (tier) {
    case 'ultra':
      return { lowerRows: 15, pitch: 0.55, nearDetail: 3, farDetail: 2, upperRows: 7, upperPitch: 0.92, nearRows: 6, occupancy: 0.955, phoneRate: 0.035 };
    case 'high':
      return { lowerRows: 15, pitch: 0.58, nearDetail: 2, farDetail: 1, upperRows: 7, upperPitch: 1.02, nearRows: 6, occupancy: 0.94, phoneRate: 0.03 };
    case 'medium':
      return { lowerRows: 15, pitch: 0.66, nearDetail: 2, farDetail: 1, upperRows: 7, upperPitch: 1.2, nearRows: 4, occupancy: 0.9, phoneRate: 0.025 };
    default:
      return { lowerRows: 15, pitch: 0.84, nearDetail: 1, farDetail: 0, upperRows: 6, upperPitch: 1.5, nearRows: 3, occupancy: 0.86, phoneRate: 0 };
  }
}

const tmp = (): RingSample => ({ x: 0, z: 0, nx: 0, nz: 0, facing: 0 });

export class ArenaSystem implements System {
  readonly name = 'arena';
  readonly order = 14;

  group = new Group();
  crowd = new Crowd();

  /** Reported by the capture harness. */
  triangles = 0;

  private ledMaterials: ShaderMaterial[] = [];
  private animMaterials: ShaderMaterial[] = [];
  private palette!: ArenaPalette;
  private textures: Texture[] = [];
  private epicentre = new Vector3(0, 1.6, 0);
  private unsubscribe: Array<() => void> = [];
  private lowerTopY = 0;
  private lowerTopOffset = 0;

  init(engine: Engine): void {
    this.group.name = 'arena';
    engine.scene.add(this.group);

    const q = engine.quality;
    const plan = planFor(q.tier);
    this.palette = makeArenaPalette();
    this.lowerTopOffset = ARENA.lower.firstOffset + plan.lowerRows * ARENA.lower.run;
    this.lowerTopY = ARENA.fasciaHeight + plan.lowerRows * ARENA.lower.rise;

    const ribbonTex = bakeRibbonStrip(2048, 64);
    const courtsideTex = bakeCourtsideStrip(2048, 128);
    const jumboTex = bakeJumbotronFace(512, 288);
    this.textures.push(ribbonTex, courtsideTex, jumboTex);

    this.buildStructure(plan);
    this.buildRibbons(ribbonTex, courtsideTex);
    this.buildCrowd(plan, q.crowdAnimated, q.crowdCount);

    const courtside = buildCourtside(this.palette, courtsideTex, SEAT_A);
    this.group.add(courtside.group);
    this.ledMaterials.push(...courtside.ledMaterials);
    this.triangles += courtside.triangles;

    const rafters = buildRafters(this.palette, jumboTex, ribbonTex);
    this.group.add(rafters.group);
    this.ledMaterials.push(...rafters.ledMaterials);
    this.triangles += rafters.triangles;

    if (q.volumetricLight) this.buildAtmosphere();
    this.buildPracticals(q.tier);
    this.bindEvents(engine);
  }

  private bindEvents(engine: Engine): void {
    const focus = (): Vector3 | undefined =>
      engine.get<{ focusPoint?: Vector3 }>('ball')?.focusPoint;
    this.unsubscribe.push(
      engine.bus.on('scored', (e) => {
        const p = focus();
        this.epicentre.set(p?.x ?? 0, 1.6, p?.z ?? 0);
        this.crowd.excite(this.epicentre, e.swish ? 0.84 : 0.66, 27, 1);
        this.crowd.setBaseline(0.075);
      }),
      engine.bus.on('dunk', (e) => {
        const p = focus();
        this.epicentre.set(p?.x ?? basketX(1), 1.6, p?.z ?? 0);
        this.crowd.excite(this.epicentre, Math.min(1, 0.8 + e.power * 0.25), 32, 1.6);
      }),
      engine.bus.on('block', () => {
        const p = focus();
        this.epicentre.set(p?.x ?? 0, 1.6, p?.z ?? 0);
        this.crowd.excite(this.epicentre, 0.74, 24, 0.8);
      }),
    );
  }

  // ---------------------------------------------------------------------------
  // Aisle topology
  // ---------------------------------------------------------------------------

  /**
   * 0 = seating, 1 = aisle, 2 = vomitory mouth. Aisles are spokes at even
   * angles about the bowl centre so a given aisle stays on one radial line all
   * the way up the rake, which is what makes the bowl read as laid out rather
   * than as noise.
   */
  private aisleCut(x: number, z: number, row: number): 0 | 1 | 2 {
    const n = ARENA.aisles;
    const k = (Math.atan2(z, x) / (Math.PI * 2)) * n + n * 4 + 0.5;
    const nearest = Math.round(k);
    const idx = ((nearest % n) + n) % n;
    const arcDist = Math.abs(k - nearest) * ((Math.PI * 2) / n) * Math.hypot(x, z);
    if (idx % ARENA.vomEvery === 0 && row < ARENA.vomRows) {
      return arcDist < VOM_HALF ? 2 : 0;
    }
    return arcDist < AISLE_HALF ? 1 : 0;
  }

  // ---------------------------------------------------------------------------
  // Architecture
  // ---------------------------------------------------------------------------

  private buildStructure(plan: TierPlan): void {
    const b: Builders = {
      matte: new MeshBuilder().attribute('color', 3, [1, 1, 1]),
      satin: new MeshBuilder().attribute('color', 3, [1, 1, 1]),
      steel: new MeshBuilder().attribute('color', 3, [1, 1, 1]),
      emissive: new MeshBuilder().attribute('color', 3, [1, 1, 1]),
    };
    const rng = makeRng(31337);

    this.buildDeck(b, rng);
    this.buildFascia(b, rng);
    this.buildRake(b, rng, plan.lowerRows, ARENA.lower.firstOffset, ARENA.lower.run, ARENA.fasciaHeight, ARENA.lower.rise, true);
    this.buildConcourse(b, rng);
    this.buildSuites(b, rng);
    this.buildRake(
      b, rng, plan.upperRows,
      this.lowerTopOffset + ARENA.mid.thickness + ARENA.upper.firstOffset,
      ARENA.upper.run, ARENA.upper.baseY, ARENA.upper.rise, false,
    );
    this.buildBackWall(b, rng, plan);

    const meshes: Mesh[] = [
      new Mesh(b.matte.build('bowl.matte'), this.palette.matte),
      new Mesh(b.satin.build('bowl.satin'), this.palette.satin),
      new Mesh(b.steel.build('bowl.steel'), this.palette.steel),
      new Mesh(
        b.emissive.build('bowl.emissive'),
        new MeshBasicMaterial({ color: 0xffffff, vertexColors: true, toneMapped: true }),
      ),
    ];
    for (const m of meshes) {
      m.castShadow = false;
      m.receiveShadow = false;
      m.matrixAutoUpdate = false;
      m.updateMatrix();
      this.triangles += MeshBuilder.tris(m.geometry as BufferGeometry);
      this.group.add(m);
    }
  }

  /** Event-level floor from under the hardwood out to the fascia. */
  private buildDeck(b: Builders, rng: () => number): void {
    const inner = ringAt(-4.6);
    const outer = ringAt(0);
    const perI = ringPerimeter(inner);
    const perO = ringPerimeter(outer);
    const n = Math.max(64, Math.round(perO / 1.2));
    const a = tmp();
    const c = tmp();
    const d = tmp();
    const e = tmp();
    for (let i = 0; i < n; i++) {
      sampleRing(inner, (i / n) * perI, a);
      sampleRing(inner, ((i + 1) / n) * perI, c);
      sampleRing(outer, ((i + 1) / n) * perO, d);
      sampleRing(outer, (i / n) * perO, e);
      b.satin.set('color', ...rgb(0x0d1017, 0.78 + rng() * 0.5));
      b.satin.quad(a.x, -0.012, a.z, c.x, -0.012, c.z, d.x, -0.012, d.z, e.x, -0.012, e.z);
    }
  }

  /** Padded front wall, walkway cap and the rail at the lip of the lower bowl. */
  private buildFascia(b: Builders, rng: () => number): void {
    const ring = ringAt(0);
    const walk = ringAt(ARENA.lower.firstOffset);
    const per = ringPerimeter(ring);
    const perW = ringPerimeter(walk);
    const h = ARENA.fasciaHeight;
    const n = Math.round(per / 0.95);
    const a = tmp();
    const c = tmp();
    const d = tmp();
    const e = tmp();

    for (let i = 0; i < n; i++) {
      const f0 = i / n;
      const f1 = (i + 1) / n;
      sampleRing(ring, f0 * per, a);
      sampleRing(ring, f1 * per, c);
      sampleRing(walk, f0 * perW, d);
      sampleRing(walk, f1 * perW, e);
      const vom = this.aisleCut(a.x, a.z, 0) === 2;
      b.satin.set('color', ...rgb(vom ? 0x080b12 : 0x121a29, 0.85 + rng() * 0.3));
      b.satin.quad(a.x, 0, a.z, c.x, 0, c.z, c.x, h, c.z, a.x, h, a.z);
      b.matte.set('color', ...rgb(0x20242e, 0.8 + rng() * 0.4));
      b.matte.quad(a.x, h, a.z, c.x, h, c.z, e.x, h, e.z, d.x, h, d.z);
    }

    // Rail: two tubes on stanchions, tight to the lip.
    const rail = new CylinderGeometry(0.023, 0.023, 1, 6);
    const post = new CylinderGeometry(0.016, 0.016, 1.08, 5);
    const steps = Math.round(per / 1.9);
    const p0 = new Vector3();
    const p1 = new Vector3();
    for (let i = 0; i < steps; i++) {
      sampleRing(ring, (i / steps) * per, a);
      sampleRing(ring, ((i + 1) / steps) * per, c);
      p0.set(a.x - a.nx * 0.13, 0, a.z - a.nz * 0.13);
      p1.set(c.x - c.nx * 0.13, 0, c.z - c.nz * 0.13);
      const len = p0.distanceTo(p1);
      const ang = Math.atan2(p1.x - p0.x, p1.z - p0.z);
      b.steel.set('color', ...rgb(0x555d6a, 0.8 + rng() * 0.4));
      for (const ry of [h + 1.0, h + 0.56]) {
        b.steel.merge(
          rail,
          new Matrix4()
            .makeTranslation((p0.x + p1.x) * 0.5, ry, (p0.z + p1.z) * 0.5)
            .multiply(new Matrix4().makeRotationY(ang))
            .multiply(new Matrix4().makeRotationX(Math.PI / 2))
            .multiply(new Matrix4().makeScale(1, len * 1.02, 1)),
        );
      }
      b.steel.merge(post, new Matrix4().makeTranslation(p0.x, h + 0.54, p0.z));
    }
    rail.dispose();
    post.dispose();
  }

  /** Treads, riser faces, aisle steps, nosing lights and vomitory tunnels. */
  private buildRake(
    b: Builders,
    rng: () => number,
    rows: number,
    firstOffset: number,
    run: number,
    baseY: number,
    rise: number,
    lower: boolean,
  ): void {
    const a = tmp();
    const c = tmp();
    const d = tmp();
    const e = tmp();

    for (let r = 0; r < rows; r++) {
      const y = baseY + r * rise;
      const inner = ringAt(firstOffset + r * run);
      const outer = ringAt(firstOffset + (r + 1) * run);
      const perI = ringPerimeter(inner);
      const perO = ringPerimeter(outer);
      const n = Math.round(perI / 1.05);

      for (let i = 0; i < n; i++) {
        const f0 = i / n;
        const f1 = (i + 1) / n;
        sampleRing(inner, f0 * perI, a);
        sampleRing(inner, f1 * perI, c);
        sampleRing(outer, f1 * perO, d);
        sampleRing(outer, f0 * perO, e);

        const cut = this.aisleCut(a.x, a.z, lower ? r : 99);

        if (cut === 2) {
          // The mouth is a hole through the front rows. Draw its floor and its
          // soffit once, from row 0, so the whole tunnel is sealed and reads as
          // a dark rectangle rather than a gap in the model.
          if (r === 0) {
            const back = ringAt(firstOffset + ARENA.vomRows * run);
            const perB = ringPerimeter(back);
            const bd = tmp();
            const be = tmp();
            sampleRing(back, f1 * perB, bd);
            sampleRing(back, f0 * perB, be);
            b.matte.set('color', ...rgb(0x090c12, 0.8 + rng() * 0.3));
            b.matte.quad(a.x, y - 0.06, a.z, c.x, y - 0.06, c.z, bd.x, y - 0.06 + rise * 0.9, bd.z, be.x, y - 0.06 + rise * 0.9, be.z);
            b.matte.set('color', ...rgb(0x0a0d14));
            // Soffit: the underside of the rows that bridge the tunnel.
            b.matte.quad(
              be.x, baseY + ARENA.vomRows * rise, be.z,
              bd.x, baseY + ARENA.vomRows * rise, bd.z,
              c.x, baseY + 2.25, c.z,
              a.x, baseY + 2.25, a.z,
            );
            // Portal frame.
            b.satin.set('color', ...rgb(0x1a1f2b));
            b.satin.box((a.x + c.x) * 0.5, baseY + 2.32, (a.z + c.z) * 0.5, 0.1, 0.1, 0.1, { faces: FACE_ALL });
          }
          continue;
        }

        b.matte.set('color', ...rgb(cut === 1 ? 0x2b313c : 0x1b2029, 0.82 + rng() * 0.36));
        b.matte.quad(a.x, y, a.z, c.x, y, c.z, d.x, y, d.z, e.x, y, e.z);
        b.matte.set('color', ...rgb(cut === 1 ? 0x232830 : 0x12161d, 0.85 + rng() * 0.3));
        b.matte.quad(e.x, y, e.z, d.x, y, d.z, d.x, y + rise, d.z, e.x, y + rise, e.z);

        if (cut === 1) {
          // Two intermediate steps: an aisle is stairs, not a ramp.
          for (let k = 1; k <= 2; k++) {
            const fx = k / 3;
            const ax = a.x + (e.x - a.x) * fx;
            const az = a.z + (e.z - a.z) * fx;
            const bx = c.x + (d.x - c.x) * fx;
            const bz = c.z + (d.z - c.z) * fx;
            const yy = y + (rise * k) / 3;
            b.matte.set('color', ...rgb(0x363d49, 0.9));
            b.matte.quad(ax, yy, az, bx, yy, bz, bx + a.nx * 0.14, yy, bz + a.nz * 0.14, ax + a.nx * 0.14, yy, az + a.nz * 0.14);
          }
          if (r % 2 === 0) {
            // Step nosing catches the house light. This is what keeps a bowl at
            // 25 luminance legible instead of a black gradient.
            b.emissive.set('color', ...rgb(0xc3cee2, 0.055));
            b.emissive.box((a.x + c.x) * 0.5, y + 0.015, (a.z + c.z) * 0.5, 0.15, 0.01, 0.15, { faces: FACE_TOP });
          }
        }
      }
    }
  }

  /** Mid concourse wall, walkway, aisle lights and exit signage. */
  private buildConcourse(b: Builders, rng: () => number): void {
    const ring = ringAt(this.lowerTopOffset);
    const outerRing = ringAt(this.lowerTopOffset + ARENA.mid.thickness);
    const per = ringPerimeter(ring);
    const perO = ringPerimeter(outerRing);
    const n = Math.round(per / 1.1);
    const top = ARENA.mid.top;
    const a = tmp();
    const c = tmp();
    const d = tmp();
    const e = tmp();

    for (let i = 0; i < n; i++) {
      const f0 = i / n;
      const f1 = (i + 1) / n;
      sampleRing(ring, f0 * per, a);
      sampleRing(ring, f1 * per, c);
      sampleRing(outerRing, f1 * perO, d);
      sampleRing(outerRing, f0 * perO, e);
      const cut = this.aisleCut(a.x, a.z, 99);
      b.matte.set('color', ...rgb(cut ? 0x090c13 : 0x171c25, 0.82 + rng() * 0.4));
      b.matte.quad(a.x, this.lowerTopY, a.z, c.x, this.lowerTopY, c.z, c.x, top, c.z, a.x, top, a.z);
      b.matte.set('color', ...rgb(0x0f1218, 0.9));
      b.matte.quad(a.x, top, a.z, c.x, top, c.z, d.x, top, d.z, e.x, top, e.z);
    }

    // Concourse spill through the tunnel mouths.
    const lights = 44;
    for (let i = 0; i < lights; i++) {
      sampleRing(ring, ((i + 0.5) / lights) * per, a);
      b.emissive.set('color', ...rgb(0xffd6a0, 0.18 + rng() * 0.1));
      b.emissive.box(a.x - a.nx * 0.07, top - 0.14, a.z - a.nz * 0.07, 0.1, 0.02, 0.1, { faces: FACE_ALL });
    }

    // Exit signs over the vomitories in the fascia — green, small, and the
    // single cheapest thing that says "this is a real building".
    const fascia = ringAt(0.1);
    for (let k = 0; k < ARENA.aisles; k += ARENA.vomEvery) {
      const ang = ((k - 0.5) / ARENA.aisles) * Math.PI * 2;
      const cs = Math.cos(ang);
      const sn = Math.sin(ang);
      const t = Math.min(fascia.hx / Math.max(Math.abs(cs), 1e-3), fascia.hz / Math.max(Math.abs(sn), 1e-3));
      b.emissive.set('color', ...rgb(0x2fd07a, 0.75));
      b.emissive.box(cs * t * 0.985, ARENA.fasciaHeight + 2.05, sn * t * 0.985, 0.2, 0.06, 0.2, { faces: FACE_ALL });
    }
  }

  /** Suite band: dark glass, mullions and the occasional lit box. */
  private buildSuites(b: Builders, rng: () => number): void {
    const ring = ringAt(this.lowerTopOffset + ARENA.mid.thickness);
    const per = ringPerimeter(ring);
    const n = Math.round(per / 1.7);
    const y0 = ARENA.mid.top;
    const y1 = ARENA.suiteTop;
    const a = tmp();
    const c = tmp();
    for (let i = 0; i < n; i++) {
      sampleRing(ring, (i / n) * per, a);
      sampleRing(ring, ((i + 1) / n) * per, c);
      b.satin.set('color', ...rgb(0x04060c, 0.75 + rng() * 0.5));
      b.satin.quad(a.x, y0, a.z, c.x, y0, c.z, c.x, y1, c.z, a.x, y1, a.z);
      if (rng() < 0.62) {
        // Suites read as *lit rooms seen through glass*: a bright ceiling line
        // and a bright counter, not a uniformly glowing pane. A full-height
        // glow at this scale looks like a lightbox nailed to the wall.
        const ix = a.nx * 0.05;
        const iz = a.nz * 0.05;
        const inset = 0.18;
        const ax = a.x + (c.x - a.x) * inset - ix;
        const az = a.z + (c.z - a.z) * inset - iz;
        const cx = c.x + (a.x - c.x) * inset - ix;
        const cz = c.z + (a.z - c.z) * inset - iz;
        b.emissive.set('color', ...rgb(0xffd9ab, 0.055 + rng() * 0.05));
        b.emissive.quad(ax, y1 - 0.95, az, cx, y1 - 0.95, cz, cx, y1 - 0.78, cz, ax, y1 - 0.78, az);
        b.emissive.set('color', ...rgb(0xf0d0a8, 0.022 + rng() * 0.02));
        b.emissive.quad(ax, y0 + 0.55, az, cx, y0 + 0.55, cz, cx, y0 + 1.15, cz, ax, y0 + 1.15, az);
        // A couple of heads in the window, so the suites are occupied.
        if (rng() < 0.5) {
          b.matte.set('color', ...rgb(0x1c1a1c));
          const hx2 = a.x + (c.x - a.x) * (0.3 + rng() * 0.4) - ix;
          const hz2 = a.z + (c.z - a.z) * (0.3 + rng() * 0.4) - iz;
          b.matte.box(hx2, y0 + 1.0, hz2, 0.11, 0.32, 0.11, { faces: FACE_ALL });
        }
      }
      b.steel.set('color', ...rgb(0x272c37, 0.9));
      b.steel.box(a.x, (y0 + y1) * 0.5, a.z, 0.05, (y1 - y0) * 0.5, 0.05, { faces: FACE_ALL });
      // Soffit up to the underside of the upper deck.
      b.matte.set('color', ...rgb(0x0e1219, 0.8 + rng() * 0.4));
      b.matte.quad(a.x, y1, a.z, c.x, y1, c.z, c.x, ARENA.upper.baseY, c.z, a.x, ARENA.upper.baseY, a.z);
    }
  }

  private buildBackWall(b: Builders, rng: () => number, plan: TierPlan): void {
    const topOffset =
      this.lowerTopOffset + ARENA.mid.thickness + ARENA.upper.firstOffset + plan.upperRows * ARENA.upper.run;
    const ring = ringAt(topOffset);
    const per = ringPerimeter(ring);
    const n = Math.round(per / 2.6);
    const y0 = ARENA.upper.baseY + plan.upperRows * ARENA.upper.rise;
    const a = tmp();
    const c = tmp();
    for (let i = 0; i < n; i++) {
      sampleRing(ring, (i / n) * per, a);
      sampleRing(ring, ((i + 1) / n) * per, c);
      // Two bands: a concourse storey behind the top row, then the shell above.
      b.matte.set('color', ...rgb(0x11151d, 0.7 + rng() * 0.6));
      b.matte.quad(a.x, y0, a.z, c.x, y0, c.z, c.x, y0 + 3.4, c.z, a.x, y0 + 3.4, a.z);
      b.matte.set('color', ...rgb(0x0c1017, 0.7 + rng() * 0.6));
      b.matte.quad(a.x, y0 + 3.4, a.z, c.x, y0 + 3.4, c.z, c.x, ARENA.roofY + 1.4, c.z, a.x, ARENA.roofY + 1.4, a.z);
      // A lit portal every few bays — the upper concourse behind the top deck.
      if (i % 3 === 1) {
        const ix = a.nx * 0.05;
        const iz = a.nz * 0.05;
        b.emissive.set('color', ...rgb(0xffd2a4, 0.035 + rng() * 0.035));
        const mx = (a.x + c.x) * 0.5;
        const mz = (a.z + c.z) * 0.5;
        b.emissive.quad(
          a.x * 0.35 + mx * 0.65 - ix, y0 + 0.35, a.z * 0.35 + mz * 0.65 - iz,
          c.x * 0.35 + mx * 0.65 - ix, y0 + 0.35, c.z * 0.35 + mz * 0.65 - iz,
          c.x * 0.35 + mx * 0.65 - ix, y0 + 1.85, c.z * 0.35 + mz * 0.65 - iz,
          a.x * 0.35 + mx * 0.65 - ix, y0 + 1.85, a.z * 0.35 + mz * 0.65 - iz,
        );
      }
      if (i % 7 === 3) {
        b.emissive.set('color', ...rgb(0x2fd07a, 0.7));
        b.emissive.box(a.x - a.nx * 0.09, y0 + 2.85, a.z - a.nz * 0.09, 0.26, 0.08, 0.26, { faces: FACE_ALL });
      }
      // Fascia lip catching the rigging light.
      b.steel.set('color', ...rgb(0x2a3040, 0.85 + rng() * 0.3));
      b.steel.box((a.x + c.x) * 0.5 - a.nx * 0.1, y0 + 3.5, (a.z + c.z) * 0.5 - a.nz * 0.1, 1.35, 0.11, 1.35, {
        faces: FACE_ALL,
        rotY: Math.atan2(a.nx, a.nz),
      });
    }
  }

  // ---------------------------------------------------------------------------
  // LED
  // ---------------------------------------------------------------------------

  private buildRibbons(ribbonTex: Texture, courtsideTex: Texture): void {
    // Courtside boards: the bright band at floor level behind the front rows.
    // In a low portrait camera this is the arena's signature light source.
    {
      const geo = buildRingStrip(ringAt(0.02), 0.2, 1.12, 1 / 5.4, 0.6);
      const mat = makeLedMaterial({ map: courtsideTex, pixelsU: 940, pixelsV: 26, scroll: 0.03, gain: 2.15 });
      this.ledMaterials.push(mat);
      const mesh = new Mesh(geo, mat);
      mesh.name = 'arena.led.courtside';
      mesh.matrixAutoUpdate = false;
      this.group.add(mesh);
      this.triangles += MeshBuilder.tris(geo);
    }

    // Baseline boards right behind each basket — much closer to camera.
    for (const sx of [1, -1] as const) {
      const g = new PlaneGeometry(8.4, 0.84, 1, 1);
      g.rotateY(sx > 0 ? -Math.PI / 2 : Math.PI / 2);
      g.translate(sx * (APRON_HX + 0.28), 0.58, 0);
      const mat = makeLedMaterial({ map: courtsideTex, pixelsU: 210, pixelsV: 22, scroll: 0.045, gain: 2.25 });
      this.ledMaterials.push(mat);
      const mesh = new Mesh(g, mat);
      mesh.name = 'arena.led.baseline';
      this.group.add(mesh);
      this.triangles += MeshBuilder.tris(g);
    }

    // Suite-level ribbon: the band that stops the upper facade reading as a void.
    {
      const geo = buildRingStrip(
        ringAt(this.lowerTopOffset + ARENA.mid.thickness - 0.05),
        ARENA.suiteTop - 0.62, ARENA.suiteTop - 0.08, 1 / 4.2, 1.0,
      );
      const mat = makeLedMaterial({ map: ribbonTex, pixelsU: 1700, pixelsV: 14, scroll: 0.038, gain: 1.6 });
      this.ledMaterials.push(mat);
      const mesh = new Mesh(geo, mat);
      mesh.name = 'arena.led.suite';
      mesh.matrixAutoUpdate = false;
      this.group.add(mesh);
      this.triangles += MeshBuilder.tris(geo);
    }

    // Upper ribbon on the mid concourse fascia.
    {
      const top = ARENA.mid.top;
      const geo = buildRingStrip(ringAt(this.lowerTopOffset - 0.03), top - 1.22, top - 0.55, 1 / 3.4, 0.85);
      const mat = makeLedMaterial({ map: ribbonTex, pixelsU: 1600, pixelsV: 16, scroll: -0.05, gain: 1.85 });
      this.ledMaterials.push(mat);
      const mesh = new Mesh(geo, mat);
      mesh.name = 'arena.led.ribbon';
      mesh.matrixAutoUpdate = false;
      this.group.add(mesh);
      this.triangles += MeshBuilder.tris(geo);
    }
  }

  // ---------------------------------------------------------------------------
  // Crowd
  // ---------------------------------------------------------------------------

  private buildCrowd(plan: TierPlan, animated: boolean, budget: number): void {
    const nearRows = Math.min(plan.nearRows, plan.lowerRows);
    const near = this.bowlPlacements(0, nearRows, ARENA.lower.firstOffset, ARENA.lower.run, ARENA.fasciaHeight, ARENA.lower.rise, plan.pitch, ARENA.vomRows);
    const far = this.bowlPlacements(nearRows, plan.lowerRows, ARENA.lower.firstOffset, ARENA.lower.run, ARENA.fasciaHeight, ARENA.lower.rise, plan.pitch, ARENA.vomRows);
    const upper = this.bowlPlacements(
      0, plan.upperRows,
      this.lowerTopOffset + ARENA.mid.thickness + ARENA.upper.firstOffset,
      ARENA.upper.run, ARENA.upper.baseY, ARENA.upper.rise, plan.upperPitch, 0,
    );
    const front = this.courtsidePlacements(plan.pitch * 1.1);

    const pods = (seats: number, share: number): number =>
      Math.max(2, Math.min(16, Math.ceil(seats / Math.max(8, Math.floor(budget * share)))));

    this.addCrowdBlock('nearBowl', near, plan.pitch, pods(near.length, 0.34), plan.nearDetail, plan.occupancy, plan.phoneRate, animated, SEAT_A);
    this.addCrowdBlock('farBowl', far, plan.pitch, pods(far.length, 0.42), plan.farDetail, plan.occupancy, plan.phoneRate, animated, SEAT_B);
    if (upper.length) {
      this.addCrowdBlock('upperBowl', upper, plan.upperPitch, pods(upper.length, 0.16), 0, plan.occupancy * 0.94, 0, animated, SEAT_B);
    }
    if (front.length) {
      this.addCrowdBlock(
        'courtside', front, plan.pitch * 1.1, pods(front.length, 0.08),
        plan.nearDetail >= 2 ? 3 : plan.nearDetail, 0.97, Math.max(plan.phoneRate, 0.05), animated, 0x0e1522,
      );
    }
  }

  private addCrowdBlock(
    name: string,
    seats: PodPlacement[],
    pitch: number,
    podSize: number,
    detail: CrowdDetail,
    occupancy: number,
    phoneRate: number,
    animated: boolean,
    seatColour: number,
  ): void {
    // Group consecutive seats into pods. A `tone < 0` entry is a run break, put
    // there by the placement walk wherever an aisle interrupts the row.
    const placements: PodPlacement[] = [];
    let run: PodPlacement[] = [];
    const flush = (): void => {
      const usable = Math.floor(run.length / podSize) * podSize;
      const start = Math.floor((run.length - usable) / 2);
      for (let i = 0; i < usable; i += podSize) {
        const p = run[start + i];
        const q = run[start + i + podSize - 1];
        placements.push({
          x: (p.x + q.x) * 0.5,
          y: p.y,
          z: (p.z + q.z) * 0.5,
          rotY: Math.atan2(Math.sin(p.rotY) + Math.sin(q.rotY), Math.cos(p.rotY) + Math.cos(q.rotY)),
          tone: p.tone,
        });
      }
      run = [];
    };
    for (const p of seats) {
      if (p.tone < 0) flush();
      else run.push(p);
    }
    flush();
    if (!placements.length) return;

    const mesh = this.crowd.add(
      { name, detail, podSize, pitch, occupancy, phoneRate, seatColor: seatColour, placements },
      animated,
    );
    this.group.add(mesh);
    this.triangles += MeshBuilder.tris(mesh.geometry as BufferGeometry) * mesh.count;
  }

  /**
   * Walk each row placing a seat every `pitch` metres, emitting a break marker
   * wherever an aisle or a vomitory interrupts the run.
   */
  private bowlPlacements(
    rowFrom: number, rowTo: number, firstOffset: number, run: number,
    baseY: number, rise: number, pitch: number, vomRowLimit: number,
  ): PodPlacement[] {
    const out: PodPlacement[] = [];
    const a = tmp();
    for (let r = rowFrom; r < rowTo; r++) {
      const ring = ringAt(firstOffset + r * run + 0.32);
      const per = ringPerimeter(ring);
      const count = Math.floor(per / pitch);
      const step = per / count;
      const y = baseY + r * rise;
      // Floor and board spill dies off fast as the rake climbs.
      const tone = 1 + Math.max(0, 0.34 - r * 0.07);
      let broke = true;
      for (let i = 0; i < count; i++) {
        sampleRing(ring, (i + 0.5) * step, a);
        if (this.aisleCut(a.x, a.z, r < vomRowLimit ? r : 99) !== 0) {
          if (!broke) out.push({ x: 0, y: 0, z: 0, rotY: 0, tone: -1 });
          broke = true;
          continue;
        }
        broke = false;
        out.push({ x: a.x, y, z: a.z, rotY: Math.atan2(a.nx, a.nz), tone });
      }
      out.push({ x: 0, y: 0, z: 0, rotY: 0, tone: -1 });
    }
    return out;
  }

  /** Four rows of courtside seats on the deck, clear of the bench and table. */
  private courtsidePlacements(pitch: number): PodPlacement[] {
    const out: PodPlacement[] = [];
    const a = tmp();
    for (let r = 0; r < 3; r++) {
      const ring = ringAt(-3.3 + r * 0.95);
      const per = ringPerimeter(ring);
      const count = Math.floor(per / pitch);
      const step = per / count;
      let broke = true;
      for (let i = 0; i < count; i++) {
        sampleRing(ring, (i + 0.5) * step, a);
        const blocked =
          (a.z < -APRON_HZ + 2.6 && r < 2 && Math.abs(a.x) < 15.4) ||
          Math.abs(a.z) > ARENA.deck.hz - 0.7 ||
          Math.abs(a.x) > ARENA.deck.hx - 0.7;
        if (blocked) {
          if (!broke) out.push({ x: 0, y: 0, z: 0, rotY: 0, tone: -1 });
          broke = true;
          continue;
        }
        broke = false;
        out.push({ x: a.x, y: 0, z: a.z, rotY: Math.atan2(a.nx, a.nz), tone: 1.6 - r * 0.1 });
      }
      out.push({ x: 0, y: 0, z: 0, rotY: 0, tone: -1 });
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // Atmosphere and practicals
  // ---------------------------------------------------------------------------

  private buildAtmosphere(): void {
    const ring = ringAt(this.lowerTopOffset + 4);
    const bottom = this.lowerTopY - 1.5;
    const geo = new CylinderGeometry(1, 1, 1, 28, 1, true);
    geo.scale(ring.hx * 0.95, ARENA.roofY - bottom + 2, ring.hz * 0.95);
    geo.translate(0, bottom + (ARENA.roofY - bottom + 2) * 0.5, 0);
    const mat = makeHazeMaterial();
    mat.uniforms.uBottom.value = bottom;
    mat.uniforms.uTop.value = ARENA.roofY;
    this.animMaterials.push(mat);
    const mesh = new Mesh(geo, mat);
    mesh.name = 'arena.haze';
    mesh.renderOrder = 3;
    mesh.frustumCulled = false;
    this.group.add(mesh);

    // Shafts under the four main banks. They die well above the hardwood —
    // atmosphere in the rigging, not god rays over the court.
    const shaftMat = makeShaftMaterial();
    this.animMaterials.push(shaftMat);
    for (const bank of LIGHT_BANKS.slice(0, 4)) {
      const g = new PlaneGeometry(bank.len * 0.9, 9.0, 1, 1);
      g.translate(0, -4.5, 0);
      if (bank.along === 'z') g.rotateY(Math.PI / 2);
      const m = new Mesh(g, shaftMat);
      m.position.set(bank.x, bank.y, bank.z);
      m.renderOrder = 4;
      m.name = 'arena.shaft';
      this.group.add(m);
    }
  }

  /**
   * The boards have to *light* something or they read as stickers. Three cheap
   * point lights — one per baseline board, one under the jumbotron — put a
   * team-coloured kick on the apron and the stanchion padding.
   */
  private buildPracticals(tier: string): void {
    for (const sx of [1, -1] as const) {
      const l = new PointLight(sx > 0 ? 0x4f7dff : 0xff7d4f, 5.0, 13, 2.2);
      l.position.set(sx * (APRON_HX - 0.35), 0.85, 0);
      l.castShadow = false;
      this.group.add(l);
    }
    if (tier === 'high' || tier === 'ultra') {
      const jl = new PointLight(0xc2d4ff, 8, 22, 2);
      jl.position.set(0, 10.6, 0);
      jl.castShadow = false;
      this.group.add(jl);
    }
  }

  // ---------------------------------------------------------------------------

  update(dt: number, _alpha: number, engine: Engine): void {
    const t = engine.elapsed;
    for (const m of this.ledMaterials) m.uniforms.uTime.value = t;
    for (const m of this.animMaterials) m.uniforms.uTime.value = t;
    this.crowd.update(dt, t);
  }

  /** Focal point for depth-of-field / camera framing helpers. */
  centre(): Vector3 {
    return new Vector3(0, 2, 0);
  }

  dispose(): void {
    for (const off of this.unsubscribe) off();
    this.unsubscribe.length = 0;
    this.crowd.dispose();
    for (const m of this.ledMaterials) m.dispose();
    for (const m of this.animMaterials) m.dispose();
    for (const t of this.textures) t.dispose();
    this.group.traverse((o) => {
      const mesh = o as Mesh;
      if (mesh.isMesh) mesh.geometry?.dispose();
    });
    this.group.removeFromParent();
  }
}
