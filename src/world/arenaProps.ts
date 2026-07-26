/**
 * Courtside: the band of detail between the sideline and the first riser.
 *
 * This is the part of the non-playing environment that a *low* camera actually
 * scrutinises, and in portrait our camera lives at 1–4 m. So it gets real
 * furniture: the scorer's table with its LED face and monitors, the two team
 * benches with their chairs, coolers, towels and ball rack, the coaching staff
 * on their feet, camera operators on tripods, photographers sitting on the
 * baseline floor, and the padded barrier boards behind them.
 *
 * Everything merges into four meshes by material.
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
  ShaderMaterial,
  PlaneGeometry,
} from 'three';

import {
  APRON_HX,
  APRON_HZ,
  FACE_ALL,
  FACE_SIDES,
  FACE_TOP,
  MeshBuilder,
} from './arenaGeometry';
import { type ArenaPalette, makeLedMaterial, rgb } from './arenaMaterials';
import { makeRng, randRange } from '../core/MathX';
import type { Texture } from 'three';

/** Front face of the scorer's table / benches, just inside the apron edge. */
const TABLE_Z = -(APRON_HZ - 0.42);
const BENCH_Z = -(APRON_HZ - 0.28);

export interface CourtsideResult {
  group: Group;
  ledMaterials: ShaderMaterial[];
  triangles: number;
}

interface Builders {
  matte: MeshBuilder;
  satin: MeshBuilder;
  steel: MeshBuilder;
  emissive: MeshBuilder;
}

// -----------------------------------------------------------------------------
// Figures
// -----------------------------------------------------------------------------

const SKIN = [0x6b4a35, 0x8d6448, 0x4a3225, 0xb08a68, 0x35231a, 0x9c7454];
const HAIR = [0x0d0b0a, 0x160f0c, 0x241a13, 0x0a0a0c, 0x3a3236];

/**
 * A seated person: officials at the table, players on the bench. Deliberately
 * blocky — at the distance courtside sits from the camera, what matters is the
 * shoulder line, the head, the knees and the hands, not the face.
 */
function seatedFigure(
  b: MeshBuilder,
  x: number,
  z: number,
  rotY: number,
  cloth: number,
  rng: () => number,
  scale = 1,
): void {
  const skin = SKIN[Math.floor(rng() * SKIN.length)];
  const hair = HAIR[Math.floor(rng() * HAIR.length)];
  const s = scale;
  const lean = randRange(rng, -0.05, 0.09);

  b.set('color', ...rgb(cloth, 0.5 + rng() * 0.2));
  // thighs
  b.box(x, 0.5 * s, z - 0.1 * s, 0.16 * s, 0.075 * s, 0.2 * s, { rotY, faces: FACE_SIDES | FACE_TOP });
  // shins
  b.box(x, 0.23 * s, z - 0.3 * s, 0.14 * s, 0.23 * s, 0.075 * s, { rotY, faces: FACE_SIDES });
  b.set('color', ...rgb(cloth, 0.85 + rng() * 0.3));
  // torso
  b.box(x, 0.79 * s, z + 0.03 * s, 0.19 * s, 0.26 * s, 0.13 * s, {
    rotY,
    faces: FACE_SIDES | FACE_TOP,
    topScaleX: 1.18,
    bottomScaleX: 0.84,
    shearZ: lean,
  });
  // arms
  for (const sgn of [-1, 1]) {
    b.box(x + sgn * 0.21 * s * Math.cos(rotY), 0.76 * s, z - sgn * 0.21 * s * Math.sin(rotY), 0.05 * s, 0.2 * s, 0.07 * s, {
      rotY,
      faces: FACE_SIDES,
    });
  }
  b.set('color', ...rgb(skin));
  b.octa(x, 1.15 * s, z + 0.01 * s, 0.096 * s, 0.115 * s, 0.1 * s, rotY);
  b.set('color', ...rgb(hair));
  b.pyramid(x, 1.155 * s, z + 0.01 * s, 0.1 * s, 0.125 * s, 0.104 * s, rotY);
  // shoes
  b.set('color', ...rgb(0x101216));
  b.box(x, 0.035 * s, z - 0.36 * s, 0.13 * s, 0.035 * s, 0.11 * s, { rotY, faces: FACE_ALL });
}

/** A standing person: coaches, staff, camera operators, ball boys. */
function standingFigure(
  b: MeshBuilder,
  x: number,
  z: number,
  rotY: number,
  cloth: number,
  rng: () => number,
  opts: { armsCrossed?: boolean; pointing?: boolean; height?: number } = {},
): void {
  const skin = SKIN[Math.floor(rng() * SKIN.length)];
  const hair = HAIR[Math.floor(rng() * HAIR.length)];
  const h = opts.height ?? randRange(rng, 1.72, 1.88);
  const s = h / 1.8;
  const cf = Math.cos(rotY);
  const sf = Math.sin(rotY);

  // legs, staggered — a symmetric stance reads as a mannequin
  b.set('color', ...rgb(0x0d1018, 0.8 + rng() * 0.4));
  for (const sgn of [-1, 1]) {
    const ox = sgn * 0.095 * s;
    const oz = sgn * randRange(rng, -0.05, 0.05) * s;
    b.box(x + ox * cf + oz * sf, 0.45 * s, z - ox * sf + oz * cf, 0.08 * s, 0.45 * s, 0.09 * s, {
      rotY,
      faces: FACE_SIDES,
    });
  }
  b.set('color', ...rgb(0x0a0b0f));
  b.box(x, 0.03 * s, z - 0.02 * s, 0.17 * s, 0.03 * s, 0.13 * s, { rotY, faces: FACE_ALL });

  // torso — jacket
  b.set('color', ...rgb(cloth, 0.8 + rng() * 0.35));
  b.box(x, 1.19 * s, z, 0.2 * s, 0.3 * s, 0.13 * s, {
    rotY,
    faces: FACE_SIDES | FACE_TOP,
    topScaleX: 1.14,
    bottomScaleX: 0.9,
    shearZ: randRange(rng, -0.03, 0.03),
  });

  // arms
  if (opts.armsCrossed) {
    b.box(x, 1.14 * s, z - 0.13 * s, 0.24 * s, 0.055 * s, 0.06 * s, { rotY, faces: FACE_SIDES });
    for (const sgn of [-1, 1]) {
      const ox = sgn * 0.23 * s;
      b.box(x + ox * cf, 1.25 * s, z - ox * sf, 0.05 * s, 0.16 * s, 0.07 * s, { rotY, faces: FACE_SIDES });
    }
  } else if (opts.pointing) {
    b.box(x + 0.25 * s * cf, 1.34 * s, z - 0.25 * s * sf, 0.19 * s, 0.05 * s, 0.06 * s, {
      rotY,
      faces: FACE_SIDES,
      shearZ: -0.5,
    });
    b.box(x - 0.23 * s * cf, 1.13 * s, z + 0.23 * s * sf, 0.05 * s, 0.24 * s, 0.07 * s, { rotY, faces: FACE_SIDES });
  } else {
    for (const sgn of [-1, 1]) {
      const ox = sgn * 0.235 * s;
      b.box(x + ox * cf, 1.14 * s, z - ox * sf, 0.05 * s, 0.25 * s, 0.075 * s, {
        rotY,
        faces: FACE_SIDES,
        shearZ: randRange(rng, -0.08, 0.08),
      });
    }
  }

  b.set('color', ...rgb(skin));
  b.octa(x, 1.62 * s, z, 0.096 * s, 0.12 * s, 0.1 * s, rotY);
  b.set('color', ...rgb(hair));
  b.pyramid(x, 1.625 * s, z, 0.1 * s, 0.13 * s, 0.104 * s, rotY);
}

/** A crouched photographer on the baseline floor, long lens up. */
const VEST = [0x2a3140, 0x1b2433, 0x4a4136, 0x33383f, 0x5a2a22, 0x2c3d33];

function photographer(b: MeshBuilder, x: number, z: number, rotY: number, rng: () => number): void {
  const cf = Math.cos(rotY);
  const sf = Math.sin(rotY);
  // Crouched, not standing: a baseline shooter sits on the floor, and a row of
  // full-height black boxes in front of the boards reads as a fence.
  const vest = VEST[Math.floor(rng() * VEST.length)];
  b.set('color', ...rgb(vest, 0.5 + rng() * 0.3));
  b.box(x, 0.19, z, 0.21, 0.19, 0.19, { rotY, faces: FACE_SIDES | FACE_TOP });
  // A knee up, which is what makes the pose read.
  b.box(x + 0.16 * sf, 0.17, z + 0.16 * cf, 0.1, 0.17, 0.12, { rotY, faces: FACE_SIDES | FACE_TOP });
  b.set('color', ...rgb(vest, 0.9 + rng() * 0.5));
  b.box(x, 0.52, z + 0.02, 0.185, 0.18, 0.13, {
    rotY,
    faces: FACE_SIDES | FACE_TOP,
    topScaleX: 1.12,
    topScaleZ: 0.85,
    shearZ: 0.09,
  });
  b.set('color', ...rgb(SKIN[Math.floor(rng() * SKIN.length)]));
  b.box(x, 0.78, z - 0.02, 0.08, 0.095, 0.085, { rotY, faces: FACE_SIDES | FACE_TOP, topScaleX: 0.9, topScaleZ: 0.9 });
  b.set('color', ...rgb(rng() < 0.45 ? 0x20242c : 0x0d0c0e));
  b.pyramid(x, 0.862, z - 0.02, 0.08, 0.07, 0.085, rotY);
  // Body + long white lens, the one bright thing on a baseline shooter.
  b.set('color', ...rgb(0x101217));
  b.box(x - 0.13 * sf, 0.72, z - 0.13 * cf, 0.075, 0.055, 0.085, { rotY, faces: FACE_ALL });
  b.set('color', ...rgb(rng() < 0.4 ? 0x9aa0a6 : 0x14161b));
  b.box(x - 0.33 * sf, 0.73, z - 0.33 * cf, 0.045, 0.045, 0.17, { rotY, faces: FACE_SIDES });
}

// -----------------------------------------------------------------------------
// Furniture
// -----------------------------------------------------------------------------

function folding(b: MeshBuilder, x: number, z: number, rotY: number, shell: number, rng: () => number): void {
  b.set('color', ...rgb(shell, 0.82 + rng() * 0.36));
  b.box(x, 0.44, z + 0.02, 0.22, 0.025, 0.21, { rotY, faces: FACE_SIDES | FACE_TOP });
  b.box(x, 0.68, z + 0.22, 0.22, 0.23, 0.028, { rotY, faces: FACE_SIDES | FACE_TOP, shearZ: 0.05 });
  b.set('color', ...rgb(0x1e222c));
  for (const sx of [-1, 1]) {
    b.box(x + sx * 0.19 * Math.cos(rotY), 0.21, z - sx * 0.19 * Math.sin(rotY), 0.018, 0.21, 0.018, {
      rotY,
      faces: FACE_SIDES,
    });
  }
}

function cooler(b: MeshBuilder, x: number, z: number, rng: () => number): void {
  b.set('color', ...rgb(0xd7621f, 0.7 + rng() * 0.2));
  b.box(x, 0.24, z, 0.16, 0.24, 0.16, { faces: FACE_SIDES });
  b.set('color', ...rgb(0xe8e9ec, 0.8));
  b.box(x, 0.5, z, 0.17, 0.03, 0.17, { faces: FACE_ALL });
  b.set('color', ...rgb(0x14171d));
  b.box(x, 0.09, z - 0.17, 0.05, 0.03, 0.02, { faces: FACE_ALL });
}

// -----------------------------------------------------------------------------

export function buildCourtside(
  palette: ArenaPalette,
  courtsideTexture: Texture,
  seatColour: number,
): CourtsideResult {
  const group = new Group();
  group.name = 'arena.courtside';
  const rng = makeRng(8801);
  const ledMaterials: ShaderMaterial[] = [];

  const b: Builders = {
    matte: new MeshBuilder().attribute('color', 3, [1, 1, 1]),
    satin: new MeshBuilder().attribute('color', 3, [1, 1, 1]),
    steel: new MeshBuilder().attribute('color', 3, [1, 1, 1]),
    emissive: new MeshBuilder().attribute('color', 3, [1, 1, 1]),
  };

  // ---------------------------------------------------------------------------
  // Scorer's table
  // ---------------------------------------------------------------------------

  const tableHalfW = 4.6;
  const tableH = 0.78;
  const tableD = 0.46;
  const tz = TABLE_Z - tableD;

  b.satin.set('color', ...rgb(0x0d1119));
  b.satin.box(0, tableH * 0.5, tz, tableHalfW, tableH * 0.5, tableD, { faces: FACE_SIDES });
  b.satin.set('color', ...rgb(0x1d222c));
  b.satin.box(0, tableH + 0.02, tz, tableHalfW + 0.05, 0.025, tableD + 0.04, { faces: FACE_ALL });

  // LED face on the court-facing side.
  {
    const g = new PlaneGeometry(tableHalfW * 2 - 0.1, tableH - 0.14);
    g.translate(0, tableH * 0.5 - 0.02, tz + tableD + 0.012);
    const mat = makeLedMaterial({
      map: courtsideTexture,
      pixelsU: 220,
      pixelsV: 14,
      scroll: 0.055,
      gain: 2.5,
    });
    ledMaterials.push(mat);
    const mesh = new Mesh(g, mat);
    mesh.name = 'arena.led.scorer';
    group.add(mesh);
  }

  // Monitors, mics and the officials behind it.
  for (let i = -3; i <= 3; i++) {
    const x = i * 1.2 + randRange(rng, -0.06, 0.06);
    b.steel.set('color', ...rgb(0x1a1e26));
    b.steel.box(x, tableH + 0.19, tz - 0.12, 0.17, 0.13, 0.02, { faces: FACE_SIDES, shearZ: -0.12 });
    b.emissive.set('color', ...rgb(0x63a8ff, 0.55 + rng() * 0.5));
    b.emissive.quad(
      x - 0.15, tableH + 0.07, tz - 0.135,
      x + 0.15, tableH + 0.07, tz - 0.135,
      x + 0.15, tableH + 0.3, tz - 0.16,
      x - 0.15, tableH + 0.3, tz - 0.16,
    );
    if (i !== 0) {
      seatedFigure(b.matte, x, tz - 0.55, 0, i % 2 === 0 ? 0x0e1220 : 0x1a1f2c, rng, 1.0);
      folding(b.matte, x, tz - 0.62, 0, 0x151a24, rng);
    }
  }
  // Shot-clock operator's horn and the possession arrow.
  b.emissive.set('color', ...rgb(0xff9a2e, 1.4));
  b.emissive.box(tableHalfW - 0.5, tableH + 0.14, tz - 0.05, 0.1, 0.05, 0.02, { faces: FACE_ALL });

  // ---------------------------------------------------------------------------
  // Team benches
  // ---------------------------------------------------------------------------

  const benchColours = [0x1c3a78, 0x7a2318];
  for (let side = 0; side < 2; side++) {
    const sx = side === 0 ? -1 : 1;
    const cx = sx * 9.4;
    const teamCol = benchColours[side];

    // Chair row.
    for (let i = 0; i < 12; i++) {
      const x = cx + (i - 5.5) * 0.62;
      folding(b.matte, x, BENCH_Z - 0.42, 0, 0x121722, rng);
      // Roughly two thirds of a bench is occupied at any moment.
      if (rng() < 0.68) {
        seatedFigure(b.matte, x, BENCH_Z - 0.5, randRange(rng, -0.22, 0.22), teamCol, rng, 1.06);
      } else if (rng() < 0.4) {
        // A towel slung over the empty chair back.
        b.matte.set('color', ...rgb(0xd8dce4, 0.55));
        b.matte.box(x, 0.72, BENCH_Z - 0.2, 0.16, 0.13, 0.03, { faces: FACE_SIDES });
      }
    }

    // Coaching staff on their feet, in front of the bench.
    standingFigure(b.matte, cx - 2.6 * sx, BENCH_Z + 0.75, sx > 0 ? -0.4 : 0.4, 0x11151f, rng, {
      pointing: true,
      height: 1.84,
    });
    standingFigure(b.matte, cx - 1.5 * sx, BENCH_Z + 0.45, 0.12 * sx, 0x171c28, rng, {
      armsCrossed: true,
      height: 1.78,
    });
    standingFigure(b.matte, cx + 1.9 * sx, BENCH_Z + 0.4, -0.18 * sx, teamCol, rng, { height: 1.9 });

    // Coolers, cups and a towel pile at the end of the bench.
    cooler(b.matte, cx + 4.1 * sx, BENCH_Z - 0.55, rng);
    cooler(b.matte, cx + 4.45 * sx, BENCH_Z - 0.5, rng);
    b.matte.set('color', ...rgb(0xe6e9ef, 0.5));
    for (let k = 0; k < 4; k++) {
      b.matte.box(
        cx + 3.6 * sx + randRange(rng, -0.12, 0.12),
        0.04 + k * 0.045,
        BENCH_Z - 0.62 + randRange(rng, -0.08, 0.08),
        0.15,
        0.022,
        0.11,
        { rotY: randRange(rng, -0.5, 0.5), faces: FACE_ALL },
      );
    }
    // Ball rack.
    b.steel.set('color', ...rgb(0x2b3140));
    b.steel.box(cx + 5.1 * sx, 0.3, BENCH_Z - 0.5, 0.35, 0.02, 0.28, { faces: FACE_ALL });
    for (const c of [-1, 1]) {
      b.steel.box(cx + 5.1 * sx + c * 0.3, 0.15, BENCH_Z - 0.5, 0.02, 0.15, 0.02, { faces: FACE_SIDES });
    }
    b.matte.set('color', ...rgb(0x8a4a1c));
    for (let k = 0; k < 3; k++) {
      b.matte.octa(cx + 5.1 * sx + (k - 1) * 0.24, 0.44, BENCH_Z - 0.5, 0.12, 0.12, 0.12);
    }
  }

  // ---------------------------------------------------------------------------
  // Camera operators and photographers
  // ---------------------------------------------------------------------------

  const tripod = new CylinderGeometry(0.02, 0.02, 1.3, 5);
  for (const [cx, cz, rot] of [
    [-APRON_HX + 1.4, -APRON_HZ + 0.6, 0.7],
    [APRON_HX - 1.4, -APRON_HZ + 0.6, -0.7],
    [0.0, APRON_HZ - 0.5, Math.PI],
  ] as const) {
    b.steel.set('color', ...rgb(0x22262f));
    for (let k = 0; k < 3; k++) {
      const a = rot + (k / 3) * Math.PI * 2;
      b.steel.merge(
        tripod,
        new Matrix4()
          .makeTranslation(cx + Math.sin(a) * 0.24, 0.65, cz + Math.cos(a) * 0.24)
          .multiply(new Matrix4().makeRotationX(0.18)),
      );
    }
    b.steel.set('color', ...rgb(0x14171d));
    b.steel.box(cx, 1.42, cz, 0.16, 0.12, 0.24, { rotY: rot, faces: FACE_ALL });
    b.steel.box(cx + Math.sin(rot) * -0.3, 1.42, cz + Math.cos(rot) * -0.3, 0.07, 0.07, 0.2, {
      rotY: rot,
      faces: FACE_SIDES,
    });
    b.emissive.set('color', ...rgb(0xff2a2a, 1.6));
    b.emissive.box(cx + 0.1, 1.55, cz, 0.018, 0.018, 0.018, { faces: FACE_ALL });
    standingFigure(b.matte, cx + 0.34, cz - 0.24, rot, 0x0f131b, rng, { height: 1.8 });
  }
  tripod.dispose();

  for (const sx of [-1, 1] as const) {
    for (let i = 0; i < 9; i++) {
      const z = (i - 4) * 1.05 + randRange(rng, -0.15, 0.15);
      if (Math.abs(z) < 1.4) continue; // leave the lane clear
      photographer(b.matte, sx * (APRON_HX - 0.55), z, sx > 0 ? -Math.PI / 2 : Math.PI / 2, rng);
    }
  }

  // ---------------------------------------------------------------------------
  // Baseline barrier boards + the padded end walls
  // ---------------------------------------------------------------------------

  for (const sx of [-1, 1] as const) {
    const x = sx * (APRON_HX + 0.35);
    b.satin.set('color', ...rgb(0x0a0d15));
    b.satin.box(x, 0.52, 0, 0.09, 0.52, 7.6, { faces: FACE_SIDES | FACE_TOP });
    b.satin.set('color', ...rgb(seatColour, 0.4));
    b.satin.box(x - sx * 0.1, 0.75, 0, 0.02, 0.14, 7.6, { faces: FACE_SIDES | FACE_TOP });
  }

  // ---------------------------------------------------------------------------

  const meshes: Mesh[] = [
    new Mesh(b.matte.build('courtside.matte'), palette.matte),
    new Mesh(b.satin.build('courtside.satin'), palette.satin),
    new Mesh(b.steel.build('courtside.steel'), palette.steel),
    new Mesh(
      b.emissive.build('courtside.emissive'),
      new MeshBasicMaterial({ color: 0xffffff, vertexColors: true, toneMapped: true }),
    ),
  ];

  let triangles = 0;
  for (const m of meshes) {
    m.castShadow = false;
    m.receiveShadow = true;
    m.matrixAutoUpdate = false;
    m.updateMatrix();
    triangles += MeshBuilder.tris(m.geometry as BufferGeometry);
    group.add(m);
  }


  return { group, ledMaterials, triangles };
}
