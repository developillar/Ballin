/**
 * Everything above the bowl: the truss ceiling, the catwalks, the light banks
 * (built where `LightingSystem` actually puts its lights, so the reflections in
 * the glass and the hardwood agree with the geometry the camera can see), the
 * hanging speaker arrays, the championship banners and the centre-hung
 * jumbotron with its hoist truss.
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
  ShaderMaterial,
  Vector3,
} from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { ARENA, FACE_ALL, FACE_SIDES, MeshBuilder } from './arenaGeometry';
import { type ArenaPalette, makeLedMaterial, rgb } from './arenaMaterials';
import { makeRng, randRange } from '../core/MathX';
import type { Texture } from 'three';

export interface RaftersResult {
  group: Group;
  ledMaterials: ShaderMaterial[];
  /** World positions of the fixture pods — handy for shafts and reflections. */
  bankCentres: Vector3[];
  triangles: number;
}

/** Where the light banks live. Mirrors the key / fill / practical plot. */
export const LIGHT_BANKS = [
  // Two long sideline banks, over the +Z and −Z catwalks.
  { x: 0, z: 10.8, len: 44, along: 'x' as const, y: ARENA.riggingY, pods: 18, gain: 3.2, tint: 0xf2f6ff },
  { x: 0, z: -10.8, len: 44, along: 'x' as const, y: ARENA.riggingY, pods: 18, gain: 2.6, tint: 0xfff2e2 },
  // Shorter cross banks over each basket, matching the two practical spots.
  { x: 11.6, z: 0, len: 15, along: 'z' as const, y: ARENA.riggingY - 1.1, pods: 6, gain: 2.9, tint: 0xfff6ea },
  { x: -11.6, z: 0, len: 15, along: 'z' as const, y: ARENA.riggingY - 1.1, pods: 6, gain: 2.9, tint: 0xfff6ea },
  // Outer wash over the lower bowl — keeps the crowd off pure black.
  { x: 0, z: 22, len: 40, along: 'x' as const, y: ARENA.riggingY + 1.2, pods: 9, gain: 1.05, tint: 0xd6e2ff },
  { x: 0, z: -22, len: 40, along: 'x' as const, y: ARENA.riggingY + 1.2, pods: 9, gain: 1.05, tint: 0xd6e2ff },
];

export function buildRafters(
  palette: ArenaPalette,
  jumboTexture: Texture,
  ribbonTexture: Texture,
): RaftersResult {
  const group = new Group();
  group.name = 'arena.rafters';
  const ledMaterials: ShaderMaterial[] = [];
  const bankCentres: Vector3[] = [];
  const rng = makeRng(1717);

  const dark = new MeshBuilder().attribute('color', 3, [1, 1, 1]);
  const steel = new MeshBuilder().attribute('color', 3, [1, 1, 1]);
  const emissive = new MeshBuilder().attribute('color', 3, [1, 1, 1]);
  const fabric = new MeshBuilder().attribute('color', 3, [1, 1, 1]);

  const roofY = ARENA.roofY;
  const halfX = 44;
  const halfZ = 36;

  // ---------------------------------------------------------------------------
  // Roof deck + primary truss grid
  // ---------------------------------------------------------------------------

  dark.set('color', ...rgb(0x0e1118));
  dark.box(0, roofY + 0.5, 0, halfX, 0.5, halfZ, { faces: FACE_ALL });

  // Long-span trusses. Top chord, bottom chord and a vertical web every 3.2 m
  // is enough structure that the ceiling reads as engineered rather than as a
  // black lid.
  const trussZ = [-30, -22, -14, -6, 2, 10, 18, 26];
  for (const z of trussZ) {
    const jitter = randRange(rng, -0.9, 0.9);
    steel.set('color', ...rgb(0x2b303c, 0.8 + rng() * 0.5));
    const zz = z + jitter;
    const top = roofY - 0.35;
    const bot = roofY - 2.2;
    steel.box(0, top, zz, halfX * 0.94, 0.16, 0.34, { faces: FACE_ALL });
    steel.box(0, bot, zz, halfX * 0.94, 0.14, 0.28, { faces: FACE_ALL });
    for (let x = -halfX * 0.9; x <= halfX * 0.9; x += 3.2) {
      steel.box(x, (top + bot) * 0.5, zz, 0.075, (top - bot) * 0.5, 0.075, { faces: FACE_SIDES });
      // Diagonal-ish web: a second post leaning the other way reads as a warren
      // truss at this distance for a quarter of the triangles.
      steel.box(x + 1.6, (top + bot) * 0.5, zz, 0.055, (top - bot) * 0.5, 0.055, {
        faces: FACE_SIDES,
        shearX: 0.9,
      });
    }
  }
  // Cross bracing the other way, sparser.
  for (let x = -36; x <= 36; x += 9) {
    steel.set('color', ...rgb(0x252a35, 0.9));
    steel.box(x, roofY - 2.6, 0, 0.13, 0.12, halfZ * 0.92, { faces: FACE_ALL });
  }

  // ---------------------------------------------------------------------------
  // Catwalks + light banks
  // ---------------------------------------------------------------------------

  for (const bank of LIGHT_BANKS) {
    const alongX = bank.along === 'x';
    const half = bank.len * 0.5;

    // Catwalk deck and rails above the bank.
    steel.set('color', ...rgb(0x30363f, 0.9));
    const deckY = bank.y + 1.5;
    if (alongX) {
      steel.box(bank.x, deckY, bank.z, half + 1.2, 0.07, 0.55, { faces: FACE_ALL });
      for (const s of [-1, 1]) {
        steel.box(bank.x, deckY + 0.5, bank.z + s * 0.5, half + 1.2, 0.03, 0.03, { faces: FACE_SIDES });
        steel.box(bank.x, deckY + 0.26, bank.z + s * 0.5, half + 1.2, 0.02, 0.02, { faces: FACE_SIDES });
      }
      for (let x = -half; x <= half; x += 2.6) {
        steel.box(bank.x + x, deckY + 0.28, bank.z + 0.5, 0.025, 0.28, 0.025, { faces: FACE_SIDES });
        steel.box(bank.x + x, deckY + 0.28, bank.z - 0.5, 0.025, 0.28, 0.025, { faces: FACE_SIDES });
      }
    } else {
      steel.box(bank.x, deckY, bank.z, 0.55, 0.07, half + 1.2, { faces: FACE_ALL });
      for (const s of [-1, 1]) {
        steel.box(bank.x + s * 0.5, deckY + 0.5, bank.z, 0.03, 0.03, half + 1.2, { faces: FACE_SIDES });
      }
    }

    // Hangers up to the truss.
    for (let k = -1; k <= 1; k += 1) {
      const ox = alongX ? k * half * 0.8 : 0;
      const oz = alongX ? 0 : k * half * 0.8;
      steel.box(bank.x + ox, (deckY + roofY - 1) * 0.5, bank.z + oz, 0.06, (roofY - 1 - deckY) * 0.5, 0.06, {
        faces: FACE_SIDES,
      });
    }

    // Fixture pods.
    for (let i = 0; i < bank.pods; i++) {
      const t = bank.pods === 1 ? 0 : (i / (bank.pods - 1)) * 2 - 1;
      const px = bank.x + (alongX ? t * half : 0);
      const pz = bank.z + (alongX ? 0 : t * half);
      const py = bank.y;
      bankCentres.push(new Vector3(px, py, pz));

      // Housing.
      dark.set('color', ...rgb(0x15181f, 0.85 + rng() * 0.3));
      dark.box(px, py + 0.26, pz, 0.78, 0.24, 0.58, { faces: FACE_ALL });
      steel.set('color', ...rgb(0x2c313c));
      steel.box(px, py + 0.52, pz, 0.06, 0.1, 0.06, { faces: FACE_SIDES });

      // Emissive lens, facing down. Deliberately above display white so the
      // bloom pass has something legitimate to grab.
      const g = bank.gain * randRange(rng, 0.9, 1.08);
      emissive.set('color', ...rgb(bank.tint, g));
      emissive.quad(
        px - 0.70, py + 0.02, pz - 0.50,
        px + 0.70, py + 0.02, pz - 0.50,
        px + 0.70, py + 0.02, pz + 0.50,
        px - 0.70, py + 0.02, pz + 0.50,
      );
      // Faint side glow on the housing skirt so it is not a floating card.
      emissive.set('color', ...rgb(bank.tint, g * 0.16));
      dark.set('color', ...rgb(0x0d0f14));
      emissive.box(px, py + 0.07, pz, 0.725, 0.06, 0.525, { faces: FACE_SIDES });
    }
  }

  // ---------------------------------------------------------------------------
  // Speaker clusters
  // ---------------------------------------------------------------------------

  for (const [sx, sz] of [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
  ] as const) {
    const cx = sx * 7.6;
    const cz = sz * 7.6;
    const topY = ARENA.riggingY + 1.0;
    dark.set('color', ...rgb(0x0e1016));
    // Hang bar.
    steel.set('color', ...rgb(0x2a2f39));
    steel.box(cx, topY + 0.1, cz, 0.7, 0.05, 0.5, { faces: FACE_ALL });
    for (let k = 0; k < 5; k++) {
      const y = topY - 0.28 - k * 0.46;
      const splay = k * 0.055;
      dark.set('color', ...rgb(0x101319, 0.85 + k * 0.05));
      dark.box(cx + sx * k * 0.06, y, cz + sz * k * 0.06, 0.66, 0.2, 0.44 - k * 0.02, {
        faces: FACE_ALL,
        shearZ: sz * splay,
      });
    }
    // Cable to the truss.
    steel.box(cx, (topY + ARENA.roofY - 1) * 0.5, cz, 0.03, (ARENA.roofY - 1 - topY) * 0.5, 0.03, {
      faces: FACE_SIDES,
    });
  }

  // ---------------------------------------------------------------------------
  // Championship banners
  // ---------------------------------------------------------------------------

  const bannerColours = [0x16305e, 0x7c2a1c, 0x16305e, 0x1d3a24, 0x16305e, 0x7c2a1c];
  for (const side of [-1, 1] as const) {
    for (let i = 0; i < 6; i++) {
      const x = side * 27.5;
      const z = (i - 2.5) * 3.4;
      const top = ARENA.riggingY + 1.3;
      const h = 3.1;
      const w = 0.85;
      const base = bannerColours[(i + (side > 0 ? 0 : 3)) % bannerColours.length];

      // Hang wire.
      steel.set('color', ...rgb(0x3a4150));
      steel.box(x, top + 0.45, z, 0.015, 0.45, 0.015, { faces: FACE_SIDES });

      // Panel — very slightly waved so six of them are not one flat wall.
      const wave = Math.sin(i * 1.7 + side) * 0.06;
      fabric.set('color', ...rgb(base, 0.85 + rng() * 0.3));
      fabric.box(x, top - h * 0.5, z, 0.02, h * 0.5, w, { faces: FACE_SIDES, shearZ: wave });
      // Gold border and a lettering block, so it reads as a banner not a flag.
      fabric.set('color', ...rgb(0xc9a227, 0.9));
      fabric.box(x + side * 0.022, top - 0.14, z, 0.008, 0.04, w * 0.86, { faces: FACE_SIDES });
      fabric.box(x + side * 0.022, top - h + 0.5, z, 0.008, 0.035, w * 0.86, { faces: FACE_SIDES });
      fabric.set('color', ...rgb(0xe7d9a8, 0.8));
      fabric.box(x + side * 0.024, top - 0.95, z, 0.006, 0.16, w * 0.5, { faces: FACE_SIDES });
      fabric.box(x + side * 0.024, top - 1.5, z, 0.006, 0.1, w * 0.34, { faces: FACE_SIDES });
      // Pennant point.
      fabric.set('color', ...rgb(base, 0.7));
      fabric.tri(
        x, top - h, z - w,
        x, top - h, z + w,
        x, top - h - 0.55, z,
      );
    }
  }

  // ---------------------------------------------------------------------------
  // Jumbotron
  // ---------------------------------------------------------------------------

  const jumboBottom = 12.6;
  const jumboTop = 16.4;
  const jumboHalf = 3.05;
  const faceInset = 0.06;

  // Hoist truss + cables.
  steel.set('color', ...rgb(0x2b303c));
  for (const [hx, hz] of [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
  ] as const) {
    steel.box(hx * 2.6, (jumboTop + ARENA.roofY - 1.5) * 0.5 + 0.3, hz * 2.6, 0.06, (ARENA.roofY - 1.5 - jumboTop) * 0.5, 0.06, {
      faces: FACE_SIDES,
    });
  }
  steel.box(0, jumboTop + 0.55, 0, 3.4, 0.14, 3.4, { faces: FACE_ALL });
  for (let k = -1; k <= 1; k++) {
    steel.box(0, jumboTop + 0.25, k * 1.7, 3.3, 0.06, 0.06, { faces: FACE_SIDES });
    steel.box(k * 1.7, jumboTop + 0.25, 0, 0.06, 0.06, 3.3, { faces: FACE_SIDES });
  }

  // Body: the dark carcass the screens are set into.
  dark.set('color', ...rgb(0x0b0d13));
  dark.box(0, (jumboTop + jumboBottom) * 0.5, 0, jumboHalf, (jumboTop - jumboBottom) * 0.5, jumboHalf, {
    faces: FACE_ALL,
  });
  // Crown + halo ring board carcass.
  dark.set('color', ...rgb(0x0e1119));
  dark.box(0, jumboTop + 0.16, 0, jumboHalf * 1.06, 0.16, jumboHalf * 1.06, { faces: FACE_ALL });
  dark.box(0, jumboBottom - 0.34, 0, jumboHalf * 0.92, 0.34, jumboHalf * 0.92, { faces: FACE_ALL });
  // Underside marker light so the belly is not a black slab from the floor.
  emissive.set('color', ...rgb(0x9fb6e8, 0.45));
  emissive.quad(
    -1.1, jumboBottom - 0.69, -1.1,
    -1.1, jumboBottom - 0.69, 1.1,
    1.1, jumboBottom - 0.69, 1.1,
    1.1, jumboBottom - 0.69, -1.1,
  );

  // Four screen faces, UV-mapped, on the LED shader.
  const faceGeos: BufferGeometry[] = [];
  const faceH = jumboTop - jumboBottom - 0.5;
  const faceW = jumboHalf * 2 - 0.35;
  for (let k = 0; k < 4; k++) {
    const a = (k * Math.PI) / 2;
    const g = new PlaneGeometry(faceW, faceH);
    const m = new Matrix4()
      .makeRotationY(a)
      .multiply(new Matrix4().makeTranslation(0, 0, jumboHalf + faceInset));
    g.applyMatrix4(new Matrix4().makeTranslation(0, (jumboTop + jumboBottom) * 0.5, 0).multiply(m));
    faceGeos.push(g);
  }
  const jumboGeo = mergeGeometries(faceGeos, false);
  if (jumboGeo) {
    const jumboMat = makeLedMaterial({
      map: jumboTexture,
      pixelsU: 128,
      pixelsV: 72,
      scroll: 0,
      gain: 1.75,
    });
    ledMaterials.push(jumboMat);
    const jumboMesh = new Mesh(jumboGeo, jumboMat);
    jumboMesh.name = 'arena.jumbotron';
    group.add(jumboMesh);
  }
  for (const g of faceGeos) g.dispose();

  // Halo ribbon under the screens — same scrolling content as the bowl ribbon,
  // which is what ties the two together in a real building.
  const haloGeos: BufferGeometry[] = [];
  for (let k = 0; k < 4; k++) {
    const a = (k * Math.PI) / 2;
    const g = new PlaneGeometry(jumboHalf * 1.84, 0.5);
    const m = new Matrix4()
      .makeRotationY(a)
      .multiply(new Matrix4().makeTranslation(0, 0, jumboHalf * 0.93 + 0.02));
    g.applyMatrix4(new Matrix4().makeTranslation(0, jumboBottom - 0.34, 0).multiply(m));
    haloGeos.push(g);
  }
  const haloGeo = mergeGeometries(haloGeos, false);
  if (haloGeo) {
    const haloMat = makeLedMaterial({
      map: ribbonTexture,
      pixelsU: 96,
      pixelsV: 8,
      scroll: -0.045,
      gain: 2.1,
    });
    ledMaterials.push(haloMat);
    group.add(new Mesh(haloGeo, haloMat));
  }
  for (const g of haloGeos) g.dispose();

  // Shot-clock / centre-hung support cones so the truss meets the body.
  const cone = new CylinderGeometry(0.09, 0.28, 0.7, 6, 1, true);
  for (const [hx, hz] of [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
  ] as const) {
    steel.set('color', ...rgb(0x232833));
    steel.merge(cone, new Matrix4().makeTranslation(hx * 2.6, jumboTop + 0.2, hz * 2.6));
  }
  cone.dispose();

  // ---------------------------------------------------------------------------

  const meshes: Mesh[] = [];
  const darkGeo = dark.build('rafters.dark');
  const steelGeo = steel.build('rafters.steel');
  const fabricGeo = fabric.build('rafters.fabric');
  const emissiveGeo = emissive.build('rafters.emissive');

  meshes.push(new Mesh(darkGeo, palette.matte));
  meshes.push(new Mesh(steelGeo, palette.steel));
  meshes.push(new Mesh(fabricGeo, palette.matte));
  meshes.push(
    new Mesh(
      emissiveGeo,
      new MeshBasicMaterial({ color: 0xffffff, vertexColors: true, toneMapped: true }),
    ),
  );
  let triangles = 0;
  for (const m of meshes) {
    m.castShadow = false;
    m.receiveShadow = false;
    m.matrixAutoUpdate = false;
    m.updateMatrix();
    triangles += MeshBuilder.tris(m.geometry as BufferGeometry);
    group.add(m);
  }

  return { group, ledMaterials, bankCentres, triangles };
}
