/**
 * Backboard, ring, net and stanchion.
 *
 * Three things here are load-bearing for the look of the whole game:
 *
 *  1. The **net** is real tubular geometry — twelve braided cords running in two
 *     opposing helices, so the mesh is a proper diamond lattice rather than a
 *     wireframe cone. One BufferGeometry, allocated once, with positions and
 *     normals rewritten in place from a verlet solve. Cords catch light, cast
 *     shadow, and glow slightly at grazing angles the way thin nylon does.
 *  2. The **glass** is a coverage-mapped pane — the painted markings are opaque,
 *     the rest is 12–30% so the bowl reads through it darkened and greened —
 *     plus a separate emissive band for the 38 mm of glass thickness you can
 *     see into, and an additive, camera-parallaxed reflection of the ceiling
 *     banks carrying both specular populations.
 *  3. The **ring breaks away**. Rim contact drives a damped hinge spring at the
 *     mount plate; the net is pinned to the ring, so it swings from the impulse.
 *
 * Owned by the hoop agent.
 */

import {
  AdditiveBlending,
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  CylinderGeometry,
  DoubleSide,
  DynamicDrawUsage,
  Group,
  InstancedMesh,
  FrontSide,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshPhysicalMaterial,
  MeshStandardMaterial,
  Object3D,
  PlaneGeometry,
  Quaternion,
  SphereGeometry,
  Texture,
  TorusGeometry,
  Vector3,
  type IUniform,
} from 'three';
import type { Engine, System } from '../core/Engine';
import { BALL, COURT, HOOP } from '../core/Constants';
import { clamp, clamp01, smoothstep } from '../core/MathX';
import {
  bakeBackboardMaps,
  bakeGlassReflection,
  bakeNetCord,
  bakeRimMaps,
  bakeVinylPad,
} from '../textures/hoopTextures';

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

/**
 * Radius of a single net cord — 3.6 mm of 120-count braided nylon. At the RIM
 * framing (~530 px/m) that lands at ~1.9 px, inside the 1.6–2.4 px window; any
 * thicker and the net starts reading as macramé.
 */
const CORD_RADIUS = 0.0018;
/**
 * Sides on the extruded cord, per tier. Six is plenty for a 2 px strand on a
 * phone; four still reads as round at that size and saves a third of the net.
 */
const CORD_SIDES: Record<string, number> = { low: 4, medium: 5, high: 6, ultra: 6 };
/** Twists per metre of cord — sets how fast the braid spirals. */
const CORD_TWISTS_PER_M = 190;

/** Hourglass profile: where the waist sits, and how tight it pulls. */
const NET_WAIST_T = 0.62;
const NET_WAIST_SCALE = 0.565;

/** Speed ceiling on any single cord knot, m/s. Purely a stability guard. */
const MAX_CORD_SPEED = 14;

/** Breakaway hinge spring. */
const RIM_OMEGA = 28;
const RIM_ZETA = 0.28;
/** Backboard shake spring — stiffer and shorter than the rim. */
const BOARD_OMEGA = 62;
const BOARD_ZETA = 0.24;

// ---------------------------------------------------------------------------
// Small geometry toolkit — everything merges down so a basket stays cheap.
// ---------------------------------------------------------------------------

const _a = new Vector3();
const _b = new Vector3();
const _c = new Vector3();
const _m4 = new Matrix4();
const _rot = new Matrix4();
const _quat = new Quaternion();
const _obj = new Object3D();
const UP = new Vector3(0, 1, 0);
const FALLBACK = new Vector3(0, 0, 1);

/** Concatenates position/normal/uv geometries into one draw call. */
function mergeGeos(geos: BufferGeometry[]): BufferGeometry {
  const flat = geos.map((g) => (g.index ? g.toNonIndexed() : g));
  let n = 0;
  for (const g of flat) n += g.getAttribute('position').count;
  const pos = new Float32Array(n * 3);
  const nor = new Float32Array(n * 3);
  const uv = new Float32Array(n * 2);
  let po = 0;
  let uo = 0;
  for (const g of flat) {
    const p = g.getAttribute('position');
    pos.set(p.array as Float32Array, po);
    nor.set(g.getAttribute('normal').array as Float32Array, po);
    uv.set(g.getAttribute('uv').array as Float32Array, uo);
    po += p.count * 3;
    uo += p.count * 2;
  }
  for (const g of geos) g.dispose();
  const out = new BufferGeometry();
  out.setAttribute('position', new BufferAttribute(pos, 3));
  out.setAttribute('normal', new BufferAttribute(nor, 3));
  out.setAttribute('uv', new BufferAttribute(uv, 2));
  out.computeBoundingSphere();
  return out;
}

function scaleUV(g: BufferGeometry, su: number, sv: number): BufferGeometry {
  const uv = g.getAttribute('uv');
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * su, uv.getY(i) * sv);
  return g;
}

function at(g: BufferGeometry, x: number, y: number, z: number): BufferGeometry {
  g.translate(x, y, z);
  return g;
}

/**
 * Orients a geometry whose local `axis` runs along the segment a → b. The basis
 * is built explicitly right-handed; a mirrored one would flip every normal.
 */
function span(g: BufferGeometry, a: Vector3, b: Vector3, axis: 'x' | 'y'): BufferGeometry {
  _a.copy(b).sub(a);
  const len = _a.length() || 1e-5;
  _a.divideScalar(len);
  const ref = Math.abs(_a.y) > 0.96 ? FALLBACK : UP;
  if (axis === 'x') {
    _c.crossVectors(_a, ref).normalize(); // Z
    _b.crossVectors(_c, _a).normalize(); // Y = Z × X
    _m4.makeBasis(_a, _b, _c);
  } else {
    _c.crossVectors(ref, _a).normalize(); // X
    _b.crossVectors(_c, _a).normalize(); // Z = X × Y
    _m4.makeBasis(_c, _a, _b);
  }
  _m4.setPosition((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2);
  g.applyMatrix4(_m4);
  return g;
}

function beamBetween(a: Vector3, b: Vector3, w: number, h: number): BufferGeometry {
  return span(new BoxGeometry(a.distanceTo(b), h, w), a, b, 'x');
}

function rodBetween(a: Vector3, b: Vector3, r0: number, r1 = r0, seg = 10): BufferGeometry {
  return span(new CylinderGeometry(r0, r1, a.distanceTo(b), seg, 1), a, b, 'y');
}

// ---------------------------------------------------------------------------
// Net data
// ---------------------------------------------------------------------------

interface NetNode {
  pos: Vector3;
  prev: Vector3;
  pinned: boolean;
  /** Rest radius from the net axis, used by the soft shape-memory pass. */
  restR: number;
  ring: number;
}

interface NetLink {
  a: number;
  b: number;
  rest: number;
  stiffness: number;
  /** Structural cord (true) versus soft knot-friction hoop (false). */
  cord: boolean;
}

interface BallProbe {
  position: Vector3;
  radius: number;
  velocity?: Vector3;
}

interface NetImpulse {
  delay: number;
  /** Azimuth the impulse is centred on, radians. */
  az: number;
  /** How tightly it focuses on that azimuth: 0 = uniform. */
  focus: number;
  /** Ring range the impulse acts on. */
  r0: number;
  r1: number;
  vx: number;
  vy: number;
  vz: number;
  /** Radial (outward) component, m/s. */
  vr: number;
}

/** A single basket: board, ring, net, and the stanchion behind it. */
export class Basket {
  readonly group = new Group();
  readonly rimCentre = new Vector3();
  readonly boardCentre = new Vector3();
  /** Outward normal of the backboard face (points onto the court). */
  readonly boardNormal = new Vector3();

  nodes: NetNode[] = [];
  links: NetLink[] = [];

  /** Live breakaway deflection, radians. Positive tips the front of the ring down. */
  rimFlex = 0;
  private rimFlexVel = 0;
  private rimTwist = 0;
  private rimTwistVel = 0;
  private boardShake = 0;
  private boardShakeVel = 0;

  private boardGroup = new Group();
  private rimPivot = new Group();
  private pivotWorld = new Vector3();

  private netMesh!: Mesh;
  private netPos!: BufferAttribute;
  private netNrm!: BufferAttribute;
  /** Node indices per continuous cord path; two opposing helical families. */
  private paths: Int32Array[] = [];
  private anchors: Vector3[] = [];
  private impulses: NetImpulse[] = [];
  private restRingR: number[] = [];

  private reflMap: Texture | null = null;
  private netDirty = true;
  private restCounter = 0;
  private asleep = false;
  private ballLast = new Vector3(0, -50, 0);

  private cosT: number[] = [];
  private sinT: number[] = [];

  private readonly rings: number;
  private readonly strands: number;
  private readonly sides: number;

  constructor(
    readonly side: 1 | -1,
    engine: Engine,
  ) {
    this.strands = HOOP.net.strands;
    this.rings = HOOP.net.segments;
    this.sides = CORD_SIDES[engine.quality.tier] ?? 6;

    const baseX = side * COURT.halfLength;
    this.boardNormal.set(-side, 0, 0);
    const boardX = baseX - side * (COURT.basketFromBaseline - HOOP.rimOffsetFromBoard - HOOP.rimRadius);
    this.boardCentre.set(boardX, HOOP.board.bottomHeight + HOOP.board.height / 2, 0);
    this.rimCentre.set(boardX - side * (HOOP.rimOffsetFromBoard + HOOP.rimRadius), HOOP.rimHeight, 0);

    // The breakaway hinge lives at the board face, on the rim's centreline.
    this.pivotWorld.set(boardX - side * (HOOP.board.thickness / 2), HOOP.rimHeight, 0);
    this.rimPivot.position.copy(this.pivotWorld);

    for (let j = 0; j < this.sides; j++) {
      const th = (j / this.sides) * Math.PI * 2;
      this.cosT.push(Math.cos(th));
      this.sinT.push(Math.sin(th));
    }

    this.group.add(this.boardGroup);
    this.boardGroup.add(this.rimPivot);

    this.buildBoard(engine);
    this.buildRim(engine);
    this.buildStanchion(baseX, engine);
    this.buildNet(engine);
  }

  // -------------------------------------------------------------------------
  // Backboard
  // -------------------------------------------------------------------------

  private buildBoard(engine: Engine): void {
    const B = HOOP.board;
    const face = this.boardCentre.x;
    const size = Math.min(1024, engine.quality.textureSize);
    const maps = bakeBackboardMaps(
      B.width,
      B.height,
      B.innerSquare,
      HOOP.rimHeight - B.bottomHeight,
      size,
    );

    // --- the glass ---------------------------------------------------------
    // Two co-planar layers, because a single alpha-blended plane cannot do both
    // jobs: anything you can see through also fades out its own reflections.
    //
    //   base  — 12–30% coverage, so the crowd reads through it but darkened and
    //           greened; the painted markings sit at full coverage in the same
    //           map, which keeps them crisp and correctly lit.
    //   spec  — additive, black diffuse, so all it ever contributes is the
    //           Fresnel reflection of the ceiling banks. This is the bright
    //           streak that makes a board read as glass instead of perspex.
    const paneGeo = new PlaneGeometry(B.width, B.height, 1, 1);
    const yaw = this.side > 0 ? -Math.PI / 2 : Math.PI / 2;

    const glass = new MeshPhysicalMaterial({
      map: maps.paint,
      alphaMap: maps.alpha,
      roughnessMap: maps.rough,
      transparent: true,
      depthWrite: false,
      roughness: 1,
      metalness: 0,
      reflectivity: 0.5,
      envMapIntensity: 0.5,
      side: DoubleSide,
    });
    const pane = new Mesh(paneGeo, glass);
    pane.rotation.y = yaw;
    pane.position.copy(this.boardCentre);
    pane.renderOrder = 2;
    this.boardGroup.add(pane);

    this.reflMap = bakeGlassReflection(B.width, B.height, size >> 1);
    const specMat = new MeshBasicMaterial({
      map: this.reflMap,
      transparent: true,
      depthWrite: false,
      blending: AdditiveBlending,
      side: FrontSide,
    });
    const spec = new Mesh(paneGeo, specMat);
    spec.rotation.y = yaw;
    spec.position.copy(this.boardCentre);
    spec.renderOrder = 4;
    this.boardGroup.add(spec);

    // --- the 38 mm edge ----------------------------------------------------
    // A separate band so we can crank the green way past what the face uses.
    const edgeMat = new MeshPhysicalMaterial({
      color: 0x63c9a2,
      roughness: 0.05,
      metalness: 0,
      reflectivity: 1,
      ior: 1.52,
      // Light piping down the pane exits at the ground edge; that glow is the
      // single most recognisable detail on a real backboard.
      emissive: 0x2b7d63,
      emissiveIntensity: 0.6,
      clearcoat: 1,
      clearcoatRoughness: 0.03,
      envMapIntensity: 2.0,
    });
    const eb = 0.013; // how much of the perimeter reads as exposed edge
    const hw = B.width / 2;
    const hh = B.height / 2;
    const t = B.thickness;
    // A whisker thicker than the pane so the two never fight for the same depth.
    const et = t * 1.06;
    const edge = mergeGeos([
      at(new BoxGeometry(et, eb, B.width), face, this.boardCentre.y + hh - eb / 2, 0),
      at(new BoxGeometry(et, eb, B.width), face, this.boardCentre.y - hh + eb / 2, 0),
      at(new BoxGeometry(et, B.height - eb * 2, eb), face, this.boardCentre.y, hw - eb / 2),
      at(new BoxGeometry(et, B.height - eb * 2, eb), face, this.boardCentre.y, -hw + eb / 2),
    ]);
    const edgeMesh = new Mesh(edge, edgeMat);
    edgeMesh.renderOrder = 3;
    this.boardGroup.add(edgeMesh);

    // --- rear channel, mount plate and bolts -------------------------------
    const alu = new MeshPhysicalMaterial({
      color: 0x9aa0a6,
      roughness: 0.24,
      metalness: 1,
      anisotropy: 0.55,
      envMapIntensity: 1.5,
      clearcoat: 0.3,
      clearcoatRoughness: 0.25,
    });
    const back = face + this.side * (t / 2 + 0.012);
    const chan: BufferGeometry[] = [
      // Slim extruded channel gripping the glass edge from behind.
      at(new BoxGeometry(0.026, 0.034, B.width), back, this.boardCentre.y + hh - 0.015, 0),
      at(new BoxGeometry(0.026, 0.034, B.width), back, this.boardCentre.y - hh + 0.015, 0),
      at(new BoxGeometry(0.026, B.height, 0.032), back, this.boardCentre.y, hw - 0.014),
      at(new BoxGeometry(0.026, B.height, 0.032), back, this.boardCentre.y, -hw + 0.014),
      // Mount plate the rim and the arms bolt into.
      at(new BoxGeometry(0.05, 0.62, 0.60), back + this.side * 0.03, HOOP.rimHeight + 0.12, 0),
      at(new BoxGeometry(0.03, 0.34, 0.90), back + this.side * 0.02, this.boardCentre.y + 0.34, 0),
    ];
    for (const dz of [-0.21, 0.21]) {
      for (const dy of [-0.2, 0.2]) {
        chan.push(
          at(
            new CylinderGeometry(0.011, 0.011, 0.05, 8).rotateZ(Math.PI / 2),
            back + this.side * 0.06,
            HOOP.rimHeight + 0.12 + dy,
            dz,
          ),
        );
      }
    }
    const chanMesh = new Mesh(mergeGeos(chan), alu);
    chanMesh.castShadow = true;
    chanMesh.receiveShadow = true;
    this.boardGroup.add(chanMesh);

    // --- padding -----------------------------------------------------------
    // Every board in the league has a vinyl wrap on the bottom edge and up the
    // lower sides. It is stitched, creased and scuffed — not an extruded box.
    const padTex = bakeVinylPad(1024, 192, {
      label: 'BALLIN',
      base: [15, 17, 24],
      panels: 7,
      // Board padding is plain dark vinyl with a sponsor print; the coloured
      // band belongs on the stanchion, not up here.
      stripe: false,
    });
    const padMat = new MeshStandardMaterial({
      map: padTex.map,
      roughnessMap: padTex.rough,
      roughness: 1,
      metalness: 0,
      envMapIntensity: 0.5,
    });
    const pt = t / 2 + 0.052;
    const bottom = B.bottomHeight;
    const padPieces = [
      // Bottom wrap. Slightly proud of the glass on both faces.
      scaleUV(
        at(new BoxGeometry(pt * 2, 0.13, B.width + 0.02), face, bottom + 0.05, 0),
        1,
        1,
      ),
      // Lower side wraps, 15 in up each edge.
      scaleUV(
        at(new BoxGeometry(pt * 2, 0.40, 0.115), face, bottom + 0.325, hw - 0.045),
        0.32,
        1,
      ),
      scaleUV(
        at(new BoxGeometry(pt * 2, 0.40, 0.115), face, bottom + 0.325, -hw + 0.045),
        0.32,
        1,
      ),
    ];
    const padMesh = new Mesh(mergeGeos(padPieces), padMat);
    padMesh.castShadow = true;
    padMesh.receiveShadow = true;
    this.boardGroup.add(padMesh);
  }

  // -------------------------------------------------------------------------
  // Ring
  // -------------------------------------------------------------------------

  private buildRim(engine: Engine): void {
    const S = this.strands;
    const size = Math.min(1024, engine.quality.textureSize);
    // The court-facing point of the ring is torus u = 0.5 on the +X basket.
    const rim = bakeRimMaps(size, this.side > 0 ? 0.5 : 0.0, S);

    const ringMat = new MeshPhysicalMaterial({
      map: rim.map,
      roughnessMap: rim.orm,
      metalnessMap: rim.orm,
      roughness: 1,
      metalness: 1,
      // Powder coat is a clear-over-colour finish: one broad lobe from the coat
      // sitting on top of the base spec, which is what makes it read as gloss
      // paint on steel rather than raw metal.
      clearcoat: 0.48,
      clearcoatRoughness: 0.17,
      // Stretch the highlight along the bar so it runs round the ring the way a
      // brushed/turned tube does.
      anisotropy: 0.55,
      anisotropyRotation: Math.PI / 2,
      envMapIntensity: 1.35,
    });

    // 12 × 80 keeps the ring's silhouette clean at RIM framing for 1,920 tris;
    // it is the most-looked-at piece of hardware in the frame, so it earns them.
    const tubular = engine.quality.textureSize >= 2048 ? 80 : 56;
    const ring = new Mesh(
      new TorusGeometry(HOOP.rimRadius, HOOP.rimTubeRadius, 12, tubular),
      ringMat,
    );
    ring.rotation.x = Math.PI / 2;
    ring.position.set(
      this.rimCentre.x - this.pivotWorld.x,
      this.rimCentre.y - this.pivotWorld.y,
      0,
    );
    ring.castShadow = true;
    ring.receiveShadow = true;
    this.rimPivot.add(ring);

    // --- hardware ----------------------------------------------------------
    // Machined steel, not a mirror: in a bowl this dark a metalness-1 surface
    // with nothing to reflect just goes black, so the hardware keeps a little
    // diffuse and a broad roughness that the overhead banks can actually catch.
    const steel = new MeshPhysicalMaterial({
      color: 0x777d85,
      roughness: 0.46,
      metalness: 0.72,
      envMapIntensity: 1.5,
      clearcoat: 0.25,
      clearcoatRoughness: 0.35,
    });
    const lx = this.rimCentre.x - this.pivotWorld.x; // ring centre, pivot-local
    const backLocal = 0;
    const plateX = lx + this.side * (HOOP.rimRadius + 0.048);
    const hw: BufferGeometry[] = [
      // Mount plate and the hinge box behind it. Compact — a real breakaway is
      // mostly hidden behind the ring, not a brick hanging off the board.
      at(new BoxGeometry(0.09, 0.05, 0.14), plateX, -0.014, 0),
      at(new BoxGeometry(0.055, 0.10, 0.12), backLocal - this.side * 0.024, -0.034, 0),
      // Hinge pin.
      at(new CylinderGeometry(0.013, 0.013, 0.20, 10), plateX - this.side * 0.03, -0.042, 0),
      // Gusset from the plate down to the ring's back arc.
      at(new BoxGeometry(0.085, 0.042, 0.042), lx + this.side * (HOOP.rimRadius * 0.64), -0.028, 0),
    ];
    // Welds: fat beads where the ring meets the plate.
    for (const dz of [-0.05, 0.05]) {
      hw.push(
        at(
          new SphereGeometry(0.011, 8, 6).scale(1.5, 0.7, 1),
          lx + this.side * (HOOP.rimRadius - 0.004),
          0,
          dz,
        ),
      );
    }
    // Bolt heads through the board plate.
    for (const dz of [-0.06, 0.06]) {
      for (const dy of [-0.042, 0.042]) {
        hw.push(
          at(
            new CylinderGeometry(0.011, 0.011, 0.026, 6).rotateZ(Math.PI / 2),
            backLocal - this.side * 0.052,
            dy - 0.038,
            dz,
          ),
        );
      }
    }
    const hwMesh = new Mesh(mergeGeos(hw), steel);
    hwMesh.castShadow = true;
    this.rimPivot.add(hwMesh);

    // --- the twelve net hooks ----------------------------------------------
    // Small welded loops under the bar. Instanced: one draw call for all twelve.
    const hookGeo = new TorusGeometry(0.014, 0.0032, 4, 7, Math.PI * 1.45);
    const hooks = new InstancedMesh(hookGeo, steel, S);
    hooks.castShadow = true;
    const hookY = -HOOP.rimTubeRadius * 0.9;
    for (let s = 0; s < S; s++) {
      const ang = (s / S) * Math.PI * 2;
      _obj.position.set(
        lx + Math.cos(ang) * HOOP.rimRadius,
        hookY,
        Math.sin(ang) * HOOP.rimRadius,
      );
      // Loop opens downward, plane containing the radial direction.
      _quat.setFromAxisAngle(UP, -ang);
      _obj.quaternion.copy(_quat);
      _obj.rotateX(Math.PI / 2);
      _obj.rotateZ(Math.PI * 0.72);
      _obj.updateMatrix();
      hooks.setMatrixAt(s, _obj.matrix);
    }
    hooks.instanceMatrix.needsUpdate = true;
    this.rimPivot.add(hooks);
  }

  // -------------------------------------------------------------------------
  // Stanchion
  // -------------------------------------------------------------------------

  private buildStanchion(baseX: number, engine: Engine): void {
    void engine;
    const s = this.side;
    const back = this.boardCentre.x + s * (HOOP.board.thickness / 2 + 0.05);
    const colX = baseX + s * 2.05;

    const steel = new MeshPhysicalMaterial({
      color: 0x565c66,
      roughness: 0.42,
      metalness: 0.7,
      envMapIntensity: 1.4,
      clearcoat: 0.35,
      clearcoatRoughness: 0.28,
    });
    const chrome = new MeshPhysicalMaterial({
      color: 0xd6dae0,
      roughness: 0.12,
      metalness: 1,
      envMapIntensity: 2.4,
    });

    const p = (x: number, y: number, z = 0) => new Vector3(x, y, z);

    // Column, boom and the gooseneck lower arm.
    const structure: BufferGeometry[] = [
      // Tapered column.
      span(new CylinderGeometry(0.20, 0.30, 4.05, 4, 1).rotateY(Math.PI / 4),
        p(colX, 0.06), p(colX, 4.11), 'y'),
      // Main boom cantilevering forward over the baseline.
      beamBetween(p(colX, 3.98), p(back + s * 0.06, 3.60), 0.42, 0.30),
      // Lower gooseneck arm.
      beamBetween(p(colX + s * -0.02, 2.62), p(back + s * 0.06, 3.04), 0.30, 0.16),
      // Knuckle where the two arms meet the board frame.
      at(new BoxGeometry(0.12, 0.78, 0.46), back + s * 0.05, 3.32, 0),
      // Column cap.
      at(new BoxGeometry(0.46, 0.10, 0.5), colX, 4.14, 0),
      // Base flange.
      at(new CylinderGeometry(0.40, 0.44, 0.12, 10), colX, 0.06, 0),
    ];
    // Diagonal braces between the two arms.
    for (const dz of [-0.19, 0.19]) {
      structure.push(beamBetween(p(colX - s * -0.5, 3.86, dz), p(back + s * 0.2, 3.12, dz), 0.06, 0.10));
    }
    const structMesh = new Mesh(mergeGeos(structure), steel);
    structMesh.castShadow = true;
    structMesh.receiveShadow = true;
    this.group.add(structMesh);

    // Hydraulic struts: dark barrel with a polished rod sliding out of it.
    const barrels: BufferGeometry[] = [];
    const rods: BufferGeometry[] = [];
    for (const dz of [-0.30, 0.30]) {
      const a0 = p(colX + s * 0.28, 1.72, dz);
      const a1 = p(colX + s * 0.02, 2.86, dz);
      const mid = a0.clone().lerp(a1, 0.55);
      barrels.push(rodBetween(a0, mid, 0.045, 0.05, 10));
      rods.push(rodBetween(mid, a1, 0.024, 0.024, 8));
      barrels.push(at(new SphereGeometry(0.035, 8, 6), a0.x, a0.y, a0.z));
      barrels.push(at(new SphereGeometry(0.03, 8, 6), a1.x, a1.y, a1.z));
    }
    const barrelMesh = new Mesh(mergeGeos(barrels), steel);
    barrelMesh.castShadow = true;
    this.group.add(barrelMesh);
    const rodMesh = new Mesh(mergeGeos(rods), chrome);
    rodMesh.castShadow = true;
    this.group.add(rodMesh);

    // Padded base wrap — the branded block everyone lands on.
    const padTex = bakeVinylPad(1024, 320, {
      label: 'BALLIN',
      base: [16, 18, 25],
      accent: [118, 34, 24],
      panels: 5,
    });
    const padMat = new MeshStandardMaterial({
      map: padTex.map,
      roughnessMap: padTex.rough,
      roughness: 1,
      metalness: 0,
      envMapIntensity: 0.45,
    });
    const px0 = baseX + s * 0.80;
    const px1 = baseX + s * 2.86;
    const pcx = (px0 + px1) / 2;
    const plen = Math.abs(px1 - px0);
    const pad = mergeGeos([
      scaleUV(at(new BoxGeometry(plen, 1.52, 1.16), pcx, 0.78, 0), 1, 1),
      // Stepped, slightly inset cap so the silhouette is not one slab.
      scaleUV(at(new BoxGeometry(plen - 0.13, 0.34, 1.05), pcx, 1.70, 0), 1, 0.24),
      // A low kick-plate skirt.
      scaleUV(at(new BoxGeometry(plen + 0.05, 0.09, 1.22), pcx, 0.045, 0), 1, 0.08),
    ]);
    const padMesh = new Mesh(pad, padMat);
    padMesh.castShadow = true;
    padMesh.receiveShadow = true;
    this.group.add(padMesh);
  }

  // -------------------------------------------------------------------------
  // Net
  // -------------------------------------------------------------------------

  /** Hourglass: full at the ring, waisted at ~62%, flaring back out to the hem. */
  private netProfile(t: number): number {
    const w = NET_WAIST_T;
    if (t <= w) return 1 + (NET_WAIST_SCALE - 1) * smoothstep(t / w);
    return (
      NET_WAIST_SCALE +
      (HOOP.net.bottomRadiusScale - NET_WAIST_SCALE) * smoothstep((t - w) / (1 - w))
    );
  }

  private buildNet(engine: Engine): void {
    const S = this.strands;
    const R = this.rings;
    const attachR = HOOP.rimRadius;
    const topY = HOOP.rimHeight - HOOP.rimTubeRadius - 0.012;

    // --- nodes -------------------------------------------------------------
    // Successive rings are offset by half a cell so the two cord families run
    // as opposing helices and the crossings form real diamonds.
    const nodes: NetNode[] = [];
    for (let r = 0; r <= R; r++) {
      const t = r / R;
      const rad = attachR * this.netProfile(t);
      this.restRingR.push(rad);
      // Rows crowd very slightly toward the hem, the way a hanging net does.
      const y = topY - HOOP.net.length * Math.pow(t, 0.96);
      for (let s = 0; s < S; s++) {
        const ang = ((s + r * 0.5) / S) * Math.PI * 2;
        const p = new Vector3(
          this.rimCentre.x + Math.cos(ang) * rad,
          y,
          this.rimCentre.z + Math.sin(ang) * rad,
        );
        nodes.push({ pos: p, prev: p.clone(), pinned: r === 0, restR: rad, ring: r });
      }
    }
    this.nodes = nodes;
    this.anchors = [];
    for (let s = 0; s < S; s++) {
      this.anchors.push(nodes[s].pos.clone().sub(this.pivotWorld));
    }

    // --- links -------------------------------------------------------------
    const idx = (r: number, s: number) => r * S + ((s % S) + S) % S;
    const links: NetLink[] = [];
    for (let r = 0; r < R; r++) {
      for (let s = 0; s < S; s++) {
        const a = idx(r, s);
        // Two cords leave every knot: one drifting right, one left.
        for (const b of [idx(r + 1, s), idx(r + 1, s - 1)]) {
          links.push({
            a,
            b,
            // Cord is a hair longer than the taut design shape, so gravity has
            // something to pull out of and the net hangs rather than stands.
            rest: nodes[a].pos.distanceTo(nodes[b].pos) * 1.006,
            stiffness: 0.96,
            cord: true,
          });
        }
      }
    }
    // Soft hoops standing in for knot friction: they keep the mesh from
    // shearing shut without stopping the ball from ballooning it open.
    for (let r = 1; r <= R; r++) {
      for (let s = 0; s < S; s++) {
        const a = idx(r, s);
        const b = idx(r, s + 1);
        links.push({
          a,
          b,
          rest: nodes[a].pos.distanceTo(nodes[b].pos),
          stiffness: r <= 2 ? 0.34 : 0.16,
          cord: false,
        });
      }
    }
    this.links = links;

    // --- paths -------------------------------------------------------------
    // Family A walks straight down the index; family B steps back one strand
    // per ring. Together they visit every cord exactly once.
    const paths: Int32Array[] = [];
    for (let k = 0; k < S; k++) {
      const pa = new Int32Array(R + 1);
      const pb = new Int32Array(R + 1);
      for (let r = 0; r <= R; r++) {
        pa[r] = idx(r, k);
        pb[r] = idx(r, k - r);
      }
      paths.push(pa, pb);
    }
    this.paths = paths;

    // --- geometry ----------------------------------------------------------
    const K = this.sides;
    const perPath = (R + 1) * K;
    const vertCount = paths.length * perPath;
    const pos = new Float32Array(vertCount * 3);
    const nrm = new Float32Array(vertCount * 3);
    const uv = new Float32Array(vertCount * 2);
    const col = new Float32Array(vertCount * 3);
    const index = new Uint16Array(paths.length * R * K * 6);

    let io = 0;
    for (let p = 0; p < paths.length; p++) {
      const base = p * perPath;
      // A little per-cord variation so no two strands are identical.
      const tone = 0.94 + ((p * 37) % 11) / 11 * 0.12;
      for (let i = 0; i <= R; i++) {
        const t = i / R;
        // Used nets are never white: bright at the top, greying to a soiled
        // hem, with the loops darkened where they saw against the hooks.
        const soil = 0.90 - 0.50 * Math.pow(t, 1.7);
        const choke = 1 - 0.26 * clamp01(1 - t / 0.10);
        const lum = clamp01(soil * choke * tone);
        for (let j = 0; j < K; j++) {
          const vi = base + i * K + j;
          uv[vi * 2] = j / K;
          uv[vi * 2 + 1] = t;
          // Soiling browns as it darkens.
          col[vi * 3] = lum;
          col[vi * 3 + 1] = lum * (0.98 - 0.06 * t);
          col[vi * 3 + 2] = lum * (0.94 - 0.17 * t);
        }
        if (i < R) {
          for (let j = 0; j < K; j++) {
            const j1 = (j + 1) % K;
            const a0 = base + i * K + j;
            const b0 = base + i * K + j1;
            const c0 = base + (i + 1) * K + j1;
            const d0 = base + (i + 1) * K + j;
            index[io++] = a0;
            index[io++] = b0;
            index[io++] = d0;
            index[io++] = b0;
            index[io++] = c0;
            index[io++] = d0;
          }
        }
      }
    }

    const geo = new BufferGeometry();
    this.netPos = new BufferAttribute(pos, 3);
    this.netNrm = new BufferAttribute(nrm, 3);
    this.netPos.setUsage(DynamicDrawUsage);
    this.netNrm.setUsage(DynamicDrawUsage);
    geo.setAttribute('position', this.netPos);
    geo.setAttribute('normal', this.netNrm);
    geo.setAttribute('uv', new BufferAttribute(uv, 2));
    geo.setAttribute('color', new BufferAttribute(col, 3));
    geo.setIndex(new BufferAttribute(index, 1));

    const cord = bakeNetCord(64);
    cord.map.repeat.set(1, HOOP.net.length * CORD_TWISTS_PER_M);
    cord.rough.repeat.copy(cord.map.repeat);
    const mat = new MeshPhysicalMaterial({
      map: cord.map,
      roughnessMap: cord.rough,
      color: 0xffffff,
      vertexColors: true,
      roughness: 1,
      metalness: 0,
      // Nylon has a soft, wide sheen rather than a metal-style highlight.
      sheen: 0.85,
      sheenRoughness: 0.55,
      sheenColor: 0xfff4e2,
      envMapIntensity: 0.85,
      side: DoubleSide,
    });

    // Thin nylon lights up where you see through the edge of a cord. Adding it
    // as emission rather than alpha keeps the net opaque — no sort order, no
    // depth fighting between 24 interleaved strands — while still giving the
    // soft, slightly glowing filament edge that reads as a real net.
    const edgeGlow: IUniform<number> = { value: 0.55 };
    this.netGlow = edgeGlow;
    mat.onBeforeCompile = (sh) => {
      sh.uniforms.uCordGlow = edgeGlow;
      sh.fragmentShader = sh.fragmentShader
        .replace('void main() {', 'uniform float uCordGlow;\nvoid main() {')
        .replace(
          '#include <emissivemap_fragment>',
          `#include <emissivemap_fragment>
           float cordFres = 1.0 - abs( dot( normalize( normal ), normalize( vViewPosition ) ) );
           totalEmissiveRadiance += diffuseColor.rgb * pow( cordFres, 2.4 ) * uCordGlow;`,
        );
    };
    mat.customProgramCacheKey = () => 'ballin-netcord';

    this.netMesh = new Mesh(geo, mat);
    this.netMesh.castShadow = true;
    this.netMesh.receiveShadow = true;
    this.netMesh.frustumCulled = false;
    this.netMesh.renderOrder = 1;
    this.group.add(this.netMesh);

    // Let the net find its own hang before the first frame is drawn.
    for (let i = 0; i < 260; i++) this.stepNet(1 / 240, engine.quality.netIterations, null);
    this.syncNetGeometry();
  }

  private netGlow: IUniform<number> = { value: 0.55 };

  // -------------------------------------------------------------------------
  // Simulation
  // -------------------------------------------------------------------------

  /** Verlet step for the net, plus rim/board springs and ball coupling. */
  stepNet(dt: number, iterations: number, ball: BallProbe | null): void {
    this.stepHardware(dt);

    // Wake on anything nearby; otherwise a settled net costs nothing.
    let probe: BallProbe | null = null;
    if (ball) {
      _a.copy(ball.position);
      if (ball.velocity) _a.addScaledVector(ball.velocity, dt);
      const dy = _a.y - HOOP.rimHeight;
      if (dy < ball.radius + 0.14 && dy > -(HOOP.net.length + ball.radius + 0.12)) {
        const dh = Math.hypot(_a.x - this.rimCentre.x, _a.z - this.rimCentre.z);
        if (dh < HOOP.rimRadius + ball.radius + 0.10) {
          probe = { position: _a.clone(), radius: ball.radius, velocity: ball.velocity };
          this.ballLast.copy(_a);
          this.asleep = false;
        }
      }
    }
    if (this.asleep && !probe && this.impulses.length === 0) return;

    this.pumpImpulses(dt);
    this.updateAnchors();

    const g = -9.80665;
    const damping = 0.9785;
    let maxV2 = 0;
    for (const n of this.nodes) {
      if (n.pinned) continue;
      const vx = (n.pos.x - n.prev.x) * damping;
      const vy = (n.pos.y - n.prev.y) * damping;
      const vz = (n.pos.z - n.prev.z) * damping;
      n.prev.copy(n.pos);
      // Hard speed ceiling. Verlet plus a stiff constraint solve can trade a
      // large impulse for an explosion; 14 m/s is well past anything a real
      // cord does and costs nothing to enforce.
      let v2 = vx * vx + vy * vy + vz * vz;
      const lim = MAX_CORD_SPEED * dt;
      if (v2 > lim * lim) {
        const k = lim / Math.sqrt(v2);
        n.pos.x += vx * k;
        n.pos.y += vy * k + g * dt * dt;
        n.pos.z += vz * k;
        v2 = lim * lim;
      } else {
        n.pos.x += vx;
        n.pos.y += vy + g * dt * dt;
        n.pos.z += vz;
      }
      if (v2 > maxV2) maxV2 = v2;
    }

    // The ball drags cord with it — this is what makes the net snap down on a
    // make instead of just being shouldered aside.
    if (probe?.velocity) {
      const reach = probe.radius + 0.075;
      for (const n of this.nodes) {
        if (n.pinned) continue;
        const d = n.pos.distanceTo(probe.position);
        if (d > reach) continue;
        const k = (1 - d / reach) * 0.55;
        n.prev.x -= probe.velocity.x * dt * k;
        n.prev.y -= probe.velocity.y * dt * k;
        n.prev.z -= probe.velocity.z * dt * k;
      }
    }

    for (let it = 0; it < iterations; it++) {
      this.solveLinks();
      if (probe) this.solveBall(probe);
    }
    this.shapeMemory();

    // Sleep once the whole net is genuinely still, not merely slow.
    const still = maxV2 < 4e-9 && !probe && this.impulses.length === 0;
    this.restCounter = still ? this.restCounter + 1 : 0;
    if (this.restCounter > 90) this.asleep = true;
    this.netDirty = true;
  }

  private solveLinks(): void {
    const links = this.links;
    const nodes = this.nodes;
    for (let i = 0; i < links.length; i++) {
      const l = links[i];
      const a = nodes[l.a];
      const b = nodes[l.b];
      const dx = b.pos.x - a.pos.x;
      const dy = b.pos.y - a.pos.y;
      const dz = b.pos.z - a.pos.z;
      const d = Math.hypot(dx, dy, dz) || 1e-6;
      // Cords take tension but not compression: a slack cord does nothing,
      // which is why a real net crumples instead of springing back like foam.
      if (l.cord && d < l.rest) continue;
      const diff = ((d - l.rest) / d) * 0.5 * l.stiffness;
      const mx = dx * diff;
      const my = dy * diff;
      const mz = dz * diff;
      if (!a.pinned) {
        a.pos.x += mx;
        a.pos.y += my;
        a.pos.z += mz;
      }
      if (!b.pinned) {
        b.pos.x -= mx;
        b.pos.y -= my;
        b.pos.z -= mz;
      }
    }
  }

  /**
   * Continuous-ish collision against the ball. Node pushout alone lets the ball
   * slice between two knots, so every cord segment is tested as a capsule and
   * the correction is split along it by barycentric weight.
   */
  private solveBall(ball: BallProbe): void {
    const nodes = this.nodes;
    const bx = ball.position.x;
    const by = ball.position.y;
    const bz = ball.position.z;
    const rr = ball.radius + CORD_RADIUS;
    const rr2 = rr * rr;

    // Knots first — cheap, and it resolves the common case.
    for (const n of nodes) {
      if (n.pinned) continue;
      const dx = n.pos.x - bx;
      const dy = n.pos.y - by;
      const dz = n.pos.z - bz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 >= rr2 || d2 < 1e-10) continue;
      const d = Math.sqrt(d2);
      const k = (rr - d) / d;
      n.pos.x += dx * k;
      n.pos.y += dy * k;
      n.pos.z += dz * k;
    }

    // Then the cord spans between them.
    const links = this.links;
    for (let i = 0; i < links.length; i++) {
      const l = links[i];
      const a = nodes[l.a];
      const b = nodes[l.b];
      if (a.pinned && b.pinned) continue;
      const ax = a.pos.x;
      const ay = a.pos.y;
      const az = a.pos.z;
      const ex = b.pos.x - ax;
      const ey = b.pos.y - ay;
      const ez = b.pos.z - az;
      const ee = ex * ex + ey * ey + ez * ez;
      if (ee < 1e-12) continue;
      let t = ((bx - ax) * ex + (by - ay) * ey + (bz - az) * ez) / ee;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const px = ax + ex * t - bx;
      const py = ay + ey * t - by;
      const pz = az + ez * t - bz;
      const d2 = px * px + py * py + pz * pz;
      if (d2 >= rr2 || d2 < 1e-10) continue;
      const d = Math.sqrt(d2);
      const push = (rr - d) / d;
      // Split so the contact point itself clears the surface exactly.
      let wa = 1 - t;
      let wb = t;
      if (a.pinned) {
        wa = 0;
        wb = 1;
      } else if (b.pinned) {
        wa = 1;
        wb = 0;
      }
      const norm = wa * wa + wb * wb;
      if (norm < 1e-6) continue;
      const ka = (wa / norm) * push;
      const kb = (wb / norm) * push;
      if (!a.pinned) {
        a.pos.x += px * ka;
        a.pos.y += py * ka;
        a.pos.z += pz * ka;
      }
      if (!b.pinned) {
        b.pos.x += px * kb;
        b.pos.y += py * kb;
        b.pos.z += pz * kb;
      }
    }
  }

  /**
   * Soft shape memory. A pure diamond lattice will shear closed under its own
   * weight, so each ring is nudged back toward its rest circumference — slowly
   * enough (τ ≈ 0.2 s) that a ball can still balloon the net wide open.
   */
  private shapeMemory(): void {
    const S = this.strands;
    const cx = this.rimCentre.x;
    const cz = this.rimCentre.z;
    for (let r = 1; r <= this.rings; r++) {
      let mean = 0;
      for (let s = 0; s < S; s++) {
        const n = this.nodes[r * S + s];
        mean += Math.hypot(n.pos.x - cx, n.pos.z - cz);
      }
      mean /= S;
      if (mean < 1e-5) continue;
      const target = this.restRingR[r];
      const kRing = r <= 2 ? 0.045 : 0.016;
      const corr = (target - mean) * kRing;
      for (let s = 0; s < S; s++) {
        const n = this.nodes[r * S + s];
        const dx = n.pos.x - cx;
        const dz = n.pos.z - cz;
        const d = Math.hypot(dx, dz) || 1e-6;
        // Ring-wide correction keeps the hourglass; a whisper of per-node pull
        // keeps it round without erasing the asymmetry of a real contact.
        const k = corr / d + ((mean - d) / d) * 0.010;
        n.pos.x += dx * k;
        n.pos.z += dz * k;
      }
    }
  }

  /** Breakaway hinge and backboard shake, both damped second-order springs. */
  private stepHardware(dt: number): void {
    this.rimFlexVel += (-RIM_OMEGA * RIM_OMEGA * this.rimFlex - 2 * RIM_ZETA * RIM_OMEGA * this.rimFlexVel) * dt;
    this.rimFlex += this.rimFlexVel * dt;
    // Real breakaways hinge down and are stopped hard on the way back up.
    if (this.rimFlex < -0.010) {
      this.rimFlex = -0.010;
      this.rimFlexVel *= -0.25;
    }
    this.rimFlex = Math.min(this.rimFlex, 0.24);

    this.rimTwistVel += (-RIM_OMEGA * 1.3 * RIM_OMEGA * 1.3 * this.rimTwist - 2 * 0.34 * RIM_OMEGA * 1.3 * this.rimTwistVel) * dt;
    this.rimTwist += this.rimTwistVel * dt;

    this.boardShakeVel += (-BOARD_OMEGA * BOARD_OMEGA * this.boardShake - 2 * BOARD_ZETA * BOARD_OMEGA * this.boardShakeVel) * dt;
    this.boardShake += this.boardShakeVel * dt;

    _rot.makeRotationY(this.rimTwist);
    _m4.makeRotationZ(this.side * this.rimFlex);
    _rot.multiply(_m4);
    this.rimPivot.quaternion.setFromRotationMatrix(_rot);
    this.boardGroup.position.set(this.boardNormal.x * this.boardShake, this.boardShake * 0.22, 0);
  }

  /** Pinned nodes ride the ring, so a flexing rim swings the net. */
  private updateAnchors(): void {
    const S = this.strands;
    const ox = this.boardGroup.position.x;
    const oy = this.boardGroup.position.y;
    for (let s = 0; s < S; s++) {
      _a.copy(this.anchors[s]).applyMatrix4(_rot);
      const n = this.nodes[s];
      // Leave `prev` behind so the motion imparts real velocity to the cord.
      n.pos.set(this.pivotWorld.x + ox + _a.x, this.pivotWorld.y + oy + _a.y, this.pivotWorld.z + _a.z);
    }
  }

  private pumpImpulses(dt: number): void {
    if (this.impulses.length === 0) return;
    for (let i = this.impulses.length - 1; i >= 0; i--) {
      const im = this.impulses[i];
      im.delay -= dt;
      if (im.delay > 0) continue;
      this.impulses.splice(i, 1);
      const cx = this.rimCentre.x;
      const cz = this.rimCentre.z;
      for (const n of this.nodes) {
        if (n.pinned) continue;
        if (n.ring < im.r0 || n.ring > im.r1) continue;
        const dx = n.pos.x - cx;
        const dz = n.pos.z - cz;
        const d = Math.hypot(dx, dz) || 1e-6;
        let w = 1;
        if (im.focus > 0) {
          const az = Math.atan2(dz, dx);
          let da = az - im.az;
          da = Math.atan2(Math.sin(da), Math.cos(da));
          w = 1 - im.focus + im.focus * Math.pow(clamp01(0.5 + 0.5 * Math.cos(da)), 1.6);
        }
        // Ramp in over the ring range so the wave has a leading edge.
        const rspan = Math.max(1, im.r1 - im.r0);
        w *= 0.45 + 0.55 * Math.sin(clamp01((n.ring - im.r0) / rspan) * Math.PI);
        n.prev.x -= (im.vx + (dx / d) * im.vr) * dt * w;
        n.prev.y -= im.vy * dt * w;
        n.prev.z -= (im.vz + (dz / d) * im.vr) * dt * w;
      }
    }
  }

  // -------------------------------------------------------------------------
  // Presentation
  // -------------------------------------------------------------------------

  /** Rebuilds the cord tubes in place. Called once per rendered frame. */
  syncNetGeometry(): void {
    if (!this.netDirty) return;
    this.netDirty = false;
    const P = this.netPos.array as Float32Array;
    const N = this.netNrm.array as Float32Array;
    const K = this.sides;
    const paths = this.paths;
    const nodes = this.nodes;
    const cx = this.rimCentre.x;
    const cz = this.rimCentre.z;
    const r = CORD_RADIUS;
    const cosT = this.cosT;
    const sinT = this.sinT;
    let vi = 0;

    for (let p = 0; p < paths.length; p++) {
      const path = paths[p];
      const last = path.length - 1;
      for (let i = 0; i <= last; i++) {
        const cur = nodes[path[i]].pos;
        const pv = nodes[path[i > 0 ? i - 1 : 0]].pos;
        const nx = nodes[path[i < last ? i + 1 : last]].pos;
        let tx = nx.x - pv.x;
        let ty = nx.y - pv.y;
        let tz = nx.z - pv.z;
        const tl = Math.hypot(tx, ty, tz) || 1;
        tx /= tl;
        ty /= tl;
        tz /= tl;

        // Reference direction is the outward radial, which never degenerates on
        // a near-vertical cord the way a world-up reference would.
        let ox = cur.x - cx;
        let oz = cur.z - cz;
        const ol = Math.hypot(ox, oz) || 1;
        ox /= ol;
        oz /= ol;

        // binormal = tangent × outward
        let bx = ty * oz;
        let by = tz * ox - tx * oz;
        let bz = -ty * ox;
        const bl = Math.hypot(bx, by, bz) || 1;
        bx /= bl;
        by /= bl;
        bz /= bl;
        // normal = binormal × tangent
        const nnx = by * tz - bz * ty;
        const nny = bz * tx - bx * tz;
        const nnz = bx * ty - by * tx;

        for (let j = 0; j < K; j++) {
          const c = cosT[j];
          const s = sinT[j];
          const dx = c * nnx + s * bx;
          const dy = c * nny + s * by;
          const dz = c * nnz + s * bz;
          const o3 = vi * 3;
          P[o3] = cur.x + dx * r;
          P[o3 + 1] = cur.y + dy * r;
          P[o3 + 2] = cur.z + dz * r;
          N[o3] = dx;
          N[o3 + 1] = dy;
          N[o3 + 2] = dz;
          vi++;
        }
      }
    }
    this.netPos.needsUpdate = true;
    this.netNrm.needsUpdate = true;
  }

  // -------------------------------------------------------------------------
  // Public hooks for gameplay / physics
  // -------------------------------------------------------------------------

  /**
   * Impulse applied to the net when the ball rips through. The ball's own
   * per-cord collision already drags the mesh down; this adds the sharp snap
   * and the recoil that flips the hem back up above the rim line.
   */
  punchNet(dir: Vector3, strength: number, at?: Vector3): void {
    this.asleep = false;
    // Physics only hands us a direction, so fall back to wherever we last saw
    // the ball inside the net — an off-centre make has to respond off-centre.
    const src = at ?? (this.ballLast.y > 0 ? this.ballLast : null);
    const az = src
      ? Math.atan2(src.z - this.rimCentre.z, src.x - this.rimCentre.x)
      : Math.atan2(dir.z, dir.x);
    const mag = clamp(strength, 0, 14);
    const focus = src ? 0.6 : 0.25;
    const R = this.rings;

    // 1. The snap. Only the lower two thirds move: on a clean make the top of
    //    the net stays composed and the bottom is yanked down after the ball.
    this.impulses.push({
      delay: 0,
      az,
      focus,
      r0: 3,
      r1: R,
      vx: dir.x * mag * 0.14,
      vy: -mag * 0.40,
      vz: dir.z * mag * 0.14,
      vr: -mag * 0.06,
    });
    // 2. The recoil. Cords go taut, the hem is thrown back up and outward, and
    //    for a moment it inverts above the rim line. If this never happens the
    //    net is not being simulated and it shows immediately.
    this.impulses.push({
      delay: 0.048,
      az,
      focus: focus * 0.35,
      r0: R - 6,
      r1: R,
      vx: 0,
      vy: mag * 0.98,
      vz: 0,
      vr: mag * 0.40,
    });
    // 3. The catch: the cords above come tight again and pull the inverted hem
    //    back down. Without this the hem simply coasts on gravity and hangs at
    //    the top of its arc, which reads as floaty rather than snappy.
    this.impulses.push({
      delay: 0.175,
      az: az + Math.PI,
      focus: focus * 0.45,
      r0: R - 6,
      r1: R,
      vx: 0,
      vy: -mag * 0.40,
      vz: 0,
      vr: -mag * 0.16,
    });
    // 4. The last wobble, roughly half the amplitude of the one before it.
    this.impulses.push({
      delay: 0.30,
      az,
      focus: focus * 0.3,
      r0: Math.max(1, R - 6),
      r1: R,
      vx: 0,
      vy: mag * 0.16,
      vz: 0,
      vr: mag * 0.07,
    });
  }

  /**
   * Drives the breakaway hinge. `speed` is the normal closing speed of the
   * contact in m/s; `at` biases the lateral twist so an off-centre hit rolls
   * the ring instead of tipping it square.
   */
  flexRim(speed: number, at?: Vector3): void {
    const s = clamp(speed, 0, 16);
    // ~1.1 mm of tip deflection per m/s of closing speed, so a routine contact
    // lands in the 2–8 mm band and only a dunk-grade impulse reaches the 9° stop.
    const peak = Math.min(0.16, s * 0.0021);
    this.rimFlexVel += peak * 46;
    if (at) {
      const dz = clamp((at.z - this.rimCentre.z) / HOOP.rimRadius, -1.4, 1.4);
      this.rimTwistVel += dz * s * 0.055;
      this.asleep = false;
      // A rattle whips the net hard and asymmetrically.
      this.impulses.push({
        delay: 0,
        az: Math.atan2(at.z - this.rimCentre.z, at.x - this.rimCentre.x),
        focus: 0.8,
        r0: 1,
        r1: this.rings,
        vx: 0,
        vy: -s * 0.05,
        vz: 0,
        vr: s * 0.10,
      });
    }
    this.asleep = false;
    this.shakeBoard(s * 0.25);
  }

  /** Backboard rattle. Amplitude is in millimetres of face travel. */
  shakeBoard(strength: number): void {
    const amp = Math.min(0.006, Math.abs(strength) * 0.0009);
    this.boardShakeVel += amp * 95;
  }

  /** Edge translucency on the cords; exposed so a VFX pass can dial it. */
  setNetGlow(v: number): void {
    this.netGlow.value = v;
  }

  /**
   * Slides the baked bank reflection against the camera. A real reflection
   * parallaxes; a decal does not, and the difference is obvious the moment the
   * camera moves even slightly.
   */
  parallaxGlass(camera: Vector3): void {
    const m = this.reflMap;
    if (!m) return;
    const dx = (camera.x - this.boardCentre.x) * -this.side;
    const dz = camera.z - this.boardCentre.z;
    const dy = camera.y - this.boardCentre.y;
    const d = Math.max(0.6, Math.abs(dx));
    m.offset.set(
      clamp((-dz / d) * 0.09, -0.06, 0.06) * this.side,
      clamp((dy / d) * 0.07, -0.05, 0.05),
    );
  }

  dispose(): void {
    this.group.traverse((o) => {
      const m = o as Mesh;
      if (m.geometry) m.geometry.dispose();
      const mat = m.material;
      if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
      else if (mat) mat.dispose();
    });
  }
}

export class HoopSystem implements System {
  readonly name = 'hoop';
  readonly order = 12;

  group = new Group();
  baskets: Basket[] = [];

  init(engine: Engine): void {
    this.group.name = 'hoops';
    engine.scene.add(this.group);
    for (const side of [1, -1] as const) {
      const b = new Basket(side, engine);
      this.baskets.push(b);
      this.group.add(b.group);
    }

    // Wire the hardware straight onto the existing bus so the rim flexes and
    // the board rattles without gameplay having to know this class exists.
    engine.bus.on('rimContact', ({ speed, position }) => {
      this.nearest(position).flexRim(speed, position);
    });
    engine.bus.on('boardContact', ({ speed, position }) => {
      this.nearest(position).shakeBoard(speed);
    });
  }

  basketFor(side: 1 | -1): Basket {
    return this.baskets.find((b) => b.side === side)!;
  }

  /** Whichever basket the event happened at. */
  nearest(p: Vector3): Basket {
    let best = this.baskets[0];
    let bd = Infinity;
    for (const b of this.baskets) {
      const d = b.rimCentre.distanceToSquared(p);
      if (d < bd) {
        bd = d;
        best = b;
      }
    }
    return best;
  }

  simulate(step: number, engine: Engine): void {
    const ballSys = engine.get<{ ballState?: { position: Vector3; velocity: Vector3 } }>('ball');
    const ball = ballSys?.ballState
      ? {
          position: ballSys.ballState.position,
          radius: BALL.radius,
          velocity: ballSys.ballState.velocity,
        }
      : null;
    for (const b of this.baskets) b.stepNet(step, engine.quality.netIterations, ball);
  }

  /** Geometry is rebuilt at frame rate, not at the 240 Hz physics rate. */
  update(_dt: number, _alpha: number, engine: Engine): void {
    for (const b of this.baskets) {
      b.syncNetGeometry();
      b.parallaxGlass(engine.camera.position);
    }
  }

  dispose(): void {
    for (const b of this.baskets) b.dispose();
    this.baskets = [];
  }
}
