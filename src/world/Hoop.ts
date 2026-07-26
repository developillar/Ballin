/**
 * Backboard, ring and net. The net is a verlet-integrated mesh so it whips
 * on contact and settles with real damping — it is the single most-watched
 * object in a basketball game, so it gets a full soft-body treatment rather
 * than a canned animation.
 *
 * Owned by the hoop agent.
 */

import {
  BufferGeometry,
  CylinderGeometry,
  DoubleSide,
  Float32BufferAttribute,
  Group,
  LineBasicMaterial,
  LineSegments,
  Mesh,
  MeshPhysicalMaterial,
  MeshStandardMaterial,
  BoxGeometry,
  TorusGeometry,
  Vector3,
} from 'three';
import type { Engine, System } from '../core/Engine';
import { BALL, COURT, HOOP } from '../core/Constants';
import { clamp } from '../core/MathX';

interface NetNode {
  pos: Vector3;
  prev: Vector3;
  pinned: boolean;
}

interface NetLink {
  a: number;
  b: number;
  rest: number;
  stiffness: number;
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
  private netLines!: LineSegments;
  private netPositions!: Float32BufferAttribute;

  constructor(
    readonly side: 1 | -1,
    engine: Engine,
  ) {
    const baseX = side * COURT.halfLength;
    this.boardNormal.set(-side, 0, 0);
    const boardX = baseX - side * (COURT.basketFromBaseline - HOOP.rimOffsetFromBoard - HOOP.rimRadius);
    this.boardCentre.set(boardX, HOOP.board.bottomHeight + HOOP.board.height / 2, 0);
    this.rimCentre.set(
      boardX - side * (HOOP.rimOffsetFromBoard + HOOP.rimRadius),
      HOOP.rimHeight,
      0,
    );

    this.buildBoard();
    this.buildRim();
    this.buildStanchion(baseX);
    this.buildNet(engine);
  }

  private buildBoard(): void {
    const glass = new MeshPhysicalMaterial({
      color: 0xdfeaf2,
      transparent: true,
      opacity: 0.16,
      roughness: 0.03,
      metalness: 0,
      transmission: 0.92,
      thickness: HOOP.board.thickness,
      ior: 1.52,
      clearcoat: 1,
      clearcoatRoughness: 0.02,
      envMapIntensity: 1.5,
      side: DoubleSide,
    });
    const board = new Mesh(
      new BoxGeometry(HOOP.board.thickness, HOOP.board.height, HOOP.board.width),
      glass,
    );
    board.position.copy(this.boardCentre);
    board.castShadow = true;
    board.receiveShadow = false;
    this.group.add(board);

    // Padded aluminium border.
    const border = new MeshStandardMaterial({ color: 0xf2f2f2, roughness: 0.34, metalness: 0.72 });
    const frameT = 0.05;
    const mk = (w: number, h: number, y: number, z: number) => {
      const m = new Mesh(new BoxGeometry(HOOP.board.thickness * 1.7, h, w), border);
      m.position.set(this.boardCentre.x, this.boardCentre.y + y, z);
      m.castShadow = true;
      this.group.add(m);
    };
    mk(HOOP.board.width, frameT, HOOP.board.height / 2, 0);
    mk(HOOP.board.width, frameT, -HOOP.board.height / 2, 0);
    mk(frameT, HOOP.board.height, 0, HOOP.board.width / 2);
    mk(frameT, HOOP.board.height, 0, -HOOP.board.width / 2);

    // Shooter's square.
    const sq = HOOP.board.innerSquare;
    const paint = new MeshStandardMaterial({ color: 0xff5a1f, roughness: 0.55, metalness: 0 });
    const bx = this.boardCentre.x - this.side * (HOOP.board.thickness / 2 + 0.004);
    const sqY = HOOP.rimHeight + sq.height / 2 - 0.03;
    const bar = (w: number, h: number, dy: number, dz: number) => {
      const m = new Mesh(new BoxGeometry(0.006, h, w), paint);
      m.position.set(bx, sqY + dy, dz);
      this.group.add(m);
    };
    bar(sq.width, sq.borderWidth, sq.height / 2, 0);
    bar(sq.width, sq.borderWidth, -sq.height / 2, 0);
    bar(sq.borderWidth, sq.height, 0, sq.width / 2);
    bar(sq.borderWidth, sq.height, 0, -sq.width / 2);
  }

  private buildRim(): void {
    const ringMat = new MeshStandardMaterial({
      color: 0xff4d16,
      roughness: 0.31,
      metalness: 0.88,
      envMapIntensity: 1.3,
    });
    const ring = new Mesh(
      new TorusGeometry(HOOP.rimRadius, HOOP.rimTubeRadius, 20, 72),
      ringMat,
    );
    ring.rotation.x = Math.PI / 2;
    ring.position.copy(this.rimCentre);
    ring.castShadow = true;
    this.group.add(ring);

    // Breakaway mount plate.
    const plate = new Mesh(
      new BoxGeometry(HOOP.rimOffsetFromBoard + 0.06, 0.085, 0.2),
      new MeshStandardMaterial({ color: 0xff4d16, roughness: 0.36, metalness: 0.85 }),
    );
    plate.position.set(
      this.rimCentre.x + this.side * (HOOP.rimRadius + HOOP.rimOffsetFromBoard * 0.5),
      HOOP.rimHeight - 0.02,
      0,
    );
    plate.castShadow = true;
    this.group.add(plate);
  }

  private buildStanchion(baseX: number): void {
    const mat = new MeshStandardMaterial({ color: 0x14161c, roughness: 0.5, metalness: 0.55 });
    const post = new Mesh(new CylinderGeometry(0.11, 0.15, 3.9, 16), mat);
    post.position.set(baseX + this.side * 1.5, 1.95, 0);
    post.castShadow = true;
    this.group.add(post);

    const arm = new Mesh(new BoxGeometry(1.9, 0.16, 0.3), mat);
    arm.position.set(baseX + this.side * 0.72, 3.55, 0);
    arm.castShadow = true;
    this.group.add(arm);

    const padMat = new MeshStandardMaterial({ color: 0x0e1015, roughness: 0.92, metalness: 0 });
    const pad = new Mesh(new BoxGeometry(0.62, 2.3, 1.5), padMat);
    pad.position.set(baseX + this.side * 1.5, 1.15, 0);
    pad.castShadow = true;
    pad.receiveShadow = true;
    this.group.add(pad);
  }

  private buildNet(engine: Engine): void {
    const S = HOOP.net.strands;
    const R = HOOP.net.segments;
    const nodes: NetNode[] = [];
    for (let r = 0; r <= R; r++) {
      const t = r / R;
      const radius = HOOP.rimRadius * (1 - t * (1 - HOOP.net.bottomRadiusScale));
      const y = HOOP.rimHeight - t * HOOP.net.length;
      for (let s = 0; s < S; s++) {
        const a = (s / S) * Math.PI * 2;
        const p = new Vector3(
          this.rimCentre.x + Math.cos(a) * radius,
          y,
          this.rimCentre.z + Math.sin(a) * radius,
        );
        nodes.push({ pos: p, prev: p.clone(), pinned: r === 0 });
      }
    }
    this.nodes = nodes;

    const links: NetLink[] = [];
    const idx = (r: number, s: number) => r * S + ((s + S) % S);
    for (let r = 0; r <= R; r++) {
      for (let s = 0; s < S; s++) {
        if (r < R) {
          const a = idx(r, s);
          const b = idx(r + 1, s);
          links.push({ a, b, rest: nodes[a].pos.distanceTo(nodes[b].pos), stiffness: 0.94 });
          // Diagonal cross-links give the net its classic diamond mesh.
          const c = idx(r + 1, s + 1);
          links.push({ a, b: c, rest: nodes[a].pos.distanceTo(nodes[c].pos), stiffness: 0.72 });
        }
        if (r > 0) {
          const a = idx(r, s);
          const b = idx(r, s + 1);
          links.push({ a, b, rest: nodes[a].pos.distanceTo(nodes[b].pos), stiffness: 0.52 });
        }
      }
    }
    this.links = links;

    const pos = new Float32Array(links.length * 6);
    const geo = new BufferGeometry();
    this.netPositions = new Float32BufferAttribute(pos, 3);
    this.netPositions.setUsage(35048 /* DynamicDrawUsage */);
    geo.setAttribute('position', this.netPositions);
    this.netLines = new LineSegments(
      geo,
      new LineBasicMaterial({ color: 0xf4f2ec, transparent: true, opacity: 0.95 }),
    );
    this.netLines.frustumCulled = false;
    this.group.add(this.netLines);
    this.syncNetGeometry();
    void engine;
  }

  /** Verlet step for the net, plus ball coupling. */
  stepNet(dt: number, iterations: number, ball: { position: Vector3; radius: number } | null): void {
    const g = -9.80665;
    const damping = 0.982;
    for (const n of this.nodes) {
      if (n.pinned) continue;
      const vx = (n.pos.x - n.prev.x) * damping;
      const vy = (n.pos.y - n.prev.y) * damping;
      const vz = (n.pos.z - n.prev.z) * damping;
      n.prev.copy(n.pos);
      n.pos.x += vx;
      n.pos.y += vy + g * dt * dt;
      n.pos.z += vz;
    }

    for (let it = 0; it < iterations; it++) {
      for (const l of this.links) {
        const a = this.nodes[l.a];
        const b = this.nodes[l.b];
        const dx = b.pos.x - a.pos.x;
        const dy = b.pos.y - a.pos.y;
        const dz = b.pos.z - a.pos.z;
        const d = Math.hypot(dx, dy, dz) || 1e-6;
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

      if (ball) {
        const r = ball.radius;
        for (const n of this.nodes) {
          if (n.pinned) continue;
          const dx = n.pos.x - ball.position.x;
          const dy = n.pos.y - ball.position.y;
          const dz = n.pos.z - ball.position.z;
          const d = Math.hypot(dx, dy, dz);
          if (d < r && d > 1e-5) {
            const k = (r - d) / d;
            n.pos.x += dx * k;
            n.pos.y += dy * k;
            n.pos.z += dz * k;
          }
        }
      }
    }
    this.syncNetGeometry();
  }

  private syncNetGeometry(): void {
    const arr = this.netPositions.array as Float32Array;
    let o = 0;
    for (const l of this.links) {
      const a = this.nodes[l.a].pos;
      const b = this.nodes[l.b].pos;
      arr[o++] = a.x;
      arr[o++] = a.y;
      arr[o++] = a.z;
      arr[o++] = b.x;
      arr[o++] = b.y;
      arr[o++] = b.z;
    }
    this.netPositions.needsUpdate = true;
  }

  /** Impulse applied to the net when the ball rips through. */
  punchNet(dir: Vector3, strength: number): void {
    for (const n of this.nodes) {
      if (n.pinned) continue;
      const k = clamp(strength * 0.02, 0, 0.08);
      n.prev.x -= dir.x * k;
      n.prev.y -= dir.y * k;
      n.prev.z -= dir.z * k;
    }
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
  }

  basketFor(side: 1 | -1): Basket {
    return this.baskets.find((b) => b.side === side)!;
  }

  simulate(step: number, engine: Engine): void {
    const ballSys = engine.get<{ ballState?: { position: Vector3 } }>('ball');
    const ball = ballSys?.ballState
      ? { position: ballSys.ballState.position, radius: BALL.radius }
      : null;
    for (const b of this.baskets) b.stepNet(step, engine.quality.netIterations, ball);
  }
}
