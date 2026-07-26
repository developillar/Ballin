/**
 * Ball rigid body: gravity, quadratic drag, Magnus lift from spin, and
 * continuous contact against the floor, the ring torus, the backboard and the
 * players. Runs on the engine's fixed 240 Hz step; rendering interpolates.
 *
 * Owned by the physics agent.
 */

import {
  Group,
  Mesh,
  MeshPhysicalMaterial,
  Quaternion,
  SphereGeometry,
  Vector3,
} from 'three';
import type { Engine, System } from '../core/Engine';
import { BALL, COURT, HOOP, PHYSICS } from '../core/Constants';
import { clamp } from '../core/MathX';
import type { HoopSystem, Basket } from '../world/Hoop';

export type BallOwner = { kind: 'free' } | { kind: 'held'; player: number } | { kind: 'shot'; by: number };

export interface BallState {
  position: Vector3;
  velocity: Vector3;
  spin: Vector3;
  orientation: Quaternion;
  owner: BallOwner;
  /** Set while the ball is inside a rim's scoring cylinder, moving down. */
  throughRim: boolean;
  resting: boolean;
}

const _tmp = new Vector3();
const _rel = new Vector3();
const _n = new Vector3();
const _vt = new Vector3();
const _dq = new Quaternion();

export class BallSystem implements System {
  readonly name = 'ball';
  readonly order = 20;

  group = new Group();
  mesh!: Mesh;
  ballState: BallState = {
    position: new Vector3(0, 1.4, 0),
    velocity: new Vector3(),
    spin: new Vector3(),
    orientation: new Quaternion(),
    owner: { kind: 'free' },
    throughRim: false,
    resting: false,
  };

  /** Where the shadow-casting key light should aim. */
  focusPoint = new Vector3();

  private prevPosition = new Vector3();
  private hoops: HoopSystem | null = null;
  private aboveRim = new Map<Basket, boolean>();

  init(engine: Engine): void {
    this.group.name = 'ball';
    engine.scene.add(this.group);
    this.hoops = engine.get<HoopSystem>('hoop') ?? null;

    // Placeholder material — the ball agent replaces this with the full
    // pebbled-leather PBR set (albedo / normal / roughness / AO).
    const mat = new MeshPhysicalMaterial({
      color: 0xc4581f,
      roughness: 0.78,
      metalness: 0,
      clearcoat: 0.18,
      clearcoatRoughness: 0.6,
      envMapIntensity: 0.9,
    });
    this.mesh = new Mesh(new SphereGeometry(BALL.radius, 64, 48), mat);
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    this.group.add(this.mesh);

    this.prevPosition.copy(this.ballState.position);
  }

  /** Launch the ball. `spin` is in rad/s; backspin is negative around +Z-ish. */
  launch(from: Vector3, velocity: Vector3, spin: Vector3, by: number): void {
    const s = this.ballState;
    s.position.copy(from);
    s.velocity.copy(velocity);
    s.spin.copy(spin);
    s.owner = { kind: 'shot', by };
    s.resting = false;
    this.prevPosition.copy(from);
  }

  hold(player: number, at: Vector3): void {
    const s = this.ballState;
    s.owner = { kind: 'held', player };
    s.position.copy(at);
    s.velocity.set(0, 0, 0);
    s.resting = false;
  }

  simulate(step: number, engine: Engine): void {
    const s = this.ballState;
    if (s.owner.kind === 'held') {
      this.focusPoint.copy(s.position);
      return;
    }
    if (s.resting) {
      this.focusPoint.copy(s.position);
      return;
    }

    this.prevPosition.copy(s.position);

    // --- Aerodynamics ---------------------------------------------------
    const v = s.velocity;
    const speed = v.length();
    if (speed > 1e-4) {
      const area = Math.PI * BALL.radius * BALL.radius;
      const q = 0.5 * PHYSICS.airDensity * area;
      // Drag: -½ ρ Cd A |v| v / m
      const dragMag = (q * BALL.dragCoefficient * speed) / BALL.mass;
      _tmp.copy(v).multiplyScalar(-dragMag * step);
      v.add(_tmp);

      // Magnus: (ω × v), scaled — backspin lifts, sidespin curves.
      _tmp.copy(s.spin).cross(v);
      const magnusMag = (q * BALL.magnusCoefficient * BALL.radius) / BALL.mass;
      v.addScaledVector(_tmp, magnusMag * step);
    }

    v.y += PHYSICS.gravity * step;
    s.spin.multiplyScalar(Math.max(0, 1 - BALL.spinDecay * step));
    s.position.addScaledVector(v, step);

    // --- Contacts ---------------------------------------------------------
    this.collideFloor();
    if (this.hoops) {
      for (const basket of this.hoops.baskets) {
        this.collideRim(basket, engine);
        this.collideBoard(basket, engine);
        this.trackScoring(basket, engine);
      }
    }
    this.collideBounds();

    // --- Rolling / rest ----------------------------------------------------
    if (
      s.position.y <= BALL.radius + 1e-3 &&
      v.length() < PHYSICS.sleepLinear &&
      s.spin.length() < PHYSICS.sleepAngular
    ) {
      s.resting = true;
      v.set(0, 0, 0);
      s.spin.set(0, 0, 0);
      s.position.y = BALL.radius;
    }

    // --- Orientation from spin --------------------------------------------
    const w = s.spin;
    const wl = w.length();
    if (wl > 1e-5) {
      _dq.setFromAxisAngle(_tmp.copy(w).divideScalar(wl), wl * step);
      s.orientation.premultiply(_dq).normalize();
    }

    this.focusPoint.copy(s.position);
  }

  private collideFloor(): void {
    const s = this.ballState;
    if (s.position.y - BALL.radius > 0) return;
    s.position.y = BALL.radius;
    if (s.velocity.y >= 0) return;

    const vn = -s.velocity.y;
    s.velocity.y = vn * BALL.restitutionFloor;

    // Tangential friction couples to spin: a backspun ball checks up.
    _vt.set(s.velocity.x, 0, s.velocity.z);
    // Surface velocity at the contact point = v_t + ω × (-r ŷ)
    const surfX = _vt.x + s.spin.z * BALL.radius;
    const surfZ = _vt.z - s.spin.x * BALL.radius;
    const surfMag = Math.hypot(surfX, surfZ);
    if (surfMag > 1e-4) {
      const jn = vn * (1 + BALL.restitutionFloor) * BALL.mass;
      const maxFric = (BALL.frictionFloor * jn) / BALL.mass;
      const dv = Math.min(maxFric, surfMag);
      const nx = surfX / surfMag;
      const nz = surfZ / surfMag;
      s.velocity.x -= nx * dv;
      s.velocity.z -= nz * dv;
      // Equal-and-opposite torque on the shell.
      const k = dv / (BALL.inertiaFactor * BALL.radius);
      s.spin.z -= nx * k;
      s.spin.x += nz * k;
    }

    this.emitBounce(vn);
  }

  private emitBounce(speed: number): void {
    if (speed < 0.5) return;
    // The audio and VFX agents subscribe to these.
    (globalThis as unknown as { __bus?: { emit: Function } }).__bus?.emit?.('floorBounce', {
      speed,
      position: this.ballState.position.clone(),
    });
  }

  /** Torus collision against the ring: the defining contact in basketball. */
  private collideRim(basket: Basket, engine: Engine): void {
    const s = this.ballState;
    _rel.copy(s.position).sub(basket.rimCentre);
    const horiz = Math.hypot(_rel.x, _rel.z);
    if (horiz < 1e-6) return;
    // Nearest point on the ring's centre circle.
    const cx = (_rel.x / horiz) * HOOP.rimRadius;
    const cz = (_rel.z / horiz) * HOOP.rimRadius;
    _n.set(_rel.x - cx, _rel.y, _rel.z - cz);
    const d = _n.length();
    const minDist = BALL.radius + HOOP.rimTubeRadius;
    if (d >= minDist || d < 1e-6) return;

    _n.divideScalar(d);
    s.position.addScaledVector(_n, minDist - d);
    const vn = s.velocity.dot(_n);
    if (vn < 0) {
      s.velocity.addScaledVector(_n, -vn * (1 + BALL.restitutionRim));
      // Rim friction bleeds tangential speed and adds spin — how a ball
      // rattles around the cylinder instead of rocketing off.
      _vt.copy(s.velocity).addScaledVector(_n, -s.velocity.dot(_n));
      const tMag = _vt.length();
      if (tMag > 1e-4) {
        const dv = Math.min(BALL.frictionRim * -vn, tMag);
        s.velocity.addScaledVector(_vt.divideScalar(tMag), -dv);
        s.spin.multiplyScalar(0.86);
      }
      engine.bus.emit('rimContact', { speed: -vn, position: s.position.clone() });
    }
  }

  private collideBoard(basket: Basket, engine: Engine): void {
    const s = this.ballState;
    const halfW = HOOP.board.width / 2;
    const bottom = HOOP.board.bottomHeight;
    const top = bottom + HOOP.board.height;
    if (s.position.z < -halfW - BALL.radius || s.position.z > halfW + BALL.radius) return;
    if (s.position.y < bottom - BALL.radius || s.position.y > top + BALL.radius) return;

    const face = basket.boardCentre.x - basket.side * (HOOP.board.thickness / 2);
    const dist = (s.position.x - face) * -basket.side;
    if (dist > BALL.radius || dist < -BALL.radius * 2) return;

    _n.copy(basket.boardNormal);
    s.position.addScaledVector(_n, BALL.radius - dist);
    const vn = s.velocity.dot(_n);
    if (vn < 0) {
      s.velocity.addScaledVector(_n, -vn * (1 + BALL.restitutionBoard));
      _vt.copy(s.velocity).addScaledVector(_n, -s.velocity.dot(_n));
      const tMag = _vt.length();
      if (tMag > 1e-4) {
        const dv = Math.min(BALL.frictionBoard * -vn, tMag);
        s.velocity.addScaledVector(_vt.divideScalar(tMag), -dv);
      }
      s.spin.multiplyScalar(0.78);
      engine.bus.emit('boardContact', { speed: -vn, position: s.position.clone() });
    }
  }

  /** Detects a clean pass down through the ring's scoring cylinder. */
  private trackScoring(basket: Basket, engine: Engine): void {
    const s = this.ballState;
    _rel.copy(s.position).sub(basket.rimCentre);
    const inside = Math.hypot(_rel.x, _rel.z) < HOOP.rimRadius - BALL.radius * 0.35;
    const wasAbove = this.aboveRim.get(basket) ?? false;

    if (inside && _rel.y > 0.02) this.aboveRim.set(basket, true);
    if (wasAbove && inside && _rel.y < -0.03 && s.velocity.y < 0) {
      this.aboveRim.set(basket, false);
      s.throughRim = true;
      basket.punchNet(_tmp.copy(s.velocity).normalize(), s.velocity.length());
      engine.bus.emit('netSwish', { position: s.position.clone() });
    }
    if (!inside && _rel.y < -0.4) this.aboveRim.set(basket, false);
  }

  private collideBounds(): void {
    const s = this.ballState;
    const lx = COURT.halfLength + COURT.apronX;
    const lz = COURT.halfWidth + COURT.apronZ;
    for (const [axis, lim] of [['x', lx], ['z', lz]] as const) {
      const p = s.position[axis];
      if (p > lim - BALL.radius) {
        s.position[axis] = lim - BALL.radius;
        if (s.velocity[axis] > 0) s.velocity[axis] *= -0.42;
      } else if (p < -lim + BALL.radius) {
        s.position[axis] = -lim + BALL.radius;
        if (s.velocity[axis] < 0) s.velocity[axis] *= -0.42;
      }
    }
    if (s.position.y > 22) {
      s.position.y = 22;
      s.velocity.y = Math.min(0, s.velocity.y);
    }
  }

  update(_dt: number, alpha: number): void {
    const s = this.ballState;
    // Render-time interpolation between the last two fixed steps.
    this.mesh.position.lerpVectors(this.prevPosition, s.position, clamp(alpha, 0, 1));
    this.mesh.quaternion.copy(s.orientation);
  }
}
