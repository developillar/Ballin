/**
 * The bowl: risers, seating, courtside tables, LED ribbon boards, the
 * jumbotron, tunnel mouths and the rafters. Everything is instanced or baked so
 * the whole arena costs a handful of draw calls.
 *
 * Owned by the arena agent.
 */

import {
  BoxGeometry,
  Color,
  Group,
  InstancedMesh,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Object3D,
  RingGeometry,
  Vector3,
} from 'three';
import type { Engine, System } from '../core/Engine';
import { COURT } from '../core/Constants';
import { makeRng, randRange } from '../core/MathX';

export class ArenaSystem implements System {
  readonly name = 'arena';
  readonly order = 14;

  group = new Group();
  crowd: InstancedMesh | null = null;

  private dummy = new Object3D();
  private crowdPhase: Float32Array = new Float32Array(0);
  private crowdBase: Float32Array = new Float32Array(0);

  init(engine: Engine): void {
    this.group.name = 'arena';
    engine.scene.add(this.group);
    this.buildBowl();
    this.buildCrowd(engine);
    this.buildRibbon();
  }

  private buildBowl(): void {
    const concrete = new MeshStandardMaterial({ color: 0x14161d, roughness: 0.94, metalness: 0 });
    const innerX = COURT.halfLength + COURT.apronX;
    const innerZ = COURT.halfWidth + COURT.apronZ;

    // Stepped risers rising away from the floor. Each row is four separate
    // bars rather than one solid box, so the bowl is genuinely hollow and the
    // camera can sit inside it.
    const rows = 22;
    const depth = 0.86;
    for (let i = 0; i < rows; i++) {
      const y = i * 0.44 + 0.22;
      const ox = innerX + 1.2 + i * depth;
      const oz = innerZ + 1.2 + i * depth;
      const h = 0.46;

      for (const sz of [1, -1]) {
        const bar = new Mesh(new BoxGeometry(ox * 2 + depth * 2, h, depth), concrete);
        bar.position.set(0, y, sz * oz);
        this.group.add(bar);
      }
      for (const sx of [1, -1]) {
        const bar = new Mesh(new BoxGeometry(depth, h, oz * 2 - depth * 2), concrete);
        bar.position.set(sx * ox, y, 0);
        this.group.add(bar);
      }
    }

    // Dark ceiling so the bowl reads as enclosed. Single-sided and facing down
    // keeps it out of the way when the camera flies above it.
    const ceilGeo = new BoxGeometry(innerX * 2 + 46, 0.6, innerZ * 2 + 46);
    const ceil = new Mesh(
      ceilGeo,
      new MeshStandardMaterial({ color: 0x080a0f, roughness: 1, metalness: 0 }),
    );
    ceil.position.y = 21;
    this.group.add(ceil);
  }

  private buildCrowd(engine: Engine): void {
    const count = engine.quality.crowdCount;
    if (count <= 0) return;
    const rng = makeRng(4242);

    // A crowd member is a cheap capsule-ish box pair; at phone scale in the
    // background this reads as a person once colour variance and motion land.
    const geo = new BoxGeometry(0.42, 0.86, 0.34);
    const mat = new MeshStandardMaterial({ roughness: 0.86, metalness: 0, vertexColors: false });
    const mesh = new InstancedMesh(geo, mat, count);
    mesh.frustumCulled = true;
    mesh.castShadow = false;
    mesh.receiveShadow = false;

    const innerX = COURT.halfLength + COURT.apronX + 2.4;
    const innerZ = COURT.halfWidth + COURT.apronZ + 2.4;
    this.crowdPhase = new Float32Array(count);
    this.crowdBase = new Float32Array(count * 3);
    const colour = new Color();
    const m = new Matrix4();

    for (let i = 0; i < count; i++) {
      const row = Math.floor(rng() * 20);
      const y = 0.9 + row * 0.44;
      const ox = innerX + row * 0.86;
      const oz = innerZ + row * 0.86;
      const side = Math.floor(rng() * 4);
      let x: number;
      let z: number;
      if (side === 0) {
        x = randRange(rng, -ox, ox);
        z = oz;
      } else if (side === 1) {
        x = randRange(rng, -ox, ox);
        z = -oz;
      } else if (side === 2) {
        x = ox;
        z = randRange(rng, -oz, oz);
      } else {
        x = -ox;
        z = randRange(rng, -oz, oz);
      }
      this.crowdBase[i * 3] = x;
      this.crowdBase[i * 3 + 1] = y;
      this.crowdBase[i * 3 + 2] = z;
      this.crowdPhase[i] = rng() * Math.PI * 2;

      this.dummy.position.set(x, y, z);
      this.dummy.rotation.y = Math.atan2(-x, -z) + randRange(rng, -0.4, 0.4);
      this.dummy.scale.setScalar(randRange(rng, 0.86, 1.12));
      this.dummy.updateMatrix();
      mesh.setMatrixAt(i, this.dummy.matrix);

      // Muted apparel with occasional team colours.
      const r = rng();
      if (r < 0.16) colour.setHSL(0.06, 0.72, 0.42);
      else if (r < 0.28) colour.setHSL(0.58, 0.5, 0.34);
      else colour.setHSL(rng(), 0.12 + rng() * 0.16, 0.14 + rng() * 0.22);
      mesh.setColorAt(i, colour);
      void m;
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    this.crowd = mesh;
    this.group.add(mesh);
  }

  private buildRibbon(): void {
    // LED ribbon around the lip of the lower bowl — a strong emissive accent
    // that sells the arena at a glance.
    const geo = new RingGeometry(0.1, 0.1, 4);
    void geo;
    const innerX = COURT.halfLength + COURT.apronX + 1.6;
    const innerZ = COURT.halfWidth + COURT.apronZ + 1.6;
    const mat = new MeshBasicMaterial({ color: 0x2f6bff, toneMapped: false });
    const bars: Array<[number, number, number, number, number]> = [
      [0, 3.2, innerZ, innerX * 2, 0.34],
      [0, 3.2, -innerZ, innerX * 2, 0.34],
    ];
    for (const [x, y, z, w, h] of bars) {
      const bar = new Mesh(new BoxGeometry(w, h, 0.12), mat);
      bar.position.set(x, y, z);
      this.group.add(bar);
    }
    for (const sx of [1, -1]) {
      const bar = new Mesh(new BoxGeometry(0.12, 0.34, innerZ * 2), mat);
      bar.position.set(sx * innerX, 3.2, 0);
      this.group.add(bar);
    }
  }

  update(_dt: number, _alpha: number, engine: Engine): void {
    if (!this.crowd || !engine.quality.crowdAnimated) return;
    const t = engine.elapsed;
    const n = this.crowd.count;
    // Idle sway; the crowd agent replaces this with reaction states.
    for (let i = 0; i < n; i += 1) {
      const p = this.crowdPhase[i];
      const x = this.crowdBase[i * 3];
      const y = this.crowdBase[i * 3 + 1];
      const z = this.crowdBase[i * 3 + 2];
      this.dummy.position.set(x, y + Math.sin(t * 1.6 + p) * 0.022, z);
      this.dummy.rotation.set(0, Math.atan2(-x, -z) + Math.sin(t * 0.7 + p) * 0.06, 0);
      this.dummy.updateMatrix();
      this.crowd.setMatrixAt(i, this.dummy.matrix);
    }
    this.crowd.instanceMatrix.needsUpdate = true;
  }

  /** Focal point for depth-of-field / camera framing helpers. */
  centre(): Vector3 {
    return new Vector3(0, 2, 0);
  }
}
