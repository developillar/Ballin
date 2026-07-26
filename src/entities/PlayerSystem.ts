/**
 * Players: procedurally generated, skinned, animated athletes.
 *
 * This system owns bodies, not motion — every rig gets its own `Animator`
 * (see `src/anim/AnimatorTypes.ts`) and this file only ever states intent.
 * What it does own:
 *
 *  - **The roster.** Five archetypes — point guard through centre — each with a
 *    distinct `BodyShape`. Two teams draw from the same five, so a five-player
 *    frame always shows at least three visibly different builds while only five
 *    body meshes are ever generated. Geometry is shared between the two players
 *    of an archetype; skeletons, animators and materials are not.
 *  - **The bakes.** One tone-neutral skin atlas for the whole league (tone
 *    arrives as `material.color`), one kit atlas per player (jersey and shorts
 *    share it, so they share a draw call), one shoe strip per team, one hair
 *    strand mask.
 *  - **The stand-in possession.** Enough game state to make a capture show a
 *    real half-court set: a ball-handler in triple threat with the ball
 *    genuinely in his hands via the animator's ball-hold IK, a defender on him,
 *    and the other eight spaced and watching the ball.
 *
 * The contract other systems rely on is unchanged: `players[]` with a stable
 * index, `position` / `facing` / `handAnchor` / `stance` per player,
 * `create()` and `handPosition()`.
 *
 * Owned by the players agent.
 */

import {
  Group,
  Matrix4,
  Object3D,
  SkinnedMesh,
  Vector3,
  type BufferGeometry,
  type Material,
} from 'three';
import type { Engine, System } from '../core/Engine';
import { BALL, PLAYER, basketX } from '../core/Constants';
import { clamp01, dampAngle, makeRng } from '../core/MathX';
import { buildSkeleton, type BodyShape, type BuiltSkeleton } from './Skeleton';
import { buildBody, type BodyBuild, type HairStyle } from './BodyMesh';
import {
  makeHairMaterial,
  makeKitMaterial,
  makeShoeMaterial,
  makeSkinMaterial,
  setSweat,
  type SkinMaterial,
} from './playerMaterials';
import {
  bakeHairMask,
  bakeSkinAtlas,
  SKIN_TONES,
  type SkinMaps,
} from '../textures/skinTextures';
import {
  AWAY_NAMES,
  AWAY_NUMBERS,
  HOME_NAMES,
  HOME_NUMBERS,
  KitBaker,
  TEAM_KITS,
  bakeShoeStrip,
  type ClothMaps,
} from '../textures/jerseyTextures';
import { Animator } from '../anim/Animator';
import type { IkIntent, LocomotionIntent } from '../anim/AnimatorTypes';

export type PlayerStance =
  | 'idle'
  | 'run'
  | 'sprint'
  | 'dribble'
  | 'triple-threat'
  | 'shoot'
  | 'jump'
  | 'land'
  | 'dunk'
  | 'layup'
  | 'defend'
  | 'block'
  | 'celebrate';

export interface PlayerRig {
  index: number;
  team: number;
  root: Group;
  position: Vector3;
  velocity: Vector3;
  facing: number;
  stance: PlayerStance;
  /** World-space point where the ball sits when this player holds it. */
  handAnchor: Object3D;
  /** 0..1 how far into the current animation. */
  phase: number;
  height: number;
  /** Vertical offset from a jump. */
  jumpY: number;
  // --- extensions ---------------------------------------------------------
  /** Motion driver. One per player; never shared. */
  animator: Animator;
  skeleton: BuiltSkeleton;
  /** Squad number, for the HUD and for debugging. */
  number: number;
  dominantHand: 'left' | 'right';
  skin: SkinMaterial;
  meshes: SkinnedMesh[];
}

/** A build. Radii, limb ratios and hair all move together. */
interface Archetype {
  role: string;
  shape: BodyShape;
  hair: HairStyle;
  /** Index into SKIN_TONES. */
  tone: number;
  hairColour: number;
}

/**
 * Five builds. The guard is short, wiry and long-legged; the centre is a full
 * 0.24 m taller with 16% more limb mass and a broader yoke. Because the mesh is
 * generated from the field rather than scaled, that is a different silhouette,
 * not a bigger copy.
 */
const ARCHETYPES: readonly Archetype[] = [
  {
    role: 'PG',
    shape: { height: 1.88, build: 0.9, shoulders: 0.96, wingspan: 1.05, legRatio: 1.03 },
    hair: 'fade',
    tone: 3,
    hairColour: 0x181310,
  },
  {
    role: 'SG',
    shape: { height: 1.96, build: 0.97, shoulders: 1.0, wingspan: 1.07, legRatio: 1.0 },
    hair: 'headband',
    tone: 1,
    hairColour: 0x2b2018,
  },
  {
    role: 'SF',
    shape: { height: 2.03, build: 1.03, shoulders: 1.04, wingspan: 1.08, legRatio: 1.01 },
    hair: 'crop',
    tone: 4,
    hairColour: 0x120f0d,
  },
  {
    role: 'PF',
    shape: { height: 2.08, build: 1.1, shoulders: 1.07, wingspan: 1.07, legRatio: 0.98 },
    hair: 'afro',
    tone: 2,
    hairColour: 0x1c1512,
  },
  {
    role: 'C',
    shape: { height: 2.14, build: 1.17, shoulders: 1.1, wingspan: 1.06, legRatio: 0.97 },
    hair: 'bald',
    tone: 3,
    hairColour: 0x141010,
  },
];

/** A half-court set: offence spaced, defence between man and basket. */
const FORMATION: ReadonlyArray<readonly [number, number]> = [
  [-8.4, 0.4], // PG, top of the key with the ball
  [-7.1, -5.2], // SG, left wing
  [-7.4, 5.4], // SF, right wing
  [-3.6, 2.6], // PF, elbow
  [-2.1, -1.9], // C, low post
];

interface BallLike {
  ballState: {
    position: Vector3;
    owner: { kind: string };
    resting: boolean;
  };
  /**
   * The ball's render transform. It is written directly after `hold()` because
   * the ball system interpolates its mesh between the last two *simulated*
   * positions, and a held ball never simulates — so the mesh would otherwise
   * lerp between the hand and wherever the ball last came to rest.
   */
  mesh?: { position: Vector3 };
  hold(player: number, at: Vector3): void;
}

const _v = new Vector3();
const _ballPoint = new Vector3();
const IDENTITY = new Matrix4();

export class PlayerSystem implements System {
  readonly name = 'players';
  readonly order = 30;

  group = new Group();
  players: PlayerRig[] = [];

  /** Index of the player currently in possession, or -1. */
  ballHandler = -1;
  /** 0..1, rises over a possession and drives the sweat specular. */
  exertion = 0.18;

  /** Reported to the harness so the triangle budget is auditable. */
  stats = { trianglesPerPlayer: 0, players: 0, drawCallsPerPlayer: 0 };

  private skin: SkinMaps | null = null;
  private kits: ClothMaps[] = [];
  private shoeMaps: ClothMaps[] = [];
  private builds: (BodyBuild | null)[] = [];
  private materials: Material[] = [];
  private skinMaterials: SkinMaterial[] = [];
  private ball: BallLike | null = null;
  private detail: 0 | 1 | 2 = 1;
  private lookTarget = new Vector3();

  init(engine: Engine): void {
    this.group.name = 'players';
    engine.scene.add(this.group);
    this.ball = engine.get<BallLike>('ball') ?? null;
    this.detail = engine.quality.playerDetail;

    const aniso = engine.anisotropy;
    const q = engine.quality;
    // The skin atlas is shared by the entire league, so it can afford the
    // resolution; the per-player kit atlas cannot.
    this.skin = bakeSkinAtlas(Math.min(q.textureSize, 1024), aniso);
    const hairMask = bakeHairMask(256);

    const skinMats = SKIN_TONES.map((t) => makeSkinMaterial(this.skin!, t));
    this.skinMaterials = skinMats;
    this.materials.push(...skinMats);

    const kitSize = q.textureSize >= 2048 ? 768 : 512;
    const bakers = TEAM_KITS.map((kit) => new KitBaker(kit, kitSize, aniso));
    const shoeMats = TEAM_KITS.map((kit) => {
      const maps = bakeShoeStrip(kit, 256, aniso);
      this.shoeMaps.push(maps);
      const m = makeShoeMaterial(maps);
      this.materials.push(m);
      return m;
    });

    // Geometry is per archetype and shared by the two players who use it; the
    // skeleton, animator and kit material are always per player.
    const geoCache: (BodyBuild | null)[] = ARCHETYPES.map(() => null);
    let totalTris = 0;

    for (let team = 0; team < 2; team++) {
      const numbers = team === 0 ? HOME_NUMBERS : AWAY_NUMBERS;
      const names = team === 0 ? HOME_NAMES : AWAY_NAMES;
      const side = team === 0 ? 1 : -1;

      for (let slot = 0; slot < ARCHETYPES.length; slot++) {
        const arch = ARCHETYPES[slot];
        const seed = 1000 + team * 97 + slot * 13;
        const skeleton = buildSkeleton(arch.shape);

        let build = geoCache[slot];
        if (!build) {
          build = buildBody({
            skeleton,
            shape: arch.shape,
            detail: this.detail,
            hair: arch.hair,
            seed: 3300 + slot * 41,
          });
          geoCache[slot] = build;
          this.builds[slot] = build;
          totalTris += build.triangles;
        }

        const kitMaps = bakers[team].bake(numbers[slot], names[slot]);
        this.kits.push(kitMaps);
        const kitMat = makeKitMaterial(kitMaps, TEAM_KITS[team]);
        this.materials.push(kitMat);
        // A tone offset per team keeps two players of the same archetype from
        // being obvious clones.
        const tone = (arch.tone + team * 2) % SKIN_TONES.length;
        const hairMat = build.hair
          ? makeHairMaterial(hairMask, team === 0 ? arch.hairColour : arch.hairColour + 0x0a0806)
          : null;
        if (hairMat) this.materials.push(hairMat);

        // Half-court set, mirrored for the defence.
        const [fx, fz] = FORMATION[slot];
        const at =
          team === 0
            ? new Vector3(basketX(1) + fx, 0, fz)
            : new Vector3(basketX(1) + fx + 1.6, 0, fz * 0.92 - 0.55);

        const rig = this.spawn({
          team,
          at,
          skeleton,
          build,
          skin: skinMats[tone],
          kitMat,
          shoeMat: shoeMats[team],
          hairMat,
          number: numbers[slot],
          seed,
          facing: side > 0 ? Math.PI / 2 : -Math.PI / 2,
        });
        rig.stance = team === 0 ? (slot === 0 ? 'triple-threat' : 'idle') : 'defend';
      }
    }

    this.ballHandler = 0;
    this.stats.players = this.players.length;
    this.stats.trianglesPerPlayer = Math.round(totalTris / ARCHETYPES.length);
    this.stats.drawCallsPerPlayer = 3 + (this.builds[0]?.hair ? 1 : 0);

    // Nothing left references the raw atlas rows; the maps stay alive on the
    // materials.
    void hairMask;
  }

  /**
   * Public creation hook kept from the original contract. Without an explicit
   * archetype it picks one by roster position so callers that only know
   * "team + spot" still get a properly built athlete.
   */
  create(team: number, at: Vector3): PlayerRig {
    const slot = this.players.length % ARCHETYPES.length;
    const arch = ARCHETYPES[slot];
    const build = this.builds[slot];
    const skeleton = buildSkeleton(arch.shape);
    if (!build || !this.skin) {
      // Called before init: hand back a transform-only rig so gameplay code
      // that just wants an anchor still works.
      const root = new Group();
      root.position.copy(at);
      const handAnchor = new Object3D();
      handAnchor.position.set(0.32, PLAYER.height * 0.82, 0.22);
      root.add(handAnchor);
      this.group.add(root);
      const rig: PlayerRig = {
        index: this.players.length,
        team,
        root,
        position: at.clone(),
        velocity: new Vector3(),
        facing: 0,
        stance: 'idle',
        handAnchor,
        phase: 0,
        height: arch.shape.height,
        jumpY: 0,
        animator: new Animator(skeleton, {
          seed: this.players.length,
          height: arch.shape.height,
          dominantHand: 'right',
        }),
        skeleton,
        number: 0,
        dominantHand: 'right',
        skin: this.skinMaterials[0],
        meshes: [],
      };
      this.players.push(rig);
      return rig;
    }
    return this.spawn({
      team,
      at,
      skeleton,
      build,
      skin: this.skinMaterials[arch.tone],
      kitMat: this.materials.find((m) => m.name === `kit${team}`) ?? this.materials[0],
      shoeMat: this.materials[0],
      hairMat: null,
      number: 0,
      seed: 7000 + this.players.length,
      facing: 0,
    });
  }

  private spawn(o: {
    team: number;
    at: Vector3;
    skeleton: BuiltSkeleton;
    build: BodyBuild;
    skin: SkinMaterial;
    kitMat: Material;
    shoeMat: Material;
    hairMat: Material | null;
    number: number;
    seed: number;
    facing: number;
  }): PlayerRig {
    const root = new Group();
    root.position.copy(o.at);
    root.rotation.y = o.facing;
    this.group.add(root);

    // Bones and skinned meshes must be siblings: with `AttachedBindMode` and an
    // identity bind matrix the skinning resolves in the rig's own space, so the
    // group transform moves the player without being applied twice.
    root.add(o.skeleton.bones[0]);
    o.skeleton.bones[0].updateMatrixWorld(true);

    const meshes: SkinnedMesh[] = [];
    const attach = (geo: BufferGeometry, mat: Material, name: string): void => {
      const mesh = new SkinnedMesh(geo, mat);
      mesh.name = name;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      root.add(mesh);
      mesh.bind(o.skeleton.skeleton, IDENTITY);
      meshes.push(mesh);
    };
    attach(o.build.body, o.skin, 'skin');
    attach(o.build.kit, o.kitMat, 'kit');
    attach(o.build.shoes, o.shoeMat, 'shoes');
    if (o.build.hair && o.hairMat) attach(o.build.hair, o.hairMat, 'hair');

    const rng = makeRng(o.seed);
    const dominantHand: 'left' | 'right' = rng() < 0.12 ? 'left' : 'right';

    // The ball anchor rides the shooting hand, so `handPosition()` reports
    // where the ball actually is rather than where a fixed offset guesses.
    const handAnchor = new Object3D();
    handAnchor.position.set(0, -0.075 * o.skeleton.height, 0.045 * o.skeleton.height);
    o.skeleton.byName[dominantHand === 'left' ? 'handL' : 'handR'].add(handAnchor);

    const rig: PlayerRig = {
      index: this.players.length,
      team: o.team,
      root,
      position: o.at.clone(),
      velocity: new Vector3(),
      facing: o.facing,
      stance: 'idle',
      handAnchor,
      phase: 0,
      height: o.skeleton.height,
      jumpY: 0,
      animator: new Animator(o.skeleton, {
        seed: o.seed,
        height: o.skeleton.height,
        dominantHand,
      }),
      skeleton: o.skeleton,
      number: o.number,
      dominantHand,
      skin: o.skin,
      meshes,
    };
    this.players.push(rig);
    return rig;
  }

  update(dt: number, _alpha: number, engine: Engine): void {
    const ballState = this.ball?.ballState;
    // Where the possession is looking. Everyone tracks it; that alone stops a
    // roster reading as ten mannequins facing the same way.
    if (ballState) this.lookTarget.copy(ballState.position);
    else this.lookTarget.set(basketX(1), 3.05, 0);

    // Sweat builds through a possession and is the difference between a fresh
    // and a fourth-quarter frame.
    this.exertion = clamp01(this.exertion + dt * 0.012);
    setSweat(this.skinMaterials, 0.15 + this.exertion * 0.85);

    const handler = this.players[this.ballHandler];
    if (handler) this.resolveBallPoint(handler, _ballPoint);

    for (const p of this.players) {
      p.phase += dt;

      const target = Math.atan2(p.velocity.x, p.velocity.z);
      if (p.velocity.lengthSq() > 0.04) p.facing = dampAngle(p.facing, target, PLAYER.turnRate, dt);
      p.root.position.set(p.position.x, p.position.y + p.jumpY, p.position.z);
      p.root.rotation.y = p.facing;
      // World matrices have to be current before the animator's IK runs, since
      // every target it is handed is in world space.
      p.root.updateMatrixWorld(true);

      const speed = p.velocity.length();
      const loco: LocomotionIntent = {
        speed,
        driftAngle: 0,
        defending: p.stance === 'defend',
        dribbling: p.stance === 'dribble',
        airborne: p.jumpY > 0.02 ? 1 : 0,
        lean: 0,
        fatigue: this.exertion * 0.6,
      };

      const holding = p.index === this.ballHandler && !!handler;
      const ik: IkIntent = {
        ball: holding ? { active: true, centre: _ballPoint, radius: BALL.radius } : null,
        lookAt: {
          active: true,
          target: this.lookTarget,
          weight: holding ? 0.35 : 0.85,
        },
        floorY: 0,
        plantFeet: p.jumpY < 0.02,
      };

      p.animator.setLocomotion(loco);
      p.animator.setIk(ik);
      p.animator.update(dt);
    }

    // Only take the ball when it is genuinely loose and settled — never yank it
    // out of a live shot.
    if (handler && ballState && ballState.owner.kind !== 'shot') {
      this.ball?.hold(handler.index, _ballPoint);
      this.ball?.mesh?.position.copy(_ballPoint);
    }

    void engine;
  }

  /**
   * Where the ball sits in a triple-threat hold: out from the chest, off the
   * dominant side, at a height that scales with the player. The animator drives
   * both hands onto this point, so the contact is exact rather than eyeballed.
   */
  private resolveBallPoint(p: PlayerRig, out: Vector3): Vector3 {
    const s = p.dominantHand === 'left' ? -1 : 1;
    const cos = Math.cos(p.facing);
    const sin = Math.sin(p.facing);
    // Forward is +Z rotated by facing; right is +X rotated by facing.
    const fx = sin;
    const fz = cos;
    const rx = cos;
    const rz = -sin;
    // Far enough off the chest that the elbows stay outside the jersey — a
    // ball tucked against the sternum drives the upper arm straight through
    // the cloth, and no amount of skinning hides that.
    out.set(
      p.position.x + fx * 0.38 + rx * 0.16 * s,
      p.position.y + p.jumpY + p.height * 0.615,
      p.position.z + fz * 0.38 + rz * 0.16 * s,
    );
    return out;
  }

  /** World position of a player's ball-carrying hand. */
  handPosition(p: PlayerRig, out = new Vector3()): Vector3 {
    p.handAnchor.updateWorldMatrix(true, false);
    return out.setFromMatrixPosition(p.handAnchor.matrixWorld);
  }

  /** Metres from a player to a world point, ignoring height. */
  distanceTo(p: PlayerRig, point: Vector3): number {
    _v.set(point.x - p.position.x, 0, point.z - p.position.z);
    return _v.length();
  }

  dispose(): void {
    for (const m of this.materials) m.dispose();
    for (const b of this.builds) {
      if (!b) continue;
      b.body.dispose();
      b.kit.dispose();
      b.shoes.dispose();
      b.hair?.dispose();
    }
    for (const k of this.kits) k.dispose();
    for (const s of this.shoeMaps) s.dispose();
    this.skin?.dispose();
    this.players = [];
  }
}
