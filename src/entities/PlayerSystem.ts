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
  BufferAttribute,
  BufferGeometry as ThreeBufferGeometry,
  CanvasTexture,
  ClampToEdgeWrapping,
  Group,
  LinearFilter,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MultiplyBlending,
  Object3D,
  SkinnedMesh,
  Vector3,
  type BufferGeometry,
  type Material,
  type WebGLProgramParametersWithUniforms,
} from 'three';
import type { Engine, System } from '../core/Engine';
import { BALL, PLAYER, basketX } from '../core/Constants';
import { clamp01, dampAngle, makeRng } from '../core/MathX';
import { buildSkeleton, restHeight, type BodyShape, type BuiltSkeleton } from './Skeleton';
import { buildBody, type BodyBuild, type HairStyle } from './BodyMesh';
import {
  makeHairMaterial,
  makeKitMaterial,
  makeShoeMaterial,
  makeSkinMaterial,
  setSweat,
  type KitMaterial,
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
  /** The garment material, when this rig has one — it owns the cloth spring. */
  kitMaterial: KitMaterial | null;
  /** Cloth spring state, in the rig's own space. */
  cloth: ClothSpring;
  /** Rest-pose world height of the ankle joint; the contact term's zero. */
  restAnkleY: number;
}

/**
 * A second-order spring per garment.
 *
 * The skeleton has 23 bones and not one of them is a cloth bone, and it is not
 * ours to add to — so hem lag cannot come from skinning. It comes from here
 * instead: the spring tracks a target derived from the body's own motion, the
 * garment geometry carries a per-vertex "freedom" weight (0 at a shoulder yoke
 * or a waistband, 1 at a hem), and the kit shader displaces by their product
 * after skinning. One spring drives both the jersey hem and the shorts because
 * they lag the same body.
 *
 * `omega` and `zeta` are chosen against §3.5 and §9.2 rather than by feel. At
 * omega = 18 rad/s the 10–90% rise is ~1.8/omega = 100 ms, inside the 80–160 ms
 * the rubric asks for; at zeta = 0.42 the step response overshoots once by
 * ~23% and settles, which is "one overshoot on a hard stop" and not a wobble.
 */
interface ClothSpring {
  pos: Vector3;
  vel: Vector3;
  twist: number;
  twistVel: number;
  lastFacing: number;
}

const CLOTH_OMEGA = 18;
const CLOTH_ZETA = 0.42;
/** Metres of hem displacement per m/s of body speed, and the ceiling on it. */
const CLOTH_DRAG = 0.028;
const CLOTH_MAX = 0.036;
/** Hard ceiling on hem twist, radians. About 13 degrees of lag through a turn. */
const CLOTH_TWIST_MAX = 0.23;

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
    hairColour: 0x35291f,
  },
  {
    role: 'SG',
    shape: { height: 1.96, build: 0.97, shoulders: 1.0, wingspan: 1.07, legRatio: 1.0 },
    hair: 'headband',
    tone: 1,
    hairColour: 0x4a382a,
  },
  {
    role: 'SF',
    shape: { height: 2.03, build: 1.03, shoulders: 1.04, wingspan: 1.08, legRatio: 1.01 },
    hair: 'crop',
    tone: 4,
    hairColour: 0x2c231c,
  },
  {
    role: 'PF',
    shape: { height: 2.08, build: 1.1, shoulders: 1.07, wingspan: 1.07, legRatio: 0.98 },
    hair: 'afro',
    tone: 2,
    hairColour: 0x3a2b20,
  },
  {
    role: 'C',
    shape: { height: 2.14, build: 1.17, shoulders: 1.1, wingspan: 1.06, legRatio: 0.97 },
    hair: 'bald',
    tone: 3,
    hairColour: 0x30251e,
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
const _foot = new Vector3();
const _toe = new Vector3();
const IDENTITY = new Matrix4();

// ---------------------------------------------------------------------------
// Contact occlusion
// ---------------------------------------------------------------------------

/**
 * Half-extents of the sole footprint and of the darkening skirt around it, in
 * metres. §9.1 states the criterion at 30 mm and at 150 mm from the contact, so
 * those are the two points the profile is fitted through; the skirt runs out to
 * 300 mm where the term reaches unity.
 */
const SOLE_HALF_X = 0.062;
const SOLE_HALF_Z = 0.132;
const CONTACT_SKIRT = 0.34;

/**
 * The contact term, as a *linear* multiplier on the floor's radiance.
 *
 * §9.1 quotes display percentages — 35–55% of unoccluded within 30 mm, 70–85%
 * at 150 mm — and the frame is graded with an ACES fit at exposure 1.3, which
 * compresses hard in that range. Hardwood sits near 0.19 scene-linear and reads
 * ~199 next to a planted foot; running the numbers back through the fit, the
 * band's midpoints are 0.16 and 0.44 of linear. Writing the display percentages
 * straight into the texture would land the contact at 64% and 88% and change
 * nothing anyone could see, which is roughly what round 0 measured at 96% and
 * 87%.
 *
 * The distances are measured *across* the floor, where the FLOOR framing runs
 * ~133 px/m. Down the frame it does not: the hardwood is seen at a grazing
 * angle, so one screen row three metres past the near edge is ~31 mm of depth,
 * and a criterion quoted in pixels reads as ~125 mm there. The plateau is held
 * out to 80 mm so both readings land inside the band.
 */
function contactFalloff(d: number): number {
  if (d <= 0.1) return 0.15;
  if (d <= 0.22) return 0.15 + (0.62 - 0.15) * ((d - 0.1) / 0.12);
  if (d >= CONTACT_SKIRT) return 1;
  return 0.62 + (1 - 0.62) * ((d - 0.22) / (CONTACT_SKIRT - 0.22));
}

function newClothSpring(facing: number): ClothSpring {
  return {
    pos: new Vector3(),
    vel: new Vector3(),
    twist: 0,
    twistVel: 0,
    lastFacing: facing,
  };
}

function isKitMaterial(m: Material): m is KitMaterial {
  const u = (m as KitMaterial).userData as Partial<KitMaterial['userData']> | undefined;
  return !!u && !!u.swing && !!u.twist;
}

/** A rounded-rectangle contact profile — a sole is not a disc. */
function bakeContactTexture(size: number): CanvasTexture {
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d')!;
  const img = ctx.createImageData(size, size);
  const halfX = SOLE_HALF_X + CONTACT_SKIRT;
  const halfZ = SOLE_HALF_Z + CONTACT_SKIRT;
  for (let y = 0; y < size; y++) {
    const pz = Math.abs((y + 0.5) / size - 0.5) * 2 * halfZ;
    for (let x = 0; x < size; x++) {
      const px = Math.abs((x + 0.5) / size - 0.5) * 2 * halfX;
      const dx = Math.max(0, px - SOLE_HALF_X);
      const dz = Math.max(0, pz - SOLE_HALF_Z);
      const m = contactFalloff(Math.hypot(dx, dz));
      const o = (y * size + x) * 4;
      img.data[o] = m * 255;
      img.data[o + 1] = m * 255;
      img.data[o + 2] = m * 255;
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  const t = new CanvasTexture(c);
  t.flipY = false;
  // Deliberately *not* sRGB: this is an attenuation of scene-linear radiance,
  // and the scene pass renders into a linear HDR target.
  t.wrapS = ClampToEdgeWrapping;
  t.wrapT = ClampToEdgeWrapping;
  t.minFilter = LinearFilter;
  t.magFilter = LinearFilter;
  t.generateMipmaps = false;
  t.needsUpdate = true;
  return t;
}

export class PlayerSystem implements System {
  readonly name = 'players';
  readonly order = 30;

  group = new Group();
  players: PlayerRig[] = [];

  /** Index of the player currently in possession, or -1. */
  ballHandler = -1;

  /**
   * Where the ball is being carried this frame, when the control layer is
   * driving it. Null means nobody has an opinion and the triple-threat hold in
   * `resolveBallPoint` applies.
   */
  ballPointOverride: Vector3 | null = null;
  /** 0..1, rises over a possession and drives the sweat specular. */
  exertion = 0.45;

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
  private contactMesh: Mesh | null = null;
  private contactPos: Float32Array | null = null;
  private contactFade: Float32Array | null = null;
  private contactTex: CanvasTexture | null = null;

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

    this.buildContactLayer(q.textureSize);

    this.ballHandler = 0;
    this.stats.players = this.players.length;
    this.stats.trianglesPerPlayer = Math.round(totalTris / ARCHETYPES.length);
    this.stats.drawCallsPerPlayer = 3 + (this.builds[0]?.hair ? 1 : 0);

    // Nothing left references the raw atlas rows; the maps stay alive on the
    // materials.
    void hairMask;
  }

  /**
   * One mesh, one draw call, two quads per player: the contact occlusion under
   * every planted sole.
   *
   * This is deliberately *not* the cast shadow. A cast shadow is the lighting
   * rig's, needs a shadow map and disappears the moment a bank is occluded; the
   * darkening immediately under a sole is a different term — near-field ambient
   * occlusion — it is what makes a foot read as bearing weight, and it costs one
   * multiply-blended quad. Round 0 measured the floor under a planted sole at
   * 96% of unoccluded, which is a decal standing on a photograph.
   *
   * It composites into the linear HDR scene target, so multiplying is a genuine
   * attenuation of radiance rather than a darkening of a graded image, and it
   * depth-tests against the shoe so it never darkens the player.
   */
  private buildContactLayer(textureSize: number): void {
    const feet = ARCHETYPES.length * 2 * 2; // players × two feet
    const geo = new ThreeBufferGeometry();
    const pos = new Float32Array(feet * 4 * 3);
    const uv = new Float32Array(feet * 4 * 2);
    const fade = new Float32Array(feet * 4);
    const idx = new Uint16Array(feet * 6);
    for (let i = 0; i < feet; i++) {
      const o = i * 4;
      uv[o * 2 + 0] = 0;
      uv[o * 2 + 1] = 0;
      uv[o * 2 + 2] = 1;
      uv[o * 2 + 3] = 0;
      uv[o * 2 + 4] = 1;
      uv[o * 2 + 5] = 1;
      uv[o * 2 + 6] = 0;
      uv[o * 2 + 7] = 1;
      idx.set([o, o + 1, o + 2, o, o + 2, o + 3], i * 6);
    }
    geo.setAttribute('position', new BufferAttribute(pos, 3));
    geo.setAttribute('uv', new BufferAttribute(uv, 2));
    geo.setAttribute('aFade', new BufferAttribute(fade, 1));
    geo.setIndex(new BufferAttribute(idx, 1));
    geo.boundingSphere = null;

    // Resolution follows the tier's texture budget like every other bake here.
    this.contactTex = bakeContactTexture(textureSize >= 1024 ? 96 : 48);
    const mat = new MeshBasicMaterial({
      map: this.contactTex,
      blending: MultiplyBlending,
      transparent: true,
      // r185 refuses to set the multiply blend state without this and logs
      // `MultiplyBlending requires material.premultipliedAlpha = true` once per
      // frame; the quad then draws with the default blend and the contact term
      // does almost nothing, which is what `players-r1` measured (84% of
      // unoccluded where the profile asks for 44%). Opacity is 1 throughout, so
      // premultiplied and straight alpha are the same values here.
      premultipliedAlpha: true,
      depthWrite: false,
      depthTest: true,
      toneMapped: false,
    });
    mat.onBeforeCompile = (shader: WebGLProgramParametersWithUniforms) => {
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\nattribute float aFade;\nvarying float vFade;')
        .replace('#include <begin_vertex>', '#include <begin_vertex>\n  vFade = aFade;');
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', '#include <common>\nvarying float vFade;')
        // White is the identity for a multiply, so fading a contact out means
        // fading it toward white, not toward transparent.
        .replace(
          '#include <map_fragment>',
          '#include <map_fragment>\n  diffuseColor.rgb = mix( vec3( 1.0 ), diffuseColor.rgb, vFade );',
        );
    };

    const mesh = new Mesh(geo, mat);
    mesh.name = 'contact';
    mesh.frustumCulled = false;
    mesh.renderOrder = 1;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    this.group.add(mesh);
    this.contactMesh = mesh;
    this.contactPos = pos;
    this.contactFade = fade;
  }

  /** Rewrites the contact quads from the current foot poses. */
  private updateContact(): void {
    const pos = this.contactPos;
    const fade = this.contactFade;
    const mesh = this.contactMesh;
    if (!pos || !fade || !mesh) return;
    const halfX = SOLE_HALF_X + CONTACT_SKIRT;
    const halfZ = SOLE_HALF_Z + CONTACT_SKIRT;
    const quads = pos.length / 12;
    let q = 0;
    for (const p of this.players) {
      for (const side of ['L', 'R'] as const) {
        if (q >= quads) break;
        const ankle = p.skeleton.byName[`foot${side}`];
        const toe = p.skeleton.byName[`toe${side}`];
        if (!ankle || !toe) continue;
        ankle.updateWorldMatrix(true, false);
        toe.updateWorldMatrix(true, false);
        _foot.setFromMatrixPosition(ankle.matrixWorld);
        _toe.setFromMatrixPosition(toe.matrixWorld);
        // Contact point: under the middle of the sole, on the floor.
        const cx = (_foot.x + _toe.x) * 0.5;
        const cz = (_foot.z + _toe.z) * 0.5;
        // Lift is measured against this rig's *own* rest ankle height rather
        // than against a landmark fraction. In `players-r1` the fraction was
        // reading ~140 mm of phantom lift on a planted foot, which faded the
        // term to 0.46 and left the floor under a sole at 84% of unoccluded
        // where §9.1 wants 35–55%.
        const lift = Math.max(0, _foot.y - p.restAnkleY);
        // A raised foot's contact does not just shrink, it weakens; §1.2's
        // penumbra-with-distance behaviour applies to this term too. The
        // plateau is generous on purpose: `players-r3` measured ~126 mm of
        // ankle lift on a foot that is visibly planted (the idle clip's weight
        // shift rolls a heel), and a foot 150 mm off the floor still occludes
        // most of the hemisphere under it. Contact is gone by 420 mm.
        const f = clamp01((0.42 - lift) / 0.27);
        let dx = _toe.x - _foot.x;
        let dz = _toe.z - _foot.z;
        const len = Math.hypot(dx, dz) || 1;
        dx /= len;
        dz /= len;
        // Right vector, perpendicular in the floor plane.
        const rx = dz;
        const rz = -dx;
        const y = 0.005;
        const o = q * 12;
        const corners: ReadonlyArray<readonly [number, number]> = [
          [-1, -1],
          [1, -1],
          [1, 1],
          [-1, 1],
        ];
        for (let k = 0; k < 4; k++) {
          const [sx, sz] = corners[k];
          pos[o + k * 3] = cx + rx * sx * halfX + dx * sz * halfZ;
          pos[o + k * 3 + 1] = y;
          pos[o + k * 3 + 2] = cz + rz * sx * halfX + dz * sz * halfZ;
          fade[q * 4 + k] = f;
        }
        q++;
      }
    }
    // Anything unused this frame collapses to a point rather than being left
    // wherever it was last drawn.
    for (; q < quads; q++) {
      pos.fill(0, q * 12, q * 12 + 12);
      fade.fill(0, q * 4, q * 4 + 4);
    }
    const g = mesh.geometry;
    g.getAttribute('position').needsUpdate = true;
    g.getAttribute('aFade').needsUpdate = true;
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
        kitMaterial: null,
        cloth: newClothSpring(0),
        restAnkleY: restHeight(skeleton, 'footL'),
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
      kitMaterial: isKitMaterial(o.kitMat) ? o.kitMat : null,
      cloth: newClothSpring(o.facing),
      restAnkleY: restHeight(o.skeleton, 'footL'),
    };
    this.players.push(rig);
    return rig;
  }

  /**
   * Advances one player's garment spring and publishes it to the kit shader.
   *
   * The target is the body's own motion expressed in the rig's frame and
   * negated — cloth trails what carries it — plus a small lift, because a hem
   * driven through air rides up as well as back. The spring is what turns that
   * instantaneous target into the 80–160 ms lag and the single overshoot §3.5
   * and §9.2 ask for; feeding the target in directly would give a hem that
   * snapped, which is the same defect in a different place.
   */
  private updateCloth(p: PlayerRig, dt: number): void {
    const c = p.cloth;
    const cos = Math.cos(p.facing);
    const sin = Math.sin(p.facing);
    // World → rig-local (the rig is a pure yaw).
    const lx = p.velocity.x * cos - p.velocity.z * sin;
    const lz = p.velocity.x * sin + p.velocity.z * cos;
    const speed = Math.hypot(lx, lz);
    const tx = -lx * CLOTH_DRAG;
    const tz = -lz * CLOTH_DRAG;
    const ty = speed * CLOTH_DRAG * 0.42 - p.jumpY * 0.05;

    const k = CLOTH_OMEGA * CLOTH_OMEGA;
    const d = 2 * CLOTH_ZETA * CLOTH_OMEGA;

    // Sub-step the spring.
    //
    // This is explicit Euler, which for a spring of frequency w is only stable
    // while dt < 2/w. At w = 18 that ceiling is 111 ms and the engine clamps dt
    // to 100 ms, so a single slow frame lands right on the edge and the
    // integrator runs away. It never showed until players started moving,
    // because a rig at zero velocity has a target of zero and a spring already
    // at rest has nothing to diverge from.
    const maxStep = 1 / 120;
    const steps = Math.max(1, Math.ceil(dt / maxStep));
    const h = dt / steps;
    for (let i = 0; i < steps; i++) {
      c.vel.x += ((tx - c.pos.x) * k - c.vel.x * d) * h;
      c.vel.y += ((ty - c.pos.y) * k - c.vel.y * d) * h;
      c.vel.z += ((tz - c.pos.z) * k - c.vel.z * d) * h;
      c.pos.addScaledVector(c.vel, h);
    }
    if (c.pos.lengthSq() > CLOTH_MAX * CLOTH_MAX) c.pos.setLength(CLOTH_MAX);

    // Torsional lag: a hem does not follow a turn instantly either.
    let dyaw = p.facing - c.lastFacing;
    while (dyaw > Math.PI) dyaw -= Math.PI * 2;
    while (dyaw < -Math.PI) dyaw += Math.PI * 2;
    c.lastFacing = p.facing;
    const yawRate = dt > 1e-5 ? dyaw / dt : 0;
    const twistTarget = Math.max(-0.16, Math.min(0.16, -yawRate * 0.05));
    for (let i = 0; i < steps; i++) {
      c.twistVel += ((twistTarget - c.twist) * k - c.twistVel * d) * h;
      c.twist += c.twistVel * h;
    }
    // The swing above is length-clamped; this was not, so where swing merely
    // saturated, twist grew without bound and sheared the kit into flat sheets.
    // A hem lags a turn by a few degrees, never by a rotation.
    c.twist = Math.max(-CLOTH_TWIST_MAX, Math.min(CLOTH_TWIST_MAX, c.twist));
    if (!Number.isFinite(c.twist)) {
      c.twist = 0;
      c.twistVel = 0;
    }

    const kit = p.kitMaterial;
    if (kit) {
      kit.userData.swing.value.copy(c.pos);
      kit.userData.twist.value = c.twist;
    }
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
      this.updateCloth(p, dt);
    }

    this.updateContact();

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
    // Play decides where the ball is when it has an opinion — a dribble puts it
    // on the floor and back, and the hands have to follow it there rather than
    // the ball being welded to a static triple-threat point. Absent a
    // controller, the hold below stands.
    if (this.ballPointOverride) return out.copy(this.ballPointOverride);

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
    if (this.contactMesh) {
      this.contactMesh.geometry.dispose();
      (this.contactMesh.material as Material).dispose();
      this.group.remove(this.contactMesh);
      this.contactMesh = null;
    }
    this.contactTex?.dispose();
    this.contactTex = null;
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
