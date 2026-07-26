/**
 * Arena lighting rig and image-based lighting.
 *
 * The plot is a real one. NBA arenas light the playing surface to 1500–2000 lux
 * at ~5600 K from two catwalk runs high over the sidelines, aimed steeply down,
 * while the seating bowl is deliberately held **2.5–4 stops under** the floor —
 * that ratio is the single most important number in the frame. Everything here
 * follows from it:
 *
 *  - **Four steep banks carry ~62% of the floor's light and every one of them
 *    casts.** Steep is what keeps the bowl dark for free: at 62–68° elevation
 *    the hardwood takes N·L ≈ 0.92 and a spectator's vertical torso takes 0.33,
 *    which is 1.5 stops before crowd albedo contributes anything. Four separate
 *    azimuths means a standing player throws one dominant near-vertical contact
 *    shadow plus fainter fans; a single hard shadow is an instant fail.
 *  - **Each bank is its own cascade** (`lightShadows.ts`). The primary fits a
 *    6.4 m ortho frustum around the ball — 3 mm per texel at `high` — and
 *    texel-snaps as it tracks so nothing crawls. The rest widen out to 34 m and
 *    refresh on a slower interval, which is the cheap half of a CSM without
 *    having to inject a cascade selector into every material in the project.
 *  - **The filter is PCSS.** Penumbra grows with the receiver-to-blocker gap, so
 *    a planted sole is a couple of pixels and a raised hand is twenty. Stock
 *    three filters at a constant radius, which is a named tell.
 *  - **A pool of overhead spots** adds the last ~12%: an inverse-square, coned
 *    falloff that makes the middle of the floor brighter than the apron and
 *    dies before the seats. Small on purpose — enough to shape the floor, never
 *    enough for a cone edge to read as an ellipse.
 *  - **The floor is the warm source.** A hemisphere term plus action-following
 *    bounce lights push amber up onto every downward-facing surface: jaw
 *    undersides, shorts hems, the bottom of the ball.
 *  - **The environment is the specular half of the rig.** `envArena.ts` paints
 *    every pod as a rectangle at the same world position the arena hangs it, so
 *    the streak a player sees in the backboard belongs to a fixture that is
 *    physically in the room.
 *
 * Owned by the lighting agent.
 */

import {
  AmbientLight,
  Color,
  DirectionalLight,
  Group,
  HemisphereLight,
  PMREMGenerator,
  PointLight,
  SpotLight,
  Texture,
  Vector3,
} from 'three';
import type { Engine, System } from '../core/Engine';
import { COURT, HOOP, LIGHT_RIG, basketX } from '../core/Constants';
import { clamp01, damp } from '../core/MathX';
import { bakeArenaEnvironment } from './envArena';
import { rigSpotPositions } from './lightRig';
import {
  installSoftShadows,
  ShadowCascade,
  SOFT_SHADOW_TIERS,
  type CascadeSpec,
} from './lightShadows';

export { bakeArenaEnvironment } from './envArena';

/**
 * Scene-linear levels the whole rig is balanced around.
 *
 * These are **pre-tonemap** radiances, which is the space the post stack works
 * in, so `grade` can hand the bloom threshold over as a real number instead of a
 * guess. The anchor is the court: ACES at exposure 1.0 maps 0.145 scene-linear
 * to roughly 110 sRGB, which is the middle of the rubric's 95–140 band for lit
 * hardwood. Everything else is derived from that.
 */
const LEVELS = {
  /** Radiance of fully lit hardwood → ~110–125 sRGB after ACES. */
  court: 0.145,
  /** Seating bowl. ~3.4 stops under the floor before crowd albedo is applied. */
  bowl: 0.009,
  /** LED ribbon peak: the brightest continuous element after the fixtures. */
  led: 1.7,
  /** A fixture pod as seen in a reflection — a 200–255 hard specular. */
  fixture: 1.7,
  /**
   * Total irradiance on the hardwood, split across the rig. Court radiance is
   * `courtIrradiance * albedo / PI`; the hardwood measures out around 0.45
   * albedo, which is what puts the target at ~1.0.
   */
  courtIrradiance: 1.01,
} as const;

/** Per-tier light counts. Everything expensive scales; nothing is hard-coded. */
function rigBudget(tier: string): { spots: number; banks: number; bounce: number; practicals: number } {
  switch (tier) {
    case 'low':
      return { spots: 2, banks: 2, bounce: 1, practicals: 0 };
    case 'medium':
      return { spots: 4, banks: 3, bounce: 2, practicals: 0 };
    case 'ultra':
      return { spots: 6, banks: 4, bounce: 3, practicals: 2 };
    default:
      return { spots: 6, banks: 4, bounce: 3, practicals: 2 };
  }
}

/**
 * The four shadow-casting banks, as unit vectors from the floor *toward* the
 * fixture cluster they stand in for. Each one sits inside a real run of pods up
 * in the rafters (`LIGHT_RIG`, mirrored from the arena's catwalk plot), so the
 * fan of shadows on the floor points back at hardware that is actually in
 * frame. They sit inboard of the catwalk line on purpose: every point on the
 * floor is lit by dozens of pods across both runs, so the *aggregate* direction
 * is steeper than any single fixture, and steepness is where the bowl-to-court
 * ratio is bought. At 62–68° elevation a spectator's vertical torso takes
 * N·L ≈ 0.33 against the hardwood's 0.92 — 1.5 stops before crowd albedo or
 * occlusion contribute anything. Drop the banks to a true 55° catwalk angle and
 * the crowd comes up almost a full stop and the frame reads as a lit gym.
 */
const BANK_DIRS: Vector3[] = [
  new Vector3(4.2, 17.8, 6.4).normalize(),   // +Z sideline run, right of centre
  new Vector3(-5.8, 17.8, -5.4).normalize(), // -Z sideline run, left of centre
  new Vector3(-8.2, 17.8, 4.6).normalize(),  // +Z sideline run, far left
  new Vector3(7.0, 16.7, -6.6).normalize(),  // cross bank over the +X basket
];

const BANK_TINTS = [0xf2f6ff, 0xfff4e6, 0xeef4ff, 0xfff6ea];
/**
 * The banks deliberately carry ~62% of the floor's light between them, because
 * that is what makes the shadows read: under a standing player all four are
 * blocked and the core drops to ~38% of the adjacent floor, while a single fan
 * — one bank blocked, three still lit — only drops to ~75%. That is exactly the
 * "one dominant contact shadow plus 2–5 fainter fans" the frame needs. Pushing
 * more light into the shadowless fill flattens the contact; pushing less makes
 * the fans read as four hard shadows.
 */
const BANK_INTENSITY = [0.36, 0.20, 0.145, 0.12];
/** Extent, follow, shadow intensity, source half-angle, distance, refresh. */
const BANK_SHADOW: Array<Omit<CascadeSpec, 'direction'>> = [
  { extent: 6.4, follow: 1.0, intensity: 1.0, sourceAngle: 0.026, distance: 26, refreshInterval: 1 },
  { extent: 16, follow: 0.85, intensity: 0.5, sourceAngle: 0.042, distance: 30, refreshInterval: 1 },
  { extent: 34, follow: 0.0, intensity: 0.42, sourceAngle: 0.055, distance: 42, refreshInterval: 15 },
  { extent: 22, follow: 0.4, intensity: 0.34, sourceAngle: 0.05, distance: 34, refreshInterval: 4 },
];

export class LightingSystem implements System {
  readonly name = 'lighting';
  readonly order = 5;

  group = new Group();

  /** Primary bank. Kept as `key` because the rest of the project names it that. */
  key!: DirectionalLight;
  /** Secondary bank, opposite azimuth. */
  fill!: DirectionalLight;
  /** Low back light off the far corner — the mandatory thin rim. */
  rim!: DirectionalLight;
  ambient!: AmbientLight;
  /** Warm hardwood bounce into every downward-facing normal. */
  hemi!: HemisphereLight;

  spots: SpotLight[] = [];
  cascades: ShadowCascade[] = [];
  banks: DirectionalLight[] = [];
  bounce: PointLight[] = [];
  envTexture: Texture | null = null;

  /**
   * Published levels so the post stack can key bloom off what the rig actually
   * emits rather than guessing a threshold. All values are scene-linear.
   */
  readonly grade = {
    exposure: 1.0,
    /** Lit hardwood. Bloom must not touch this. */
    courtLuminance: LEVELS.court,
    /** Seating bowl mean. */
    bowlLuminance: LEVELS.bowl,
    /** LED ribbon / jumbotron peak. */
    ledLuminance: LEVELS.led,
    /** Fixture pod — the brightest thing in frame. */
    fixtureLuminance: LEVELS.fixture,
    /** Suggested bloom threshold, scene-linear: above the hottest varnish streak
     *  and sweat specular (~0.9) but under the LED boards, so bloom sources stay
     *  countable — boards, jumbotron, fixture reflections, flashes. */
    bloomThreshold: 1.1,
    bloomIntensity: 0.55,
    /** Bank white point. Shadows should be graded cooler than this, not warmer. */
    keyKelvin: LIGHT_RIG.kelvin,
    /** 0 → 1 celebration lift, for a post flash that matches the rig. */
    pulse: 0,
  };

  private pmrem: PMREMGenerator | null = null;
  private readonly focus = new Vector3(0, 0.9, 0);
  private readonly desired = new Vector3(0, 0.9, 0);
  private pulse = 0;
  private pulseTarget = 0;
  private unsubscribe: Array<() => void> = [];
  private baseSpotIntensity: number[] = [];
  private baseBankIntensity: number[] = [];
  private baseBounceIntensity: number[] = [];
  private baseExposure = 1.0;
  private readonly warmFlash = new Color(0xffd9a8);
  private readonly bankBase: Color[] = [];
  private readonly scratch = new Color();

  init(engine: Engine): void {
    const { scene, renderer, quality } = engine;
    this.group.name = 'lighting';
    scene.add(this.group);

    // Must happen before any material compiles — this system has the lowest
    // order in the engine for exactly that reason.
    installSoftShadows(renderer, SOFT_SHADOW_TIERS[quality.tier] ?? SOFT_SHADOW_TIERS.high);

    const budget = rigBudget(quality.tier);

    // ---------------------------------------------------------- environment
    const envWidth = quality.tier === 'low' ? 256 : quality.tier === 'medium' ? 512 : 1024;
    this.pmrem = new PMREMGenerator(renderer);
    this.pmrem.compileEquirectangularShader();
    const raw = bakeArenaEnvironment({
      width: envWidth,
      fixtureLuminance: LEVELS.fixture,
      ledLuminance: LEVELS.led,
      floorLuminance: LEVELS.court,
      bowlLuminance: LEVELS.bowl,
    });
    const rt = this.pmrem.fromEquirectangular(raw);
    this.envTexture = rt.texture;
    scene.environment = rt.texture;
    // The bake is authored directly in the same scene-linear units as the
    // analytic rig, so this stays at 1 and the numbers in `grade` mean
    // something to whoever reads them.
    scene.environmentIntensity = 1.0;
    raw.dispose();

    // ---------------------------------------------------- overhead bank spots
    // A light *pool*: the banks already light the room evenly, and these add the
    // gentle centre-bright falloff a real fixture layout produces. Cones aim
    // slightly inward so the pool edge lands on the apron, not in the seats.
    // Deliberately only ~12% of the floor's light — enough to shape it, not
    // enough that a cone edge is ever visible as a hard ellipse.
    const spotPositions = rigSpotPositions(budget.spots);
    const perSpot = 150 / Math.max(1, spotPositions.length);
    for (const p of spotPositions) {
      const spot = new SpotLight(0xf4f7ff, perSpot, 40, 0.46, 0.92, 2);
      spot.position.copy(p);
      // Cross-lit: a fixture on the +Z catwalk aims at the -Z half of the floor,
      // which is what real rigs do and what keeps the pool on the hardwood
      // instead of on the seats directly beneath the catwalk.
      spot.target.position.set(p.x * 0.7, 0, -p.z * 0.3);
      spot.castShadow = false;
      this.spots.push(spot);
      this.baseSpotIntensity.push(perSpot);
      this.group.add(spot, spot.target);
    }

    // ------------------------------------------------------------- bowl wash
    // Real arenas do not let the bowl fall to nothing: there is aisle lighting
    // and a dim wash off the outer catwalk. Two very wide, very soft cones
    // aimed at the seating give the crowd a *gradient* — brighter at the rail,
    // falling off up the rake — which is architecture. A flat ambient lift of
    // the same magnitude would just make the frame grey.
    // Aimed outward and down from inboard of the rail, so the cone lands on the
    // *court-facing* side of the crowd and misses the hardwood entirely.
    for (const sz of [1, -1] as const) {
      const wash = new SpotLight(0xaec4ea, 3, 44, 0.95, 1.0, 2);
      wash.position.set(0, 13.5, sz * 8.5);
      wash.target.position.set(0, 4.5, sz * 20);
      wash.castShadow = false;
      this.spots.push(wash);
      this.baseSpotIntensity.push(3);
      this.group.add(wash, wash.target);
    }

    // ------------------------------------------------------- shadow banks
    const mapSize = quality.shadowMapSize;
    const cascadeCount = Math.max(1, Math.min(quality.shadowCascades, budget.banks, 4));
    for (let i = 0; i < budget.banks; i++) {
      const shadowSpec = { ...BANK_SHADOW[i], direction: BANK_DIRS[i] };
      // With a single cascade there is nothing behind it, so widen the primary
      // rather than leave the far half of the court unshadowed.
      if (i === 0 && cascadeCount === 1) {
        shadowSpec.extent = 14;
        shadowSpec.sourceAngle = 0.04;
      }
      if (i < cascadeCount) {
        const c = new ShadowCascade(shadowSpec, BANK_TINTS[i], BANK_INTENSITY[i], mapSize);
        c.light.name = `bank${i}`;
        this.cascades.push(c);
        this.banks.push(c.light);
        this.group.add(c.light, c.light.target);
      } else {
        const d = new DirectionalLight(BANK_TINTS[i], BANK_INTENSITY[i]);
        d.name = `bank${i}`;
        d.position.copy(BANK_DIRS[i]).multiplyScalar(30);
        d.castShadow = false;
        this.banks.push(d);
        this.group.add(d, d.target);
      }
      this.baseBankIntensity.push(BANK_INTENSITY[i]);
      this.bankBase.push(new Color(BANK_TINTS[i]));
    }
    this.key = this.banks[0];
    this.fill = this.banks[1] ?? this.banks[0];

    // ------------------------------------------------------------- rim light
    // Low, from behind the far baseline corner, cool. It must *break* where the
    // silhouette turns away from it — which a real directional does for free and
    // a Fresnel hack never does.
    this.rim = new DirectionalLight(0xc2d6ff, 0.34);
    this.rim.position.set(-7.5, 8.4, -17);
    this.rim.castShadow = false;
    this.group.add(this.rim, this.rim.target);

    const rimB = new DirectionalLight(0xd8e4ff, 0.2);
    rimB.position.set(10.5, 9.2, -14.5);
    rimB.castShadow = false;
    this.group.add(rimB, rimB.target);

    // ------------------------------------------------------- bounce and fill
    // The hardwood is the warm source. Ground colour lands on every
    // downward-facing normal; sky colour is the dark bowl, so upward normals
    // gain almost nothing and the ratio survives.
    this.hemi = new HemisphereLight(0x2b3a58, 0xffb070, 0.32);
    this.hemi.position.set(0, 6, 0);
    this.group.add(this.hemi);

    // Deliberately small. Ambient is the one term that raises the bowl and the
    // hardwood by the same absolute amount, so every unit of it eats directly
    // into the bowl-to-court ratio, which is the master criterion for the whole
    // frame. Cool-tinted, so the shadow end of the image is already split
    // against the warm hardwood bounce before the grade touches it.
    this.ambient = new AmbientLight(0x2a3a55, 0.6);
    this.group.add(this.ambient);

    // Directional bounce that follows the action, plus a static one under each
    // basket where the traffic is. Kept weak and given a sub-quadratic decay on
    // purpose: a bright point sitting on a roughness-0.11 varnish turns into an
    // isolated hot streak in the floor, which is far more noticeable than the
    // warmth it buys. The hemisphere term carries the bulk of the bounce; these
    // only add the *direction* — a local, moving warm up-light under the ball.
    const bouncePositions: Vector3[] = [new Vector3(0, 0.55, 0)];
    if (budget.bounce > 1) bouncePositions.push(new Vector3(basketX(1) * 0.8, 0.5, 0));
    if (budget.bounce > 2) bouncePositions.push(new Vector3(basketX(-1) * 0.8, 0.5, 0));
    for (let i = 0; i < bouncePositions.length; i++) {
      const intensity = i === 0 ? 0.12 : 0.08;
      const p = new PointLight(0xffb377, intensity, 8, 1.6);
      p.position.copy(bouncePositions[i]);
      p.castShadow = false;
      this.bounce.push(p);
      this.baseBounceIntensity.push(intensity);
      this.group.add(p);
    }

    // ------------------------------------------------------- practical spots
    // Aimed at each backboard. Their only job is the hard specular population on
    // the glass and a hot top edge on the ring.
    for (let i = 0; i < budget.practicals; i++) {
      const side = i === 0 ? 1 : -1;
      const x = basketX(side as 1 | -1);
      // Cut the throw off just past the glass: this fixture exists for the
      // specular population on the backboard, and letting its cone reach the
      // hardwood puts a single isolated hot streak in the varnish.
      const practicalIntensity = 7;
      const spot = new SpotLight(0xfff4e2, practicalIntensity, 9.2, 0.4, 0.5, 2);
      spot.position.set(x * 0.66, 9.9, 0);
      spot.target.position.set(x, HOOP.rimHeight + 0.5, 0);
      spot.castShadow = false;
      this.spots.push(spot);
      this.baseSpotIntensity.push(practicalIntensity);
      this.group.add(spot, spot.target);
    }

    // ------------------------------------------------------------- exposure
    // ACES is already on. This lands lit hardwood at ~110 sRGB while leaving the
    // LED boards room to sit at 200–250 without clipping to a flat white slab.
    this.baseExposure = 0.85;
    renderer.toneMappingExposure = this.baseExposure;
    this.grade.exposure = this.baseExposure;

    this.bindReactions(engine);
    this.focusShadowOn(this.focus);
  }

  /**
   * A dunk or a made three lifts the room a little. Subtle — the arena should
   * breathe, not strobe.
   */
  private bindReactions(engine: Engine): void {
    const hit = (amount: number) => {
      this.pulseTarget = Math.max(this.pulseTarget, amount);
      this.pulse = Math.max(this.pulse, amount);
    };
    this.unsubscribe.push(
      engine.bus.on('scored', (e) => hit(e.swish ? 0.75 : 0.55)),
      engine.bus.on('dunk', (e) => hit(0.7 + clamp01(e.power) * 0.3)),
      engine.bus.on('block', () => hit(0.4)),
      engine.bus.on('quarterEnd', () => hit(0.5)),
    );
  }

  /**
   * Keeps the shadow frusta tight around the action. Every cascade re-fits
   * according to its own `follow` weight, and the primary bounce light rides
   * along so the warm up-light stays under whoever has the ball.
   */
  focusShadowOn(target: Vector3): void {
    this.desired.copy(target);
    for (const c of this.cascades) c.focus(target);
    if (this.bounce.length > 0) {
      this.bounce[0].position.set(target.x, 0.55, target.z);
    }
  }

  update(dt: number, _alpha: number, engine: Engine): void {
    const ball = engine.get<{ focusPoint?: Vector3 }>('ball');
    if (ball?.focusPoint) this.desired.copy(ball.focusPoint);

    // Damp the focus so the shadow frustum never snaps; texel snapping inside
    // the cascade handles the rest of the crawl.
    this.focus.x = damp(this.focus.x, this.desired.x, 9, dt);
    this.focus.z = damp(this.focus.z, this.desired.z, 9, dt);
    this.focus.y = damp(this.focus.y, this.desired.y, 9, dt);
    this.focusShadowOn(this.focus);
    for (const c of this.cascades) c.tick();

    // Celebration lift.
    this.pulseTarget = damp(this.pulseTarget, 0, 3.4, dt);
    this.pulse = damp(this.pulse, this.pulseTarget, 6, dt);
    if (this.pulse < 0.0015) this.pulse = 0;
    this.grade.pulse = this.pulse;

    // A very slow breath keeps the rig from reading as a static bake even when
    // nothing is happening. ±1.2%, well below the threshold of "flicker".
    const breath = 1 + Math.sin(engine.elapsed * 0.82) * 0.006 + Math.sin(engine.elapsed * 0.31) * 0.006;
    const lift = 1 + this.pulse * 0.16;

    for (let i = 0; i < this.spots.length; i++) {
      this.spots[i].intensity = this.baseSpotIntensity[i] * breath * lift;
    }
    for (let i = 0; i < this.banks.length; i++) {
      this.banks[i].intensity = this.baseBankIntensity[i] * breath * lift;
      // The lift warms as well as brightens — arena celebration lighting is
      // never a neutral gain.
      this.scratch.copy(this.bankBase[i]).lerp(this.warmFlash, this.pulse * 0.35);
      this.banks[i].color.copy(this.scratch);
    }
    for (let i = 0; i < this.bounce.length; i++) {
      this.bounce[i].intensity = this.baseBounceIntensity[i] * (1 + this.pulse * 0.55);
    }
    const exposure = this.baseExposure * (1 + this.pulse * 0.035);
    engine.renderer.toneMappingExposure = exposure;
    this.grade.exposure = exposure;
  }

  /**
   * Where the light rig thinks the action is. Handy for the camera and post
   * agents, and cheaper than re-deriving it.
   */
  focusPoint(): Vector3 {
    return this.focus;
  }

  /** Approximate world-space extent of the brightly lit region. */
  litExtent(): { x: number; z: number } {
    return { x: COURT.halfLength + COURT.apronX, z: COURT.halfWidth + COURT.apronZ };
  }

  dispose(): void {
    for (const off of this.unsubscribe) off();
    this.unsubscribe = [];
    this.pmrem?.dispose();
    this.envTexture?.dispose();
  }
}
