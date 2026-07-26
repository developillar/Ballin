/**
 * Arena lighting rig and image-based lighting.
 *
 * The plot is a real one. NBA arenas light the playing surface to 1500–2000 lux
 * at ~5600 K from two catwalk runs high over the sidelines, aimed steeply down,
 * while the seating bowl is deliberately held **2.5–4 stops under** the floor —
 * that ratio is the single most important number in the frame. Everything here
 * follows from it:
 *
 *  - **Four steep banks carry ~58% of the floor's light, and they carry it
 *    evenly.** Steep is what keeps the bowl dark for free: at 62–68° elevation
 *    the hardwood takes N·L ≈ 0.92 and a spectator's vertical torso takes 0.33,
 *    which is 1.5 stops before crowd albedo contributes anything. A directional
 *    light also has no distance term and no cone, so every square metre of
 *    hardwood receives exactly the same irradiance from it — which is why the
 *    floor's *brightness* lives here and not in the spot pool.
 *  - **Each bank is its own cascade** (`lightShadows.ts`), on four separate
 *    azimuths, so a standing player should throw one dominant near-vertical
 *    contact shadow plus 2–5 fainter fans. **VERIFIED FALSE IN A CAPTURE, round
 *    2: no cast shadow reaches the floor at all.** three's shadow pass is
 *    issuing zero draws into the maps even though it is called every frame with
 *    three valid shadow lights and three's own frustum test puts 39 casters
 *    inside the primary cascade. The receiving end is provably fine — forcing
 *    `getShadow()` to a constant darkens the hardwood — and three unrelated
 *    shadow implementations fail identically, so the fault is upstream of this
 *    rig. Full evidence and the reproduction are in the `lightShadows.ts`
 *    header. Until that is fixed the frame has no contact shadows and §1.2 and
 *    §9.1 cannot pass; the shadow-side numbers below are intent, not measurement.
 *  - **The filter is PCSS.** Penumbra should grow with the receiver-to-blocker
 *    gap, so a planted sole is a couple of pixels and a raised hand is twenty.
 *    Stock three filters at a constant radius, which is a named tell. Also
 *    unverifiable while the maps come back empty.
 *  - **A pool of overhead spots** adds the last ~6%, and it is deliberately the
 *    most boring light in the room. Every cone is a 58° half-angle aimed at the
 *    court's long axis with `penumbra = 1`, so its edge lands ~28 m from the aim
 *    point — off the hardwood, off the apron, out of frame — and all that is
 *    left on the floor is the smooth `smoothstep` shoulder. Their sum is one
 *    soft dome that fades toward the sidelines. There is no cone edge anywhere
 *    on the floor, because there is no cone edge within 15 m of the floor.
 *  - **The floor is the warm source, and it is a hemisphere, not a point.**
 *    Amber lands on every downward-facing normal — jaw undersides, shorts hems,
 *    the bottom of the ball — from a `HemisphereLight`, which three evaluates
 *    into irradiance only. That matters: a *punctual* bounce light sitting 0.5 m
 *    over a roughness-0.1 varnish is a specular hazard, and it wrote a visible
 *    hard orange arc across the floor in the round-1 captures (its distance
 *    cutoff sphere cutting the floor plane). Diffuse-only terms cannot do that.
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
import { poolAims } from './lightRig';
import {
  installSoftShadows,
  ShadowCascade,
  SOFT_SHADOW_TIERS,
  type CascadeSpec,
} from './lightShadows';

export { bakeArenaEnvironment } from './envArena';

/**
 * THE EXPOSURE CONTRACT.
 *
 * Scene-linear (**pre-tonemap**) radiances the whole rig is balanced around,
 * plus the one exposure number that maps them onto the display. Every other
 * system that needs to know how bright something should be reads these through
 * `LightingSystem.grade` rather than guessing, so there is exactly one place
 * where the frame's exposure is decided.
 *
 * The chain, stated once so it can be checked rather than believed:
 *
 *   scene-linear radiance  ×  `exposure`  →  ACES filmic  →  sRGB encode
 *
 * The anchor is the hardwood. `court` × `exposure` = 0.24 goes through three's
 * ACES fit to ≈ 0.17 display-linear, which encodes to **≈ 112 sRGB** — the
 * middle of the rubric's 95–140 band (§1.1). `bowl` × `exposure` = 0.050 lands
 * at **≈ 32 sRGB**, the middle of the 18–45 band. On the display side, where the
 * rubric measures, that pair is **≈ 3.7 stops** apart, because the ACES shoulder
 * compresses the court end harder than the bowl end and *widens* a scene-linear
 * ratio of 2.25 stops into the required 2.5–4.
 *
 * Two consequences worth stating because they are easy to get backwards:
 *
 *  - **Exposure cannot fix the court-to-bowl ratio.** It multiplies both. The
 *    ratio is bought with steep banks, a bowl that no bank reaches, and an
 *    ambient term small enough not to matter — see `BANK_DIRS` and `ambient`.
 *  - **A blown-looking floor is almost never the exposure.** Round 1 measured
 *    the *mean* hardwood at 88 (under the band) while individual streaks hit
 *    165 and desaturated. That is specular spread, not level: the fix was to
 *    take energy out of the low, grazing, punctual sources that were painting
 *    those streaks and put it into steep banks and diffuse-only terms, then
 *    raise exposure to land the now-even floor in the band.
 */
const LEVELS = {
  /** Radiance of fully lit hardwood → ~112 sRGB at `exposure`. */
  court: 0.19,
  /**
   * Seating bowl mean → ~32 sRGB at `exposure`. `Crowd.ts` carries its own
   * lighting model and owns the figure it actually renders; this is the number
   * the environment bake paints the bowl at, so reflections of the bowl in the
   * varnish and in chrome agree with the crowd standing in it.
   */
  bowl: 0.04,
  /** LED ribbon peak → ~232 sRGB: bright, headroom for bloom, never clipped. */
  led: 1.7,
  /** A fixture pod as seen in a reflection — a 200–255 hard specular. */
  fixture: 1.7,
  /**
   * Total irradiance on an up-facing patch of hardwood, summed over the whole
   * rig. Court radiance is `courtIrradiance * albedo / PI`; the hardwood
   * measures out around 0.45 albedo, which is what puts `court` where it is.
   * The pool spots size themselves off this, so their share of the floor stays
   * a stated fraction instead of a magic candela number.
   */
  courtIrradiance: 1.32,
  /**
   * `renderer.toneMappingExposure`. Owned here and nowhere else.
   */
  exposure: 1.3,
} as const;

/**
 * Share of the floor's irradiance carried by the analytic pool.
 *
 * Small on purpose. The pool exists to give the middle of the floor a gentle
 * lift over the apron; everything else is carried by the banks, which are
 * directional and therefore *perfectly* even across the hardwood. Round 1 ran
 * the pool at roughly a fifth of the floor's light through six narrow cones and
 * the result was readable ellipses.
 */
const POOL_SHARE = 0.055;
/** Half-angle of a pool cone. At 17.8 m the edge is 28 m out — nowhere near the floor. */
const POOL_ANGLE = 1.02;

/** Per-tier light counts. Everything expensive scales; nothing is hard-coded. */
function rigBudget(tier: string): { spots: number; banks: number; practicals: number } {
  switch (tier) {
    case 'low':
      return { spots: 2, banks: 2, practicals: 0 };
    case 'medium':
      return { spots: 3, banks: 3, practicals: 1 };
    case 'ultra':
      return { spots: 5, banks: 4, practicals: 2 };
    default:
      return { spots: 5, banks: 4, practicals: 2 };
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

/**
 * Bank white points. §1.3 asks for 5600–6000 K overall: on a grey card at court
 * level, B − R must land **+3 to +10** in 8-bit sRGB. Weighted by
 * `BANK_INTENSITY` these average **B − R ≈ +8**. The round-1 set averaged −0.6
 * — neutral-to-warm — which, on top of warm maple and warm floor bounce, put the
 * frame's top quintile at R − B = +33 against §8.3's +4..+12 ceiling.
 */
const BANK_TINTS = [0xeff4ff, 0xfaf8ff, 0xecf2ff, 0xfff5e9];
/**
 * The banks deliberately carry ~58% of the floor's light between them, because
 * that is what makes the shadows read: under a standing player all four are
 * blocked and the core drops to ~42% of the adjacent floor, while a single fan
 * — one bank blocked, three still lit — only drops to ~76%. That is exactly the
 * "one dominant contact shadow plus 2–5 fainter fans" the frame needs (§1.2).
 * Pushing more light into the shadowless fill flattens the contact; pushing less
 * makes the fans read as four hard shadows.
 *
 * They also carry it *evenly*: a directional light has no distance term and no
 * cone, so every square metre of hardwood receives exactly the same irradiance
 * from it. That is the whole reason the floor's brightness now lives here rather
 * than in the spot pool.
 */
const BANK_INTENSITY = [0.54, 0.31, 0.222, 0.18];
/** Extent, follow, shadow intensity, source half-angle, distance, refresh. */
const BANK_SHADOW: Array<Omit<CascadeSpec, 'direction'>> = [
  { extent: 6.4, follow: 1.0, intensity: 1.0, sourceAngle: 0.026, distance: 26, refreshInterval: 1 },
  { extent: 16, follow: 0.85, intensity: 0.9, sourceAngle: 0.042, distance: 30, refreshInterval: 1 },
  { extent: 34, follow: 0.0, intensity: 0.72, sourceAngle: 0.07, distance: 42, refreshInterval: 15 },
  { extent: 22, follow: 0.4, intensity: 0.8, sourceAngle: 0.055, distance: 34, refreshInterval: 4 },
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
  /**
   * Shadowless, straight down, neutral. Carries even diffuse floor light with
   * effectively zero specular for any broadcast-height camera: the half-vector
   * between a near-vertical light and a near-horizontal view is 45° off the
   * floor normal, which is far outside a roughness-0.1 clear-coat lobe. It is
   * the cheapest way to raise the hardwood without adding another streak.
   */
  top!: DirectionalLight;
  ambient!: AmbientLight;
  /** Warm hardwood bounce into every downward-facing normal. */
  hemi!: HemisphereLight;

  spots: SpotLight[] = [];
  cascades: ShadowCascade[] = [];
  banks: DirectionalLight[] = [];
  /**
   * Kept for the systems that read it. Empty by design: see the header — the
   * warm floor bounce is a `HemisphereLight`, because a punctual light near a
   * glossy floor is a specular artefact waiting to happen and was one.
   */
  bounce: PointLight[] = [];
  envTexture: Texture | null = null;

  /**
   * Published levels so the post stack can key bloom off what the rig actually
   * emits rather than guessing a threshold. All values are scene-linear.
   */
  readonly grade = {
    exposure: LEVELS.exposure as number,
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
    // analytic rig, so the numbers in `grade` mean something to whoever reads
    // them. Held a little under 1 all the same: the environment is the *only*
    // thing supplying the floor's grazing reflection, and at unity the ceiling
    // and courtside bands came back across the bottom third of a FLOOR frame
    // hot enough to desaturate the maple to sRGB saturation 0.09. The 10% that
    // comes off here goes back into the banks above, which are diffuse-even and
    // cannot streak.
    scene.environmentIntensity = 0.9;
    raw.dispose();

    // ---------------------------------------------------- overhead bank spots
    // The *pool*: the last few percent of the floor's light, and the only part
    // of it that varies across the hardwood. Three properties make it a falloff
    // instead of an ellipse, and all three matter:
    //
    //   1. `penumbra = 1` — three's spot term becomes
    //      `smoothstep(cos(angle), 1, cos(theta))`, i.e. a smooth roll from full
    //      on the axis to zero at the edge, with no inner cone and no ring.
    //   2. `POOL_ANGLE = 1.02 rad` — from 17.8 m the edge sits 28 m from the aim
    //      point. The court's far corner is 19 m away, so the hardwood only ever
    //      sees the top two thirds of that shoulder: a wide, gentle dome.
    //   3. `distance = 0` — no cutoff sphere. The round-1 rig ran a bounce light
    //      with `distance = 8`, and where that sphere sliced the floor plane it
    //      drew a hard orange arc 8 m across, centred on the ball. three's
    //      cutoff is `pow2(1 - pow4(d/cut))`; the quartic makes it abrupt right
    //      at the limit, which is exactly where it meets a flat receiver.
    const aims = poolAims(budget.spots);
    const h = LIGHT_RIG.bankHeight;
    // Sized off the stated share rather than a magic number. 0.55 is the mean of
    // the cone × cosine × inverse-square term over the playing surface, measured
    // from the layout above; it exists so `POOL_SHARE` means what it says.
    const perSpot = (LEVELS.courtIrradiance * POOL_SHARE * h * h) / (Math.max(1, aims.length) * 0.55);
    for (const aim of aims) {
      const spot = new SpotLight(0xeef3ff, perSpot, 0, POOL_ANGLE, 1.0, 2);
      spot.position.copy(aim.from);
      spot.target.position.copy(aim.at);
      spot.castShadow = false;
      this.spots.push(spot);
      this.baseSpotIntensity.push(perSpot);
      this.group.add(spot, spot.target);
    }

    // ------------------------------------------------------------- bowl wash
    // Real arenas do not let the bowl fall to nothing: there is aisle lighting
    // and a dim wash off the rail. Two soft cones aimed up the rake give the
    // seating a *gradient* — brighter at the front rows, falling off toward the
    // upper deck — which is architecture. A flat ambient lift of the same
    // magnitude would just make the frame grey, and would lift the hardwood by
    // the same absolute amount, which is the one thing §1.1 cannot afford.
    //
    // These sit **at the rail and aim outward and up**. That is not a style
    // choice, it is the only placement for which no floor spill is possible: the
    // fixture is outboard of the apron edge, so every point on the hardwood lies
    // in the opposite half-space from the cone axis — more than 90° off it —
    // and a 35° cone cannot reach any of them no matter how soft it is. The
    // round-1 wash hung *inboard* at 13.5 m and aimed down and out through a 54°
    // cone, which put a pale blue tongue of light on the varnish beside the
    // stanchion. Geometry, not tuning, is what fixes that.
    const railZ = COURT.halfWidth + COURT.apronZ + 0.6;
    for (const sz of [1, -1] as const) {
      const washIntensity = 0.62;
      // Sub-quadratic decay: at true inverse-square a fixture this close to the
      // front row is 40× brighter there than eight rows up, which reads as a
      // hot band, not a rake.
      const wash = new SpotLight(0x9db4da, washIntensity, 34, 0.62, 1.0, 1.15);
      wash.position.set(0, 2.35, sz * railZ);
      wash.target.position.set(0, 9.2, sz * 24);
      wash.castShadow = false;
      this.spots.push(wash);
      this.baseSpotIntensity.push(washIntensity);
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

    // -------------------------------------------------------- overhead fill
    // Straight down, shadowless, neutral. This is the term that lets the floor
    // be *bright* without being *streaky*. A directional light contributes the
    // same irradiance to every square metre of a horizontal plane, and its
    // specular lobe for a broadcast camera sits 45° off the floor normal, which
    // a roughness-0.1 clear coat does not respond to at all. The 3° tilt is so
    // the one place it could theoretically bloom — a shot looking straight down
    // — is not exactly under it.
    this.top = new DirectionalLight(0xf0f5ff, 0.20);
    this.top.position.set(1.1, 20, -0.7);
    this.top.castShadow = false;
    this.group.add(this.top, this.top.target);

    // ------------------------------------------------------------- rim light
    // From behind the far baseline corner, cool. It must *break* where the
    // silhouette turns away from it — which a real directional does for free and
    // a Fresnel hack never does.
    //
    // Raised and dimmed hard from round 1 (8.4 m → 17.5 m, 0.34 → 0.13). This
    // is the single most expensive light in the frame per unit of intensity,
    // and the reason is geometric: the camera looks up-court, so the mirror
    // direction of the near hardwood points up-court too, at roughly the
    // camera's own elevation. A back light at a *similar* elevation on the
    // *opposite* azimuth puts the half-vector within a couple of degrees of the
    // floor normal — N·H ≈ 0.995 — which on a clear coat is the peak of the
    // specular lobe. At round-1 values it painted a soft white sheet over the
    // near wood that measured sRGB saturation 0.09: maple with the colour
    // washed out of it. Every stop of elevation moves the half-vector off the
    // normal quadratically, so 22° → 42° plus a 2.6× cut takes roughly an order
    // of magnitude out of that sheet while still raking the shoulders from
    // behind. The lost floor irradiance goes into `top`, which cannot streak.
    this.rim = new DirectionalLight(0xc2d6ff, 0.13);
    this.rim.position.set(-9.5, 17.5, -17);
    this.rim.castShadow = false;
    this.group.add(this.rim, this.rim.target);

    const rimB = new DirectionalLight(0xd8e4ff, 0.085);
    rimB.position.set(12.5, 16.5, -14.5);
    rimB.castShadow = false;
    this.group.add(rimB, rimB.target);

    // ------------------------------------------------------- bounce and fill
    // The hardwood is the warm source, and it is a *hemisphere*. Ground colour
    // lands on every downward-facing normal — jaw undersides, shorts hems, the
    // bottom of the ball (§1.3) — sky colour is the dark bowl, so upward normals
    // gain almost nothing and the ratio survives.
    //
    // three evaluates a HemisphereLight into `irradiance` only: no specular
    // lobe, no distance term, no cutoff sphere. That is precisely why the bounce
    // lives here now. The round-1 rig used PointLights at 0.5 m, and a punctual
    // source that close to a roughness-0.1 varnish does two visible things — a
    // hot specular smear, and a hard ring where its `distance` cutoff sphere
    // cuts the floor plane. Both showed up in the capture as an orange arc
    // sweeping out from under the ball-handler. A diffuse-only term buys the
    // same warmth and cannot draw either artefact.
    this.hemi = new HemisphereLight(0x24314c, 0xffb478, 0.62);
    this.hemi.position.set(0, 6, 0);
    this.group.add(this.hemi);

    // Deliberately small. Ambient is the one term that raises the bowl and the
    // hardwood by the same absolute amount, so every unit of it eats directly
    // into the bowl-to-court ratio, which is the master criterion for the whole
    // frame. Cool-tinted, so the shadow end of the image is already split
    // against the warm hardwood bounce before the grade touches it. Its real job
    // is the §1.1 floor on the deepest arena shadow: 6–16, never 0.
    this.ambient = new AmbientLight(0x27364f, 0.34);
    this.group.add(this.ambient);

    // ------------------------------------------------------- practical spots
    // Aimed at each backboard. Their only job is the hard specular population on
    // the glass (§1.4b) and a hot top edge on the ring.
    for (let i = 0; i < budget.practicals; i++) {
      const side = i === 0 ? 1 : -1;
      const x = basketX(side as 1 | -1);
      // Cut the throw off just past the glass: this fixture exists for the
      // specular population on the backboard, and letting its cone reach the
      // hardwood puts a single isolated hot streak in the varnish.
      const practicalIntensity = 4.5;
      // penumbra 1: at 0.5 the cone drew its own soft-but-readable ellipse on
      // the glass and the padding, which is the same tell as the floor pools.
      const spot = new SpotLight(0xfff4e2, practicalIntensity, 9.2, 0.4, 1.0, 2);
      spot.position.set(x * 0.66, 9.9, 0);
      spot.target.position.set(x, HOOP.rimHeight + 0.5, 0);
      spot.castShadow = false;
      this.spots.push(spot);
      this.baseSpotIntensity.push(practicalIntensity);
      this.group.add(spot, spot.target);
    }

    // ------------------------------------------------------------- exposure
    // ACES is already on. See LEVELS: this is the single number that maps the
    // rig's scene-linear radiances onto the display, and it lands lit hardwood
    // in §1.1's 95–140 band while leaving the LED boards room to sit at 200–250
    // without clipping to a flat white slab.
    this.baseExposure = LEVELS.exposure;
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
    // `bounce` is empty by design (see the field). Kept so a future directional
    // bounce can ride the action without another caller having to change.
    if (this.bounce.length > 0) this.bounce[0].position.set(target.x, 0.55, target.z);
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
