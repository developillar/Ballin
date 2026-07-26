/**
 * The hardwood.
 *
 * A broadcast NBA floor is not a wood texture on a plane. It is 2-1/4 in
 * milled maple strips with staggered butt joints, buried under several coats
 * of high-gloss polyurethane that were sanded along the boards — which is why
 * the floor looks *wet* on television and why the overhead banks smear into
 * long grain-aligned streaks rather than round dots. The paint is under that
 * coat, not on it, so a highlight crossing the sideline never changes shape.
 *
 * The material here is built to reproduce exactly that:
 *
 *  - a whole-court bake (albedo + wear + two signed distance fields for the
 *    line work + a coat-roughness field + AO) from `courtBake`;
 *  - a tiling maple detail texture carrying grain and seams in a real height
 *    field, converted to a tangent-space normal map, so the milling catches
 *    light instead of being painted on;
 *  - a single anisotropic specular lobe standing in for the varnish (the wood
 *    beneath is buried in resin, so there is physically only one interface),
 *    with a light clearcoat over it for the second, tighter lobe;
 *  - a planar reflection rendered from a mirrored camera on the tiers that can
 *    afford it, blurred by roughness and distance, stretched along the boards,
 *    with a Fresnel gain that goes near-mirror at grazing angles. Below that
 *    tier the same code path falls back to the PMREM environment alone.
 *
 * Owned by the court agent.
 */

import {
  CanvasTexture,
  ClampToEdgeWrapping,
  Color,
  Group,
  HalfFloatType,
  LinearMipmapLinearFilter,
  LinearSRGBColorSpace,
  Matrix4,
  Mesh,
  MeshPhysicalMaterial,
  PerspectiveCamera,
  Plane,
  PlaneGeometry,
  RepeatWrapping,
  SRGBColorSpace,
  Vector2,
  Vector3,
  Vector4,
  WebGLRenderTarget,
  type Camera,
  type Scene,
  type WebGLRenderer,
} from 'three';
import type { Engine, System } from '../core/Engine';
import { COURT } from '../core/Constants';
import {
  bakeCourt,
  LOGO_RADIUS,
  ROUGH_BASE,
  ROUGH_RANGE,
  SDF_RANGE,
  type CourtBake,
} from '../textures/courtBake';
import {
  BOARD_WIDTH,
  PANEL_BOARDS,
  PANEL_LENGTH,
  SEAM_HALF,
  TILE_LENGTH,
  TILE_WIDTH,
} from '../textures/courtWood';

const MIRROR_PLANE = new Plane(new Vector3(0, 1, 0), 0);

export class CourtSystem implements System {
  readonly name = 'court';
  readonly order = 10;

  group = new Group();
  floor!: Mesh;
  material!: MeshPhysicalMaterial;

  /** Playing surface height. The milled relief is sub-millimetre and normal-map only. */
  readonly surfaceY = 0;

  private engine!: Engine;
  private bake!: CourtBake;
  private textures: CanvasTexture[] = [];

  // --- planar reflection ---
  private reflectionRT: WebGLRenderTarget | null = null;
  private reflectCamera = new PerspectiveCamera();
  private reflectEvery = 1;
  private reflecting = false;
  private lastReflectFrame = -1;
  private readonly _pos = new Vector3();
  private readonly _look = new Vector3();
  private readonly _up = new Vector3();
  private readonly _rot = new Matrix4();
  private readonly _plane = new Plane();
  private readonly _clip = new Vector4();
  private readonly _q = new Vector4();

  private readonly uniforms = {
    uMask: { value: null as CanvasTexture | null },
    uLogoMap: { value: null as CanvasTexture | null },
    /**
     * The painted lane. Mixed lighter and flatter than a brand navy for the
     * same reason the centre mark is (see `bakeCentreLogo`): §8.3's shadow
     * split lifts blue and drops red across the frame's darkest quartile, and a
     * deep blue that measures 64% on the swatch comes back past §2.5's 72%
     * ceiling once the grade has been through it.
     */
    uKeyColor: { value: new Color('#31506d') },
    uLineColor: { value: new Color('#e4ddcc') },
    /** x: paint bleed (texels), y: grain telegraph, z: normal flatten, w: key opacity. */
    uPaint: { value: new Vector4(0.22, 0.2, 0.72, 0.95) },
    /** x: board tone amp, y: grain tone amp, z: roughness base, w: roughness range. */
    uWood: { value: new Vector4(0.16, 1.55, ROUGH_BASE, ROUGH_RANGE) },
    /**
     * x: board width (m), y: grain→roughness, z: board grid origin (m),
     * w: coat veil — how far the top of the floor's range is pulled toward
     * neutral.
     *
     * A gloss coat's first-surface reflection is achromatic and rises with how
     * hard the lobe is being driven, so what comes back through it is the
     * maple's hue *diluted*: a wet broadcast floor is amber through the
     * mid-tones and close to white in its streaks. Doing that here rather than
     * leaving it to the tone curve is what lets the wood stay honestly warm
     * where §8.3 wants the mid-tones honest while the top fifth of the range —
     * which the hardwood owns in every framing, and which is what §8.3's
     * highlight split actually measures — goes neutral.
     */
    uBoard: { value: new Vector4(BOARD_WIDTH, 0.16, 0, 0.8) },
    /**
     * Milled joinery, all drawn analytically off the world position.
     *
     * x: groove half-width (m), y: strip-seam darkening, z: butt-joint
     * darkening, w: butt-joint pitch spread (m).
     *
     * These used to live in the tiling grain texture and none of them survived
     * to a frame. A groove is 2.2 mm: two of the sixty-four texels this tile
     * spends across a board, and about half a texel along it, so the near floor
     * — which minifies the tile roughly two to one — averaged the seams away and
     * never had the butt joints in the first place. From a coordinate they are
     * exact at any distance and box-filter down cleanly instead of aliasing.
     */
    uSeam: { value: new Vector4(SEAM_HALF, 0.5, 0.34, 1.35) },
    /**
     * Portable-floor panel module. x: panel length along the boards (m),
     * y: boards per panel across, z: cross-joint darkening, w: extra darkening
     * on the strip seam that is also a panel edge.
     *
     * §2.1 asks for the panel grid at 25–40% of the board-seam contrast, which
     * is what z and w are set against uSeam.y to give.
     */
    uPanel: { value: new Vector4(PANEL_LENGTH, PANEL_BOARDS, 0.16, 0.3) },
    /**
     * Specular shoulder knees: direct, indirect (env), clearcoat.
     *
     * These are asymptotes, in linear radiance, on how bright the *coat* alone
     * may get. They are the single most important numbers on this material.
     * The floor is a mirror-bright dielectric under a rig carrying dozens of
     * hot practicals, and specular is achromatic: let it run and it lays a
     * white sheet over the maple, which is exactly how hardwood ends up
     * reading as pale laminate no matter what the albedo says. Held here so
     * the coat can still go wet-looking without ever outrunning the wood.
     */
    uSpec: { value: new Vector3(0.9, 0.22, 0.5) },
    /**
     * The varnish tint. Poured polyurethane on maple is amber and several
     * coats deep, so almost everything the eye reads as "the highlight" has
     * been through the film twice — down and back — and comes out warm. Only
     * the thin first-surface term is neutral. Tinting the specular is what
     * keeps a bank reflection reading as *wet amber floor* instead of as a
     * grey blowout sitting on top of one.
     *
     * Held much closer to neutral than it was. The hardwood specular owns the
     * top fifth of the range in every framing, so §8.3's highlight split is
     * measuring this number almost directly — at (1, 0.925, 0.795) it read
     * 18.9–24.3 against a 4–12 target on all three frames.
     */
    uCoat: { value: new Color(1.0, 0.978, 0.94) },
    uSdfRange: { value: SDF_RANGE },
    uLogoR: { value: LOGO_RADIUS },
    uReflTex: { value: null as unknown },
    uReflMx: { value: new Matrix4() },
    /**
     * x: how much of the environment's specular the planar tap stands in for,
     * y: roughness→LOD, z: max LOD, w: streak half-length in reflection UV.
     *
     * x is the share of the environment probe's specular the tap stands in
     * for. It is not 1: the reflection frustum is the camera's, so the tap sees
     * nothing outside the frame and nothing behind it, and taking the whole
     * env term away costs the grazing near floor light it genuinely receives.
     * What it is *not* any more is a blend against the finished shading.
     * Blending it against the whole outgoing radiance, which is what this did,
     * threw the maple's diffuse away exactly where Fresnel is highest: the near
     * hardwood measured 56 against 175 for the same floor at twenty metres, and
     * the bottom fifth of a FLOOR framing came back as a dim reflection of the
     * far bowl instead of as wood.
     */
    uReflParams: { value: new Vector4(0.5, 5.5, 4.5, 0.07) },
  };

  init(engine: Engine): void {
    this.engine = engine;
    this.group.name = 'court';
    engine.scene.add(this.group);

    const q = engine.quality;
    // The whole-court map is always magnified in play, so height (across the
    // boards) is what buys line crispness; width follows the real aspect so
    // the distance fields stay isotropic.
    const mapH = Math.max(256, q.textureSize >> 1);
    const detailSize = Math.max(256, Math.min(1024, q.textureSize >> 1));
    // The centre mark is ~4 m across and the FLOOR framing magnifies it to
    // roughly 185 screen px/m, so it gets its own texture rather than the
    // whole-court bake's ~50 texels/m — the difference between a painted logo
    // and a blurred sticker. At 768 it was sampling one-for-one with the
    // screen, which is exactly the density at which the wordmark, the ball
    // glyph and the wear arcs all sit on the filter's cutoff; 1024 puts a
    // texel-and-a-third behind every pixel and the artwork resolves.
    const logoSize = Math.max(256, Math.min(1024, q.textureSize >> 1));
    this.bake = bakeCourt(mapH, detailSize, logoSize);

    const albedo = new CanvasTexture(this.bake.albedo);
    albedo.colorSpace = SRGBColorSpace;

    const mask = new CanvasTexture(this.bake.mask);
    mask.colorSpace = LinearSRGBColorSpace;

    const detail = new CanvasTexture(this.bake.detail.canvas);
    detail.colorSpace = LinearSRGBColorSpace;

    const totalW = COURT.length + COURT.apronX * 2;
    const totalD = COURT.width + COURT.apronZ * 2;
    detail.repeat.set(totalW / TILE_LENGTH, totalD / TILE_WIDTH);

    for (const t of [albedo, mask, detail]) {
      t.wrapS = t.wrapT = RepeatWrapping;
      t.anisotropy = engine.anisotropy;
      t.minFilter = LinearMipmapLinearFilter;
      t.generateMipmaps = true;
      t.needsUpdate = true;
      this.textures.push(t);
    }
    this.uniforms.uMask.value = mask;

    // Sampled analytically off world XZ, so it clamps rather than wraps.
    const logo = new CanvasTexture(this.bake.logo);
    logo.colorSpace = SRGBColorSpace;
    logo.wrapS = logo.wrapT = ClampToEdgeWrapping;
    logo.anisotropy = engine.anisotropy;
    logo.minFilter = LinearMipmapLinearFilter;
    logo.generateMipmaps = true;
    logo.needsUpdate = true;
    this.textures.push(logo);
    this.uniforms.uLogoMap.value = logo;

    // The plane's V runs from the +Z edge of the deck (`uv.y` is 1 at local +Y,
    // which rotateX(-90°) sends to world -Z), so the grain tile's board grid is
    // measured *down* from +totalD/2 — and `TILE_BOARDS` divides the tile
    // exactly, so board boundaries land on integer values of
    // (totalD/2 - z) / BOARD_WIDTH.
    //
    // The analytic grid used to count *up* from -totalD/2, and since
    // totalD / BOARD_WIDTH = 367.57 is not an integer the two disagreed by
    // 23 mm — 42% of a board. Every tone step therefore landed mid-board, which
    // is the blotching the per-board scatter exists to avoid, and the analytic
    // seams and the milled ones would have been drawn in different places.
    this.uniforms.uBoard.value.z = totalD * 0.5;

    const mat = new MeshPhysicalMaterial({
      map: albedo,
      normalMap: detail,
      normalScale: new Vector2(1.05, 1.05),
      // Polyurethane, not wood: one smooth interface with a real IOR.
      roughness: ROUGH_BASE,
      metalness: 0,
      ior: 1.52,
      // Sanding runs with the boards, so the specular lobe is stretched along
      // +U — which is the court's long axis.
      //
      // three.js widens the *other* axis for us:
      // `alphaT = mix(roughness², 1, anisotropy²)`. So anisotropy alone sets the
      // floor under the broad axis, and 0.38 pinned it at sqrt(0.1444) = 0.38
      // no matter how smooth the coat was — 0.39–0.45 once the roughness field
      // was folded in, against §2.3's 0.30 ceiling, which is why the highlight
      // was a wide dull smear with no core. At 0.25 with the tight axis at
      // 0.055–0.14 the pair lands at 0.26–0.28 broad against 0.06–0.14 tight:
      // a 2.1:1–4.6:1 lobe, inside §2.3's 2.5:1–6:1 wherever the coat is not
      // scuffed and duller in the lanes, which is what §2.3 asks for.
      anisotropy: 0.25,
      anisotropyRotation: 0,
      // The second lobe is deliberately weak and *not* sharp.
      //
      // three's clearcoat is isotropic, so every photon it returns is a round
      // highlight on a floor whose whole character is that its highlights are
      // not round — and worse, an isotropic lobe this tight reflects the
      // environment's ribbon band as a hard, unshouldered horizontal stripe
      // across the near floor at constant depth. That stripe, not the planar
      // reflection, was the "specular streak running across the boards" the
      // review measured: it survived with the direct, indirect and planar terms
      // all switched off. Broadened, halved, and put through the same shoulder
      // as everything else below.
      clearcoat: 0.09,
      clearcoatRoughness: 0.16,
      // Note: three r185 overwrites this with `scene.environmentIntensity` for
      // any material that does not carry its own envMap, which this does not —
      // so the floor's IBL gain is owned by `Lighting.ts`, not by this number.
      // Kept at the scene value so the code does not claim otherwise.
      envMapIntensity: 0.26,
    });
    mat.name = 'hardwood';

    if (q.floorReflections && q.reflectionResolution > 0) {
      mat.defines = { ...(mat.defines ?? {}), USE_PLANAR_REFLECTION: '' };
      // The wide five-tap stretch is five dependent texture reads per floor
      // pixel and the floor is most of a portrait frame, so the two tiers that
      // have to hold 60 fps on a phone take the three-tap version. It keeps the
      // board-aligned smear; it just resolves the far end of it more coarsely.
      if (q.tier === 'high' || q.tier === 'ultra') {
        mat.defines.WIDE_REFLECTION = '';
      }
      this.reflectEvery = q.tier === 'medium' ? 2 : 1;
      this.buildReflectionTarget(engine);
    }
    mat.onBeforeCompile = (shader) => this.patch(shader);
    // The tap count is a define, so it has to be part of the key or the two
    // variants share a compiled program.
    mat.customProgramCacheKey = () =>
      `ballin-hardwood-2${mat.defines?.WIDE_REFLECTION !== undefined ? '-wide' : ''}`;
    this.material = mat;

    const geo = new PlaneGeometry(totalW, totalD, 24, 14);
    geo.rotateX(-Math.PI / 2);
    this.floor = new Mesh(geo, mat);
    this.floor.receiveShadow = true;
    this.floor.castShadow = false;
    this.floor.name = 'hardwood';
    this.floor.matrixAutoUpdate = false;
    this.floor.updateMatrix();
    this.floor.onBeforeRender = (renderer, scene, camera) =>
      this.renderReflection(renderer, scene, camera);
    this.group.add(this.floor);
  }

  // -------------------------------------------------------------------------
  // Shader
  // -------------------------------------------------------------------------

  private patch(shader: {
    uniforms: Record<string, { value: unknown }>;
    vertexShader: string;
    fragmentShader: string;
  }): void {
    Object.assign(shader.uniforms, this.uniforms);

    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        /* glsl */ `#include <common>
varying vec3 vWPos;
#ifdef USE_PLANAR_REFLECTION
uniform mat4 uReflMx;
varying vec4 vReflCoord;
#endif`,
      )
      .replace(
        '#include <project_vertex>',
        /* glsl */ `#include <project_vertex>
vWPos = ( modelMatrix * vec4( transformed, 1.0 ) ).xyz;
#ifdef USE_PLANAR_REFLECTION
vReflCoord = uReflMx * vec4( vWPos, 1.0 );
#endif`,
      );

    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        /* glsl */ `#include <common>
uniform sampler2D uMask;
uniform sampler2D uLogoMap;
uniform vec3 uKeyColor;
uniform vec3 uLineColor;
uniform vec4 uPaint;
uniform vec4 uWood;
uniform vec4 uBoard;
uniform vec4 uSeam;
uniform vec4 uPanel;
uniform vec3 uSpec;
uniform vec3 uCoat;
uniform float uSdfRange;
uniform float uLogoR;
varying vec3 vWPos;
#ifdef USE_PLANAR_REFLECTION
uniform sampler2D uReflTex;
uniform vec4 uReflParams;
uniform mat4 uReflMx;
varying vec4 vReflCoord;
#endif

float courtHash( float n ) {
  return fract( sin( n * 12.9898 ) * 43758.5453123 );
}

/**
 * Box-filtered periodic groove.
 *
 * "d" is the distance to the nearest joint and "halfW" the groove's half-width,
 * both in period units; "foot" is the pixel's footprint in the same units. Wider
 * than the pixel the groove is drawn at full strength; narrower, it keeps its
 * total darkening and spreads it across the pixel. That is what lets a 2.2 mm
 * seam be a hard 1–2 px line on the near floor and fade honestly to sub-pixel by
 * mid-court instead of either aliasing into moiré or being filtered to nothing —
 * which is what happened to it while it lived in the tiling texture.
 */
float courtGroove( float d, float halfW, float foot ) {
  float w = max( foot, halfW * 2.0 );
  return saturate( halfW * 2.0 / w ) * ( 1.0 - smoothstep( 0.0, w * 0.5, d ) );
}

// Signed-distance coverage. Line work is barely 2.6 texels wide in the bake,
// so thresholding a distance field is the only way it stays a 1-pixel edge
// under magnification. The bleed term is the paint wicking into the grain.
float courtEdge( float s, float range, float bleed ) {
  float d = ( s - 0.5 ) * range;
  float w = min( fwidth( d ), 2.5 ) * 0.6 + bleed;
  return smoothstep( -w, w, d );
}`,
      )
      .replace(
        '#include <map_fragment>',
        /* glsl */ `#include <map_fragment>

vec4 courtMask = texture2D( uMask, vMapUv );
vec4 courtDetail = texture2D( normalMap, vNormalMapUv );

// Per-board tone scatter, computed analytically off world Z so it is crisp at
// any texel density and phase-locked to the tile's seams: the plane's V runs
// down from the +Z deck edge, which uBoard.z carries, and TILE_BOARDS divides
// the tile exactly, so integer boardCoord *is* a milled seam. Faded out once a
// pixel spans more than a board, which is the only honest way to antialias it.
float boardCoord = ( uBoard.z - vWPos.z ) / uBoard.x;
float boardFoot = max( fwidth( vWPos.z ) / uBoard.x, 1e-5 );
float boardId = floor( boardCoord );
float boardTone = ( courtHash( boardId * 1.13 ) - 0.5 ) * 2.0;
boardTone *= ( courtHash( boardId * 2.71 + 5.3 ) > 0.92 ) ? 2.3 : 1.0;
boardTone *= uWood.x * smoothstep( 1.7, 0.3, boardFoot );

// --- joinery ---------------------------------------------------------------
// Strip seams, staggered butt joints and the portable floor's panel grid, all
// drawn from the world coordinate. None of the three survived the trip through
// the tiling texture: a 2.2 mm groove is two texels of the sixty-four the tile
// spends across a board and half a texel along it, and the near floor minifies
// the tile about two to one, so the seams filtered to a 1.8-unit ripple and the
// butt joints were gone before the bake finished. Drawn here they are exact.
float seamId = floor( boardCoord + 0.5 );
float dSeam = abs( boardCoord - seamId );
float seamHalf = uSeam.x / uBoard.x;
// Core plus lip. The groove itself is 2.2 mm — 1.2 px at the near floor, and
// §7.4 puts a 3–4 px near-field circle of confusion over exactly that ground,
// so a bare core loses three quarters of its depth before it reaches the frame.
// The lip is the milled bevel and the finish that pools in it: three times as
// wide at under half the depth, which is both what the joint physically looks
// like and what survives the blur.
float seam = max(
  courtGroove( dSeam, seamHalf, boardFoot ),
  courtGroove( dSeam, seamHalf * 3.8, boardFoot ) * 0.45 );
// A panel edge is a strip edge on a real deck, so it is the same seam cut a
// little deeper rather than a line of its own wandering across the boards.
seam *= 1.0 + uPanel.w * step( mod( seamId, uPanel.y ), 0.5 );

// Butt joints. Strips are finite and the ends are staggered board to board, so
// the pitch and the phase are both hashed off the board index — never a column
// of aligned joints, which §2.1 names as the tell.
float jPitch = 1.35 + courtHash( boardId * 4.19 + 1.7 ) * uSeam.w;
float jCoord = ( vWPos.x + courtHash( boardId * 9.31 + 0.4 ) * jPitch ) / jPitch;
float jFoot = max( fwidth( vWPos.x ) / jPitch, 1e-5 );
float jFrac = fract( jCoord );
float joint = courtGroove( min( jFrac, 1.0 - jFrac ), uSeam.x * 0.8 / jPitch, jFoot );

// Panel cross joints run unbroken across every board at the same station,
// which is what distinguishes them from the staggered butt joints.
float pCoord = vWPos.x / uPanel.x;
float pFoot = max( fwidth( vWPos.x ) / uPanel.x, 1e-5 );
float pFrac = fract( pCoord );
float panelCut = courtGroove( min( pFrac, 1.0 - pFrac ), uSeam.x * 0.55 / uPanel.x, pFoot );

float millCut = max( seam * uSeam.y, max( joint * uSeam.z, panelCut * uPanel.z ) );

// Buff and drag marks.
//
// The whole-court bake holds the *distribution* of wear — traffic-weighted, so
// the paint is black with rubber and the corners are clean — but at ~50
// texels/m a 200 mm scuff is four texels, and the near floor magnifies that
// bake tenfold, so all of it dissolves into haze exactly where the camera is
// closest. This resamples the grain tile at a long, shallow, rotated scale to
// get the high-frequency half back: soft streaks a few hundred millimetres
// long lying with the direction of play. It lands almost entirely in
// roughness, because that is what shoe-polished coat actually is — the albedo
// barely moves and the marks show only as the highlight breaking up over them.
vec2 buffUv = vec2(
  vWPos.x * 0.052 + vWPos.z * 0.021,
  vWPos.z * 0.138 - vWPos.x * 0.0055 );
float courtBuff = texture2D( normalMap, buffUv ).a - 0.5;

float courtGrain = ( courtDetail.a - 0.5 ) * uWood.y;
// Late wood is not just darker maple, it is *browner* maple — the dense
// summer growth holds more extractive and drinks more of the amber finish. So
// the grain and the plank scatter are applied with a hue slope: where the wood
// goes dark it also loses blue and gains red, which is what stops a tonal
// grain from reading as a grey pencil rubbing over a flat colour.
vec3 woodTint = vec3( 0.72, 1.0, 1.55 );
diffuseColor.rgb *= 1.0 + ( boardTone + courtGrain ) * woodTint + courtBuff * 0.075;
diffuseColor.rgb *= courtMask.a * ( 1.0 - millCut );

// Paint is pigment on sanded wood, then varnish. The grain, the scuffs and the
// joinery under it still modulate what comes back.
float telegraph = ( 1.0 + ( boardTone + courtGrain ) * uPaint.y ) * ( 1.0 - millCut * uPaint.y );

// --- centre logo -------------------------------------------------------
// Goes down before the line work, so the division line and the centre circle
// are painted across it exactly as they are on a real deck. Alpha is paint
// coverage: where the mark has worn through, bare maple comes back with its
// own grain rather than the logo merely fading toward a lighter flat colour.
vec2 logoUv = vec2( 0.5 + vWPos.x / ( uLogoR * 2.0 ), 0.5 - vWPos.z / ( uLogoR * 2.0 ) );
vec4 logoTex = texture2D( uLogoMap, logoUv );
float logoPaint = logoTex.a *
  ( 1.0 - smoothstep( uLogoR * 0.985, uLogoR * 1.01, length( vWPos.xz ) ) );
diffuseColor.rgb = mix( diffuseColor.rgb, logoTex.rgb * telegraph, logoPaint );

float keyPaint = courtEdge( courtMask.r, uSdfRange, uPaint.x ) * uPaint.w;
float linePaint = courtEdge( courtMask.g, uSdfRange, uPaint.x );
diffuseColor.rgb = mix( diffuseColor.rgb, uKeyColor * telegraph, keyPaint );
diffuseColor.rgb = mix( diffuseColor.rgb, uLineColor * telegraph, linePaint );

float courtPaint = max( max( keyPaint, linePaint ), logoPaint );`,
      )
      .replace(
        '#include <roughnessmap_fragment>',
        /* glsl */ `#include <roughnessmap_fragment>
roughnessFactor = uWood.z + courtMask.b * uWood.w;
// Late wood sits fractionally rougher, so the grain modulates the highlight
// and not only the base colour.
roughnessFactor += ( 0.5 - courtDetail.a ) * uBoard.y;
// The joinery breaks the coat as well as darkening it — a seam that only
// darkens reads as a printed line rather than a groove.
roughnessFactor += seam * 0.07 + joint * 0.055 + panelCut * 0.024;
// Sole-polished streaks: rougher where the coat has been abraded, glassier in
// the lanes nobody walks. Weighted by the bake's own traffic field so it stays
// out of the corners and off the apron.
roughnessFactor -= courtBuff * ( 0.045 + courtMask.b * 0.07 );
// Paint fills the grain: marginally smoother under the same coat.
roughnessFactor *= mix( 1.0, 0.9, courtPaint );
// The ceiling is §2.3's broad-axis cap read back through three's
// alphaT = mix(roughness², 1, anisotropy²): past ~0.15 tight the wide axis
// leaves the 0.18–0.30 band and the highlight stops having a core.
roughnessFactor = clamp( roughnessFactor, 0.05, 0.26 );`,
      )
      .replace(
        '#include <normal_fragment_maps>',
        /* glsl */ `#include <normal_fragment_maps>
normal = normalize( mix( normal, nonPerturbedNormal, courtPaint * uPaint.z ) );`,
      )
      .replace(
        '#include <lights_fragment_end>',
        /* glsl */ `#include <lights_fragment_end>
// Polyurethane is a mirror-bright interface and the arena carries very hot
// practicals, so a single fixture can otherwise blow a whole board to flat
// white and take the grain with it. Roll the floor's *specular* off through a
// soft shoulder — diffuse is untouched, so the maple keeps its tone and the
// highlight keeps its streak shape instead of clipping into a slab.
reflectedLight.directSpecular /= 1.0 + reflectedLight.directSpecular / uSpec.x;
reflectedLight.indirectSpecular /= 1.0 + reflectedLight.indirectSpecular / uSpec.y;
// ...and then tint what is left. Several coats of amber varnish sit between
// the eye and the maple, so the bulk of the returned specular has been
// filtered twice on its way down and back. Without this the highlight is pure
// achromatic white, which desaturates the wood underneath it in proportion to
// how bright the floor is — i.e. it bleaches hardest exactly where the frame
// is looking.
//
// The tint releases toward neutral as the lobe gets hot, because the two
// specular terms have different paths: the dim, broad part of the highlight is
// mostly light that went down through the film and came back, and is doubly
// filtered; the searing core is the first-surface reflection off the top of
// the coat, which never entered it and is the colour of the fixture. Holding
// the amber all the way to the peak also drives §8.3's highlight split well
// past +12 on any frame the floor dominates.
reflectedLight.directSpecular *= mix(
  uCoat, vec3( 1.0 ), saturate( dot( reflectedLight.directSpecular, vec3( 0.9 ) ) ) );
reflectedLight.indirectSpecular *= mix(
  uCoat, vec3( 1.0 ), saturate( dot( reflectedLight.indirectSpecular, vec3( 0.9 ) ) ) );
#ifdef USE_CLEARCOAT
// The clearcoat lobe stands in for the thin, un-yellowed top surface, so it
// stays neutral. It is the only genuinely white highlight on the floor.
//
// The *indirect* half of it needs the shoulder more than the direct half does:
// it is an isotropic mirror of a room whose brightest feature is a continuous
// horizontal LED band, and unshouldered it lays that band across the near floor
// as a hard stripe running the wrong way across the grain.
clearcoatSpecularDirect /= 1.0 + clearcoatSpecularDirect / uSpec.z;
clearcoatSpecularIndirect /= 1.0 + clearcoatSpecularIndirect / ( uSpec.z * 0.35 );
#endif`,
      )
      .replace(
        '#include <lights_physical_fragment>',
        /* glsl */ `#include <lights_physical_fragment>
#ifdef USE_CLEARCOAT
material.clearcoatRoughness = clamp(
  material.clearcoatRoughness + courtMask.b * 0.07, 0.03, 0.45 );
#endif`,
      )
      .replace(
        '#include <opaque_fragment>',
        /* glsl */ `
#ifdef USE_PLANAR_REFLECTION
{
  vec2 ruv = vReflCoord.xy / max( vReflCoord.w, 1e-4 );
  float valid = step( 0.0, vReflCoord.w );
  // Fade at the edges of the reflection frustum so nothing pops at the border.
  float edge = smoothstep( 0.0, 0.05, ruv.x ) * smoothstep( 1.0, 0.95, ruv.x ) *
               smoothstep( 0.0, 0.04, ruv.y ) * smoothstep( 1.0, 0.96, ruv.y );

  // The board axis, carried into reflection UV.
  //
  // uReflMx is affine in world position, so a step along +X is a *constant*
  // offset in homogeneous reflection space and the projected direction comes
  // out exact rather than guessed in screen space. This matters: §2.4's
  // "blurred more vertically than horizontally" means blurred *along the
  // boards*, and the boards run at 20–75° from screen horizontal across the
  // near floor of a portrait frame. Offsetting the taps in the target's own
  // vertical, which is what this did, smeared the result along screen-x once
  // the projective mapping was applied — across the grain, the opposite of an
  // anisotropic coat, and it put the measured streak axis 40° off the boards.
  vec4 rTan = vReflCoord + uReflMx * vec4( 0.35, 0.0, 0.0, 0.0 );
  vec2 along = rTan.xy / max( rTan.w, 1e-4 ) - ruv;
  along = normalize( along + vec2( 1e-7, 1e-7 ) );

  float viewDist = length( vViewPosition );
  float lod0 = clamp( material.roughness * uReflParams.y + viewDist * 0.02,
                      0.0, uReflParams.z );
  // Five taps along the boards with the blur growing outward. Height above the
  // deck maps to distance along this axis in the mirrored image, so the tap
  // that lands on a shoe is near-sharp and the ones a body-length up are a
  // smear — which is exactly §2.4's sharpness falloff, for free.
  float sm = uReflParams.w * ( 0.25 + material.roughness * 4.0 );
#ifdef WIDE_REFLECTION
  vec3 refl = textureLod( uReflTex, ruv, lod0 ).rgb * 0.34;
  refl += textureLod( uReflTex, ruv + along * sm, lod0 + 0.85 ).rgb * 0.19;
  refl += textureLod( uReflTex, ruv - along * sm, lod0 + 0.85 ).rgb * 0.19;
  refl += textureLod( uReflTex, ruv + along * sm * 2.4, lod0 + 1.9 ).rgb * 0.14;
  refl += textureLod( uReflTex, ruv - along * sm * 2.4, lod0 + 1.9 ).rgb * 0.14;
#else
  vec3 refl = textureLod( uReflTex, ruv, lod0 ).rgb * 0.46;
  refl += textureLod( uReflTex, ruv + along * sm * 1.6, lod0 + 1.4 ).rgb * 0.27;
  refl += textureLod( uReflTex, ruv - along * sm * 1.6, lod0 + 1.4 ).rgb * 0.27;
#endif
  // A varnish reflection carries the *structure* of the room — dark bowl,
  // bright ceiling — not the crowd's shirt colours, which at this blur would
  // read as coloured bruises on the wood.
  float reflLum = dot( refl, vec3( 0.2126, 0.7152, 0.0722 ) );
  refl = mix( vec3( reflLum ), refl, 0.7 );
  // Seen through the film, like everything else off this floor — and released
  // toward neutral in the hot core for the same reason as the direct lobe.
  refl *= mix( uCoat, vec3( 1.0 ), saturate( reflLum * 2.1 ) );

  // Schlick against a real dielectric coat, not a remapped one. This is the
  // grazing-angle gain §2.3 asks for: ~4% face-on, climbing at the horizon.
  float fres = pow( 1.0 - saturate( dot( geometryNormal, geometryViewDir ) ), 5.0 );
  float F = 0.04 + 0.96 * fres;
  float k = uReflParams.x * edge * valid * ( 1.0 - smoothstep( 14.0, 34.0, viewDist ) );

  // The planar tap *stands in for part of the environment probe*, not for the
  // frame.
  //
  // It is a better estimate of what the coat sees than a PMREM of the whole
  // room, so it displaces the indirect specular term and nothing else.
  // Blending it against the finished outgoing radiance — which is what this
  // did — discards the maple's diffuse in proportion to Fresnel, i.e. hardest
  // exactly where a portrait frame spends most of its hardwood pixels, and the
  // near floor came back as a dim picture of the far bowl instead of as wood.
  // Written this way the diffuse is untouched, the grazing gain is a real
  // Fresnel-weighted reflection *added to* the boards, and a player standing on
  // the floor darkens the coat under his shoes because he is occluding the
  // room, which is the physical reason a reflection is visible at all.
  //
  // The shoulder is the one the environment lobe gets. The ribbon boards are
  // the hottest thing in the room and a planar tap hands them back at full HDR
  // range, so without it the near floor comes back striped with hard bright
  // bands of reflected LED — which is the one structure in a portrait frame
  // that runs *across* the boards.
  vec3 planarSpec = refl * F;
  planarSpec /= 1.0 + planarSpec / uSpec.y;
  outgoingLight += saturate( k ) * ( planarSpec - reflectedLight.indirectSpecular );
}
#endif
// The film veils, per uBoard.w. Applied to everything the floor returns —
// paint included, because the paint is under the same coat and §2.5's whole
// point is that it behaves the same.
{
  float coatL = dot( outgoingLight, vec3( 0.2126, 0.7152, 0.0722 ) );
  outgoingLight = mix( outgoingLight, vec3( coatL ),
                       uBoard.w * smoothstep( 0.17, 0.52, coatL ) );
}
#include <opaque_fragment>`,
      );
  }

  // -------------------------------------------------------------------------
  // Planar reflection
  // -------------------------------------------------------------------------

  private buildReflectionTarget(engine: Engine): void {
    const h = engine.quality.reflectionResolution;
    const aspect = engine.pixelWidth / Math.max(1, engine.pixelHeight);
    const w = Math.max(64, Math.round(h * (aspect > 0 ? aspect : 9 / 19.5)));
    this.reflectionRT?.dispose();
    const rt = new WebGLRenderTarget(w, h, {
      type: HalfFloatType,
      depthBuffer: true,
      stencilBuffer: false,
      generateMipmaps: true,
      minFilter: LinearMipmapLinearFilter,
    });
    rt.texture.name = 'court.reflection';
    this.reflectionRT = rt;
    this.uniforms.uReflTex.value = rt.texture;
  }

  /**
   * Mirrors the active camera through the floor plane and renders the arena
   * into the reflection target. Runs from `onBeforeRender` so it always sees
   * the camera the frame is actually being drawn with, and guards against
   * re-entry so the nested render cannot recurse.
   */
  private renderReflection(renderer: WebGLRenderer, scene: Scene, camera: Camera): void {
    const rt = this.reflectionRT;
    if (!rt || this.reflecting) return;
    if (camera !== this.engine.camera) return;
    const frame = this.engine.frame;
    if (frame === this.lastReflectFrame) return;
    if (frame % this.reflectEvery !== 0 && this.lastReflectFrame >= 0) return;

    const main = camera as PerspectiveCamera;
    this._pos.setFromMatrixPosition(main.matrixWorld);
    if (this._pos.y <= 0.05) return; // camera at or under the deck

    this._rot.extractRotation(main.matrixWorld);
    this._look.set(0, 0, -1).applyMatrix4(this._rot).add(this._pos);
    this._up.set(0, 1, 0).applyMatrix4(this._rot);

    const cam = this.reflectCamera;
    cam.position.set(this._pos.x, -this._pos.y, this._pos.z);
    cam.up.set(-this._up.x, this._up.y, -this._up.z);
    cam.lookAt(this._look.x, -this._look.y, this._look.z);
    cam.near = main.near;
    cam.far = main.far;
    cam.fov = main.fov;
    cam.aspect = main.aspect;
    cam.updateMatrixWorld(true);
    cam.projectionMatrix.copy(main.projectionMatrix);

    // Oblique near plane, so nothing under the deck can bleed into the mirror.
    this._plane.copy(MIRROR_PLANE).applyMatrix4(cam.matrixWorldInverse);
    this._clip.set(
      this._plane.normal.x,
      this._plane.normal.y,
      this._plane.normal.z,
      this._plane.constant,
    );
    const pm = cam.projectionMatrix.elements;
    this._q.set(
      (Math.sign(this._clip.x) + pm[8]) / pm[0],
      (Math.sign(this._clip.y) + pm[9]) / pm[5],
      -1,
      (1 + pm[10]) / pm[14],
    );
    const denom = this._clip.dot(this._q);
    if (Math.abs(denom) > 1e-6) {
      this._clip.multiplyScalar(2 / denom);
      pm[2] = this._clip.x;
      pm[6] = this._clip.y;
      pm[10] = this._clip.z + 1 - 0.004;
      pm[14] = this._clip.w;
    }

    this.reflecting = true;
    this.lastReflectFrame = frame;

    const prevTarget = renderer.getRenderTarget();
    const prevFace = renderer.getActiveCubeFace();
    const prevMip = renderer.getActiveMipmapLevel();
    const prevShadowAuto = renderer.shadowMap.autoUpdate;
    const prevXr = renderer.xr.enabled;

    renderer.xr.enabled = false;
    renderer.shadowMap.autoUpdate = false; // the frame's maps are already current
    this.floor.visible = false;

    renderer.setRenderTarget(rt);
    renderer.clear();
    renderer.render(scene, cam);

    this.floor.visible = true;
    renderer.setRenderTarget(prevTarget, prevFace, prevMip);
    renderer.shadowMap.autoUpdate = prevShadowAuto;
    renderer.xr.enabled = prevXr;
    this.reflecting = false;

    // Bias · projection · view — sampled with the floor's world position.
    const m = this.uniforms.uReflMx.value;
    m.set(0.5, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 1);
    m.multiply(cam.projectionMatrix);
    m.multiply(cam.matrixWorldInverse);
  }

  resize(_width: number, _height: number, engine: Engine): void {
    if (this.reflectionRT) this.buildReflectionTarget(engine);
  }

  // -------------------------------------------------------------------------

  /**
   * World-space floor height. The court is a true plane: every bit of milled
   * relief is under half a millimetre and lives in the normal map, so physics
   * and AI get a flat, deterministic surface rather than a lie about
   * displacement that is not in the geometry.
   */
  heightAt(_p: Vector3): number {
    return this.surfaceY;
  }

  /** True when a world position is inside the 94 × 50 ft playing surface. */
  inBounds(p: Vector3): boolean {
    return Math.abs(p.x) <= COURT.halfLength && Math.abs(p.z) <= COURT.halfWidth;
  }

  dispose(): void {
    this.floor?.geometry.dispose();
    this.material?.dispose();
    for (const t of this.textures) t.dispose();
    this.textures.length = 0;
    this.reflectionRT?.dispose();
    this.reflectionRT = null;
    this.group.removeFromParent();
  }
}
