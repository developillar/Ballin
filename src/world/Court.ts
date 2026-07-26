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
import { BOARD_WIDTH, TILE_LENGTH, TILE_WIDTH } from '../textures/courtWood';

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
    uKeyColor: { value: new Color('#22405f') },
    uLineColor: { value: new Color('#e4ddcc') },
    /** x: paint bleed (texels), y: grain telegraph, z: normal flatten, w: key opacity. */
    uPaint: { value: new Vector4(0.22, 0.2, 0.72, 0.95) },
    /** x: board tone amp, y: grain tone amp, z: roughness base, w: roughness range. */
    uWood: { value: new Vector4(0.135, 1.55, ROUGH_BASE, ROUGH_RANGE) },
    /** x: board width (m), y: grain→roughness, z: board grid origin (m), w: spare. */
    uBoard: { value: new Vector4(BOARD_WIDTH, 0.26, 0, 0) },
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
    uSpec: { value: new Vector3(0.72, 0.34, 0.5) },
    /**
     * The varnish tint. Poured polyurethane on maple is amber and several
     * coats deep, so almost everything the eye reads as "the highlight" has
     * been through the film twice — down and back — and comes out warm. Only
     * the thin first-surface term is neutral. Tinting the specular is what
     * keeps a bank reflection reading as *wet amber floor* instead of as a
     * grey blowout sitting on top of one.
     */
    uCoat: { value: new Color(1.0, 0.925, 0.795) },
    uSdfRange: { value: SDF_RANGE },
    uLogoR: { value: LOGO_RADIUS },
    uReflTex: { value: null as unknown },
    uReflMx: { value: new Matrix4() },
    /**
     * x: strength at full Fresnel, y: roughness→blur, z: max LOD.
     *
     * x is capped well under 1 on purpose. A physical coat really does go
     * near-mirror at the horizon line, but the planar tap carries none of the
     * floor's own signal, so at a blend much past ~0.4 the grazing near floor —
     * the biggest, closest, most scrutinised patch of hardwood in a portrait
     * frame — loses its grain, its seams and its scuffs and goes smooth.
     */
    uReflParams: { value: new Vector3(0.6, 13, 5) },
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
    // The centre mark is ~4 m across and the FLOOR framing magnifies it, so it
    // gets its own texture at ~190 texels/m rather than the whole-court bake's
    // ~50 — the difference between a painted logo and a blurred sticker.
    const logoSize = Math.max(256, Math.min(768, q.textureSize >> 1));
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

    // The tiling grain wraps from the -Z edge of the deck, so the analytic
    // per-board tone has to be phased from the same origin or every tone step
    // lands mid-board instead of on a seam — which reads as blotching rather
    // than as plank scatter.
    this.uniforms.uBoard.value.z = totalD * 0.5;

    const mat = new MeshPhysicalMaterial({
      map: albedo,
      normalMap: detail,
      normalScale: new Vector2(1.05, 1.05),
      // Polyurethane, not wood: one smooth interface with a real IOR.
      roughness: 0.11,
      metalness: 0,
      ior: 1.52,
      // Sanding runs with the boards, so the specular lobe is stretched along
      // +U — which is the court's long axis. At roughness ~0.13 that puts the
      // lobe at ~0.12 across the boards and ~0.42 along them: a 3.5:1 streak,
      // inside §2.3's 2.5:1–6:1.
      anisotropy: 0.38,
      anisotropyRotation: 0,
      clearcoat: 0.16,
      clearcoatRoughness: 0.1,
      // The environment is a whole bright arena and this term is achromatic
      // and covers every texel of the floor at once, so it is the cheapest way
      // in the material to bleach the maple. Kept low; the shaped, tinted
      // planar reflection carries the room instead.
      envMapIntensity: q.floorReflections ? 0.17 : 0.3,
    });
    mat.name = 'hardwood';

    if (q.floorReflections && q.reflectionResolution > 0) {
      mat.defines = { ...(mat.defines ?? {}), USE_PLANAR_REFLECTION: '' };
      this.reflectEvery = q.tier === 'medium' ? 2 : 1;
      this.buildReflectionTarget(engine);
    }
    mat.onBeforeCompile = (shader) => this.patch(shader);
    mat.customProgramCacheKey = () => 'ballin-hardwood-1';
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
uniform vec3 uSpec;
uniform vec3 uCoat;
uniform float uSdfRange;
uniform float uLogoR;
varying vec3 vWPos;
#ifdef USE_PLANAR_REFLECTION
uniform sampler2D uReflTex;
uniform vec3 uReflParams;
varying vec4 vReflCoord;
#endif

float courtHash( float n ) {
  return fract( sin( n * 12.9898 ) * 43758.5453123 );
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
// any texel density and phase-locked to the tile's seams (uBoard.z carries the
// deck's -Z edge, which is where the grain tile wraps from). Faded out once a
// pixel spans more than a board, which is the only honest way to antialias it.
float boardCoord = ( vWPos.z + uBoard.z ) / uBoard.x;
float boardFoot = fwidth( boardCoord );
float boardId = floor( boardCoord );
float boardTone = ( courtHash( boardId * 1.13 ) - 0.5 ) * 2.0;
boardTone *= ( courtHash( boardId * 2.71 + 5.3 ) > 0.92 ) ? 2.6 : 1.0;
boardTone *= uWood.x * smoothstep( 1.7, 0.3, boardFoot );

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
vec3 woodTint = vec3( 0.78, 1.0, 1.42 );
diffuseColor.rgb *= 1.0 + ( boardTone + courtGrain ) * woodTint + courtBuff * 0.075;
diffuseColor.rgb *= courtMask.a;

// Paint is pigment on sanded wood, then varnish. The grain and the scuffs
// under it still modulate what comes back.
float telegraph = 1.0 + ( boardTone + courtGrain ) * uPaint.y;

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
// Sole-polished streaks: rougher where the coat has been abraded, glassier in
// the lanes nobody walks. Weighted by the bake's own traffic field so it stays
// out of the corners and off the apron.
roughnessFactor -= courtBuff * ( 0.09 + courtMask.b * 0.13 );
// Paint fills the grain: marginally smoother under the same coat.
roughnessFactor *= mix( 1.0, 0.9, courtPaint );
roughnessFactor = clamp( roughnessFactor, 0.03, 0.7 );`,
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
clearcoatSpecularDirect /= 1.0 + clearcoatSpecularDirect / uSpec.z;
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
  float viewDist = length( vViewPosition );
  float lod = clamp( material.roughness * uReflParams.y + viewDist * 0.055,
                     0.0, uReflParams.z );
  // Three vertical taps: an anisotropic coat smears its reflections along the
  // boards, which in every broadcast framing runs away from camera.
  float sm = ( 0.0022 + 0.0016 * lod );
  vec3 refl = textureLod( uReflTex, ruv, lod ).rgb * 0.5;
  refl += textureLod( uReflTex, ruv + vec2( 0.0, sm ), lod + 0.7 ).rgb * 0.25;
  refl += textureLod( uReflTex, ruv - vec2( 0.0, sm ), lod + 0.7 ).rgb * 0.25;
  // A varnish reflection carries the *structure* of the room — dark bowl,
  // bright ceiling — not the crowd's shirt colours, which at this blur would
  // read as coloured bruises on the wood.
  float reflLum = dot( refl, vec3( 0.2126, 0.7152, 0.0722 ) );
  refl = mix( vec3( reflLum ), refl, 0.7 );
  // Seen through the film, like everything else off this floor — and released
  // toward neutral in the hot core for the same reason as the direct lobe.
  refl *= mix( uCoat, vec3( 1.0 ), saturate( reflLum * 2.1 ) );

  // Schlick against a real dielectric coat, not a remapped one. This is the
  // grazing-angle gain §2.3 asks for: ~4% face-on, climbing toward a mirror at
  // the horizon line.
  float fres = pow( 1.0 - saturate( dot( geometryNormal, geometryViewDir ) ), 5.0 );
  float k = uReflParams.x * ( 0.035 + 0.965 * fres ) * edge * valid;
  k *= 1.0 - smoothstep( 11.0, 32.0, viewDist );
  k *= saturate( 1.0 - material.roughness * 2.2 );
  // *Blend*, do not add. Adding a mirror image on top of a fully shaded
  // surface is energy the floor never received, and because the reflected room
  // is bright and near-neutral it lands as a white sheet over the near
  // hardwood — the exact grazing angles a broadcast framing spends most of its
  // pixels on. Replacing the shading instead means the boards genuinely carry
  // the dark bowl and the bright ceiling, and can never be brighter than the
  // room they are reflecting.
  outgoingLight = mix( outgoingLight, refl, saturate( k ) );
}
#endif
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
