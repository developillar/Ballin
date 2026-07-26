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
    uKeyColor: { value: new Color('#274465') },
    uLineColor: { value: new Color('#e2dbcb') },
    /** x: paint bleed (texels), y: grain telegraph, z: normal flatten, w: key opacity. */
    uPaint: { value: new Vector4(0.22, 0.18, 0.72, 0.95) },
    /** x: board tone amp, y: grain tone amp, z: roughness base, w: roughness range. */
    uWood: { value: new Vector4(0.1, 1.3, ROUGH_BASE, ROUGH_RANGE) },
    /** x: board width (m), y: grain→roughness, z/w: spare. */
    uBoard: { value: new Vector4(BOARD_WIDTH, 0.26, 0, 0) },
    /** Specular shoulder knees: direct, indirect (env), clearcoat. */
    uSpec: { value: new Vector3(1.7, 2.4, 1.5) },
    uSdfRange: { value: SDF_RANGE },
    uLogoR: { value: LOGO_RADIUS },
    uReflTex: { value: null as unknown },
    uReflMx: { value: new Matrix4() },
    /** x: strength, y: roughness→blur, z: max LOD. */
    uReflParams: { value: new Vector3(0.4, 13, 5) },
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
    this.bake = bakeCourt(mapH, detailSize);

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

    const mat = new MeshPhysicalMaterial({
      map: albedo,
      normalMap: detail,
      normalScale: new Vector2(1.25, 1.25),
      // Polyurethane, not wood: one smooth interface with a real IOR.
      roughness: 0.11,
      metalness: 0,
      ior: 1.52,
      // Sanding runs with the boards, so the specular lobe is stretched along
      // +U — which is the court's long axis. At roughness ~0.13 that puts the
      // lobe at ~0.13 across the boards and ~0.33 along them: a 2.5:1 streak.
      anisotropy: 0.3,
      anisotropyRotation: 0,
      clearcoat: 0.2,
      clearcoatRoughness: 0.1,
      envMapIntensity: q.floorReflections ? 0.34 : 0.58,
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
uniform vec3 uKeyColor;
uniform vec3 uLineColor;
uniform vec4 uPaint;
uniform vec4 uWood;
uniform vec4 uBoard;
uniform vec3 uSpec;
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
// any texel density and phase-locked to the tile's seams. Faded out once a
// pixel spans more than a board, which is the only honest way to antialias it.
float boardCoord = vWPos.z / uBoard.x;
float boardFoot = fwidth( boardCoord );
float boardId = floor( boardCoord );
float boardTone = ( courtHash( boardId * 1.13 ) - 0.5 ) * 2.0;
boardTone *= ( courtHash( boardId * 2.71 + 5.3 ) > 0.92 ) ? 2.6 : 1.0;
boardTone *= uWood.x * smoothstep( 1.7, 0.3, boardFoot );

float courtGrain = ( courtDetail.a - 0.5 ) * uWood.y;
diffuseColor.rgb *= 1.0 + boardTone + courtGrain;
diffuseColor.rgb *= courtMask.a;

float keyPaint = courtEdge( courtMask.r, uSdfRange, uPaint.x ) * uPaint.w;
float linePaint = courtEdge( courtMask.g, uSdfRange, uPaint.x );
// Paint is pigment on sanded wood, then varnish. The grain and the scuffs
// under it still modulate what comes back.
float telegraph = 1.0 + ( boardTone + courtGrain ) * uPaint.y;
diffuseColor.rgb = mix( diffuseColor.rgb, uKeyColor * telegraph, keyPaint );
diffuseColor.rgb = mix( diffuseColor.rgb, uLineColor * telegraph, linePaint );

float logoPaint = 1.0 - smoothstep( uLogoR * 0.985, uLogoR * 1.015, length( vWPos.xz ) );
float courtPaint = max( max( keyPaint, linePaint ), logoPaint * 0.85 );`,
      )
      .replace(
        '#include <roughnessmap_fragment>',
        /* glsl */ `#include <roughnessmap_fragment>
roughnessFactor = uWood.z + courtMask.b * uWood.w;
// Late wood sits fractionally rougher, so the grain modulates the highlight
// and not only the base colour.
roughnessFactor += ( 0.5 - courtDetail.a ) * uBoard.y;
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
#ifdef USE_CLEARCOAT
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
  refl = mix( vec3( dot( refl, vec3( 0.2126, 0.7152, 0.0722 ) ) ), refl, 0.55 );

  float fres = pow( 1.0 - saturate( dot( geometryNormal, geometryViewDir ) ), 5.0 );
  float k = uReflParams.x * mix( 0.03, 1.0, fres ) * edge * valid;
  k *= 1.0 - smoothstep( 11.0, 32.0, viewDist );
  k *= saturate( 1.0 - material.roughness * 2.2 );
  outgoingLight += refl * max( k, 0.0 );
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
