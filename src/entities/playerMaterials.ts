/**
 * Player materials: skin, kit, shoes and hair.
 *
 * The interesting one is skin. `MeshPhysicalMaterial` gets us a plausible base,
 * but three things it does not do are exactly the three things that identify
 * real-time skin, so all three are patched in:
 *
 *  1. **Subsurface warmth at the terminator.** Light that enters the skin,
 *     scatters and re-emerges does so a millimetre or two away, which shows as
 *     a reddening *band* where the lit side rolls into shadow — hue toward
 *     5–20°, saturation up, over roughly 6–20 px at gameplay distance. It is
 *     modelled here as the difference between a wrapped diffuse response and
 *     the Lambert one, which is non-zero only in that band and exactly zero on
 *     the fully lit side and in the core shadow.
 *
 *     Round 0 had this term and it did not read, for an arithmetic reason worth
 *     stating: it was scaled by `material.diffuseColor`, and on the deep tone
 *     that is ~0.08 linear, so the band's absolute contribution was under two
 *     sRGB units — below the film grain. Scatter is not albedo. What comes back
 *     out of skin has been filtered by roughly *half* the absorption path an
 *     albedo multiply implies, so the scale here is `sqrt(diffuseColor)` times a
 *     bright scatter tint. That keeps the band tracking the tone (deeper skin
 *     still shows a shorter, less saturated one) while leaving it visible.
 *  2. **A second specular lobe.** Skin has a broad oil sheen (roughness
 *     ~0.4–0.55, from the roughness map) *plus* sharp sweat highlights
 *     (roughness ~0.08–0.15) sitting on top of it. One lobe alone reads as
 *     plastic. Sweat is driven by a 0..1 uniform so it can build over a
 *     possession, masked by a per-vertex region weight and a texture channel,
 *     so it beads on the forehead, deltoids and forearms rather than uniformly.
 *  3. **An eye.** The sclera is not skin, and painting it into a tone-neutral
 *     atlas cannot work: the per-player tone multiply takes any white down to
 *     the tone's own luminance. The atlas therefore carries an eye *mask* in
 *     the data map's blue channel and the shader substitutes the material —
 *     sclera, iris and a cornea roughness that produces a real catchlight from
 *     the nearest bank.
 *
 * Plus a back-scatter term for ears, nostril wings and finger webbing, which is
 * what makes a backlit hand read as flesh.
 *
 * The kit and the hair carry their own patches for the same reason — a knit
 * transmits light through a single layer, and hair is anisotropic — and the kit
 * additionally owns the cloth spring, which is where hem lag comes from.
 *
 * Owned by the players agent.
 */

import {
  Color,
  DoubleSide,
  FrontSide,
  MeshPhysicalMaterial,
  Vector2,
  Vector3,
  type IUniform,
  type WebGLProgramParametersWithUniforms,
} from 'three';
import type { HairMaps, SkinMaps, SkinTone } from '../textures/skinTextures';
import type { ClothMaps, TeamKit } from '../textures/jerseyTextures';

export interface SkinMaterial extends MeshPhysicalMaterial {
  /** 0..1 — how wet the player is. Rises over a possession. */
  userData: { sweat: IUniform<number> } & Record<string, unknown>;
}

/**
 * A kit material with a live cloth spring. `swing` is a displacement in the
 * rig's own space applied to garment vertices in proportion to how free they
 * are (0 at the shoulder yoke and the waistband, 1 at a hem); `twist` is the
 * same thing about the vertical axis, so a hem lags a turn as well as a stride.
 */
export interface KitMaterial extends MeshPhysicalMaterial {
  userData: {
    swing: IUniform<Vector3>;
    twist: IUniform<number>;
  } & Record<string, unknown>;
}

const SKIN_VERT_PARS = /* glsl */ `
attribute vec2 aFlesh;
varying vec2 vFlesh;
`;

const SKIN_FRAG_PARS = /* glsl */ `
varying vec2 vFlesh;
uniform float uSweat;
uniform float uWrap;
uniform float uSss;
uniform vec3 uSubsurface;
uniform float uSweatRough;
uniform float uTranslucency;
uniform vec3 uSclera;
uniform vec3 uIris;
// The packed skin data map again under our own name: three declares the
// roughnessMap sampler after the lighting chunks, so it is not in scope here.
uniform sampler2D uSkinData;

// Replaces the physical direct-lighting response for skin. The base response is
// untouched — this adds only what a dielectric BRDF cannot express.
void RE_Direct_Skin(
  const in IncidentLight directLight,
  const in vec3 geometryPosition,
  const in vec3 geometryNormal,
  const in vec3 geometryViewDir,
  const in vec3 geometryClearcoatNormal,
  const in PhysicalMaterial material,
  inout ReflectedLight reflectedLight
) {
  RE_Direct_Physical( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );

  float ndl = dot( geometryNormal, directLight.direction );
  float lam = max( ndl, 0.0 );

  #ifdef USE_MAP
    vec3 skinData = texture2D( uSkinData, vMapUv ).rgb;
  #else
    vec3 skinData = vec3( 0.5, 0.5, 0.0 );
  #endif

  // --- Terminator scatter ------------------------------------------------
  // A wrapped diffuse lobe minus the Lambert one: zero where the surface is
  // fully lit, zero deep in shadow, peaked exactly at the terminator. Scaled by
  // sqrt(albedo) rather than albedo — see the file header.
  float wrapped = clamp( ( ndl + uWrap ) / ( 1.0 + uWrap ), 0.0, 1.0 );
  float band = max( wrapped - lam, 0.0 ) * ( 1.0 - lam );
  vec3 scatter = sqrt( material.diffuseColor ) * uSubsurface;
  reflectedLight.directDiffuse +=
    directLight.color * scatter * ( band * uSss * RECIPROCAL_PI );

  // --- Back-scatter through thin parts ----------------------------------
  // Ears, nostril wings, the webbing between spread fingers. vFlesh.y is a
  // measured thickness from the body field, not a per-part constant.
  vec3 tl = normalize( directLight.direction + geometryNormal * 0.3 );
  float trans = pow( clamp( dot( geometryViewDir, -tl ), 0.0, 1.0 ), 3.0 );
  reflectedLight.directDiffuse +=
    directLight.color * uSubsurface * ( trans * vFlesh.y * uTranslucency * RECIPROCAL_PI );

  // --- Sweat: the sharp second specular lobe -----------------------------
  float wet = uSweat * vFlesh.x * skinData.r;
  // --- Cornea: sharper still, and always on where there is an eye --------
  float eye = smoothstep( 0.25, 0.5, skinData.b );
  if ( wet > 0.001 || eye > 0.001 ) {
    vec3 hv = normalize( directLight.direction + geometryViewDir );
    float nh = max( dot( geometryNormal, hv ), 0.0 );
    float nv = max( dot( geometryNormal, geometryViewDir ), 1e-3 );
    float nl = max( ndl, 1e-3 );
    float voh = max( dot( geometryViewDir, hv ), 0.0 );
    float F = 0.035 + 0.965 * pow( 1.0 - voh, 5.0 );
    float V = 0.5 / ( nl + nv );
    if ( wet > 0.001 ) {
      float a = uSweatRough * uSweatRough;
      float a2 = a * a;
      float den = nh * nh * ( a2 - 1.0 ) + 1.0;
      float D = a2 / ( 3.141592 * den * den );
      reflectedLight.directSpecular += directLight.color * ( D * V * F * lam * wet * 0.3 );
    }
    if ( eye > 0.001 ) {
      float ae = 0.055 * 0.055;
      float a2e = ae * ae;
      float dene = nh * nh * ( a2e - 1.0 ) + 1.0;
      float De = a2e / ( 3.141592 * dene * dene );
      reflectedLight.directSpecular += directLight.color * ( De * V * F * lam * eye * 0.5 );
    }
  }
}

#undef RE_Direct
#define RE_Direct RE_Direct_Skin
`;

/**
 * One material per skin tone. The albedo atlas is tone-neutral, so the whole
 * roster shares a single bake and the tone arrives as `color`.
 */
export function makeSkinMaterial(maps: SkinMaps, tone: SkinTone): SkinMaterial {
  const sweat: IUniform<number> = { value: 0.18 };

  const mat = new MeshPhysicalMaterial({
    map: maps.albedo,
    normalMap: maps.normal,
    normalScale: new Vector2(1.0, 1.0),
    roughnessMap: maps.data,
    color: new Color(tone.color).convertSRGBToLinear(),
    roughness: tone.oilRoughness / 0.47,
    metalness: 0,
    vertexColors: true,
    // The broad, low, grazing-angle sheen of skin oil. Under the sweat lobe it
    // gives the two-population specular the rubric asks for. 0.16 was too shy
    // for sebum: §3.3 calls the broad lobe one of the *two* required
    // populations, and at 0.16 the second one was carrying the whole effect.
    sheen: 0.32,
    sheenRoughness: 0.55,
    sheenColor: new Color(0xffe2c4),
    specularIntensity: tone.specular,
    envMapIntensity: 0.5,
    side: FrontSide,
  }) as SkinMaterial;

  mat.userData = { sweat };

  mat.onBeforeCompile = (shader: WebGLProgramParametersWithUniforms) => {
    shader.uniforms.uSweat = sweat;
    shader.uniforms.uWrap = { value: tone.wrap };
    shader.uniforms.uSss = { value: tone.sss };
    shader.uniforms.uSubsurface = {
      value: new Color(tone.subsurface).convertSRGBToLinear(),
    };
    shader.uniforms.uSweatRough = { value: 0.11 };
    shader.uniforms.uSkinData = { value: maps.data };
    shader.uniforms.uTranslucency = { value: 0.55 };
    // Not white: the sclera is a wet, slightly warm off-white, and it is the
    // one surface on a player that must not take the skin tone multiply.
    shader.uniforms.uSclera = { value: new Vector3(0.93, 0.89, 0.85) };
    shader.uniforms.uIris = { value: new Vector3(0.6, 0.44, 0.31) };

    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${SKIN_VERT_PARS}`)
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n  vFlesh = aFlesh;');

    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <lights_physical_pars_fragment>',
        `#include <lights_physical_pars_fragment>\n${SKIN_FRAG_PARS}`,
      )
      // The eye substitution has to land after the albedo and the vertex colour
      // are folded in and before `lights_physical_fragment` builds the
      // PhysicalMaterial out of them.
      .replace(
        '#include <roughnessmap_fragment>',
        /* glsl */ `#include <roughnessmap_fragment>
        #ifdef USE_MAP
        {
          float eyeMask = texture2D( uSkinData, vMapUv ).b;
          float sclera = smoothstep( 0.72, 0.90, eyeMask );
          float iris = smoothstep( 0.26, 0.46, eyeMask ) * ( 1.0 - sclera );
          diffuseColor.rgb = mix( diffuseColor.rgb, uSclera, sclera );
          diffuseColor.rgb = mix( diffuseColor.rgb, uIris, iris );
          roughnessFactor = mix( roughnessFactor, 0.06, max( sclera, iris ) );
        }
        #endif`,
      );
  };

  return mat;
}

const KIT_VERT_PARS = /* glsl */ `
attribute vec2 aCloth;
uniform vec3 uClothSwing;
uniform float uClothTwist;
`;

/**
 * Cloth's own direct-lighting term.
 *
 * A single layer of knit polyester is not opaque: a measurable fraction of the
 * light that hits it comes out the other side, which is why a jersey's shadow
 * side is never as dark as the shadow side of the body under it and why the
 * terminator on cloth sits *lower contrast* than the one on skin next to it.
 * Three's dielectric BRDF has no term for that at all, so the shaded side of a
 * garment is pure Lambert falloff plus ambient, which is most of why round 0's
 * home white measured a flat 144 from collarbone to hem.
 */
const KIT_FRAG_PARS = /* glsl */ `
uniform float uClothWrap;

void RE_Direct_Cloth(
  const in IncidentLight directLight,
  const in vec3 geometryPosition,
  const in vec3 geometryNormal,
  const in vec3 geometryViewDir,
  const in vec3 geometryClearcoatNormal,
  const in PhysicalMaterial material,
  inout ReflectedLight reflectedLight
) {
  RE_Direct_Physical( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );

  float ndl = dot( geometryNormal, directLight.direction );
  float lam = max( ndl, 0.0 );
  float wrapped = clamp( ( ndl + uClothWrap ) / ( 1.0 + uClothWrap ), 0.0, 1.0 );
  reflectedLight.directDiffuse +=
    directLight.color * material.diffuseColor * ( max( wrapped - lam, 0.0 ) * RECIPROCAL_PI );
}

#undef RE_Direct
#define RE_Direct RE_Direct_Cloth
`;

/** Cloth: polyester knit, with a Charlie sheen rather than a dielectric lobe. */
export function makeKitMaterial(maps: ClothMaps, kit: TeamKit): KitMaterial {
  const warm = kit.name === 'home' ? 0xfff1e2 : 0xdfe6ff;
  const swing: IUniform<Vector3> = { value: new Vector3() };
  const twist: IUniform<number> = { value: 0 };

  const mat = new MeshPhysicalMaterial({
    map: maps.albedo,
    normalMap: maps.normal,
    normalScale: new Vector2(0.85, 0.85),
    roughnessMap: maps.data,
    roughness: 1,
    metalness: 0,
    // The garment carries its own sky-occlusion in the vertex colour: a hem is
    // 10–25% down on a chest because the shoulder shelf and the torso itself
    // block most of the upper hemisphere from it (§1.2).
    vertexColors: true,
    // Knit is not matte diffuse: it has a broad forward-scattering sheen that
    // lifts hard at the silhouette, which is most of what says "fabric".
    // Kept at the low end of §3.4's "broad, low, forward-scattering sheen".
    // Pushed to 0.95 it bought a few sRGB units of brightness and cost the
    // drape its contrast, because sheen does not scale with the vertex colour
    // that carries the folds.
    sheen: 0.78,
    sheenRoughness: 0.6,
    sheenColor: new Color(warm),
    specularIntensity: 0.55,
    envMapIntensity: 0.7,
    // The armhole and the neck are open edges; both sides can be seen.
    side: DoubleSide,
  }) as KitMaterial;

  mat.userData = { swing, twist };

  mat.onBeforeCompile = (shader: WebGLProgramParametersWithUniforms) => {
    shader.uniforms.uClothSwing = swing;
    shader.uniforms.uClothTwist = twist;
    shader.uniforms.uClothWrap = { value: 0.42 };

    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${KIT_VERT_PARS}`)
      // After skinning, so the swing is a *lag* on top of the pose rather than
      // something the skin cluster immediately undoes.
      .replace(
        '#include <skinning_vertex>',
        /* glsl */ `#include <skinning_vertex>
        {
          float freedom = aCloth.x;
          vec3 lag = uClothSwing * freedom;
          lag.x += -transformed.z * uClothTwist * freedom;
          lag.z += transformed.x * uClothTwist * freedom;
          transformed += lag;
        }`,
      );

    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <lights_physical_pars_fragment>',
      `#include <lights_physical_pars_fragment>\n${KIT_FRAG_PARS}`,
    );
  };

  return mat;
}

/** Shoes: four materials in one draw call, selected by the strip's v band. */
export function makeShoeMaterial(maps: ClothMaps): MeshPhysicalMaterial {
  return new MeshPhysicalMaterial({
    map: maps.albedo,
    normalMap: maps.normal,
    normalScale: new Vector2(1.1, 1.1),
    roughnessMap: maps.data,
    roughness: 1,
    metalness: 0,
    sheen: 0.25,
    sheenRoughness: 0.7,
    sheenColor: new Color(0xffffff),
    specularIntensity: 0.7,
    envMapIntensity: 0.85,
    side: DoubleSide,
  });
}

/**
 * Hair's direct lighting.
 *
 * Two things separate hair from a brown plastic cap, and neither is expressible
 * in a dielectric BRDF:
 *
 *  - **Anisotropy.** The highlight is a *band* running across the flow
 *    direction, not a point. Kajiya–Kay against a tangent derived per pixel —
 *    hair on a scalp flows down the meridian, so the tangent is the world up
 *    projected onto the tangent plane, and no extra attribute is needed.
 *  - **Rim penetration.** The outer few strands are optically thin. Light from
 *    behind comes *through* them, which is what makes the outline of a head
 *    separate from a dark bowl. It is scaled by a Fresnel so it lives at the
 *    silhouette and by the light's own direction so it is not a uniform glow —
 *    §10 tell 4.
 */
const HAIR_FRAG_PARS = /* glsl */ `
uniform vec3 uHairTint;
uniform float uHairShift;
uniform float uHairRim;

void RE_Direct_Hair(
  const in IncidentLight directLight,
  const in vec3 geometryPosition,
  const in vec3 geometryNormal,
  const in vec3 geometryViewDir,
  const in vec3 geometryClearcoatNormal,
  const in PhysicalMaterial material,
  inout ReflectedLight reflectedLight
) {
  RE_Direct_Physical( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );

  vec3 up = vec3( 0.0, 1.0, 0.0 );
  vec3 flow = up - geometryNormal * dot( geometryNormal, up );
  float flowLen = length( flow );
  vec3 T = flowLen > 1e-3 ? flow / flowLen : normalize( cross( geometryNormal, vec3( 1.0, 0.0, 0.0 ) ) );
  // Shifting the tangent along the normal is what moves the band off the
  // geometric centre line and gives the two lobes their offset.
  vec3 T1 = normalize( T + geometryNormal * uHairShift );
  vec3 T2 = normalize( T - geometryNormal * uHairShift * 2.2 );

  float ndl = max( dot( geometryNormal, directLight.direction ), 0.0 );

  float tl1 = dot( T1, directLight.direction );
  float tv1 = dot( T1, geometryViewDir );
  float kk1 = max( sqrt( max( 0.0, 1.0 - tl1 * tl1 ) ) * sqrt( max( 0.0, 1.0 - tv1 * tv1 ) ) - tl1 * tv1, 0.0 );
  float tl2 = dot( T2, directLight.direction );
  float tv2 = dot( T2, geometryViewDir );
  float kk2 = max( sqrt( max( 0.0, 1.0 - tl2 * tl2 ) ) * sqrt( max( 0.0, 1.0 - tv2 * tv2 ) ) - tl2 * tv2, 0.0 );

  float primary = pow( kk1, 48.0 );
  float secondary = pow( kk2, 11.0 );
  reflectedLight.directSpecular +=
    directLight.color * uHairTint * ( ( primary * 0.15 + secondary * 0.045 ) * ( 0.2 + 0.8 * ndl ) );

  // Rim penetration: transmitted, warm, and only where the silhouette turns
  // away from the viewer.
  float fres = pow( clamp( 1.0 - abs( dot( geometryNormal, geometryViewDir ) ), 0.0, 1.0 ), 2.6 );
  float through = pow( clamp( dot( -geometryViewDir, normalize( directLight.direction + geometryNormal * 0.35 ) ), 0.0, 1.0 ), 2.2 );
  reflectedLight.directDiffuse +=
    directLight.color * uHairTint * ( fres * through * uHairRim * RECIPROCAL_PI );
}

#undef RE_Direct
#define RE_Direct RE_Direct_Hair
`;

/**
 * Hair shells. Alpha-tested rather than blended so there is no sort order to
 * get wrong, and the strand mask's own shading supplies the banded, anisotropic
 * highlight that longer styles need.
 */
export function makeHairMaterial(maps: HairMaps, colour: number): MeshPhysicalMaterial {
  const m = new MeshPhysicalMaterial({
    map: maps.shade,
    alphaMap: maps.alpha,
    alphaTest: 0.36,
    color: new Color(colour).convertSRGBToLinear(),
    roughness: 0.62,
    metalness: 0,
    // The dielectric sheen stays low: the banded highlight is the Kajiya–Kay
    // lobe below, and doubling it up here is what makes shell hair read as
    // moulded plastic.
    sheen: 0.22,
    sheenRoughness: 0.42,
    sheenColor: new Color(0x8e7a63),
    specularIntensity: 0.4,
    envMapIntensity: 0.45,
    side: DoubleSide,
  });
  m.transparent = false;
  m.onBeforeCompile = (shader: WebGLProgramParametersWithUniforms) => {
    // Warmer than the strand colour: light that has been through a hair shaft
    // comes back carrying the melanin, which is why a rim on dark hair is
    // orange-brown and not white.
    shader.uniforms.uHairTint = { value: new Color(0xffdcbc).convertSRGBToLinear() };
    shader.uniforms.uHairShift = { value: 0.18 };
    shader.uniforms.uHairRim = { value: 0.75 };
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <lights_physical_pars_fragment>',
      `#include <lights_physical_pars_fragment>\n${HAIR_FRAG_PARS}`,
    );
  };
  return m;
}

/** Drives the sweat uniform on every skin material in one call. */
export function setSweat(materials: readonly SkinMaterial[], amount: number): void {
  const v = amount < 0 ? 0 : amount > 1 ? 1 : amount;
  for (const m of materials) m.userData.sweat.value = v;
}
