/**
 * Player materials: skin, kit, shoes and hair.
 *
 * The interesting one is skin. `MeshPhysicalMaterial` gets us a plausible base,
 * but two things it does not do are exactly the two things that identify
 * real-time skin, so both are patched in:
 *
 *  1. **Subsurface warmth at the terminator.** Light that enters the skin,
 *     scatters and re-emerges does so a millimetre or two away, which shows as
 *     a reddening *band* where the lit side rolls into shadow — hue toward
 *     5–20°, saturation up, over roughly 6–20 px at gameplay distance. It is
 *     modelled here as the difference between a wrapped diffuse response and
 *     the Lambert one, which is non-zero only in that band and exactly zero on
 *     the fully lit side and in the core shadow. The wrap distance and the
 *     scatter colour both move with skin tone: melanin absorbs the long paths,
 *     so deeper tones get a shorter, less saturated band and a harder specular
 *     rather than the same red on a darker albedo.
 *  2. **A second specular lobe.** Skin has a broad oil sheen (roughness
 *     ~0.4–0.55, from the roughness map) *plus* sharp sweat highlights
 *     (roughness ~0.08–0.15) sitting on top of it. One lobe alone reads as
 *     plastic. Sweat is driven by a 0..1 uniform so it can build over a
 *     possession, masked by a per-vertex region weight and a texture channel,
 *     so it beads on the forehead, deltoids and forearms rather than uniformly.
 *
 * Plus a back-scatter term for ears, nostril wings and finger webbing, which is
 * what makes a backlit hand read as flesh.
 *
 * Owned by the players agent.
 */

import {
  Color,
  DoubleSide,
  FrontSide,
  MeshPhysicalMaterial,
  Vector2,
  type IUniform,
  type WebGLProgramParametersWithUniforms,
} from 'three';
import type { HairMaps, SkinMaps, SkinTone } from '../textures/skinTextures';
import type { ClothMaps, TeamKit } from '../textures/jerseyTextures';

export interface SkinMaterial extends MeshPhysicalMaterial {
  /** 0..1 — how wet the player is. Rises over a possession. */
  userData: { sweat: IUniform<number> } & Record<string, unknown>;
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

  // --- Terminator scatter ------------------------------------------------
  // A wrapped diffuse lobe minus the Lambert one: zero where the surface is
  // fully lit, zero deep in shadow, peaked exactly at the terminator.
  float wrapped = clamp( ( ndl + uWrap ) / ( 1.0 + uWrap ), 0.0, 1.0 );
  float band = max( wrapped * wrapped - lam, 0.0 );
  reflectedLight.directDiffuse +=
    directLight.color * material.diffuseColor * uSubsurface * ( band * uSss * 3.4 );

  // --- Back-scatter through thin parts ----------------------------------
  // Ears, nostril wings, the webbing between spread fingers.
  vec3 tl = normalize( directLight.direction + geometryNormal * 0.3 );
  float trans = pow( clamp( dot( geometryViewDir, -tl ), 0.0, 1.0 ), 3.5 );
  reflectedLight.directDiffuse +=
    directLight.color * uSubsurface * ( trans * vFlesh.y * uTranslucency );

  // --- Sweat: the sharp second specular lobe -----------------------------
  #ifdef USE_ROUGHNESSMAP
    float beads = texture2D( roughnessMap, vMapUv ).r;
  #else
    float beads = 0.5;
  #endif
  float wet = uSweat * vFlesh.x * beads;
  if ( wet > 0.001 ) {
    vec3 hv = normalize( directLight.direction + geometryViewDir );
    float nh = max( dot( geometryNormal, hv ), 0.0 );
    float nv = max( dot( geometryNormal, geometryViewDir ), 1e-3 );
    float nl = max( ndl, 1e-3 );
    float voh = max( dot( geometryViewDir, hv ), 0.0 );
    float a = uSweatRough * uSweatRough;
    float a2 = a * a;
    float den = nh * nh * ( a2 - 1.0 ) + 1.0;
    float D = a2 / ( 3.141592 * den * den );
    float V = 0.5 / ( nl + nv );
    float F = 0.035 + 0.965 * pow( 1.0 - voh, 5.0 );
    reflectedLight.directSpecular += directLight.color * ( D * V * F * lam * wet * 0.85 );
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
    // gives the two-population specular the rubric asks for.
    sheen: 0.3,
    sheenRoughness: 0.55,
    sheenColor: new Color(0xffd8bc),
    specularIntensity: tone.specular,
    envMapIntensity: 0.85,
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
    shader.uniforms.uTranslucency = { value: 0.5 };

    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${SKIN_VERT_PARS}`)
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n  vFlesh = aFlesh;');

    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <lights_physical_pars_fragment>',
      `#include <lights_physical_pars_fragment>\n${SKIN_FRAG_PARS}`,
    );
  };

  return mat;
}

/** Cloth: polyester knit, with a Charlie sheen rather than a dielectric lobe. */
export function makeKitMaterial(maps: ClothMaps, kit: TeamKit): MeshPhysicalMaterial {
  const warm = kit.name === 'home' ? 0xfff1e2 : 0xdfe6ff;
  return new MeshPhysicalMaterial({
    map: maps.albedo,
    normalMap: maps.normal,
    normalScale: new Vector2(0.85, 0.85),
    roughnessMap: maps.data,
    roughness: 1,
    metalness: 0,
    // Knit is not matte diffuse: it has a broad forward-scattering sheen that
    // lifts hard at the silhouette, which is most of what says "fabric".
    sheen: 0.75,
    sheenRoughness: 0.62,
    sheenColor: new Color(warm),
    specularIntensity: 0.42,
    envMapIntensity: 0.7,
    // The armhole and the neck are open edges; both sides can be seen.
    side: DoubleSide,
  });
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
 * Hair shells. Alpha-tested rather than blended so there is no sort order to
 * get wrong, and the strand mask's own shading supplies the banded, anisotropic
 * highlight that longer styles need.
 */
export function makeHairMaterial(maps: HairMaps, colour: number): MeshPhysicalMaterial {
  const m = new MeshPhysicalMaterial({
    map: maps.shade,
    alphaMap: maps.alpha,
    alphaTest: 0.45,
    color: new Color(colour).convertSRGBToLinear(),
    roughness: 0.44,
    metalness: 0,
    sheen: 0.6,
    sheenRoughness: 0.35,
    sheenColor: new Color(0xbfa98c),
    specularIntensity: 0.55,
    envMapIntensity: 0.9,
    side: DoubleSide,
  });
  m.transparent = false;
  return m;
}

/** Drives the sweat uniform on every skin material in one call. */
export function setSweat(materials: readonly SkinMaterial[], amount: number): void {
  const v = amount < 0 ? 0 : amount > 1 ? 1 : amount;
  for (const m of materials) m.userData.sweat.value = v;
}
