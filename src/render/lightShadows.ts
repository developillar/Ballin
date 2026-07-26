/**
 * Soft shadows and cascades.
 *
 * Two problems with the stock setup:
 *
 *  - `PCFSoftShadowMap` is deprecated in r185 (the renderer silently downgrades
 *    it to `PCFShadowMap` and warns), and even the current PCF path filters at a
 *    **constant** radius. Constant-softness shadows are a named tell: a planted
 *    foot needs a ≤3 px penumbra while the same player's raised hand needs
 *    12–30 px. So we take the shadow map over: `BasicShadowMap` gives us the raw
 *    depth texture as a plain `sampler2D`, and the chunk below implements PCSS —
 *    a Vogel-disc blocker search, a penumbra estimate from the blocker distance,
 *    then a Vogel-disc PCF at that radius.
 *
 *  - One 16 m shadow camera at phone resolution is ~8 mm per texel and mushy at
 *    the point of contact. `ShadowCascade` fits a tight ortho frustum around the
 *    action for the primary bank and hands the wide, static coverage to the
 *    secondary banks, which is the cheap half of a CSM without having to inject
 *    a cascade selector into every material in the project.
 *
 * Owned by the lighting agent.
 */

import {
  BasicShadowMap,
  DirectionalLight,
  ShaderChunk,
  Vector3,
  type WebGLRenderer,
} from 'three';

export interface SoftShadowOptions {
  /** Taps in the blocker search. 0 disables PCSS and falls back to fixed-radius PCF. */
  blockerSamples: number;
  /** Taps in the filter. */
  filterSamples: number;
  /** Hard floor on the filter radius — the contact-sharp end. */
  minTexels: number;
  /** Hard ceiling on the filter radius — keeps the far end affordable. */
  maxTexels: number;
  /** Blocker search radius, in texels. */
  searchTexels: number;
}

export const SOFT_SHADOW_TIERS: Record<string, SoftShadowOptions> = {
  low: { blockerSamples: 0, filterSamples: 4, minTexels: 1.1, maxTexels: 1.1, searchTexels: 0 },
  medium: { blockerSamples: 6, filterSamples: 10, minTexels: 0.75, maxTexels: 14, searchTexels: 7 },
  high: { blockerSamples: 10, filterSamples: 16, minTexels: 0.7, maxTexels: 22, searchTexels: 9 },
  ultra: { blockerSamples: 12, filterSamples: 24, minTexels: 0.65, maxTexels: 28, searchTexels: 11 },
};

function chunkSource(o: SoftShadowOptions): string {
  const pcss = o.blockerSamples > 0;
  return /* glsl */ `
#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif

#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif

#ifdef USE_SHADOWMAP

	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif

	#if NUM_SPOT_LIGHT_SHADOWS > 0
		uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif

	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform samplerCube pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif

	// Interleaved gradient noise: rotates the disc per pixel so the low tap
	// count dissolves into dither instead of banding into visible rings.
	float ballinIGN( vec2 p ) {
		return fract( 52.9829189 * fract( dot( p, vec2( 0.06711056, 0.00583715 ) ) ) );
	}

	vec2 ballinVogel( const in int index, const in float count, const in float phi ) {
		float r = sqrt( ( float( index ) + 0.5 ) / count );
		float theta = float( index ) * 2.399963229728653 + phi;
		return vec2( cos( theta ), sin( theta ) ) * r;
	}

	float ballinDepth( sampler2D shadowMap, vec2 uv ) {
		return texture2D( shadowMap, uv ).r;
	}

	/**
	 * PCSS. \`shadowRadius\` is repurposed by \`ShadowCascade\` to carry
	 * "penumbra texels per unit of normalised depth gap", which folds the
	 * bank's angular size, the map resolution and the frustum extent into one
	 * number the shader can multiply straight through.
	 */
	float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {

		shadowCoord.xyz /= shadowCoord.w;

		#ifdef USE_REVERSED_DEPTH_BUFFER
			shadowCoord.z -= shadowBias;
		#else
			shadowCoord.z += shadowBias;
		#endif

		bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
		if ( ! ( inFrustum && shadowCoord.z <= 1.0 && shadowCoord.z >= 0.0 ) ) return 1.0;

		vec2 texel = vec2( 1.0 ) / shadowMapSize;
		float phi = ballinIGN( gl_FragCoord.xy ) * PI2;
		float filterTexels = ${o.minTexels.toFixed(3)};

		${pcss ? /* glsl */ `
		// ---- blocker search
		float blockerSum = 0.0;
		float blockerCount = 0.0;
		for ( int i = 0; i < ${o.blockerSamples}; i ++ ) {
			vec2 off = ballinVogel( i, ${o.blockerSamples}.0, phi ) * ${o.searchTexels.toFixed(2)} * texel;
			float d = ballinDepth( shadowMap, shadowCoord.xy + off );
			#ifdef USE_REVERSED_DEPTH_BUFFER
				if ( d > shadowCoord.z ) { blockerSum += d; blockerCount += 1.0; }
			#else
				if ( d < shadowCoord.z ) { blockerSum += d; blockerCount += 1.0; }
			#endif
		}
		if ( blockerCount < 0.5 ) return 1.0;

		// ---- penumbra estimate from the receiver-to-blocker gap
		float gap = abs( shadowCoord.z - blockerSum / blockerCount );
		filterTexels = clamp( shadowRadius * gap, ${o.minTexels.toFixed(3)}, ${o.maxTexels.toFixed(2)} );
		` : ''}

		// ---- filter
		float shadow = 0.0;
		for ( int i = 0; i < ${o.filterSamples}; i ++ ) {
			vec2 off = ballinVogel( i, ${o.filterSamples}.0, phi + 1.7 ) * filterTexels * texel;
			float d = ballinDepth( shadowMap, shadowCoord.xy + off );
			#ifdef USE_REVERSED_DEPTH_BUFFER
				shadow += step( d, shadowCoord.z );
			#else
				shadow += step( shadowCoord.z, d );
			#endif
		}
		shadow /= ${o.filterSamples}.0;

		// Feather the last few percent of the cascade so the boundary between a
		// tight follower and a wide static bank never shows as a hard line.
		vec2 lo = smoothstep( vec2( 0.0 ), vec2( 0.045 ), shadowCoord.xy );
		vec2 hi = vec2( 1.0 ) - smoothstep( vec2( 0.955 ), vec2( 1.0 ), shadowCoord.xy );
		float edge = lo.x * lo.y * hi.x * hi.y;
		edge *= 1.0 - smoothstep( 0.94, 1.0, shadowCoord.z );
		shadow = mix( 1.0, shadow, edge );

		return mix( 1.0, shadow, shadowIntensity );

	}

	#if NUM_POINT_LIGHT_SHADOWS > 0
	float getPointShadow( samplerCube shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {

		vec3 lightToPosition = shadowCoord.xyz;
		vec3 absVec = abs( lightToPosition );
		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );
		float shadow = 1.0;

		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {

			float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
			dp += shadowBias;

			float depth = textureCube( shadowMap, normalize( lightToPosition ) ).r;

			#ifdef USE_REVERSED_DEPTH_BUFFER
				depth = 1.0 - depth;
			#endif

			shadow = step( dp, depth );

		}

		return mix( 1.0, shadow, shadowIntensity );

	}
	#endif

#endif
`;
}

let installed = false;

/**
 * Swaps three's shadow chunk for the PCSS version and puts the renderer on the
 * raw-depth shadow path it needs. Must run before anything compiles a material,
 * which is why `LightingSystem` has the lowest `order` in the engine.
 */
export function installSoftShadows(renderer: WebGLRenderer, opts: SoftShadowOptions): void {
  renderer.shadowMap.enabled = true;
  // BasicShadowMap keeps `compareFunction` off, so the depth texture binds as a
  // plain sampler2D and the blocker search can actually read depth values.
  renderer.shadowMap.type = BasicShadowMap;
  renderer.shadowMap.autoUpdate = true;
  if (installed) return;
  ShaderChunk.shadowmap_pars_fragment = chunkSource(opts);
  installed = true;
}

export interface CascadeSpec {
  /** Unit vector from the lit point *toward* the bank. */
  direction: Vector3;
  /** Half-width of the ortho frustum, metres. */
  extent: number;
  /** 0 = pinned to the court centre, 1 = fully tracks the focus point. */
  follow: number;
  /** How dark this bank's shadow is. The key is 1; the fans are much lighter. */
  intensity: number;
  /** Angular half-size of the emitting bank, radians. Drives penumbra growth. */
  sourceAngle: number;
  /** Distance the virtual light sits back along `direction`. */
  distance: number;
  /** Frames between shadow-map refreshes. 1 for the follower, more for static. */
  refreshInterval: number;
}

/**
 * One shadow-casting bank with a fitted, texel-snapped ortho frustum.
 *
 * Snapping matters: without it, moving the frustum by a fraction of a texel
 * makes every shadow edge in the scene crawl, which is far more visible than
 * the resolution you gained by fitting tightly in the first place.
 */
export class ShadowCascade {
  readonly light: DirectionalLight;
  readonly spec: CascadeSpec;

  private readonly right = new Vector3();
  private readonly up = new Vector3();
  private readonly centre = new Vector3();
  private readonly snapped = new Vector3();
  private texelWorld = 0.01;
  private frames = 0;

  constructor(spec: CascadeSpec, colour: number, intensity: number, mapSize: number) {
    this.spec = spec;
    this.light = new DirectionalLight(colour, intensity);
    this.light.castShadow = true;

    // Basis in the light's image plane, used for texel snapping.
    const d = spec.direction.clone().normalize();
    this.right.crossVectors(new Vector3(0, 1, 0), d);
    if (this.right.lengthSq() < 1e-6) this.right.set(1, 0, 0);
    this.right.normalize();
    this.up.crossVectors(d, this.right).normalize();

    this.resize(mapSize);
    this.focus(new Vector3(0, 0, 0));
  }

  /** Re-derives every resolution-dependent number: extent, bias, penumbra rate. */
  resize(mapSize: number): void {
    const s = this.spec;
    const shadow = this.light.shadow;
    shadow.mapSize.set(mapSize, mapSize);
    shadow.camera.left = -s.extent;
    shadow.camera.right = s.extent;
    shadow.camera.top = s.extent;
    shadow.camera.bottom = -s.extent;
    shadow.camera.near = 0.5;
    shadow.camera.far = s.distance + s.extent * 1.6 + 8;
    shadow.camera.updateProjectionMatrix();
    shadow.intensity = s.intensity;
    shadow.autoUpdate = s.refreshInterval <= 1;
    // A throttled cascade still has to render once before it is ever valid.
    shadow.needsUpdate = true;

    this.texelWorld = (s.extent * 2) / mapSize;

    // Depth bias scaled to the frustum, so the tight cascade keeps its contact
    // shadows glued to the sole while the wide one still avoids acne.
    const depthRange = shadow.camera.far - shadow.camera.near;
    shadow.bias = -(0.9 * this.texelWorld) / depthRange;
    // Normal bias in world units — one and a half texels, which pushes the
    // sample off the surface without detaching the shadow (peter-panning).
    shadow.normalBias = this.texelWorld * 1.5;
    // Penumbra texels per unit of normalised depth gap. See the chunk above.
    shadow.radius = (2 * Math.tan(s.sourceAngle) * mapSize * depthRange) / (s.extent * 2);
  }

  /** Fits the frustum around `point`, snapped to the shadow map's texel grid. */
  focus(point: Vector3): void {
    const s = this.spec;
    this.centre.set(point.x * s.follow, 0, point.z * s.follow);

    const cr = this.centre.dot(this.right);
    const cu = this.centre.dot(this.up);
    const cd = this.centre.dot(s.direction);
    const t = this.texelWorld;
    this.snapped
      .set(0, 0, 0)
      .addScaledVector(this.right, Math.round(cr / t) * t)
      .addScaledVector(this.up, Math.round(cu / t) * t)
      .addScaledVector(s.direction, cd);

    this.light.target.position.copy(this.snapped);
    this.light.position.copy(this.snapped).addScaledVector(s.direction, s.distance);
    this.light.target.updateMatrixWorld();
    this.light.updateMatrixWorld();
  }

  /** Throttles the wide, static banks so they are not re-rendered every frame. */
  tick(): void {
    if (this.spec.refreshInterval <= 1) return;
    this.frames++;
    if (this.frames % this.spec.refreshInterval === 0) this.light.shadow.needsUpdate = true;
  }
}
