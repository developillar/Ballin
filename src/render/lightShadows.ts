/**
 * Soft shadows and cascades.
 *
 * ---------------------------------------------------------------------------
 * ROUND 1: the reason no shadow ever reached the floor, and the fix.
 *
 * The round-2 header claimed three's shadow pass was issuing zero draws. That
 * was wrong, and so was the conclusion drawn from it. The pass renders fine.
 * What did not work was the **read**.
 *
 * Isolated reproduction (a 256² page, one plane, one box, one directional
 * light, no Ballin code, run in the capture harness's own browser —
 * headless Chromium on ANGLE/SwiftShader). Darkest plane pixel over brightest,
 * inside the box's shadow footprint:
 *
 *   | shadow map type                          | darkest / brightest |
 *   |------------------------------------------|---------------------|
 *   | `PCFShadowMap` (sampler2DShadow)         | **0.904** — shadow  |
 *   | `BasicShadowMap` (stock, sampler2D)      | 1.000 — no shadow   |
 *   | `BasicShadowMap` + the PCSS chunk below  | 1.000 — no shadow   |
 *
 * In r185 the shadow map is a `WebGLRenderTarget` whose depth attachment is a
 * `DepthTexture(UnsignedIntType, DepthFormat)`, and `WebGLLights` binds
 * `shadow.map.depthTexture` — never the colour attachment. On the `BasicShadowMap`
 * path three sets `compareFunction = null` and the chunk samples that
 * DEPTH_COMPONENT24 texture as a plain `sampler2D`. Under this rasteriser that
 * read does not come back as a usable depth value, so every tap looks unoccluded
 * and `getShadow()` returns exactly 1.0 — which is precisely what the reviewer
 * measured on the hardwood (under a planted foot 96.12 against 95.53 beside it,
 * ratio 1.006). It also explains why `VSMShadowMap` failed identically: three's
 * VSM pre-pass reads that same native depth texture as `sampler2D shadow_pass`.
 * Three implementations failing the same way was not evidence of a fault
 * upstream of sampling; it was three variants of the same unsupported read.
 *
 * So this file is now on the **comparison sampler**: `PCFShadowMap`, which makes
 * three allocate the depth texture with `LessEqualCompare` and bind it as a
 * `sampler2DShadow`. Every tap is a hardware depth compare (and, with
 * `LinearFilter`, a free 2×2 PCF), which is both correct here and the fast path
 * on a phone.
 *
 * That costs the raw depth read PCSS normally uses for its blocker search, so
 * the search is rebuilt on top of the comparison: sampling at
 * `receiverDepth − dz` is occluded exactly when a blocker sits more than `dz`
 * nearer the light, so a short ladder of `dz` values brackets the
 * receiver-to-blocker gap without ever reading a depth value. The filter radius
 * comes from that gap, so a planted sole still filters at ~1 texel while a
 * raised hand opens out to the cascade's ceiling — the contact hardening §1.2
 * asks for, kept off the deprecated `PCFSoftShadowMap` path and its constant
 * radius.
 * ---------------------------------------------------------------------------
 *
 * The other half of the file is cascades. One 16 m shadow camera at phone
 * resolution is ~8 mm per texel and mushy at the point of contact.
 * `ShadowCascade` fits a tight, texel-snapped ortho frustum around the action
 * for the primary bank and hands wide static coverage to the secondary banks —
 * the cheap half of a CSM without injecting a cascade selector into every
 * material in the project.
 *
 * Owned by the lighting agent.
 */

import {
  DirectionalLight,
  PCFShadowMap,
  ShaderChunk,
  Vector3,
  type WebGLRenderer,
} from 'three';

export interface SoftShadowOptions {
  /**
   * Rungs in the blocker-gap ladder. 0 disables contact hardening and falls
   * back to a fixed-radius filter at `minTexels`.
   */
  probeLevels: number;
  /** Taps per rung of the ladder. */
  probeSamples: number;
  /** Taps in the filter. */
  filterSamples: number;
  /** Hard floor on the filter radius — the contact-sharp end. */
  minTexels: number;
  /** Hard ceiling on the filter radius — keeps the far end affordable. */
  maxTexels: number;
  /** Blocker search radius, in texels, at the widest rung. */
  searchTexels: number;
}

/**
 * Tap budgets per tier. Every tap is a texture fetch on the shadow map, so the
 * totals are what a phone actually pays: `probeLevels * probeSamples +
 * filterSamples` per shadowed light. `low` spends 4, `medium` 16, `high` 22,
 * `ultra` 34 — and `Quality.shadowCascades` (1 / 2 / 3 / 4) multiplies it.
 */
export const SOFT_SHADOW_TIERS: Record<string, SoftShadowOptions> = {
  low: { probeLevels: 0, probeSamples: 0, filterSamples: 4, minTexels: 1.1, maxTexels: 1.1, searchTexels: 0 },
  medium: { probeLevels: 2, probeSamples: 4, filterSamples: 8, minTexels: 0.75, maxTexels: 14, searchTexels: 7 },
  high: { probeLevels: 2, probeSamples: 5, filterSamples: 12, minTexels: 0.7, maxTexels: 22, searchTexels: 9 },
  ultra: { probeLevels: 3, probeSamples: 6, filterSamples: 16, minTexels: 0.65, maxTexels: 28, searchTexels: 11 },
};

function chunkSource(o: SoftShadowOptions): string {
  const pcss = o.probeLevels > 0 && o.probeSamples > 0;
  return /* glsl */ `
#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif

#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif

#ifdef USE_SHADOWMAP

	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform sampler2DShadow directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
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
		uniform sampler2DShadow spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
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
		uniform samplerCubeShadow pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
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

	/**
	 * PCSS over a comparison sampler.
	 *
	 * \`shadowRadius\` is repurposed by \`ShadowCascade\` to carry "penumbra
	 * texels per unit of normalised depth gap", which folds the bank's angular
	 * size, the map resolution and the frustum extent into one number. That also
	 * makes \`maxTexels / shadowRadius\` the exact gap at which the penumbra
	 * saturates, so the blocker ladder self-scales to each cascade and never
	 * probes further than it can use.
	 */
	float getShadow( sampler2DShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {

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
		// ---- blocker gap, bracketed by comparison rather than read
		// A tap at ( receiver - dz ) reports "occluded" exactly when a blocker
		// sits more than dz nearer the light, so walking dz outward brackets the
		// receiver-to-blocker gap without ever needing a raw depth value.
		float gapMax = ${o.maxTexels.toFixed(2)} / max( shadowRadius, 1e-4 );
		float gap = 0.0;
		for ( int k = 1; k <= ${o.probeLevels}; k ++ ) {
			float rung = float( k ) / ${o.probeLevels}.0;
			float dz = gapMax * rung;
			float occ = 0.0;
			for ( int i = 0; i < ${o.probeSamples}; i ++ ) {
				vec2 off = ballinVogel( i, ${o.probeSamples}.0, phi ) * ${o.searchTexels.toFixed(2)} * ( 0.35 + 0.65 * rung ) * texel;
				#ifdef USE_REVERSED_DEPTH_BUFFER
					occ += 1.0 - texture( shadowMap, vec3( shadowCoord.xy + off, shadowCoord.z + dz ) );
				#else
					occ += 1.0 - texture( shadowMap, vec3( shadowCoord.xy + off, shadowCoord.z - dz ) );
				#endif
			}
			occ /= ${o.probeSamples}.0;
			// smoothstep, not a raw fraction: a rung that is mostly occluded means
			// the blocker is at least that far away, and scaling by the fraction
			// would systematically under-read the gap at every rung.
			gap = max( gap, dz * smoothstep( 0.08, 0.62, occ ) );
		}
		filterTexels = clamp( shadowRadius * gap, ${o.minTexels.toFixed(3)}, ${o.maxTexels.toFixed(2)} );
		` : ''}

		// ---- filter
		float shadow = 0.0;
		for ( int i = 0; i < ${o.filterSamples}; i ++ ) {
			vec2 off = ballinVogel( i, ${o.filterSamples}.0, phi + 1.7 ) * filterTexels * texel;
			shadow += texture( shadowMap, vec3( shadowCoord.xy + off, shadowCoord.z ) );
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
	float getPointShadow( samplerCubeShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {

		vec3 lightToPosition = shadowCoord.xyz;
		vec3 bd3D = normalize( lightToPosition );
		vec3 absVec = abs( lightToPosition );
		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );
		float shadow = 1.0;

		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {

			#ifdef USE_REVERSED_DEPTH_BUFFER
				float dp = ( shadowCameraNear * ( shadowCameraFar - viewSpaceZ ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
				dp -= shadowBias;
			#else
				float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
				dp += shadowBias;
			#endif

			float texelSize = shadowRadius / shadowMapSize.x;
			vec3 absDir = abs( bd3D );
			vec3 tangent = absDir.x > absDir.z ? vec3( 0.0, 1.0, 0.0 ) : vec3( 1.0, 0.0, 0.0 );
			tangent = normalize( cross( bd3D, tangent ) );
			vec3 bitangent = cross( bd3D, tangent );
			float phi = ballinIGN( gl_FragCoord.xy ) * PI2;

			shadow = 0.0;
			for ( int i = 0; i < 5; i ++ ) {
				vec2 s = ballinVogel( i, 5.0, phi );
				shadow += texture( shadowMap, vec4( bd3D + ( tangent * s.x + bitangent * s.y ) * texelSize, dp ) );
			}
			shadow *= 0.2;

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
 * comparison-sampler shadow path. Must run before anything compiles a material,
 * which is why `LightingSystem` has the lowest `order` in the engine.
 */
export function installSoftShadows(renderer: WebGLRenderer, opts: SoftShadowOptions): void {
  renderer.shadowMap.enabled = true;
  // PCFShadowMap — NOT Basic, and NOT the deprecated PCFSoftShadowMap. This is
  // the only three path that allocates the shadow depth texture with a compare
  // function and binds it as a `sampler2DShadow`; the Basic path's plain
  // `sampler2D` read of a DEPTH_COMPONENT24 texture returns no usable depth on
  // this rasteriser and silently made every shadow term exactly 1.0. See the
  // file header for the isolated reproduction. The chunk above replaces three's
  // 5-tap fixed-radius filter, so nothing but the sampler type is stock.
  renderer.shadowMap.type = PCFShadowMap;
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
    // shadows glued to the sole while the wide one still avoids acne. With a
    // comparison sampler the hardware also does a 2×2 tap per fetch, so the
    // bias has to cover half a texel of slope on top of the depth quantisation.
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
