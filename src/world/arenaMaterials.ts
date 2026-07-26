/**
 * Procedural materials for the arena: the LED surfaces (ribbon boards, courtside
 * boards, scorer's table face, jumbotron), the upper-bowl haze, and the handful
 * of shared standard materials the architecture is built from.
 *
 * The LED shader is the important one. A real board is an array of discrete
 * emitters behind a black mask, so it has to carry (a) a visible pixel lattice,
 * (b) content that moves, and (c) a brightness that sits above display white so
 * the bloom pass has something legitimate to grab — without blowing the whole
 * band to a flat clipped slab.
 *
 * Owned by the arena agent.
 */

import {
  AdditiveBlending,
  BackSide,
  CanvasTexture,
  Color,
  DoubleSide,
  LinearFilter,
  MeshStandardMaterial,
  RepeatWrapping,
  ShaderMaterial,
  SRGBColorSpace,
  Texture,
  Vector2,
} from 'three';
import { makeRng } from '../core/MathX';

export const TEAM_HOME_COLOR = new Color('#2d5cc8');
export const TEAM_AWAY_COLOR = new Color('#c8452b');
export const TEAM_ACCENT_COLOR = new Color('#e8a32c');

// -----------------------------------------------------------------------------
// Canvas content
// -----------------------------------------------------------------------------

function canvas(w: number, h: number): { c: HTMLCanvasElement; g: CanvasRenderingContext2D } {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const g = c.getContext('2d')!;
  return { c, g };
}

function toTexture(c: HTMLCanvasElement, repeatX = 1): CanvasTexture {
  const t = new CanvasTexture(c);
  t.colorSpace = SRGBColorSpace;
  t.wrapS = RepeatWrapping;
  t.wrapT = RepeatWrapping;
  t.repeat.x = repeatX;
  t.magFilter = LinearFilter;
  t.minFilter = LinearFilter;
  t.generateMipmaps = false;
  t.needsUpdate = true;
  return t;
}

/**
 * The scrolling ribbon strip: sponsor word marks, colour blocks, chevron
 * wipes and a score panel, laid out along a very wide strip that the shader
 * then pans. Text is drawn with the platform font — still procedural, still
 * zero bytes of downloaded asset.
 */
export function bakeRibbonStrip(width = 2048, height = 64): CanvasTexture {
  const { c, g } = canvas(width, height);
  const rng = makeRng(9021);
  g.fillStyle = '#04060c';
  g.fillRect(0, 0, width, height);

  const words = ['BALLIN', 'COURTSIDE', 'HOME OF THE', 'TIP-OFF', 'SEASON PASS', 'DOWNTOWN'];
  const palette = ['#2d5cc8', '#c8452b', '#e8a32c', '#12203f', '#1c8f6a', '#0d1424', '#7a2318'];

  let x = 0;
  let w = 0;
  while (x < width) {
    const kind = rng() * 1.12;
    if (kind < 0.34) {
      // Word mark on a flat field.
      const bg = palette[Math.floor(rng() * palette.length)];
      const word = words[Math.floor(rng() * words.length)];
      g.font = `700 ${Math.round(height * 0.62)}px system-ui, -apple-system, Segoe UI, sans-serif`;
      w = g.measureText(word).width + height * 2.6;
      g.fillStyle = bg;
      g.fillRect(x, 0, w, height);
      g.fillStyle = '#e8eefc';
      g.textBaseline = 'middle';
      g.fillText(word, x + height * 0.7, height * 0.54);
    } else if (kind < 0.58) {
      // Chevron wipe.
      w = height * (5 + rng() * 5);
      const a = palette[Math.floor(rng() * palette.length)];
      g.fillStyle = '#0a1226';
      g.fillRect(x, 0, w, height);
      g.fillStyle = a;
      for (let k = 0; k < 10; k++) {
        const cx = x + k * (height * 0.62);
        if (cx > x + w) break;
        g.beginPath();
        g.moveTo(cx, height);
        g.lineTo(cx + height * 0.34, 0);
        g.lineTo(cx + height * 0.62, 0);
        g.lineTo(cx + height * 0.28, height);
        g.closePath();
        g.fill();
      }
    } else if (kind < 0.78) {
      // Stat / score panel.
      w = height * (5.5 + rng() * 3);
      g.fillStyle = '#080c18';
      g.fillRect(x, 0, w, height);
      g.fillStyle = '#e8a32c';
      g.font = `700 ${Math.round(height * 0.5)}px system-ui, sans-serif`;
      g.fillText(`${Math.floor(rng() * 40 + 60)}`, x + height * 0.4, height * 0.55);
      g.fillStyle = '#8fa4c8';
      g.font = `600 ${Math.round(height * 0.3)}px system-ui, sans-serif`;
      g.fillText('PTS', x + height * 1.7, height * 0.55);
      g.fillStyle = '#2d5cc8';
      g.fillRect(x + w - height * 0.9, height * 0.18, height * 0.5, height * 0.64);
    } else {
      // Colour band stack.
      w = height * (2.6 + rng() * 3.4);
      const bands = 2 + Math.floor(rng() * 3);
      for (let b = 0; b < bands; b++) {
        g.fillStyle = palette[Math.floor(rng() * palette.length)];
        g.globalAlpha = 0.85;
        g.fillRect(x, (b / bands) * height, w, height / bands + 1);
      }
      g.globalAlpha = 1;
    }
    x += w + 2;
  }

  return toTexture(c);
}

/** Courtside board content: bigger, bolder, fewer elements — it is 900 mm tall. */
export function bakeCourtsideStrip(width = 2048, height = 128): CanvasTexture {
  const { c, g } = canvas(width, height);
  const rng = makeRng(551);
  g.fillStyle = '#03050a';
  g.fillRect(0, 0, width, height);
  const words = ['BALLIN', 'PLAYOFFS', 'GAME NIGHT', 'THE HOUSE', 'RISE UP'];
  const palette = ['#2d5cc8', '#c8452b', '#e8a32c', '#0f1b3a', '#f2f5ff'];
  let x = 0;
  while (x < width) {
    const bg = palette[Math.floor(rng() * palette.length)];
    const word = words[Math.floor(rng() * words.length)];
    g.font = `800 ${Math.round(height * 0.56)}px system-ui, -apple-system, sans-serif`;
    const w = g.measureText(word).width + height * 1.2;
    g.fillStyle = bg;
    g.fillRect(x, 0, w, height);
    // A slab of contrast so the board is never one flat colour.
    g.fillStyle = 'rgba(0,0,0,0.45)';
    g.fillRect(x, height * 0.74, w, height * 0.26);
    g.fillStyle = bg === '#f2f5ff' ? '#08101f' : '#f4f8ff';
    g.textBaseline = 'middle';
    g.fillText(word, x + height * 0.6, height * 0.42);
    x += w + 4;
  }
  return toTexture(c);
}

/** The jumbotron face: a scoreboard graphic that reads at a glance. */
export function bakeJumbotronFace(width = 512, height = 288): CanvasTexture {
  const { c, g } = canvas(width, height);
  g.fillStyle = '#050810';
  g.fillRect(0, 0, width, height);

  // Upper 60%: "video" — an abstract court-side plate with a warm key.
  const grad = g.createLinearGradient(0, 0, 0, height * 0.58);
  grad.addColorStop(0, '#1a2c4e');
  grad.addColorStop(0.55, '#3d5478');
  grad.addColorStop(1, '#0b1220');
  g.fillStyle = grad;
  g.fillRect(0, 0, width, height * 0.58);
  const rng = makeRng(3311);
  for (let i = 0; i < 90; i++) {
    g.fillStyle = `rgba(${180 + rng() * 60 | 0},${150 + rng() * 70 | 0},${110 + rng() * 60 | 0},${0.05 + rng() * 0.16})`;
    const w = 6 + rng() * 34;
    g.fillRect(rng() * width, rng() * height * 0.58, w, 4 + rng() * 12);
  }
  g.fillStyle = 'rgba(255,232,190,0.20)';
  g.fillRect(0, height * 0.40, width, height * 0.06);

  // Lower band: the score bug.
  g.fillStyle = '#070b16';
  g.fillRect(0, height * 0.58, width, height * 0.42);
  g.fillStyle = '#2d5cc8';
  g.fillRect(0, height * 0.58, width * 0.30, height * 0.42);
  g.fillStyle = '#c8452b';
  g.fillRect(width * 0.70, height * 0.58, width * 0.30, height * 0.42);

  g.textBaseline = 'middle';
  g.textAlign = 'center';
  g.fillStyle = '#f4f8ff';
  g.font = `800 ${Math.round(height * 0.13)}px system-ui, sans-serif`;
  g.fillText('HOM', width * 0.15, height * 0.68);
  g.fillText('AWY', width * 0.85, height * 0.68);
  g.font = `800 ${Math.round(height * 0.22)}px system-ui, sans-serif`;
  g.fillText('88', width * 0.15, height * 0.87);
  g.fillText('84', width * 0.85, height * 0.87);

  g.fillStyle = '#e8a32c';
  g.font = `800 ${Math.round(height * 0.20)}px system-ui, sans-serif`;
  g.fillText('4:37', width * 0.5, height * 0.72);
  g.fillStyle = '#9db0d4';
  g.font = `700 ${Math.round(height * 0.11)}px system-ui, sans-serif`;
  g.fillText('4TH', width * 0.5, height * 0.90);

  return toTexture(c);
}

/** Championship banner: a hung felt panel with a year and a word. */
export function bakeBannerAtlas(cols = 4, rows = 2, cell = 128): CanvasTexture {
  const { c, g } = canvas(cols * cell, rows * cell * 2);
  const rng = makeRng(77);
  const years = ['1994', '2001', '2008', '2013', '2017', '2022', '2024', '2026'];
  const kinds = ['CHAMPIONS', 'DIVISION', 'CONFERENCE', 'CHAMPIONS'];
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const x = i * cell;
      const y = j * cell * 2;
      const h = cell * 2;
      const base = rng() < 0.5 ? '#16305e' : '#7c2a1c';
      g.fillStyle = base;
      g.fillRect(x, y, cell, h);
      g.strokeStyle = '#c9a227';
      g.lineWidth = cell * 0.05;
      g.strokeRect(x + cell * 0.08, y + cell * 0.08, cell * 0.84, h - cell * 0.16);
      g.textAlign = 'center';
      g.textBaseline = 'middle';
      g.fillStyle = '#e7d9a8';
      g.font = `800 ${Math.round(cell * 0.2)}px system-ui, sans-serif`;
      g.fillText(kinds[(i + j) % kinds.length], x + cell * 0.5, y + h * 0.34);
      g.font = `800 ${Math.round(cell * 0.28)}px system-ui, sans-serif`;
      g.fillText(years[(i + j * cols) % years.length], x + cell * 0.5, y + h * 0.55);
      // Bottom point of the pennant.
      g.fillStyle = '#04060c';
      g.beginPath();
      g.moveTo(x, y + h * 0.86);
      g.lineTo(x + cell * 0.5, y + h);
      g.lineTo(x + cell, y + h * 0.86);
      g.lineTo(x + cell, y + h);
      g.lineTo(x, y + h);
      g.closePath();
      g.fill();
    }
  }
  const t = toTexture(c);
  t.wrapS = t.wrapT = RepeatWrapping;
  return t;
}

// -----------------------------------------------------------------------------
// LED material
// -----------------------------------------------------------------------------

export interface LedOptions {
  map: Texture;
  /** Emitters across the board's U axis — drives the lattice frequency. */
  pixelsU: number;
  pixelsV: number;
  /** Scroll speed in UV units per second. */
  scroll?: number;
  /** Scene-linear multiplier. > 1 pushes the board above display white. */
  gain?: number;
  /** UV tiling of the strip across the mesh. */
  repeat?: Vector2;
  side?: typeof DoubleSide | undefined;
}

/**
 * Emissive LED surface. `toneMapped` stays *on*: a board that skips the tone
 * curve keeps full saturation all the way to 255, which is the signature of a
 * missing tonemapper. Instead we push scene-linear gain above 1 and let the
 * ACES shoulder desaturate the core, so the board clips to near-white ringed by
 * its own colour — exactly what a camera does to an LED wall.
 */
export function makeLedMaterial(o: LedOptions): ShaderMaterial {
  const mat = new ShaderMaterial({
    uniforms: {
      uMap: { value: o.map },
      uTime: { value: 0 },
      uScroll: { value: o.scroll ?? 0.06 },
      uGain: { value: o.gain ?? 2.4 },
      uPixels: { value: new Vector2(o.pixelsU, o.pixelsV) },
      uRepeat: { value: o.repeat ?? new Vector2(1, 1) },
      uFlicker: { value: 0 },
    },
    side: o.side,
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      varying vec3 vWorld;
      void main() {
        vUv = uv;
        vec4 wp = modelMatrix * vec4( position, 1.0 );
        vWorld = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      #include <common>
      uniform sampler2D uMap;
      uniform float uTime;
      uniform float uScroll;
      uniform float uGain;
      uniform vec2 uPixels;
      uniform vec2 uRepeat;
      varying vec2 vUv;
      varying vec3 vWorld;

      void main() {
        vec2 uv = vUv * uRepeat;
        uv.x += uTime * uScroll;
        vec3 c = texture2D( uMap, uv ).rgb;

        // Discrete emitters behind a black mask. Fades out as soon as one
        // screen pixel spans more than an emitter, so it never moires.
        vec2 cell = fract( vUv * uPixels );
        vec2 fw = fwidth( vUv * uPixels );
        float vis = 1.0 - smoothstep( 0.30, 0.75, max( fw.x, fw.y ) );
        float gx = smoothstep( 0.0, 0.16, cell.x ) * smoothstep( 1.0, 0.84, cell.x );
        float gy = smoothstep( 0.0, 0.16, cell.y ) * smoothstep( 1.0, 0.84, cell.y );
        c *= mix( 1.0, 0.82 + 0.18 * gx * gy, vis );

        // Emitters are Lambertian-ish but the mask cuts them off at grazing
        // angles, so a board seen edge-on dims instead of staying full blast.
        vec3 V = normalize( cameraPosition - vWorld );
        vec3 N = normalize( cross( dFdx( vWorld ), dFdy( vWorld ) ) );
        float graze = abs( dot( N, V ) );
        float fall = mix( 0.34, 1.0, smoothstep( 0.0, 0.55, graze ) );

        vec3 lit = c * uGain * fall;
        // Scan refresh: a very faint rolling bar, the way a camera sees LED.
        lit *= 1.0 - 0.05 * sin( vWorld.y * 9.0 - uTime * 5.0 );

        gl_FragColor = vec4( lit, 1.0 );
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }
    `,
  });
  mat.name = 'arena.led';
  return mat;
}

// -----------------------------------------------------------------------------
// Haze
// -----------------------------------------------------------------------------

/**
 * Upper-bowl atmosphere. A back-faced shell that adds a height-banded, very low
 * density glow so the rigging reads through air and the far side of the room
 * falls off. Deliberately confined above the seating lip — visible shafts over
 * the hardwood are a concert, not a basketball game.
 */
export function makeHazeMaterial(): ShaderMaterial {
  const mat = new ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uBottom: { value: 7.0 },
      uTop: { value: 21.0 },
      uColor: { value: new Color('#7f96c4') },
      uDensity: { value: 0.095 },
    },
    transparent: true,
    depthWrite: false,
    blending: AdditiveBlending,
    side: BackSide,
    vertexShader: /* glsl */ `
      varying vec3 vWorld;
      void main() {
        vec4 wp = modelMatrix * vec4( position, 1.0 );
        vWorld = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      #include <common>
      uniform float uTime;
      uniform float uBottom;
      uniform float uTop;
      uniform vec3 uColor;
      uniform float uDensity;
      varying vec3 vWorld;

      float hash13( vec3 p ) {
        p = fract( p * 0.1031 );
        p += dot( p, p.yzx + 33.33 );
        return fract( ( p.x + p.y ) * p.z );
      }

      void main() {
        float h = smoothstep( uBottom, uBottom + 3.0, vWorld.y ) *
                  ( 1.0 - smoothstep( uTop - 5.0, uTop + 1.0, vWorld.y ) );
        float d = length( cameraPosition - vWorld );
        // Nothing within arm's reach of the camera contributes, or a camera
        // that wanders into the shell gets a full-frame milk filter.
        float far = smoothstep( 14.0, 58.0, d );
        // Slow drift so the haze is not a static gradient.
        float drift = 0.82 + 0.18 * sin( vWorld.x * 0.09 + uTime * 0.13 ) *
                              sin( vWorld.z * 0.11 - uTime * 0.09 );
        float a = uDensity * h * far * drift;
        // Dither: a 0.02 alpha gradient over 900 px bands like mad without it.
        a += ( hash13( floor( gl_FragCoord.xyz * vec3( 1.0, 1.0, 0.0 ) + vec3( 0.0, 0.0, 3.0 ) ) ) - 0.5 ) * 0.004;
        vec3 c = uColor * a;
        gl_FragColor = vec4( c, 1.0 );
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }
    `,
  });
  mat.name = 'arena.haze';
  return mat;
}

/**
 * A single light-bank shaft: a soft cone of scattered light under a fixture,
 * fading out well above the seating lip.
 */
export function makeShaftMaterial(): ShaderMaterial {
  const mat = new ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uColor: { value: new Color('#dce8ff') },
      uStrength: { value: 0.016 },
    },
    transparent: true,
    depthWrite: false,
    blending: AdditiveBlending,
    side: DoubleSide,
    vertexShader: /* glsl */ `
      varying vec3 vWorld;
      varying vec2 vUv;
      void main() {
        vUv = uv;
        vec4 wp = modelMatrix * vec4( position, 1.0 );
        vWorld = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      #include <common>
      uniform vec3 uColor;
      uniform float uStrength;
      uniform float uTime;
      varying vec3 vWorld;
      varying vec2 vUv;
      void main() {
        // Radial softness across the cone, plus a vertical fade so the shaft
        // dies out before it reaches the hardwood.
        float r = abs( vUv.x * 2.0 - 1.0 );
        float radial = pow( 1.0 - r, 2.2 );
        float vert = smoothstep( 0.0, 0.35, vUv.y ) * ( 1.0 - smoothstep( 0.55, 1.0, vUv.y ) );
        float flick = 0.94 + 0.06 * sin( uTime * 0.7 + vWorld.x * 0.3 );
        // A shaft is a volume seen edge-on; a flat card facing the camera is a
        // slab of milk. Kill it as the surface turns to face us, and kill it
        // near the camera.
        vec3 V = normalize( cameraPosition - vWorld );
        vec3 N = normalize( cross( dFdx( vWorld ), dFdy( vWorld ) ) );
        float edgeOn = 1.0 - abs( dot( N, V ) );
        float near = smoothstep( 6.0, 20.0, length( cameraPosition - vWorld ) );
        gl_FragColor = vec4( uColor * ( uStrength * radial * vert * flick * edgeOn * near ), 1.0 );
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }
    `,
  });
  mat.name = 'arena.shaft';
  return mat;
}

// -----------------------------------------------------------------------------
// Architecture materials
// -----------------------------------------------------------------------------

/**
 * Four shared materials cover every square metre of arena architecture. Albedo
 * lives in the vertex colour, so a single merged mesh can carry concrete, seat
 * plastic, painted steel and vinyl padding without splitting the draw call —
 * and every piece gets its own tonal jitter, which is what stops a 40 m wall
 * from sampling to one flat value.
 */
export interface ArenaPalette {
  /** Concrete, seat shells, risers, vinyl, fabric. */
  matte: MeshStandardMaterial;
  /** Painted steel, plastic trim, padding — a soft sheen. */
  satin: MeshStandardMaterial;
  /** Rails, truss, rigging, camera bodies. */
  steel: MeshStandardMaterial;
  /** Suite glass and the dark specular bands. */
  gloss: MeshStandardMaterial;
}

export function makeArenaPalette(): ArenaPalette {
  const matte = new MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.94,
    metalness: 0,
    vertexColors: true,
  });
  // Nothing in a real arena is at zero: there is aisle lighting, exit signage
  // and concourse bounce everywhere. A hair of emissive keeps the deepest
  // geometry off the black point instead of crushing to a shapeless void.
  matte.emissive = new Color(0x070a12);
  matte.envMapIntensity = 0.55;
  matte.name = 'arena.matte';

  const satin = new MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.55,
    metalness: 0.04,
    vertexColors: true,
  });
  satin.emissive = new Color(0x060810);
  satin.envMapIntensity = 0.6;
  satin.name = 'arena.satin';

  const steel = new MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.36,
    metalness: 0.86,
    vertexColors: true,
  });
  steel.envMapIntensity = 0.8;
  steel.name = 'arena.steel';

  const gloss = new MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.13,
    metalness: 0.35,
    vertexColors: true,
  });
  gloss.envMapIntensity = 0.7;
  gloss.name = 'arena.gloss';

  return { matte, satin, steel, gloss };
}

const _c = new Color();

/** Linear RGB triple for a hex, ready to be written into a vertex-colour attribute. */
export function rgb(hex: number, scale = 1): [number, number, number] {
  _c.setHex(hex);
  return [_c.r * scale, _c.g * scale, _c.b * scale];
}
