/**
 * The crowd.
 *
 * At the distances a portrait gameplay camera sees a seating bowl — 18 to 45 m,
 * so a spectator is 12 to 40 px tall — faces are meaningless and silhouette,
 * density and motion are everything. So this builds *bodies*: a seated pitch
 * with a lap, a leaned-back torso, shoulders wider than the waist, a head and a
 * hair cap that breaks the outline, arms down the sides, and the seat itself
 * underneath so an empty place still reads as a seat rather than a hole.
 *
 * The economics: one InstancedMesh per bowl block, where an *instance* is a pod
 * of several adjacent seats rather than one person. That gets 8 000 people out
 * of 1 700 instances, one draw call each. Every per-person difference — height,
 * girth, yaw, clothing hue and value, skin tone, hair, whether they are on a
 * phone, whether they are standing right now, and their idle phase and
 * frequency — is derived in the vertex shader from a hash of (instance seed,
 * seat index). Nothing is a per-frame CPU matrix write; the only thing the CPU
 * touches each frame is one float per pod, the excitement level, which is what
 * drives the reaction wave.
 *
 * Owned by the arena agent.
 */

import {
  Color,
  DynamicDrawUsage,
  InstancedBufferAttribute,
  InstancedMesh,
  Object3D,
  ShaderMaterial,
  StaticDrawUsage,
  Vector2,
  Vector3,
} from 'three';
import { FACE_ALL, FACE_OPEN_BOTTOM, FACE_SIDES, FACE_TOP, MeshBuilder } from './arenaGeometry';

// Vertex part tags. Kept as floats because they ride in an attribute.
const PART_SEAT = 0;
const PART_LAP = 1;
const PART_TORSO = 2;
const PART_HEAD = 3;
const PART_HAIR = 4;
const PART_ARM = 5;
const PART_PHONE = 6;
const PART_SHIN = 7;

/** Local shoulder height — the mirror line for a raised arm. */
const SHOULDER_Y = 0.94;

export type CrowdDetail = 0 | 1 | 2 | 3;

export interface PodPlacement {
  x: number;
  y: number;
  z: number;
  rotY: number;
  /** Local exposure multiplier: courtside sits closer to the floor spill. */
  tone: number;
}

export interface CrowdBlockSpec {
  name: string;
  detail: CrowdDetail;
  /** Seats per instance. */
  podSize: number;
  /** Seat pitch along the row, metres. */
  pitch: number;
  /** 0–1 fraction of seats that hold a person. */
  occupancy: number;
  /** Fraction of spectators holding a lit phone. */
  phoneRate: number;
  /** Base seat shell colour. */
  seatColor: number;
  placements: PodPlacement[];
}

// -----------------------------------------------------------------------------
// Geometry
// -----------------------------------------------------------------------------

/**
 * One pod: `podSize` seats side by side along local +X, all facing local −Z
 * (toward the court). Local origin sits on the row tread at the pod's centre.
 */
function buildPod(podSize: number, pitch: number, detail: CrowdDetail, seatColor: number): MeshBuilder {
  const b = new MeshBuilder();
  b.attribute('aPart', 1, [PART_SEAT]);
  b.attribute('aSeat', 1, [0]);
  b.attribute('aBaseX', 1, [0]);
  b.attribute('aSeatCol', 3, [0, 0, 0]);

  const col = new Color(seatColor);
  b.set('aSeatCol', col.r, col.g, col.b);

  const halfSeat = pitch * 0.5;
  for (let k = 0; k < podSize; k++) {
    const bx = (k - (podSize - 1) * 0.5) * pitch;
    b.set('aSeat', k);
    b.set('aBaseX', bx);

    // --- the seat itself -----------------------------------------------------
    if (detail >= 1) {
      b.set('aPart', PART_SEAT);
      // Pan: a single upward quad. The underside is never seen from the court.
      const pw = halfSeat * 0.86;
      b.quad(
        bx - pw, 0.44, 0.03,
        bx + pw, 0.44, 0.03,
        bx + pw, 0.435, 0.42,
        bx - pw, 0.435, 0.42,
      );
      // Back, raked back 8°, with a visible gap to its neighbour.
      b.box(bx, 0.665, 0.425, pw, 0.225, 0.035, {
        faces: detail >= 2 ? FACE_SIDES | FACE_TOP : FACE_SIDES,
        shearZ: 0.055,
      });
    }

    // --- the person ----------------------------------------------------------
    b.set('aPart', PART_LAP);
    if (detail >= 1) {
      // Thighs: hips at the seat back, knees forward and slightly down.
      b.box(bx, 0.5, 0.06, 0.165, 0.075, 0.19, { faces: FACE_SIDES | FACE_TOP });
    }
    if (detail >= 3) {
      b.set('aPart', PART_SHIN);
      b.box(bx, 0.21, -0.125, 0.14, 0.21, 0.07, { faces: FACE_SIDES });
    }

    b.set('aPart', PART_TORSO);
    b.box(bx, 0.77, 0.19, 0.185, 0.25, 0.125, {
      faces: detail >= 1 ? FACE_OPEN_BOTTOM : FACE_ALL,
      // Shoulders wider than the waist and narrower front-to-back than the
      // chest: without the second taper the top face is a slab and a bowl full
      // of them reads as a field of envelopes.
      topScaleX: 1.22,
      bottomScaleX: 0.82,
      topScaleZ: 0.84,
      shearZ: 0.045,
    });

    if (detail >= 2) {
      b.set('aPart', PART_ARM);
      for (const s of [-1, 1]) {
        b.box(bx + s * 0.207, 0.75, 0.15, 0.05, 0.19, 0.07, { faces: FACE_SIDES });
      }
    }

    // A diamond reads as a diamond at 20 px. A slightly tapered box under a
    // hair cap reads as a head.
    b.set('aPart', PART_HEAD);
    b.box(bx, 1.135, 0.155, 0.083, 0.098, 0.088, {
      faces: FACE_SIDES | FACE_TOP,
      topScaleX: 0.92,
      topScaleZ: 0.92,
      bottomScaleX: 0.86,
      bottomScaleZ: 0.86,
    });

    b.set('aPart', PART_HAIR);
    b.pyramid(bx, 1.222, 0.155, 0.081, 0.078, 0.085);

    if (detail >= 2) {
      // A phone held at chest height. Collapsed for everyone who is not on one.
      b.set('aPart', PART_PHONE);
      b.quad(
        bx - 0.028, 0.83, -0.055,
        bx + 0.028, 0.83, -0.055,
        bx + 0.028, 0.915, -0.045,
        bx - 0.028, 0.915, -0.045,
      );
    }
  }
  return b;
}

// -----------------------------------------------------------------------------
// Material
// -----------------------------------------------------------------------------

const CROWD_VERT = /* glsl */ `
  attribute float aPart;
  attribute float aSeat;
  attribute float aBaseX;
  attribute vec3  aSeatCol;
  attribute float iSeed;
  attribute float iReact;
  attribute float iTone;

  uniform float uTime;
  uniform float uOccupancy;
  uniform float uPhoneRate;
  uniform float uExposure;
  uniform float uPhoneGain;
  uniform vec3  uKeyDir;
  uniform vec3  uKeyCol;
  uniform vec3  uFillDir;
  uniform vec3  uFillCol;
  uniform vec3  uAmbTop;
  uniform vec3  uAmbBot;
  uniform vec3  uSpillCol;
  uniform vec3  uHazeCol;
  uniform vec2  uHazeRange;
  uniform float uHazeAmount;
  uniform float uRim;
  uniform float uAnimate;

  varying vec3 vCol;

  float rnd( float n ) { return fract( sin( n * 12.9898 ) * 43758.5453 ); }

  // Clothing. A real bowl is overwhelmingly dark — coats, hoodies, charcoal —
  // with a minority in team colour and a thin scattering of white shirts. Get
  // that distribution wrong in either direction and the crowd reads as confetti
  // or as a black wall.
  vec3 clothColour( float k, float v ) {
    vec3 c;
    if ( k < 0.46 ) {
      // Dark neutrals, faintly cool.
      float g = 0.020 + v * 0.055;
      c = vec3( g * 0.94, g * 0.97, g * 1.14 );
    } else if ( k < 0.64 ) {
      // Mid charcoal / denim / olive.
      float g = 0.070 + v * 0.075;
      c = mix( vec3( g, g * 1.02, g * 1.22 ), vec3( g * 1.1, g * 1.0, g * 0.78 ), step( 0.5, v ) );
    } else if ( k < 0.755 ) {
      c = vec3( 0.028, 0.075, 0.245 ) * ( 0.6 + v * 0.9 );      // home blue
    } else if ( k < 0.83 ) {
      c = vec3( 0.30, 0.055, 0.028 ) * ( 0.6 + v * 0.9 );        // away red
    } else if ( k < 0.875 ) {
      c = vec3( 0.34, 0.20, 0.030 ) * ( 0.6 + v * 0.8 );         // gold
    } else if ( k < 0.955 ) {
      float g = 0.30 + v * 0.30;                                  // white shirts
      c = vec3( g, g * 0.985, g * 0.96 );
    } else {
      c = vec3( 0.05 + v * 0.20, 0.15 + v * 0.15, 0.07 + v * 0.08 ); // odd greens
    }
    return c;
  }

  void main() {
    float part = aPart;
    bool isSeat = part < 0.5;

    // --- per-person identity ------------------------------------------------
    float id   = iSeed * 61.0 + aSeat * 1.37;
    float kOcc = rnd( id + 1.0 );
    float kSize= rnd( id + 2.0 );
    float kYaw = rnd( id + 3.0 );
    float kLat = rnd( id + 4.0 );
    float kPh  = rnd( id + 5.0 );
    float kFreq= rnd( id + 6.0 );
    float kCol = rnd( id + 7.0 );
    float kVal = rnd( id + 8.0 );
    float kSkin= rnd( id + 9.0 );
    float kHair= rnd( id + 10.0 );
    float kStand = rnd( id + 11.0 );
    float kPhone = rnd( id + 12.0 );
    float kArms  = rnd( id + 13.0 );

    float present = step( kOcc, uOccupancy );

    vec3 p = position;
    vec3 n = normal;
    float rel = 0.0;

    if ( !isSeat ) {
      // Collapse absent spectators and un-held phones into the seat pan.
      float hasPhone = step( kPhone, uPhoneRate );
      float keep = present * mix( 1.0, hasPhone, step( 5.5, part ) * step( part, 6.5 ) );

      vec3 q = vec3( p.x - aBaseX, p.y, p.z );

      // Standing. People do not rise together: each has their own trigger
      // level, so excitement sweeps up through a section instead of flipping it.
      float stand = smoothstep( kStand * 0.82, kStand * 0.82 + 0.30, iReact ) * present;

      bool isLeg = ( part > 0.5 && part < 1.5 ) || part > 6.5;
      if ( isLeg ) {
        // Thighs and shins straighten into legs.
        q.y = mix( q.y, q.y * 1.92, stand );
        q.z = mix( q.z, q.z * 0.42 + 0.06, stand );
      } else {
        q.y += 0.44 * stand;
        // Arms go up for a fraction of the people who stand.
        float raise = stand * step( kArms, 0.42 );
        if ( part > 4.5 && part < 5.5 ) {
          q.y = mix( q.y, 2.0 * ( ${SHOULDER_Y.toFixed(3)} + 0.44 * stand ) - q.y, raise );
          q.x *= mix( 1.0, 1.22, raise );
        }
        // Lean forward when up.
        q.z -= ( q.y - 0.5 ) * 0.10 * stand;
      }

      // Size: height and girth vary independently, so the row is not a wave of
      // one silhouette scaled up and down.
      float hScale = 0.90 + kSize * 0.21;
      float wScale = 0.90 + kVal * 0.22;
      q.y *= hScale;
      q.x *= wScale;
      q.z *= mix( 0.94, 1.10, kSize );

      // Yaw, so neighbours do not present identical faces to the camera.
      float yaw = ( kYaw - 0.5 ) * 0.62 + sin( uTime * ( 0.19 + kFreq * 0.2 ) + kPh * 12.0 ) * 0.10 * uAnimate;
      float cy = cos( yaw ), sy = sin( yaw );
      vec3 r = vec3( q.x * cy + q.z * sy, q.y, -q.x * sy + q.z * cy );
      n = vec3( n.x * cy + n.z * sy, n.y, -n.x * sy + n.z * cy );

      // Idle life: a slow lean plus a small bob, every one at its own phase and
      // rate. Amplitude scales with height off the seat so the lap stays put.
      float ph = kPh * 6.2831853;
      float fr = 0.62 + kFreq * 0.62;
      float sway = sin( uTime * fr + ph ) * 0.026 * uAnimate;
      float bob  = sin( uTime * fr * 1.7 + ph * 1.9 ) * 0.010 * uAnimate;
      // Standing people bounce noticeably harder.
      sway *= 1.0 + stand * 2.6;
      bob  *= 1.0 + stand * 3.4;
      float lever = max( r.y - 0.45, 0.0 );
      r.x += sway * lever;
      r.z += sway * 0.45 * lever;
      r.y += bob * min( lever * 2.0, 1.0 );

      r.x += ( kLat - 0.5 ) * 0.09;
      r.z += ( kSize - 0.5 ) * 0.07;

      p = vec3( aBaseX + r.x, r.y, r.z ) * keep;
      p += vec3( aBaseX, 0.42, 0.24 ) * ( 1.0 - keep );
      rel = r.y;
    }

    vec3 albedo;
    float emissive = 0.0;

    if ( isSeat ) {
      // Seat shells carry their own scatter so a bank of empties is not a
      // single flat value.
      albedo = aSeatCol * ( 0.78 + rnd( id + 21.0 ) * 0.46 );
    } else if ( part > 2.5 && part < 3.5 ) {
      // Skin. Range of tones, and the SSS-ish warmth is baked into the hue.
      float tone = 0.30 + kSkin * kSkin * 0.85;
      albedo = vec3( 0.235, 0.135, 0.092 ) * tone + vec3( 0.012, 0.006, 0.004 );
    } else if ( part > 3.5 && part < 4.5 ) {
      // Hair: mostly near-black, occasional brown, rare grey.
      float g = kHair < 0.72 ? 0.012 + kHair * 0.02 : ( kHair < 0.93 ? 0.05 : 0.16 );
      albedo = vec3( g * 1.12, g * 0.95, g * 0.86 );
    } else if ( part > 5.5 && part < 6.5 ) {
      albedo = vec3( 0.55, 0.62, 0.95 );
      emissive = uPhoneGain * ( 0.72 + 0.28 * sin( uTime * 3.1 + kPh * 20.0 ) );
    } else {
      albedo = clothColour( kCol, kVal );
      // Trousers read darker than tops almost universally.
      if ( part < 1.5 || part > 6.5 ) albedo *= 0.55;
    }

    // --- transform ----------------------------------------------------------
    vec4 world = modelMatrix * instanceMatrix * vec4( p, 1.0 );
    mat3 im = mat3( instanceMatrix );
    vec3 N = normalize( mat3( modelMatrix ) * ( im * n ) );

    // --- lighting -----------------------------------------------------------
    // The bowl is not lit by the court banks; it catches their spill, the
    // house lights and a warm kick off the hardwood. Deliberately shaped so
    // shoulders and the tops of heads separate and laps fall away.
    float ndK = max( dot( N, uKeyDir ), 0.0 );
    float ndF = max( dot( N, uFillDir ), 0.0 );
    float hemi = 0.5 + 0.5 * N.y;
    // Self-occlusion down the body: rows shadow each other and a seated torso
    // shadows its own lap.
    float ao = mix( 0.30, 1.0, smoothstep( 0.15, 1.15, rel ) );
    if ( isSeat ) ao = 0.42;

    vec3 lit = albedo * (
        uKeyCol * ndK
      + uFillCol * ndF
      + uAmbTop * hemi
      + uAmbBot * ( 1.0 - hemi )
    ) * ao;

    // Courtside LED / hardwood spill: strongest on the lowest rows, which is
    // what iTone encodes.
    lit += albedo * uSpillCol * max( iTone - 1.0, 0.0 ) * ( 0.35 + 0.65 * ndF );

    // A thin backlight so silhouettes cut against the row behind them.
    vec3 V = normalize( cameraPosition - world.xyz );
    float fres = pow( 1.0 - clamp( dot( N, V ), 0.0, 1.0 ), 3.0 );
    lit += uKeyCol * fres * uRim * ( 0.25 + albedo );

    lit *= uExposure * iTone;
    lit += albedo * emissive;

    // Depth cueing — far stands lose contrast, which is most of what makes a
    // big room read as big.
    float d = length( cameraPosition - world.xyz );
    float haze = smoothstep( uHazeRange.x, uHazeRange.y, d ) * uHazeAmount;
    // Floor the bowl off the black point: crushed regions with no detail are a
    // named tell, and a real building always has some ambient spill.
    vCol = max( mix( lit, uHazeCol, haze ), vec3( 0.0042, 0.0046, 0.0062 ) );

    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const CROWD_FRAG = /* glsl */ `
  #include <common>
  varying vec3 vCol;
  void main() {
    gl_FragColor = vec4( max( vCol, vec3( 0.0 ) ), 1.0 );
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`;

export interface CrowdUniformSet {
  uTime: { value: number };
  uOccupancy: { value: number };
  uPhoneRate: { value: number };
  uExposure: { value: number };
  uPhoneGain: { value: number };
  uKeyDir: { value: Vector3 };
  uKeyCol: { value: Color };
  uFillDir: { value: Vector3 };
  uFillCol: { value: Color };
  uAmbTop: { value: Color };
  uAmbBot: { value: Color };
  uSpillCol: { value: Color };
  uHazeCol: { value: Color };
  uHazeRange: { value: Vector2 };
  uHazeAmount: { value: number };
  uRim: { value: number };
  uAnimate: { value: number };
}

function makeCrowdMaterial(occupancy: number, phoneRate: number, animate: boolean): ShaderMaterial {
  const mat = new ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uOccupancy: { value: occupancy },
      uPhoneRate: { value: phoneRate },
      // Tuned against §1.1: the bowl has to land 2.5–4 stops under the hardwood.
      uExposure: { value: 4.4 },
      uPhoneGain: { value: 2.6 },
      uKeyDir: { value: new Vector3(0.36, 0.86, 0.36).normalize() },
      uKeyCol: { value: new Color(0.30, 0.325, 0.40) },
      uFillDir: { value: new Vector3(-0.5, 0.42, -0.75).normalize() },
      uFillCol: { value: new Color(0.14, 0.135, 0.16) },
      uAmbTop: { value: new Color(0.095, 0.110, 0.152) },
      uAmbBot: { value: new Color(0.080, 0.066, 0.052) },
      uSpillCol: { value: new Color(0.5, 0.42, 0.34) },
      uHazeCol: { value: new Color(0.030, 0.037, 0.055) },
      uHazeRange: { value: new Vector2(22, 74) },
      uHazeAmount: { value: 0.55 },
      uRim: { value: 0.10 },
      uAnimate: { value: animate ? 1 : 0 },
    },
    vertexShader: CROWD_VERT,
    fragmentShader: CROWD_FRAG,
  });
  mat.name = 'arena.crowd';
  return mat;
}

// -----------------------------------------------------------------------------
// System-facing crowd
// -----------------------------------------------------------------------------

interface Block {
  mesh: InstancedMesh;
  react: InstancedBufferAttribute;
  /** Pod world XZ, for the reaction wave. */
  px: Float32Array;
  pz: Float32Array;
  /** Current excitement, 0–1. */
  level: Float32Array;
  /** Scheduled wave arrival time and amplitude. */
  waveAt: Float32Array;
  waveAmp: Float32Array;
}

export class Crowd {
  readonly blocks: Block[] = [];
  readonly materials: ShaderMaterial[] = [];

  private time = 0;
  private baseline = 0.055;
  private _dummy = new Object3D();

  /** Total spectators actually modelled, for the perf report. */
  people = 0;
  seats = 0;

  add(spec: CrowdBlockSpec, animated: boolean): InstancedMesh {
    const count = spec.placements.length;
    if (count === 0) throw new Error('crowd block with no placements');

    const geo = buildPod(spec.podSize, spec.pitch, spec.detail, spec.seatColor).build(
      `crowd.${spec.name}`,
    );
    const mat = makeCrowdMaterial(spec.occupancy, spec.phoneRate, animated);
    this.materials.push(mat);

    const mesh = new InstancedMesh(geo, mat, count);
    mesh.name = `crowd.${spec.name}`;
    mesh.frustumCulled = true;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    mesh.instanceMatrix.setUsage(StaticDrawUsage);

    const seed = new Float32Array(count);
    const react = new Float32Array(count);
    const tone = new Float32Array(count);
    const px = new Float32Array(count);
    const pz = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      const p = spec.placements[i];
      this._dummy.position.set(p.x, p.y, p.z);
      this._dummy.rotation.set(0, p.rotY, 0);
      this._dummy.scale.setScalar(1);
      this._dummy.updateMatrix();
      mesh.setMatrixAt(i, this._dummy.matrix);
      seed[i] = ((i * 0.6180339887 + 0.1237) % 1) * 0.97 + 0.013;
      tone[i] = p.tone;
      px[i] = p.x;
      pz[i] = p.z;
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.computeBoundingSphere();

    geo.setAttribute('iSeed', new InstancedBufferAttribute(seed, 1));
    const reactAttr = new InstancedBufferAttribute(react, 1);
    reactAttr.setUsage(DynamicDrawUsage);
    geo.setAttribute('iReact', reactAttr);
    geo.setAttribute('iTone', new InstancedBufferAttribute(tone, 1));

    this.blocks.push({
      mesh,
      react: reactAttr,
      px,
      pz,
      level: new Float32Array(count),
      waveAt: new Float32Array(count).fill(-1),
      waveAmp: new Float32Array(count),
    });

    this.people += count * spec.podSize * spec.occupancy;
    this.seats += count * spec.podSize;
    return mesh;
  }

  /**
   * Fire a reaction. The wave leaves the epicentre at `speed` m/s so the
   * section under the play comes up first and the far corner follows about a
   * second later, which is exactly how a real building sounds and looks.
   */
  excite(epicentre: Vector3, amplitude: number, speed = 26, spread = 1): void {
    for (const b of this.blocks) {
      for (let i = 0; i < b.px.length; i++) {
        const dx = b.px[i] - epicentre.x;
        const dz = b.pz[i] - epicentre.z;
        const d = Math.hypot(dx, dz);
        const falloff = 1 / (1 + Math.pow(d / (26 * spread), 2.1));
        const amp = amplitude * (0.32 + 0.68 * falloff);
        const at = this.time + d / speed;
        // A stronger, earlier wave wins.
        if (b.waveAt[i] < 0 || amp > b.waveAmp[i] * 0.9) {
          b.waveAt[i] = at;
          b.waveAmp[i] = amp;
        }
      }
    }
  }

  /** Slow ambient excitement, e.g. a tight game late. */
  setBaseline(v: number): void {
    this.baseline = v;
  }

  update(dt: number, elapsed: number): void {
    this.time = elapsed;
    for (const m of this.materials) m.uniforms.uTime.value = elapsed;

    const attack = 1 - Math.exp(-9 * dt);
    const release = 1 - Math.exp(-1.3 * dt);

    for (const b of this.blocks) {
      const n = b.level.length;
      const arr = b.react.array as Float32Array;
      let dirty = false;
      for (let i = 0; i < n; i++) {
        let target = this.baseline;
        const at = b.waveAt[i];
        if (at >= 0) {
          const age = elapsed - at;
          if (age >= 0) {
            // Fast rise, ~1.6 s hold, slow settle back into the seats.
            const env =
              age < 0.22 ? age / 0.22 : Math.exp(-Math.max(0, age - 0.22) * 0.62);
            target = Math.max(target, b.waveAmp[i] * env);
            if (age > 6) b.waveAt[i] = -1;
          }
        }
        const cur = b.level[i];
        const next = cur + (target - cur) * (target > cur ? attack : release);
        if (Math.abs(next - arr[i]) > 0.002) dirty = true;
        b.level[i] = next;
        arr[i] = next;
      }
      if (dirty) b.react.needsUpdate = true;
    }
  }

  /** Bulk uniform poke — used to tune bowl exposure from one place. */
  setUniform(name: string, value: unknown): void {
    for (const m of this.materials) {
      const u = m.uniforms[name];
      if (u) u.value = value as never;
    }
  }

  dispose(): void {
    for (const b of this.blocks) {
      b.mesh.geometry.dispose();
      b.mesh.dispose();
    }
    for (const m of this.materials) m.dispose();
    this.blocks.length = 0;
    this.materials.length = 0;
  }
}
