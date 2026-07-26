/**
 * The locomotion blend tree.
 *
 * One continuous parametric gait rather than a handful of clips with cross-
 * fades between them. Speed picks a pair of rungs on a ladder (walk → jog →
 * run → sprint) and blends them at a *shared phase*, so there is never a moment
 * where two cycles are fighting each other out of sync. Direction blends the
 * same way against a backpedal and a lateral shuffle.
 *
 * The important number in here is **stride length**. Each gait clip knows how
 * far the body travels in one cycle, measured from the forward kinematics of
 * its own leg curves (see `clips.ts::measureStride`). Playing the cycle at
 * `speed / strideLength` cycles per second makes the stride rate match the
 * ground speed by construction — which is the only way to make foot-skating
 * (rubric §9.1, tell #47) go away rather than be hidden.
 */

import { LOCOMOTION } from './clips';
import { blendPose, copyPose, makePose, sampleClip, type Clip, type Pose } from './Pose';
import { clamp01, damp, smoothstep } from '../core/MathX';

export interface GaitInput {
  /** Ground speed, m/s. */
  speed: number;
  /** Direction of travel relative to facing, radians. 0 = straight ahead. */
  driftAngle: number;
  defending: boolean;
  airborne: number;
  fatigue: number;
  /** Standing height, metres — stride lengths are stored as height fractions. */
  height: number;
}

/** Which foot, and how firmly it is on the floor right now. */
export interface FootState {
  /** 0 = swinging, 1 = fully weighted. */
  contact: number;
  /** Normalised time within this foot's stance, 0..1. */
  stanceT: number;
}

const REF_HEIGHT = 1.98;

export class GaitBlender {
  /** Cycle phase; the left foot strikes at 0, the right at 0.5. */
  phase = 0;
  /** Ground distance covered by one cycle, metres. */
  strideMetres = 1.5;
  /** Cycles per second currently being played. */
  cadence = 0;
  /** Blended stance fraction for the current speed. */
  stanceFraction = 0.4;
  /** Name of the dominant rung, for debugging. */
  dominant = 'idle';

  readonly left: FootState = { contact: 1, stanceT: 0 };
  readonly right: FootState = { contact: 1, stanceT: 0.5 };

  private readonly a = makePose();
  private readonly b = makePose();
  private readonly c = makePose();
  private readonly out = makePose();
  private smoothedSpeed = 0;
  private smoothedDrift = 0;
  private groundedness = 1;
  private prevPhase = 0;
  private idleTime = 0;
  private readonly strideBias: number;

  /** Fires when a foot strikes the floor. `force` is 0..1. */
  onPlant: ((foot: 'left' | 'right', force: number) => void) | null = null;

  constructor(strideBias = 1) {
    this.strideBias = strideBias;
  }

  /** Advances the tree and returns the blended locomotion pose. */
  update(dt: number, input: GaitInput): Pose {
    const heightScale = input.height / REF_HEIGHT;
    this.smoothedSpeed = damp(this.smoothedSpeed, input.speed, 11, dt);
    this.smoothedDrift = damp(this.smoothedDrift, input.driftAngle, 9, dt);
    const speed = this.smoothedSpeed;
    const grounded = 1 - clamp01(input.airborne);
    this.groundedness = damp(this.groundedness, grounded, 14, dt);

    // A tired player takes shorter, busier steps: bias the rung selection down
    // so the same ground speed is covered with more, smaller strides.
    const fatigue = clamp01(input.fatigue);
    const selectSpeed = speed * (1 + fatigue * 0.34) / heightScale;

    if (input.defending) return this.updateDefensive(dt, input, speed, heightScale);

    // --- Pick the two rungs and their weights ------------------------------
    const gaits = LOCOMOTION.gaits;
    let hi = 0;
    while (hi < gaits.length - 1 && (gaits[hi].refSpeed ?? 0) < selectSpeed) hi++;
    const lo = Math.max(0, hi - 1);
    const sLo = gaits[lo].refSpeed ?? 0;
    const sHi = gaits[hi].refSpeed ?? 1;
    const k = hi === lo ? 0 : clamp01((selectSpeed - sLo) / Math.max(0.001, sHi - sLo));
    this.dominant = (k > 0.5 ? gaits[hi] : gaits[lo]).name;

    const strideLo = gaits[lo].strideLength ?? 1;
    const strideHi = gaits[hi].strideLength ?? 1;
    const stanceLo = gaits[lo].stanceFraction ?? 0.4;
    const stanceHi = gaits[hi].stanceFraction ?? 0.4;
    // Fatigue shortens the stride directly as well as biasing the rung.
    const strideFrac = (strideLo + (strideHi - strideLo) * k) * (1 - fatigue * 0.12) * this.strideBias;
    this.stanceFraction = stanceLo + (stanceHi - stanceLo) * k;
    this.strideMetres = Math.max(0.15, strideFrac * input.height);

    // --- Advance the phase at exactly ground speed --------------------------
    this.prevPhase = this.phase;
    const moving = speed > 0.12 && this.groundedness > 0.35;
    this.cadence = moving ? speed / this.strideMetres : 0;
    if (moving) {
      this.phase = (this.phase + this.cadence * dt) % 1;
    } else if (speed <= 0.12) {
      // Settle the cycle onto a plausible standing phase instead of freezing
      // mid-swing with one foot in the air.
      const target = this.phase < 0.5 ? 0 : 1;
      this.phase = damp(this.phase, target, 7, dt) % 1;
    }

    this.updateFeet(speed);

    // --- Blend the rungs, then fold in direction and airborne ---------------
    sampleClip(this.a, gaits[lo], this.phase);
    if (k > 0.001) {
      sampleClip(this.b, gaits[hi], this.phase);
      blendPose(this.a, this.a, this.b, k);
    }

    // Direction. Backpedal and shuffle are separate cycles on the same phase.
    const fwd = Math.cos(this.smoothedDrift);
    const lat = Math.sin(this.smoothedDrift);
    const backW = clamp01(-fwd) * clamp01(speed / 1.2);
    const latW = Math.abs(lat) * clamp01(speed / 1.2);
    if (backW > 0.002) {
      sampleClip(this.b, LOCOMOTION.backpedal, this.phase);
      blendPose(this.a, this.a, this.b, backW);
    }
    if (latW > 0.002) {
      sampleClip(this.b, lat > 0 ? LOCOMOTION.slideL : LOCOMOTION.slideR, this.phase);
      blendPose(this.a, this.a, this.b, latW * 0.85);
    }

    // Idle. Below a walk the cycle has nothing to say, so fade into a settled
    // standing pose rather than playing a walk at rate 0.
    this.idleTime += dt;
    const moveW = smoothstep(clamp01((speed - 0.14) / 0.62));
    if (moveW < 0.999) {
      sampleClip(this.c, LOCOMOTION.idle, this.idleTime / LOCOMOTION.idle.duration);
      blendPose(this.a, this.c, this.a, moveW);
    }

    // Airborne overrides everything below the waist.
    const airW = clamp01(input.airborne);
    if (airW > 0.002) {
      sampleClip(this.b, LOCOMOTION.air, this.phase * 0.5);
      blendPose(this.a, this.a, this.b, smoothstep(airW));
    }

    copyPose(this.out, this.a);
    return this.out;
  }

  /** Defensive stance and slide share the same tree but a different rung set. */
  private updateDefensive(dt: number, input: GaitInput, speed: number, heightScale: number): Pose {
    this.dominant = speed > 0.4 ? 'slide' : 'stance';
    const slide: Clip = this.smoothedDrift >= 0 ? LOCOMOTION.slideL : LOCOMOTION.slideR;
    this.strideMetres = Math.max(0.2, (slide.strideLength ?? 0.62) * input.height);
    this.stanceFraction = slide.stanceFraction ?? 0.55;
    this.prevPhase = this.phase;
    const moving = speed > 0.25;
    this.cadence = moving ? speed / this.strideMetres : 0.55;
    this.phase = (this.phase + this.cadence * dt) % 1;
    this.updateFeet(speed);

    sampleClip(this.a, LOCOMOTION.stance, (this.phase * 0.6) % 1);
    const w = smoothstep(clamp01((speed - 0.2) / 0.9));
    if (w > 0.002) {
      sampleClip(this.b, slide, this.phase);
      blendPose(this.a, this.a, this.b, w);
    }
    const airW = clamp01(input.airborne);
    if (airW > 0.002) {
      sampleClip(this.b, LOCOMOTION.air, this.phase * 0.5);
      blendPose(this.a, this.a, this.b, smoothstep(airW));
    }
    void heightScale;
    copyPose(this.out, this.a);
    return this.out;
  }

  private updateFeet(speed: number): void {
    const stance = this.stanceFraction;
    const grounded = this.groundedness;
    const force = clamp01(speed / 7.5) * grounded;
    this.applyFoot(this.left, this.phase, 0, stance, grounded);
    this.applyFoot(this.right, this.phase, 0.5, stance, grounded);
    if (grounded > 0.5 && this.cadence > 0.01) {
      if (crossed(this.prevPhase, this.phase, 0)) this.onPlant?.('left', force);
      if (crossed(this.prevPhase, this.phase, 0.5)) this.onPlant?.('right', force);
    }
  }

  private applyFoot(
    foot: FootState,
    phase: number,
    offset: number,
    stance: number,
    grounded: number,
  ): void {
    const p = ((phase - offset) % 1 + 1) % 1;
    if (p >= stance) {
      foot.contact = 0;
      foot.stanceT = 1;
      return;
    }
    const u = p / Math.max(1e-4, stance);
    foot.stanceT = u;
    // Ramp the weight on at heel strike and off from mid-stance, so the world
    // lock never grabs or releases the foot in a single frame — and so it has
    // let go by the time the heel lifts and the ankle stops being the pivot.
    foot.contact = smoothstep(u / 0.14) * (1 - smoothstep((u - 0.42) / 0.2)) * grounded;
  }
}

/** True when `mark` lies in the wrapped interval (a, b]. */
function crossed(a: number, b: number, mark: number): boolean {
  if (a === b) return false;
  return a <= b ? a < mark && mark <= b : mark > a || mark <= b;
}
