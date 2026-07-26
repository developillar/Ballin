/**
 * The animation driver — one per player.
 *
 * Gameplay never authors poses. It states intent ("moving at 5.2 m/s, drifting
 * 20° right, dribbling") and fires discrete actions ("shoot at the rim"); this
 * resolves that into a locomotion blend, layers any running action over the
 * body, adds the secondary motion that stops a body reading as a mannequin,
 * applies IK for contact, and reports back the frames gameplay cares about —
 * the release, the foot plants, the apex of a jump.
 *
 * The stack, in the order it composes:
 *
 *   1. `GaitBlender` — a continuous speed/direction blend at a phase locked to
 *      ground speed, so stride rate matches translation exactly.
 *   2. A procedural **momentum layer** — trunk pitch from acceleration, roll
 *      from turning. Physically derived (`θ ≈ atan(a/g)`), so a player leans
 *      into a start and rocks back into a stop without any authored clip.
 *   3. The **action layer**, either additive over the upper body (a pass on the
 *      move) or a masked override of the whole body (a jump shot).
 *   4. `SecondaryMotion` — breathing, fatigue, fidgets, contact and landing
 *      shocks.
 *   5. **IK** — world-space foot locking (the thing that actually kills foot
 *      skating), the two-handed ball hold with real hand orientation, and a
 *      clamped look-at.
 *
 * Owned by the animation agent. See AnimatorTypes.ts for the contract.
 */

import { Quaternion, Vector3 } from 'three';
import { BONE_INDEX, CHAINS, LANDMARK, type BoneName, type BuiltSkeleton } from '../entities/Skeleton';
import {
  ClipPlayer,
  addPose,
  applyPose,
  blendPoseMasked,
  copyPose,
  identityPose,
  makeMask,
  makePose,
  sampleClip,
  upperBodyMask,
  type BoneMask,
  type Pose,
} from './Pose';
import { solveBallHold, solveLegPlant, solveLookAt } from './IK';
import { GaitBlender } from './Locomotion';
import { SecondaryMotion, addBone } from './Secondary';
import { clamp, clamp01, damp, makeRng, smootherstep, wrapAngle } from '../core/MathX';
import type {
  ActionKind,
  ActionParams,
  AnimatorEvents,
  AnimatorOptions,
  IAnimator,
  IkIntent,
  LocomotionIntent,
} from './AnimatorTypes';
import { LOCOMOTION, actionClip } from './clips';

const _fwd = new Vector3();
const _left = new Vector3();
const _target = new Vector3();
const _axis = new Vector3();
const _quat = new Quaternion();
const _tmp = new Vector3();
const UP = new Vector3(0, 1, 0);
const GRAVITY = 9.80665;

const IDLE_INTENT: LocomotionIntent = {
  speed: 0,
  driftAngle: 0,
  defending: false,
  dribbling: false,
  airborne: 0,
  lean: 0,
  fatigue: 0,
};

const DEFAULT_IK: IkIntent = { ball: null, lookAt: null, floorY: 0, plantFeet: true };

/** Actions a newly triggered action is allowed to cut off. */
const INTERRUPTIBLE = new Set<ActionKind>([
  'crossover', 'spin', 'stepback', 'celebrate', 'dejected', 'postUp', 'boxOut',
  'closeout', 'cut', 'pivot', 'jumpStop', 'hardStop', 'accelBurst', 'euroStep',
]);

/** Actions the animator may raise on its own from locomotion alone. */
const AUTO_ACTIONS = new Set<ActionKind>(['cut', 'hardStop', 'jumpStop', 'accelBurst']);

interface FootLock {
  active: boolean;
  point: Vector3;
  weight: number;
}

export class Animator implements IAnimator {
  readonly skeleton: BuiltSkeleton;

  events: AnimatorEvents = {};

  /** Set false to stop the animator raising its own plants, cuts and stops. */
  autoTransitions = true;

  private readonly opts: AnimatorOptions;
  private readonly upperMask: BoneMask = upperBodyMask();
  private readonly fullMask: BoneMask = makeMask(1);
  private readonly trunkMask: BoneMask;
  private readonly dribbleMask: BoneMask;

  private locomotion: LocomotionIntent = { ...IDLE_INTENT };
  private ik: IkIntent = { ...DEFAULT_IK };

  private readonly gaitTree: GaitBlender;
  private readonly secondary: SecondaryMotion;
  private action = new ClipPlayer();

  private actionKind: ActionKind | null = null;
  private actionAuto = false;
  private actionWeight = 0;
  private actionElapsed = 0;
  private actionDuration = 0;
  private actionRegion: 'upper' | 'full' = 'upper';
  private actionAdditive = true;

  private readonly workPose: Pose = makePose();
  private readonly momentum: Pose = makePose();
  private readonly dribblePose: Pose = makePose();
  private finalPose: Pose = makePose();

  private smoothedSpeed = 0;
  private accel = 0;
  private turnRate = 0;
  private prevDrift = 0;
  private leanPitch = 0;
  private leanRoll = 0;
  private dribblePhase = 0;
  private elapsed = 0;
  private lift = 0;
  private lastPlantSide: 'left' | 'right' = 'right';

  private readonly lockL: FootLock = { active: false, point: new Vector3(), weight: 0 };
  private readonly lockR: FootLock = { active: false, point: new Vector3(), weight: 0 };

  /** Per-player idiosyncrasy so a roster does not move in lockstep. */
  private readonly idleOffset: number;

  constructor(skeleton: BuiltSkeleton, options: AnimatorOptions) {
    this.skeleton = skeleton;
    this.opts = options;
    const rng = makeRng(options.seed);
    this.idleOffset = rng() * Math.PI * 2;
    // ±8% on stride length: taller striders and choppier ones, from one seed.
    this.gaitTree = new GaitBlender(0.94 + rng() * 0.12);
    this.secondary = new SecondaryMotion(options.seed);
    this.gaitTree.onPlant = (foot, force) => this.onFootPlant(foot, force);

    this.trunkMask = makeMask(0);
    for (const b of ['hips', 'spine', 'chest', 'upperChest', 'neck', 'head'] as const) {
      this.trunkMask[boneIndex(b)] = 1;
    }
    this.dribbleMask = makeMask(0);
    const side = options.dominantHand === 'left' ? 'L' : 'R';
    for (const b of [`clavicle${side}`, `upperArm${side}`, `foreArm${side}`, `hand${side}`] as const) {
      this.dribbleMask[boneIndex(b)] = 1;
    }
    this.dribbleMask[boneIndex(`clavicle${side}`)] = 0.4;

    this.action.onEvent = (name) => this.onActionEvent(name);
  }

  get currentClip(): string | null {
    return this.actionKind ? this.action.clipName : this.gaitTree.dominant;
  }

  get busy(): boolean {
    return this.actionKind !== null && !this.actionAuto;
  }

  get rootLift(): number {
    return this.lift;
  }

  /** Cycles per second the gait is currently running at. Diagnostics. */
  get cadence(): number {
    return this.gaitTree.cadence;
  }

  /** Ground distance one full stride cycle covers, metres. Diagnostics. */
  get strideLength(): number {
    return this.gaitTree.strideMetres;
  }

  setLocomotion(intent: LocomotionIntent): void {
    this.locomotion = intent;
  }

  setIk(intent: IkIntent): void {
    this.ik = intent;
  }

  trigger(kind: ActionKind, params: ActionParams = {}): boolean {
    if (this.actionKind && !INTERRUPTIBLE.has(this.actionKind)) return false;
    const clip = actionClip(kind, params, this.opts);
    if (!clip) return false;
    this.actionKind = kind;
    this.actionAuto = false;
    this.actionElapsed = 0;
    this.actionDuration = clip.duration / (params.rate ?? 1);
    this.actionRegion = clip.region ?? 'upper';
    this.actionAdditive = (clip.layer ?? 'additive') === 'additive';
    this.action.play(clip, params.blendIn ?? (this.actionAdditive ? 0.07 : 0.11), params.rate ?? 1);
    return true;
  }

  cancelAction(): void {
    this.actionKind = null;
    this.actionAuto = false;
    this.actionWeight = 0;
    this.actionElapsed = 0;
  }

  /** A knock from a defender, a screen, a box-out. `dir` is world space. */
  react(dir: Vector3, strength: number): void {
    this.skeleton.byName.hips.getWorldQuaternion(_quat);
    _fwd.set(0, 0, 1).applyQuaternion(_quat);
    _left.set(1, 0, 0).applyQuaternion(_quat);
    _tmp.copy(dir);
    const mag = _tmp.length();
    if (mag > 1e-5) _tmp.divideScalar(mag);
    this.secondary.hit(
      { back: -_tmp.dot(_fwd), side: _tmp.dot(_left), lift: _tmp.y },
      clamp01(strength),
    );
  }

  private onFootPlant(foot: 'left' | 'right', force: number): void {
    this.lastPlantSide = foot;
    const lock = foot === 'left' ? this.lockL : this.lockR;
    lock.active = false; // re-captured on the next IK pass
    this.secondary.footStrike(force);
    this.events.onFootPlant?.(foot, force);
  }

  private onActionEvent(name: string): void {
    const kind = this.actionKind;
    if (!kind) return;
    if (name === 'release') this.events.onRelease?.(kind);
    else if (name === 'apex') this.events.onApex?.(kind);
    else if (name === 'land') {
      this.events.onLand?.(kind);
      this.secondary.footStrike(0.85);
    } else if (name === 'plantA' || name === 'plantB') {
      const side = name === 'plantA' ? this.lastPlantSide : this.lastPlantSide === 'left' ? 'right' : 'left';
      this.onFootPlant(side, 0.8);
    }
  }

  update(dt: number): void {
    const step = Math.max(1e-5, Math.min(dt, 0.1));
    this.elapsed += step;
    const loco = this.locomotion;

    // --- Momentum bookkeeping ---------------------------------------------
    const prevSpeed = this.smoothedSpeed;
    this.smoothedSpeed = damp(this.smoothedSpeed, loco.speed, 12, step);
    const rawAccel = (this.smoothedSpeed - prevSpeed) / step;
    this.accel = damp(this.accel, clamp(rawAccel, -40, 40), 10, step);
    const dDrift = wrapAngle(loco.driftAngle - this.prevDrift) / step;
    this.prevDrift = loco.driftAngle;
    this.turnRate = damp(this.turnRate, clamp(dDrift, -12, 12), 8, step);

    this.maybeAutoTransition(step);

    // --- 1. Locomotion -----------------------------------------------------
    const gaitPose = this.gaitTree.update(step, {
      speed: loco.speed,
      driftAngle: loco.driftAngle,
      defending: loco.defending,
      airborne: loco.airborne,
      fatigue: loco.fatigue,
      height: this.skeleton.height,
    });
    copyPose(this.workPose, gaitPose);

    // --- 2. Momentum layer -------------------------------------------------
    this.buildMomentum(step, loco);
    addPose(this.workPose, this.workPose, this.momentum, 1, this.trunkMask);

    // --- 3. Dribble --------------------------------------------------------
    if (loco.dribbling && this.actionWeight < 0.5 && loco.airborne < 0.5) {
      const beats = this.gaitTree.cadence > 0.05 ? this.gaitTree.cadence : 1.55;
      const before = this.dribblePhase;
      this.dribblePhase = (this.dribblePhase + beats * step) % 1;
      const clip = this.opts.dominantHand === 'left' ? LOCOMOTION.dribbleL : LOCOMOTION.dribbleR;
      sampleClip(this.dribblePose, clip, this.dribblePhase);
      const w = (1 - this.actionWeight) * clamp01(1 - loco.airborne * 2);
      addPose(this.workPose, this.workPose, this.dribblePose, w, this.dribbleMask);
      // Two touches per cycle, on the push-down beat.
      for (const mark of [0.1, 0.6]) {
        if (crossed(before, this.dribblePhase, mark)) {
          this.events.onDribbleTouch?.(this.opts.dominantHand, clamp01(0.5 + this.smoothedSpeed / 12));
        }
      }
    }

    // --- 4. Action layer ---------------------------------------------------
    if (this.actionKind) {
      this.actionElapsed += step;
      const t = clamp01(this.actionElapsed / Math.max(1e-3, this.actionDuration));
      // In fast, out slower, so the follow-through reads before the gait returns.
      const inT = this.actionAdditive ? 0.1 : 0.14;
      this.actionWeight = Math.min(smootherstep(t / inT), smootherstep((1 - t) / 0.24));
      const pose = this.action.update(step);
      const mask = this.actionRegion === 'full' ? this.fullMask : this.upperMask;
      if (this.actionAdditive) {
        addPose(this.finalPose, this.workPose, pose, this.actionWeight, mask);
      } else {
        blendPoseMasked(this.finalPose, this.workPose, pose, this.actionWeight, mask, this.actionWeight);
      }
      if (t >= 1) {
        const done = this.actionKind;
        this.actionKind = null;
        this.actionAuto = false;
        this.actionWeight = 0;
        this.events.onActionEnd?.(done);
      }
    } else {
      this.actionWeight = 0;
      copyPose(this.finalPose, this.workPose);
    }

    // --- 5. Secondary motion ----------------------------------------------
    const sec = this.secondary.update(step, this.smoothedSpeed, loco.fatigue, this.actionWeight);
    addPose(this.finalPose, this.finalPose, sec, 1);

    this.lift = this.finalPose.rootOffset.y;
    applyPose(this.skeleton, this.finalPose);
    this.applyIk(step);
  }

  // -------------------------------------------------------------------------

  /**
   * Trunk attitude from momentum. A body accelerating at `a` has to lean by
   * `atan(a/g)` to put the ground reaction through its centre of mass, and the
   * same relation run backwards is what makes a hard stop rock the chest back.
   * Deriving it rather than authoring it means every speed change gets the
   * right amount of lean for free.
   */
  private buildMomentum(dt: number, loco: LocomotionIntent): void {
    const airborne = clamp01(loco.airborne);
    const wantPitch = clamp(Math.atan2(this.accel, GRAVITY), -0.42, 0.48) * (1 - airborne);
    // Banking into a turn: the faster and tighter the arc, the more roll.
    const bank = clamp(this.turnRate * this.smoothedSpeed / (GRAVITY * 2.4), -0.3, 0.3);
    const wantRoll = clamp(bank + loco.lean * 0.22, -0.34, 0.34) * (1 - airborne);
    this.leanPitch = damp(this.leanPitch, wantPitch, 7, dt);
    this.leanRoll = damp(this.leanRoll, wantRoll, 6, dt);

    identityPose(this.momentum);
    const p = this.leanPitch;
    const r = this.leanRoll;
    if (Math.abs(p) > 1e-4 || Math.abs(r) > 1e-4) {
      // The pelvis takes a little, the spine most of it, and the head undoes
      // enough of it that the eyes stay on the play.
      addBone(this.momentum, 'hips', p * 0.16, 0, r * 0.3);
      addBone(this.momentum, 'spine', p * 0.34, 0, r * 0.3);
      addBone(this.momentum, 'chest', p * 0.3, 0, r * 0.24);
      addBone(this.momentum, 'upperChest', p * 0.2, 0, r * 0.16);
      addBone(this.momentum, 'neck', -p * 0.3, 0, -r * 0.3);
      addBone(this.momentum, 'head', -p * 0.42, 0, -r * 0.42);
    }
    // Deceleration also drops the hips — you cannot stop standing tall.
    const brake = clamp01(-this.accel / 22) * (1 - airborne);
    this.momentum.rootOffset.y -= brake * 0.06 * this.skeleton.height / 1.98;
  }

  /**
   * Raises footwork the gameplay layer did not ask for: a plant-and-cut when
   * the heading swings hard at speed, a two-foot stop when the player brakes
   * hard, and a first-step burst out of a standstill. Gameplay actions always
   * win — these only fire when nothing else owns the body.
   */
  private maybeAutoTransition(dt: number): void {
    void dt;
    if (!this.autoTransitions) return;
    if (this.actionKind && !this.actionAuto) return;
    if (this.actionKind && this.actionAuto && this.actionElapsed < this.actionDuration * 0.6) return;
    if (this.locomotion.airborne > 0.3) return;

    const speed = this.smoothedSpeed;
    // Hard deceleration: PLAYER.deceleration is 30 m/s², so half of that is
    // unambiguously a stop rather than an ease-off.
    if (this.accel < -15 && speed > 1.6) {
      this.fireAuto(speed > 4.6 ? 'hardStop' : 'jumpStop', {});
      return;
    }
    // A hard change of heading while carrying speed is a plant and a push.
    if (Math.abs(this.turnRate) > 2.6 && speed > 2.8) {
      this.fireAuto('cut', { direction: this.turnRate });
      return;
    }
    // First step out of a standstill.
    if (this.accel > 9 && speed > 0.35 && speed < 2.6) {
      this.fireAuto('accelBurst', { direction: this.lastPlantSide === 'left' ? -1 : 1 });
    }
  }

  private fireAuto(kind: ActionKind, params: ActionParams): void {
    if (!AUTO_ACTIONS.has(kind)) return;
    const clip = actionClip(kind, params, this.opts);
    if (!clip) return;
    this.actionKind = kind;
    this.actionAuto = true;
    this.actionElapsed = 0;
    this.actionDuration = clip.duration;
    this.actionRegion = clip.region ?? 'full';
    this.actionAdditive = (clip.layer ?? 'override') === 'additive';
    this.action.play(clip, 0.12, 1);
  }

  // -------------------------------------------------------------------------

  private applyIk(dt: number): void {
    const sk = this.skeleton;
    const ik = this.ik;

    const chest = sk.byName.upperChest;
    chest.updateWorldMatrix(true, false);
    chest.getWorldQuaternion(_quat);
    _fwd.set(0, 0, 1).applyQuaternion(_quat).normalize();
    _left.set(1, 0, 0).applyQuaternion(_quat).normalize();

    if (ik.plantFeet) {
      const grounded = 1 - clamp01(this.locomotion.airborne);
      // A full-body override (a jump shot, a dunk) owns the legs; the lock only
      // guards against the floor while it runs.
      const lockScale = grounded * (this.actionRegion === 'full' ? 1 - this.actionWeight : 1);
      this.plantLeg('legL', this.lockL, this.gaitTree.left.contact * lockScale, ik.floorY, dt);
      this.plantLeg('legR', this.lockR, this.gaitTree.right.contact * lockScale, ik.floorY, dt);
    }

    // Ball hold — skipped while an action clip is driving the arms.
    if (ik.ball?.active && this.actionWeight < 0.4) {
      const shootIsLeft = this.opts.dominantHand === 'left';
      const shoot = shootIsLeft ? CHAINS.armL : CHAINS.armR;
      const guide = shootIsLeft ? CHAINS.armR : CHAINS.armL;
      solveBallHold(
        [sk.byName[shoot[0]], sk.byName[shoot[1]], sk.byName[shoot[2]]],
        [sk.byName[guide[0]], sk.byName[guide[1]], sk.byName[guide[2]]],
        ik.ball.centre,
        ik.ball.radius,
        _fwd,
        _left,
        shootIsLeft,
        (1 - this.actionWeight / 0.4) * 0.95,
        0.062 * (sk.height / 1.98),
      );
    }

    if (ik.lookAt?.active) {
      // Chest and neck share the turn so the head is never doing all of it.
      const w = clamp01(ik.lookAt.weight);
      solveLookAt(sk.byName.neck, ik.lookAt.target, _axis.set(0, 0, 1), { yaw: 0.5, pitch: 0.34 }, w * 0.45);
      solveLookAt(sk.byName.head, ik.lookAt.target, _axis.set(0, 0, 1), { yaw: 0.85, pitch: 0.52 }, w);
    }
  }

  /**
   * World-space foot locking.
   *
   * When a foot takes weight its world position is captured, and for as long as
   * it is in contact the leg is solved back to that captured point while the
   * body translates past it. Stride rate already matches ground speed
   * (`Locomotion.ts`), so the correction is small — but "small" is not "zero",
   * and the rubric's limit is 15 mm of drift over half a second. This makes it
   * exactly zero for as long as the leg can reach, and releases cleanly rather
   * than stretching when it cannot.
   */
  private plantLeg(
    chain: 'legL' | 'legR',
    lock: FootLock,
    contact: number,
    floorY: number,
    dt: number,
  ): void {
    const sk = this.skeleton;
    const [thighName, shinName, footName] = CHAINS[chain];
    const thigh = sk.byName[thighName];
    const foot = sk.byName[footName];
    foot.updateWorldMatrix(true, false);
    _target.setFromMatrixPosition(foot.matrixWorld);

    const ankleHeight = sk.height * LANDMARK.ankle;
    const groundY = floorY + ankleHeight;

    lock.weight = damp(lock.weight, clamp01(contact), 18, dt);

    if (contact > 0.02) {
      if (!lock.active) {
        lock.active = true;
        lock.point.copy(_target);
        lock.point.y = groundY;
      }
      // Release rather than stretch: past this the leg would visibly straighten
      // and snap, which is worse than a few millimetres of slip.
      thigh.updateWorldMatrix(true, false);
      _tmp.setFromMatrixPosition(thigh.matrixWorld);
      const legLen = (LANDMARK.thigh + LANDMARK.shin) * sk.height;
      if (_tmp.distanceTo(lock.point) > legLen * 0.997) {
        lock.active = false;
        lock.weight = 0;
      }
    } else {
      lock.active = false;
    }

    if (lock.active && lock.weight > 0.01) {
      _target.copy(lock.point);
    } else if (_target.y >= groundY - 0.004) {
      // Swing phase and above the floor: the clip is right about where it goes.
      return;
    } else {
      // Below the floor — push it back out even with no lock.
      _target.y = groundY;
      lock.weight = Math.max(lock.weight, 0.85);
    }

    sk.byName.hips.getWorldQuaternion(_quat);
    _fwd.set(0, 0, 1).applyQuaternion(_quat).normalize();
    _tmp.set(chain === 'legL' ? 1 : -1, 0, 0).applyQuaternion(_quat).normalize();
    solveLegPlant(
      thigh,
      sk.byName[shinName],
      foot,
      { target: _target, normal: UP, weight: clamp01(lock.weight) },
      _fwd,
      _tmp,
    );
  }
}

// ---------------------------------------------------------------------------

const boneIndex = (name: BoneName): number => BONE_INDEX[name];

/** True when `mark` lies in the wrapped interval (a, b]. */
function crossed(a: number, b: number, mark: number): boolean {
  if (a === b) return false;
  return a <= b ? a < mark && mark <= b : mark > a || mark <= b;
}
