/**
 * The animation driver — one per player.
 *
 * Gameplay never authors poses. It states intent ("moving at 5.2 m/s, drifting
 * 20° right, dribbling") and fires discrete actions ("shoot at the rim"); this
 * resolves that into a locomotion blend, layers any running action over the
 * upper body, applies IK for contact, and reports back the frames gameplay
 * cares about — the release, the foot plants, the apex of a jump.
 *
 * Owned by the animation agent. See AnimatorTypes.ts for the contract.
 */

import { Quaternion, Vector3 } from 'three';
import { CHAINS, type BuiltSkeleton } from '../entities/Skeleton';
import {
  ClipPlayer,
  addPose,
  applyPose,
  copyPose,
  makePose,
  upperBodyMask,
  type BoneMask,
  type Clip,
  type Pose,
} from './Pose';
import { solveBallHold, solveLegPlant, solveLookAt } from './IK';
import { clamp01, damp, makeRng, smootherstep } from '../core/MathX';
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
const _right = new Vector3();
const _target = new Vector3();
const _axis = new Vector3();
const _quat = new Quaternion();
const UP = new Vector3(0, 1, 0);

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
const INTERRUPTIBLE = new Set<ActionKind>(['crossover', 'spin', 'stepback', 'celebrate', 'dejected']);

/** True when `mark` lies in the wrapped interval (a, b]. */
function crossed(a: number, b: number, mark: number): boolean {
  return a <= b ? a < mark && mark <= b : mark > a || mark <= b;
}

export class Animator implements IAnimator {
  readonly skeleton: BuiltSkeleton;

  events: AnimatorEvents = {};

  private readonly opts: AnimatorOptions;
  private readonly mask: BoneMask = upperBodyMask();

  private locomotion: LocomotionIntent = { ...IDLE_INTENT };
  private ik: IkIntent = { ...DEFAULT_IK };

  private gait = new ClipPlayer();
  private action = new ClipPlayer();
  private lastGait: Clip = LOCOMOTION.idle;

  private actionKind: ActionKind | null = null;
  private actionWeight = 0;
  private actionElapsed = 0;
  private actionDuration = 0;

  private finalPose: Pose = makePose();

  private stridePhase = 0;
  private smoothedSpeed = 0;
  private elapsed = 0;
  private lift = 0;

  /** Per-player idiosyncrasy so a roster does not move in lockstep. */
  private readonly idleOffset: number;
  private readonly strideBias: number;

  constructor(skeleton: BuiltSkeleton, options: AnimatorOptions) {
    this.skeleton = skeleton;
    this.opts = options;
    const rng = makeRng(options.seed);
    this.idleOffset = rng() * Math.PI * 2;
    this.strideBias = 0.92 + rng() * 0.16;
    this.gait.play(LOCOMOTION.idle, 0);
    this.action.onEvent = (name) => this.onActionEvent(name);
  }

  get currentClip(): string | null {
    return this.actionKind ? this.action.clipName : this.gait.clipName;
  }

  get busy(): boolean {
    return this.actionKind !== null;
  }

  get rootLift(): number {
    return this.lift;
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
    this.actionElapsed = 0;
    this.actionDuration = clip.duration / (params.rate ?? 1);
    this.action.play(clip, 0.08, params.rate ?? 1);
    return true;
  }

  cancelAction(): void {
    this.actionKind = null;
    this.actionWeight = 0;
    this.actionElapsed = 0;
  }

  private onActionEvent(name: string): void {
    const kind = this.actionKind;
    if (!kind) return;
    if (name === 'release') this.events.onRelease?.(kind);
    else if (name === 'apex') this.events.onApex?.(kind);
  }

  update(dt: number): void {
    this.elapsed += dt;
    const loco = this.locomotion;

    // --- Locomotion --------------------------------------------------------
    this.smoothedSpeed = damp(this.smoothedSpeed, loco.speed, 9, dt);
    const speed = this.smoothedSpeed;

    const gaitClip = this.pickGait(loco, speed);
    // Stride rate scales with speed so feet do not skate: the run cycle covers
    // about 2.05 m of ground per loop at rate 1.
    const strideRate = speed > 0.15 ? Math.max(0.55, speed / 2.05) * this.strideBias : 1;
    this.gait.play(gaitClip, gaitClip === this.lastGait ? 0 : 0.16, strideRate);
    this.lastGait = gaitClip;
    const gaitPose = this.gait.update(dt);

    if (speed > 0.15 && loco.airborne < 0.5) {
      const before = this.stridePhase;
      this.stridePhase = (this.stridePhase + strideRate * dt) % 1;
      if (crossed(before, this.stridePhase, 0.08)) {
        this.events.onFootPlant?.('left', clamp01(speed / 7));
      }
      if (crossed(before, this.stridePhase, 0.58)) {
        this.events.onFootPlant?.('right', clamp01(speed / 7));
      }
    }

    // --- Action layer ------------------------------------------------------
    if (this.actionKind) {
      this.actionElapsed += dt;
      const t = clamp01(this.actionElapsed / Math.max(1e-3, this.actionDuration));
      // Ease in fast, out a little slower, so the follow-through reads.
      this.actionWeight = Math.min(smootherstep(t / 0.12), smootherstep((1 - t) / 0.22));
      const pose = this.action.update(dt);
      addPose(this.finalPose, gaitPose, pose, this.actionWeight, this.mask);
      if (t >= 1) {
        const done = this.actionKind;
        this.actionKind = null;
        this.actionWeight = 0;
        this.events.onActionEnd?.(done);
      }
    } else {
      copyPose(this.finalPose, gaitPose);
    }

    // A body that holds perfectly still reads as a mannequin. A continuous
    // breath costs nothing and fixes it; it fades out as the player picks up
    // speed and the stride takes over.
    const settle = 1 - clamp01(speed / 2);
    this.finalPose.rootOffset.y += Math.sin(this.idleOffset + this.elapsed * 1.7) * 0.012 * settle;
    // Lean into cuts.
    this.finalPose.rootOffset.x += loco.lean * 0.035;

    this.lift = this.finalPose.rootOffset.y;
    applyPose(this.skeleton, this.finalPose);
    this.applyIk();
  }

  private pickGait(loco: LocomotionIntent, speed: number): Clip {
    if (loco.airborne > 0.5) return LOCOMOTION.air;
    if (loco.defending) return speed > 0.4 ? LOCOMOTION.slide : LOCOMOTION.stance;
    if (loco.dribbling) {
      if (speed > 5.4) return LOCOMOTION.dribbleSprint;
      if (speed > 0.5) return LOCOMOTION.dribbleRun;
      return LOCOMOTION.dribbleIdle;
    }
    if (speed > 6.4) return LOCOMOTION.sprint;
    if (speed > 2.2) return LOCOMOTION.run;
    if (speed > 0.35) return LOCOMOTION.jog;
    return LOCOMOTION.idle;
  }

  private applyIk(): void {
    const sk = this.skeleton;
    const ik = this.ik;

    const chest = sk.byName.upperChest;
    chest.updateWorldMatrix(true, false);
    chest.getWorldQuaternion(_quat);
    _fwd.set(0, 0, 1).applyQuaternion(_quat).normalize();
    _right.set(1, 0, 0).applyQuaternion(_quat).normalize();

    if (ik.plantFeet && this.locomotion.airborne < 0.4) {
      this.plantLeg('legL', ik.floorY);
      this.plantLeg('legR', ik.floorY);
    }

    // Ball hold — skipped while an action clip is driving the arms.
    if (ik.ball?.active && this.actionWeight < 0.35) {
      const shoot = this.opts.dominantHand === 'left' ? CHAINS.armL : CHAINS.armR;
      const guide = this.opts.dominantHand === 'left' ? CHAINS.armR : CHAINS.armL;
      solveBallHold(
        [sk.byName[shoot[0]], sk.byName[shoot[1]], sk.byName[shoot[2]]],
        [sk.byName[guide[0]], sk.byName[guide[1]], sk.byName[guide[2]]],
        ik.ball.centre,
        ik.ball.radius,
        _fwd,
        _right,
        1 - this.actionWeight,
      );
    }

    if (ik.lookAt?.active) {
      solveLookAt(
        sk.byName.head,
        ik.lookAt.target,
        _axis.set(0, 0, 1),
        { yaw: 1.05, pitch: 0.62 },
        ik.lookAt.weight,
      );
    }
  }

  /**
   * Corrects a foot that the clip has driven below the floor. A foot above the
   * floor is mid-stride and the clip is right about where it belongs.
   */
  private plantLeg(chain: 'legL' | 'legR', floorY: number): void {
    const sk = this.skeleton;
    const [thighName, shinName, footName] = CHAINS[chain];
    const foot = sk.byName[footName];
    foot.updateWorldMatrix(true, false);
    _target.setFromMatrixPosition(foot.matrixWorld);

    const ankleHeight = sk.height * 0.039;
    if (_target.y >= floorY + ankleHeight - 0.004) return;
    _target.y = floorY + ankleHeight;

    sk.byName.hips.getWorldQuaternion(_quat);
    _fwd.set(0, 0, 1).applyQuaternion(_quat).normalize();
    solveLegPlant(
      sk.byName[thighName],
      sk.byName[shinName],
      foot,
      { target: _target, normal: UP, weight: 1 },
      _fwd,
    );
  }
}
