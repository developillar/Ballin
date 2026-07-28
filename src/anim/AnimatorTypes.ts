/**
 * The contract between the character layer and the animation layer.
 *
 * `PlayerSystem` owns bodies; `Animator` owns motion. They meet here and
 * nowhere else, so either side can be rewritten without touching the other.
 *
 * The division of labour: gameplay tells the animator *intent* ("I am moving at
 * 5.2 m/s and I want to shoot"), never poses. The animator resolves intent into
 * a blended pose, writes it onto the skeleton, applies IK, and reports back the
 * moments gameplay actually cares about — the frame the ball leaves the hand,
 * the frame a foot plants.
 */

import type { Vector3 } from 'three';
import type { BuiltSkeleton } from '../entities/Skeleton';

/**
 * Discrete one-shot actions. Locomotion is continuous and set separately.
 *
 * The list is append-only: callers that only ever trigger the original set keep
 * compiling, and the animator falls back to locomotion for any kind it cannot
 * build a clip for.
 */
export type ActionKind =
  // --- original set -------------------------------------------------------
  | 'shoot'
  | 'jumpShot'
  | 'layup'
  | 'dunk'
  | 'pass'
  | 'bouncePass'
  | 'block'
  | 'steal'
  | 'rebound'
  | 'crossover'
  | 'stepback'
  | 'spin'
  | 'celebrate'
  | 'dejected'
  // --- finishes -----------------------------------------------------------
  | 'fadeaway'
  | 'floater'
  | 'hookShot'
  | 'fingerRoll'
  | 'reverseLayup'
  | 'euroStep'
  | 'dunkTwoHand'
  | 'dunkTomahawk'
  // --- post and passing ---------------------------------------------------
  | 'postUp'
  | 'dropStep'
  | 'overheadPass'
  // --- defence ------------------------------------------------------------
  | 'closeout'
  | 'contest'
  | 'boxOut'
  // --- footwork transitions ----------------------------------------------
  | 'cut'
  | 'pivot'
  | 'jumpStop'
  | 'hardStop'
  | 'accelBurst';

export interface ActionParams {
  /** World point the action is aimed at — the rim for a shot, a team-mate for a pass. */
  target?: Vector3;
  /** 0..1 release quality; drives follow-through confidence and flourish. */
  quality?: number;
  /** Playback rate multiplier. */
  rate?: number;
  /** Which hand leads. */
  hand?: 'left' | 'right';
  /** For dunks: how much hang and reach to commit to. */
  power?: number;
  /**
   * Signed direction the move goes, in radians relative to facing. Only the
   * sign is used today (cut left / cut right, spin direction, euro-step lead).
   */
  direction?: number;
  /** Two-foot gather. Dunks and contests read very differently off one foot. */
  twoFoot?: boolean;
  /** Free-form variant selector — `'quick'` shot, `'betweenLegs'` handle, etc. */
  style?: string;
  /** Overrides the cross-fade into the action, in seconds. */
  blendIn?: number;
}

/** Continuous locomotion intent, refreshed every frame. */
export interface LocomotionIntent {
  /** Ground speed in m/s. */
  speed: number;
  /** Direction of travel relative to facing, radians. 0 = forward. */
  driftAngle: number;
  /** True while the player is in a defensive slide. */
  defending: boolean;
  /** True while dribbling — swaps in the ball-handling upper body. */
  dribbling: boolean;
  /** 0 = grounded, 1 = fully airborne. */
  airborne: number;
  /** Signed lean for cuts and hard stops, roughly -1..1. */
  lean: number;
  /** Fatigue 0..1 — bleeds energy out of the idle and the stride. */
  fatigue: number;
}

/** Where the animator should place hands and eyes, resolved by gameplay. */
export interface IkIntent {
  /** Two-handed ball hold. Ignored while an action clip owns the arms. */
  ball: { active: boolean; centre: Vector3; radius: number } | null;
  /** Head and eye tracking. */
  lookAt: { active: boolean; target: Vector3; weight: number } | null;
  /** Floor height under each foot, for planting on a non-flat surface. */
  floorY: number;
  /** Disable foot planting while airborne. */
  plantFeet: boolean;
}

export interface AnimatorEvents {
  /** The frame the ball should leave the hand. */
  onRelease?: (kind: ActionKind) => void;
  /** A foot struck the floor; `force` is 0..1. */
  onFootPlant?: (foot: 'left' | 'right', force: number) => void;
  /** The action clip finished and the animator is back on locomotion. */
  onActionEnd?: (kind: ActionKind) => void;
  /** Peak of a jump — the moment a dunk should attach the ball to the rim. */
  onApex?: (kind: ActionKind) => void;
  /** The frame a jump's feet retake the floor. Optional; ignore it if unused. */
  onLand?: (kind: ActionKind) => void;
  /** The dribble hand reached the ball. `force` is 0..1. */
  onDribbleTouch?: (hand: 'left' | 'right', force: number) => void;
}

export interface AnimatorOptions {
  /** Seeds per-player idiosyncrasy: stride timing, shot form, idle fidgets. */
  seed: number;
  /** Standing height in metres; clip amplitudes scale off it. */
  height: number;
  /** Handedness. */
  dominantHand: 'left' | 'right';
}

/**
 * The animation driver. One instance per player.
 *
 * Call order per frame is fixed and matters:
 *   setLocomotion() → setIk() → update(dt)
 * `update` writes bone rotations onto the skeleton it was constructed with and
 * leaves world matrices current, so the skinned mesh can be drawn immediately.
 */
export interface IAnimator {
  readonly skeleton: BuiltSkeleton;
  /** Name of the clip currently dominating the blend, for debugging. */
  readonly currentClip: string | null;
  /** True while a one-shot action owns the upper body. */
  readonly busy: boolean;
  /** Vertical offset the animation wants applied to the root, in metres. */
  readonly rootLift: number;

  events: AnimatorEvents;

  setLocomotion(intent: LocomotionIntent): void;
  setIk(intent: IkIntent): void;
  /** Fires a one-shot action; returns false if one is already running and cannot be interrupted. */
  trigger(kind: ActionKind, params?: ActionParams): boolean;
  /** Cancels any running action and returns to locomotion. */
  cancelAction(): void;
  update(dt: number): void;

  /**
   * Optional: a physical knock. `dir` is in world space, `strength` 0..1. The
   * animator turns it into a decaying whole-body reaction on top of whatever
   * else is playing. Safe to ignore — it is not part of the required contract.
   */
  react?(dir: Vector3, strength: number): void;
}
