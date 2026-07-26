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

/** Discrete one-shot actions. Locomotion is continuous and set separately. */
export type ActionKind =
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
  | 'dejected';

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
}
