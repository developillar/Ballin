/**
 * Player control.
 *
 * This is the layer that was missing: `Input` resolved a stick and buttons,
 * `PlayerSystem` read `velocity` and `facing` off each rig and fed them to the
 * animator, and nothing in between ever wrote them. Every player stood still
 * because no code connected the two ends.
 *
 * What lives here is *intent*, and only intent. It writes velocity, facing,
 * stance and the ball's carry point onto the rigs; it never touches a bone, a
 * clip or a material. The character layer turns that into motion and the
 * animator turns it into a pose, exactly as they already did.
 *
 * Two decisions worth stating:
 *
 * **This runs on the variable step, not the fixed one.** The animator consumes
 * locomotion once per frame, and if movement were integrated at 240 Hz while
 * the animator sampled it at 60, the stride rate and the distance covered would
 * disagree and the feet would skate. Matching the animator's rate is worth more
 * here than determinism is.
 *
 * **A shot is released by the animation, not by the button.** Letting go of the
 * button starts the jump-shot clip; the ball leaves on the clip's `release`
 * event, which is where the hand is actually at the top of the motion. Firing
 * the ball the instant the button comes up is the single most obvious tell that
 * a shot is a button press with an animation played next to it.
 */

import { Vector3 } from 'three';
import type { Engine, System } from '../core/Engine';
import { BALL, COURT, PLAYER, SHOT } from '../core/Constants';
import { clamp, clamp01 } from '../core/MathX';
import type { ActionKind } from '../anim/AnimatorTypes';

/** The slice of a player rig this layer writes to. */
interface RigView {
  index: number;
  team: number;
  position: Vector3;
  velocity: Vector3;
  facing: number;
  stance: string;
  jumpY: number;
  height: number;
  animator: {
    events: { onRelease?: (kind: ActionKind) => void; onFootPlant?: (foot: 'left' | 'right', force: number) => void };
    trigger(kind: ActionKind, params?: Record<string, unknown>): boolean;
    readonly busy: boolean;
  };
}

interface PlayersView {
  players: RigView[];
  ballHandler: number;
  /** Set by this system so the character layer carries the ball where play says. */
  ballPointOverride: Vector3 | null;
  handPosition(p: RigView, out?: Vector3): Vector3;
}

interface BallView {
  ballState: { position: Vector3; velocity: Vector3; owner: { kind: string } };
  launch(from: Vector3, velocity: Vector3, spin: Vector3, by: number): void;
}

interface GameView {
  shotMeter: number;
  shotCharging: boolean;
  possession: number;
  /**
   * Set by this system so the rules layer stops firing the ball the instant the
   * button comes up and waits for the animation's release frame instead.
   */
  deferReleaseToAnimation: boolean;
  releaseShot(engine: Engine, meter: number, from?: Vector3): void;
}

/** How far in front of the chest the ball rides while being dribbled. */
const DRIBBLE_REACH = 0.34;
/** Bounces per second, at rest and at full speed. */
const DRIBBLE_RATE = [1.9, 3.1] as const;

export class PlayControlSystem implements System {
  readonly name = 'playcontrol';
  // Between the ball (20) and the characters (30): the rigs must carry this
  // frame's intent before PlayerSystem reads them.
  readonly order = 25;

  /** Index of the rig the human is driving. */
  controlled = 0;

  private players: PlayersView | null = null;
  private ball: BallView | null = null;
  private game: GameView | null = null;

  private readonly desired = new Vector3();
  private readonly forward = new Vector3();
  private readonly right = new Vector3();
  private readonly carry = new Vector3();
  private readonly hand = new Vector3();

  private dribblePhase = 0;
  /** Set while a shot clip is running, so the release event knows what to do. */
  private pendingShot: { meter: number } | null = null;
  private wiredRelease = new Set<number>();

  init(engine: Engine): void {
    this.players = engine.get<PlayersView>('players') ?? null;
    this.ball = engine.get<BallView>('ball') ?? null;
    this.game = engine.get<GameView>('game') ?? null;
    // With a controller present the animation owns the release frame.
    if (this.game) this.game.deferReleaseToAnimation = true;
  }

  update(dt: number, _alpha: number, engine: Engine): void {
    const players = this.players;
    if (!players || players.players.length === 0) return;

    const me = players.players[this.controlled];
    if (!me) return;

    this.wireRelease(me, engine);
    this.drive(me, dt, engine);
    this.handleActions(me, engine);
    this.carryBall(me, dt, players, engine);
  }

  /**
   * Moves the controlled player from the stick.
   *
   * The stick is screen-relative and the camera swings around the court, so the
   * input is resolved against the camera's own basis. Pushing up always means
   * "away from me" regardless of which way the broadcast camera is facing —
   * anything else and the controls invert as the camera crosses half court.
   */
  private drive(me: RigView, dt: number, engine: Engine): void {
    const stick = engine.input.move;

    this.forward.set(0, 0, -1).applyQuaternion(engine.camera.quaternion).setY(0);
    if (this.forward.lengthSq() < 1e-6) this.forward.set(0, 0, -1);
    this.forward.normalize();
    this.right.set(1, 0, 0).applyQuaternion(engine.camera.quaternion).setY(0).normalize();

    // Stick y is positive downward on screen, which is backward in the world.
    this.desired.set(0, 0, 0).addScaledVector(this.right, stick.x).addScaledVector(this.forward, -stick.y);

    const push = Math.min(1, this.desired.length());
    if (push > 1e-3) this.desired.normalize();

    // A player carrying the ball gives up a little top speed, which is both
    // true and the thing that makes a defender able to stay in front.
    const holding = this.players?.ballHandler === me.index;
    const top = (holding ? PLAYER.topSpeed * 0.92 : PLAYER.topSpeed) * push;

    const wanted = this.desired.multiplyScalar(top);
    const rate = push > 1e-3 ? PLAYER.acceleration : PLAYER.deceleration;
    // Approach the target velocity at a bounded rate rather than snapping to
    // it, so a hard change of direction costs time and reads as weight.
    const dv = rate * dt;
    const diff = wanted.sub(me.velocity);
    const len = diff.length();
    if (len > dv) diff.multiplyScalar(dv / len);
    me.velocity.add(diff);
    if (me.velocity.lengthSq() < 1e-4) me.velocity.set(0, 0, 0);

    me.position.addScaledVector(me.velocity, dt);

    // Keep everyone on the floor. The apron is playable; the seating is not.
    const lx = COURT.halfLength + COURT.apronX * 0.5;
    const lz = COURT.halfWidth + COURT.apronZ * 0.5;
    if (Math.abs(me.position.x) > lx) {
      me.position.x = clamp(me.position.x, -lx, lx);
      me.velocity.x = 0;
    }
    if (Math.abs(me.position.z) > lz) {
      me.position.z = clamp(me.position.z, -lz, lz);
      me.velocity.z = 0;
    }

    const speed = me.velocity.length();
    me.stance = engine.input.actions.defend.held
      ? 'defend'
      : this.players?.ballHandler === me.index
        ? 'dribble'
        : speed > 0.4
          ? 'run'
          : 'idle';
  }

  /** Starts the shot clip on release; the ball leaves on the clip's own event. */
  private handleActions(me: RigView, engine: Engine): void {
    const game = this.game;
    if (!game) return;

    const shoot = engine.input.actions.shoot;
    const holding = this.players?.ballHandler === me.index;

    // This runs before the rules layer, so the charge is still live and
    // `shotMeter` still holds the value the player released on.
    if (shoot.released && game.shotCharging && holding && !this.pendingShot) {
      this.pendingShot = { meter: game.shotMeter };
      const quality = 1 - Math.abs(game.shotMeter - SHOT.windowCentre) / Math.max(SHOT.windowCentre, 1e-3);
      me.animator.trigger('jumpShot', { quality: clamp01(quality) });
    }
  }

  /**
   * Subscribes to the animator's release frame once per rig. The animator owns
   * the timing; this only says what to do when it arrives.
   */
  private wireRelease(me: RigView, engine: Engine): void {
    if (this.wiredRelease.has(me.index)) return;
    this.wiredRelease.add(me.index);

    const previous = me.animator.events.onRelease;
    me.animator.events.onRelease = (kind) => {
      previous?.(kind);
      const shot = this.pendingShot;
      if (!shot) return;
      this.pendingShot = null;
      this.players?.handPosition(me, this.hand);
      this.game?.releaseShot(engine, shot.meter, this.hand);
    };
  }

  /**
   * Drives the ball's carry point.
   *
   * Standing still it sits in a triple-threat hold. Moving, it bounces: the
   * point drops to the floor and back on a rate that scales with speed, and the
   * animator's ball-hold IK follows it, so the hand stays on the ball through
   * the whole cycle instead of the ball being welded to a static hand.
   */
  private carryBall(me: RigView, dt: number, players: PlayersView, engine: Engine): void {
    if (players.ballHandler !== me.index) {
      players.ballPointOverride = null;
      return;
    }
    // A shot in flight owns the ball; do not drag it back to the hand.
    if (this.ball && this.ball.ballState.owner.kind === 'shot') {
      players.ballPointOverride = null;
      return;
    }

    const speed = me.velocity.length();
    const moving = speed > 0.35;
    if (!moving && !this.pendingShot) {
      // Let the character layer's own triple-threat point stand.
      players.ballPointOverride = null;
      this.dribblePhase = 0;
      return;
    }
    if (this.pendingShot) {
      players.ballPointOverride = null;
      return;
    }

    const hz = DRIBBLE_RATE[0] + (DRIBBLE_RATE[1] - DRIBBLE_RATE[0]) * clamp01(speed / PLAYER.topSpeed);
    const before = this.dribblePhase;
    this.dribblePhase = (this.dribblePhase + hz * dt) % 1;

    // Height traces a bounce: fast down, fast up, with the ball spending most
    // of the cycle high where the hand meets it.
    const t = this.dribblePhase;
    const bounce = Math.abs(Math.cos(t * Math.PI));
    const low = BALL.radius;
    const high = me.height * 0.52;
    const y = low + (high - low) * bounce;

    // Out in front, on the outside hand, leaning further ahead the faster the
    // player is going — a runner pushes the ball out to meet their stride.
    const lead = DRIBBLE_REACH + clamp01(speed / PLAYER.topSpeed) * 0.42;
    this.carry.set(Math.sin(me.facing) * lead, y, Math.cos(me.facing) * lead).add(me.position);
    players.ballPointOverride = this.carry;

    // The contact itself, once per cycle at the bottom.
    if (before > t) {
      engine.bus.emit('dribble', { speed: 4 + speed, position: this.carry.clone().setY(BALL.radius) });
    }
  }
}
