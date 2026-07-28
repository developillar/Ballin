/**
 * Team AI — the nine players the human is not driving.
 *
 * `PlayControl` connected the stick to one rig. This connects the *other nine*,
 * which up to now were placed once at spawn and never moved again: a half-court
 * set frozen at the instant of the screenshot that produced it.
 *
 * It follows `PlayControl`'s contract exactly, because that contract is the
 * thing that keeps the character and animation layers rewritable:
 *
 * **Intent only.** This writes `velocity`, `facing` and `stance` onto rigs, and
 * the ball's carry point when an AI is handling it. It never touches a bone, a
 * clip, a material or a pose. `PlayerSystem` turns intent into motion and
 * `Animator` turns motion into a pose, exactly as they already did.
 *
 * **The variable step, not the fixed one.** Same reason as `PlayControl`: the
 * animator consumes locomotion once per frame, and integrating movement at
 * 240 Hz while sampling it at 60 makes stride rate and distance covered
 * disagree, which is skating.
 *
 * **Ownership.** `PlayControl` owns the rig at `PlayControl.controlled`. This
 * owns every other rig and writes to no other. The controlled index is *read*
 * from `PlayControl` rather than assumed, so the two can never disagree about
 * who has the stick.
 *
 * The behaviour is deliberately a short list of always-on rules rather than a
 * state machine. Every player, every frame, is doing exactly one of five
 * things, and which one is a pure function of the possession:
 *
 *   1. **Chase** — the ball is nobody's, and he is his team's nearest man to
 *      where it is going to be.
 *   2. **Handle** — he has the ball: attack the rim when his defender is beaten,
 *      hold the perimeter when he is not, and move it on before the possession
 *      goes stale.
 *   3. **Space** — his team has the ball and he does not: take a standing spot,
 *      stay out of the lane, and slide off the line when a defender stands in
 *      the passing lane to it.
 *   4. **Mark** — the other team has the ball: get and stay goal-side of the man
 *      he is assigned, tight on the handler and sagging with a help shade off it.
 *   5. **Settle** — nothing above applies; hold position and watch the ball.
 *
 * Owned by the team-AI agent.
 */

import { Vector3 } from 'three';
import type { Engine, System } from '../core/Engine';
import { BALL, PHYSICS, PLAYER } from '../core/Constants';
import { clamp, clamp01, dampAngle } from '../core/MathX';
import type { TierName } from '../core/Quality';
import type { ActionKind } from '../anim/AnimatorTypes';
import {
  BOUNDS,
  RIM,
  RIM_GROUND,
  SPOTS,
  flatDistance,
  goalSidePoint,
  landingPoint,
  laneEscape,
  segmentDistanceXZ,
} from './aiCourt';

/** The slice of a player rig this layer writes to. Mirrors `PlayControl`'s. */
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
    events: { onRelease?: (kind: ActionKind) => void };
    trigger(kind: ActionKind, params?: Record<string, unknown>): boolean;
    readonly busy: boolean;
  };
}

interface PlayersView {
  players: RigView[];
  /** Index of the player in possession, or -1 while the ball is loose. */
  ballHandler: number;
  ballPointOverride: Vector3 | null;
  handPosition(p: RigView, out?: Vector3): Vector3;
}

interface BallView {
  ballState: {
    position: Vector3;
    velocity: Vector3;
    spin: Vector3;
    owner: { kind: string };
    resting: boolean;
  };
}

interface ControlView {
  controlled: number;
}

// ---------------------------------------------------------------------------
// Tuning. Every distance is in metres, every duration in seconds.
// ---------------------------------------------------------------------------

/** How often assignments are re-solved. The only per-tier cost this has. */
const REPLAN_INTERVAL: Record<TierName, number> = {
  low: 0.5,
  medium: 0.34,
  high: 0.2,
  ultra: 0.16,
};

/** On-ball defender: close enough to contest, far enough not to be blown by. */
const GAP_ON_BALL = 1.4;
/** Off-ball defender: a step-and-a-half of sag. */
const GAP_OFF_BALL = 2.15;
/** Perpendicular help shade toward the ball, on and off the ball. */
const SHADE_ON_BALL = 0.18;
const SHADE_OFF_BALL = 0.85;

/** Bodies stop approaching each other at about here. */
const SEPARATION_RADIUS = 1.25;

/** How close a player must be to a live loose ball to take it. */
const CATCH_RADIUS = 0.62;
/** The intended receiver of a pass gets a bigger hoop to catch through. */
const CATCH_RADIUS_TARGET = 1.15;
/** A receiver has to be about this still to be thrown to. */
const SETTLED_SPEED = 1.6;
/**
 * Longest throw this offence will attempt.
 *
 * A half-court set does not throw cross-court. Measured: a 14.7 m pass spends
 * a full second crossing ground four defenders are standing on and was taken
 * 5 m short of the target. The widest legitimate pass here — corner to opposite
 * wing — is under 11 m.
 */
const MAX_PASS = 11;
/** Above this the ball is over everybody's head and cannot be claimed. */
const CATCH_HEIGHT = 2.15;

/** Speed a chest pass leaves the hand at. */
const PASS_SPEED = 12.5;
/** A handler moves the ball on somewhere between these, by index. */
const HOLD_MIN = 1.7;
const HOLD_MAX = 2.9;
/** A handler this closely guarded gives it up early. */
const PRESSURE_RADIUS = 1.15;
/** The passer cannot re-claim his own pass for this long. */
const PASS_LOCKOUT = 0.55;
/**
 * How much of a pass belongs to the man it was thrown to, as a fraction of the
 * throw's length.
 *
 * A loose ball is claimed by whoever is within arm's reach of it, and the
 * nearest body to a ball that has just left a hand is always the defender
 * guarding the passer. Measured over eight AI possessions with no protection at
 * all, *every single pass* was taken by the on-ball defender 0.30 s after
 * release, 4–9 m short of where it was aimed — not an interception, the ball
 * magnetised to whoever happened to be standing next to the throw.
 *
 * Stated as a fraction rather than as a duration because a duration protects a
 * short pass completely and a long one barely at all. Past the halfway mark the
 * ball is contestable, which is where a defender jumping a route actually
 * makes the play.
 */
const PASS_PROTECT_FRACTION = 0.55;
/** Hard ceiling on that: a pass nobody catches becomes an ordinary loose ball. */
const PASS_WINDOW = 1.6;

/** Beaten defender → drive; guarded → hold the perimeter, this far from the rim. */
const DRIVE_STANDOFF = 1.7;
const HOLD_STANDOFF = 5.8;
const BEATEN_BY = 2.6;

/** A defender standing this near the ball→spot line is blocking it. */
const LANE_BLOCK_RADIUS = 0.95;

/** Arrive: full speed beyond this, tapering inside it, stopped inside the dead zone. */
const SLOW_RADIUS = 1.5;
const ARRIVE_DEAD_ZONE = 0.14;

const _to = new Vector3();
const _push = new Vector3();
const _target = new Vector3();
const _wanted = new Vector3();
const _diff = new Vector3();
const _escape = new Vector3();
const _hand = new Vector3();
const _lead = new Vector3();
const _chase = new Vector3();
const _carry = new Vector3();

export class TeamAISystem implements System {
  readonly name = 'teamai';
  /**
   * Between `PlayControl` (25) and `PlayerSystem` (30). Every rig must be
   * carrying this frame's intent before the character layer reads it, and the
   * human's rig must already be resolved before the nine react to it.
   */
  readonly order = 26;

  private players: PlayersView | null = null;
  private ball: BallView | null = null;
  private control: ControlView | null = null;

  /** Defensive assignment: `mark[defenderIndex] = offensiveIndex`. */
  private mark: number[] = [];
  /** Offensive assignment: `spot[playerIndex] = index into SPOTS`. */
  private spot: number[] = [];
  /** Seconds the current handler has held the ball. */
  private holdFor = 0;
  /** Which team was last on offence — survives the loose-ball window. */
  private lastOffence = 0;
  /** Why the ball is loose, so a catch is not reported as a rebound. */
  private looseFrom: 'none' | 'shot' | 'pass' = 'none';
  /** Where the last pass was aimed — the receiver runs here, not at the ball. */
  private readonly passTarget = new Vector3();
  /** Where it was thrown from, so "how far along is it" is answerable. */
  private readonly passOrigin = new Vector3();
  /** Bounce phase for an AI handler's dribble. */
  private dribblePhase = 0;

  private replanIn = 0;
  private replanInterval = 0.34;

  /** A pass that has been started and is waiting for the clip's release frame. */
  private pendingPass: { from: number; to: number; expires: number } | null = null;
  private lockout = { player: -1, until: 0 };
  /** While `elapsed` is under this, a live pass is the receiver's alone. */
  private protectUntil = 0;
  private incoming = -1;
  private elapsed = 0;
  private wired = new Set<number>();
  /** Set by the animator's release frame; consumed at the top of the next update. */
  private releaseArmed = false;
  /** Ball position at the end of the previous update, for the swept catch test. */
  private readonly prevBall = new Vector3();
  private prevBallValid = false;

  /**
   * True while a pass is still on its way to the man it was thrown to.
   *
   * It has to expire. `looseFrom` alone does not: a pass that is dropped, or
   * thrown to a receiver who cannot get to it, would otherwise stay "a pass"
   * forever — the receiver stands on the aim point, the other team is told not
   * to chase, and the ball lies on the floor for the rest of the game. Measured
   * once, exactly that: 35 s of a live ball nobody was allowed to want.
   */
  private get passLive(): boolean {
    return this.looseFrom === 'pass' && this.elapsed < this.protectUntil;
  }

  /** True while the ball is still in the passer's half of the throw. */
  private get passProtected(): boolean {
    if (!this.passLive) return false;
    const travelled = flatDistance(this.ball!.ballState.position, this.passOrigin);
    return travelled < flatDistance(this.passTarget, this.passOrigin) * PASS_PROTECT_FRACTION;
  }

  init(engine: Engine): void {
    this.players = engine.get<PlayersView>('players') ?? null;
    this.ball = engine.get<BallView>('ball') ?? null;
    this.control = engine.get<ControlView>('playcontrol') ?? null;
    this.replanInterval = REPLAN_INTERVAL[engine.quality.tier] ?? 0.34;

    const rigs = this.players?.players ?? [];
    this.mark = new Array(rigs.length).fill(-1);
    this.spot = new Array(rigs.length).fill(-1);
    if (this.players) this.lastOffence = rigs[this.players.ballHandler]?.team ?? 0;
  }

  update(dt: number, _alpha: number, engine: Engine): void {
    const players = this.players;
    if (!players) return;
    const rigs = players.players;
    if (rigs.length === 0) return;
    const ball = this.ball;
    if (!ball) return;

    this.elapsed += dt;
    // Rosters are built in `init`, but `create()` is a public hook — grow rather
    // than index off the end if somebody adds a player later.
    if (this.mark.length < rigs.length) {
      while (this.mark.length < rigs.length) this.mark.push(-1);
      while (this.spot.length < rigs.length) this.spot.push(-1);
    }

    const human = this.control?.controlled ?? -1;
    for (const p of rigs) if (p.index !== human) this.wirePass(p, engine);

    // Passes are thrown *here*, at the top of this system's update, and never
    // from inside the animator callback that arms them.
    //
    // `PlayerSystem.update` reads `ballHandler` once at the top of its own frame
    // and then re-welds the ball into that player's hands at the bottom of it —
    // and the animator's release event fires in between, from inside its loop.
    // A pass thrown from the callback therefore had its velocity zeroed and the
    // ball put straight back in the passer's hand, by a `handler` reference
    // captured before the pass existed. Measured: eight passes in a row that
    // never flew, every one of them "intercepted" 6-9 m from the aim point by
    // whoever wandered into the stalled ball first. Deferring by one frame costs
    // ~16 ms of release latency and makes the throw real.
    this.servicePass(players, engine);

    this.resolvePossession(dt, rigs, players, engine);

    const handler = players.ballHandler;
    const held = handler >= 0 && !!rigs[handler];
    if (held) this.lastOffence = rigs[handler].team;
    const offence = this.lastOffence;

    this.replanIn -= dt;
    if (this.replanIn <= 0) {
      this.replanIn = this.replanInterval;
      this.plan(rigs, offence, held ? handler : -1, human);
    }

    // Where a loose ball is going, and who on each team is going after it.
    const chaseA = this.chaserFor(rigs, 0, human, held);
    const chaseB = this.chaserFor(rigs, 1, human, held);

    for (const p of rigs) {
      if (p.index === human) continue;
      if (!held && (p.index === chaseA || p.index === chaseB)) this.chase(p, dt, rigs);
      else if (held && p.index === handler) this.handle(p, dt, rigs, players, engine);
      else if (p.team === offence) this.space(p, dt, rigs, held ? rigs[handler] : null);
      else this.defend(p, dt, rigs, held ? handler : -1);
      this.clampToFloor(p);
    }

    // The handler's carry point, when the handler is not the human.
    if (held && handler !== human) this.carry(rigs[handler], dt, players, engine);

    // Close the frame by remembering where the ball was, so next frame's catch
    // test can ask what it swept through rather than where it landed.
    this.prevBall.copy(ball.ballState.position);
    this.prevBallValid = true;
  }

  // -------------------------------------------------------------------------
  // Possession
  // -------------------------------------------------------------------------

  /**
   * The one piece of shared state this system owns: who has the ball.
   *
   * `PlayerSystem` welds the ball into `ballHandler`'s hands every frame and
   * nothing ever cleared that index, so a rebound was not a rebound — the ball
   * teleported back to whoever shot it the moment the rules layer let go of it.
   * A shot therefore *drops* possession here, and it is picked up again by
   * whoever actually gets to the ball.
   */
  private resolvePossession(dt: number, rigs: RigView[], players: PlayersView, engine: Engine): void {
    const s = this.ball!.ballState;

    if (s.owner.kind === 'shot') {
      // In the air and nobody's. Clearing the index is what makes the rebound
      // live; the claim below cannot fire until the rules layer frees the ball.
      if (players.ballHandler >= 0) {
        players.ballHandler = -1;
        this.holdFor = 0;
        this.incoming = -1;
        this.looseFrom = 'shot';
      }
      return;
    }

    if (players.ballHandler >= 0) {
      this.holdFor += dt;
      return;
    }

    // Loose. Whoever the ball passed within arm's reach of takes it — measured
    // against the path it swept this frame, not the point it happens to be at.
    const from = this.prevBallValid ? this.prevBall : s.position;
    if (Math.min(from.y, s.position.y) > CATCH_HEIGHT) return;
    const protectedPass = this.passProtected;
    let best = -1;
    let bestD = Infinity;
    for (const p of rigs) {
      if (p.index === this.lockout.player && this.elapsed < this.lockout.until) continue;
      if (protectedPass && p.index !== this.incoming) continue;
      const d = segmentDistanceXZ(p.position.x, p.position.z, from, s.position);
      const reach = p.index === this.incoming ? CATCH_RADIUS_TARGET : CATCH_RADIUS;
      if (d < reach && d < bestD) {
        best = p.index;
        bestD = d;
      }
    }
    if (best < 0) return;

    const team = rigs[best].team;
    if (this.looseFrom === 'pass') this.closePass(best, s.position);
    if (this.looseFrom === 'shot') {
      engine.bus.emit('rebound', { team, player: best, offensive: team === this.lastOffence });
    }
    if (team !== this.lastOffence) engine.bus.emit('possessionChanged', { team });

    players.ballHandler = best;
    this.holdFor = 0;
    this.incoming = -1;
    this.looseFrom = 'none';
    this.lockout.player = -1;
  }

  /** The team's nearest available body to where a loose ball is going. */
  private chaserFor(rigs: RigView[], team: number, human: number, held: boolean): number {
    if (held) return -1;
    // A pass is a loose ball with an address on it. The man it was thrown to
    // goes and meets it even when a team-mate happens to be nearer; the other
    // team does *not* send anybody, because five defenders converging on a ball
    // that is already on its way to a specific player is not defence, it is a
    // swarm — they keep marking, and the closeout on the new handler happens on
    // the next re-plan, which is where it belongs.
    if (this.passLive && this.incoming >= 0) {
      if (rigs[this.incoming]?.team !== team) return -1;
      return this.incoming === human ? -1 : this.incoming;
    }
    const s = this.ball!.ballState;
    landingPoint(s.position, s.velocity, CATCH_HEIGHT, PHYSICS.gravity, _chase);
    let best = -1;
    let bestD = Infinity;
    for (const p of rigs) {
      if (p.team !== team || p.index === human) continue;
      const d = flatDistance(p.position, _chase);
      if (d < bestD) {
        best = p.index;
        bestD = d;
      }
    }
    return best;
  }

  // -------------------------------------------------------------------------
  // Assignment
  // -------------------------------------------------------------------------

  /**
   * Re-solves both assignments — who guards whom, and who stands where.
   *
   * Greedy over every pair sorted by cost, which for five-on-five is 25
   * comparisons and gives the same answer every time it is handed the same
   * court. Two bonuses shape it: the ball-handler is cheap to guard, so
   * *somebody* always takes him; and the pairing a player already has is cheap
   * to keep, so a defender does not swap men because an opponent drifted 10 cm.
   *
   * The human is in neither pool. He is a wildcard the other nine react to —
   * they separate from him, they read him as a passing-lane blocker, he can be
   * passed to and he can pick up a loose ball — but nothing is ever *assigned*
   * to him, because an assignment nobody executes is worse than none: it would
   * hold a defensive matchup open that no AI is covering, and hold a standing
   * spot that no AI is allowed to take.
   */
  private plan(rigs: RigView[], offenceTeam: number, handler: number, human: number): void {
    const offence: RigView[] = [];
    const defence: RigView[] = [];
    for (const p of rigs) {
      if (p.index === human) continue;
      (p.team === offenceTeam ? offence : defence).push(p);
    }
    if (offence.length === 0 || defence.length === 0) return;

    // A player who has switched sides is carrying the assignment he had on the
    // other one. Clear both directions first, so `mark` and `spot` never hold a
    // stale answer that a reader — or a diagnostic — could believe.
    for (const o of offence) this.mark[o.index] = -1;
    for (const d of defence) this.spot[d.index] = -1;
    if (human >= 0 && human < this.mark.length) {
      this.mark[human] = -1;
      this.spot[human] = -1;
    }

    // --- who guards whom ---------------------------------------------------
    const pairs: { d: number; o: number; cost: number }[] = [];
    for (const d of defence) {
      for (const o of offence) {
        let cost = flatDistance(d.position, o.position);
        if (o.index === handler) cost -= 2.5;
        if (this.mark[d.index] === o.index) cost -= 1.2;
        pairs.push({ d: d.index, o: o.index, cost });
      }
    }
    pairs.sort((a, b) => a.cost - b.cost || a.d - b.d || a.o - b.o);
    const tookD = new Set<number>();
    const tookO = new Set<number>();
    for (const p of pairs) {
      if (tookD.has(p.d) || tookO.has(p.o)) continue;
      tookD.add(p.d);
      tookO.add(p.o);
      this.mark[p.d] = p.o;
    }
    // Uneven squads: anyone left over doubles the ball.
    const fallback = handler >= 0 ? handler : offence[0].index;
    for (const d of defence) if (!tookD.has(d.index)) this.mark[d.index] = fallback;

    // --- who stands where --------------------------------------------------
    const ballAt = handler >= 0 ? rigs[handler]?.position ?? null : null;
    const spotPairs: { p: number; s: number; cost: number }[] = [];
    for (const o of offence) {
      if (o.index === handler) continue;
      for (let s = 0; s < SPOTS.length; s++) {
        _target.set(SPOTS[s].x, 0, SPOTS[s].z);
        let cost = flatDistance(o.position, _target);
        // Leave the handler room to work: the spot he is standing on is not a
        // spot anybody else should be walking to.
        if (ballAt && flatDistance(ballAt, _target) < 3.4) cost += 6;
        if (this.spot[o.index] === s) cost -= 1.2;
        spotPairs.push({ p: o.index, s, cost });
      }
    }
    spotPairs.sort((a, b) => a.cost - b.cost || a.p - b.p || a.s - b.s);
    const tookP = new Set<number>();
    const tookS = new Set<number>();
    for (const q of spotPairs) {
      if (tookP.has(q.p) || tookS.has(q.s)) continue;
      tookP.add(q.p);
      tookS.add(q.s);
      this.spot[q.p] = q.s;
    }
  }

  // -------------------------------------------------------------------------
  // Behaviours
  // -------------------------------------------------------------------------

  /** Go and get it. */
  private chase(p: RigView, dt: number, rigs: RigView[]): void {
    this.chasePoint(p, _chase);
    this.steer(p, _chase, PLAYER.topSpeed, dt, rigs);
    this.face(p, _chase, dt);
    p.stance = p.velocity.lengthSq() > 0.16 ? 'run' : 'idle';
  }

  /**
   * Where a chaser should run, which is never simply "at the ball".
   *
   * A pass is chased to *the point it was thrown to* — the receiver goes to the
   * catch and waits for it. Everything else is led: a ball above head height is
   * chased to where it comes down, and a ball below it is chased to where it
   * will be by the time this player can get there. Pure pursuit of a moving ball
   * is what puts a chaser permanently a stride behind it, and it is what turned
   * half of all completed passes into gifts for the nearest defender.
   */
  private chasePoint(p: RigView, out: Vector3): Vector3 {
    if (this.passLive && p.index === this.incoming) return out.copy(this.passTarget);
    const s = this.ball!.ballState;
    if (s.position.y > CATCH_HEIGHT) {
      return landingPoint(s.position, s.velocity, CATCH_HEIGHT, PHYSICS.gravity, out);
    }
    const tau = clamp(flatDistance(p.position, s.position) / PLAYER.topSpeed, 0, 1.2);
    out.set(s.position.x + s.velocity.x * tau, 0, s.position.z + s.velocity.z * tau);
    out.x = clamp(out.x, -BOUNDS.x, BOUNDS.x);
    out.z = clamp(out.z, -BOUNDS.z, BOUNDS.z);
    return out;
  }

  /**
   * Handle the ball: attack the rim when the defender has been beaten, hold the
   * perimeter when he has not, and give it up before the possession goes stale.
   */
  private handle(
    p: RigView,
    dt: number,
    rigs: RigView[],
    players: PlayersView,
    engine: Engine,
  ): void {
    const defender = this.nearestOpponent(p, rigs);
    const pressure = defender ? flatDistance(p.position, defender.position) : 99;
    const standoff = pressure > BEATEN_BY ? DRIVE_STANDOFF : HOLD_STANDOFF;

    // A point `standoff` metres out from the rim along the line he is already
    // on: he attacks straight at the basket rather than orbiting it.
    let ux = p.position.x - RIM_GROUND.x;
    let uz = p.position.z - RIM_GROUND.z;
    const len = Math.hypot(ux, uz) || 1e-4;
    ux /= len;
    uz /= len;
    _target.set(RIM_GROUND.x + ux * standoff, 0, RIM_GROUND.z + uz * standoff);

    this.steer(p, _target, PLAYER.topSpeed * 0.92, dt, rigs);
    this.face(p, RIM_GROUND, dt);
    p.stance = 'dribble';

    // Move it on. Deterministic per player so two handlers never share a clock.
    const patience = HOLD_MIN + ((p.index * 7) % 13) / 12 * (HOLD_MAX - HOLD_MIN);
    const forced = pressure < PRESSURE_RADIUS && this.holdFor > 0.6;
    if (!this.pendingPass && (this.holdFor > patience || forced)) {
      const receiver = this.bestPass(p, rigs, this.holdFor > patience * 2.2);
      if (receiver >= 0) this.startPass(p, receiver, players, engine);
    }
  }

  /** Off the ball: take a spot, keep out of the lane, keep the lane to the ball open. */
  private space(p: RigView, dt: number, rigs: RigView[], handler: RigView | null): void {
    const s = this.spot[p.index];
    const spot = SPOTS[s >= 0 ? s : p.index % SPOTS.length];
    _target.set(spot.x, 0, spot.z);

    // If a defender is parked on the line from the ball to this spot, slide
    // along the arc rather than standing behind him waiting for a pass that
    // cannot arrive.
    if (handler) {
      const blocker = this.lineBlocker(handler.position, _target, rigs, p.team);
      if (blocker) {
        let dx = _target.x - handler.position.x;
        let dz = _target.z - handler.position.z;
        const d = Math.hypot(dx, dz) || 1e-4;
        dx /= d;
        dz /= d;
        // Perpendicular, pointing away from the man in the way.
        const px = dz;
        const pz = -dx;
        const side = Math.sign(
          (blocker.position.x - handler.position.x) * px +
            (blocker.position.z - handler.position.z) * pz,
        ) || 1;
        _target.x -= px * side * 1.35;
        _target.z -= pz * side * 1.35;
      }
    }

    // And never loiter in the paint.
    laneEscape(_target.x, _target.z, _escape);
    _target.add(_escape);

    this.steer(p, _target, PLAYER.topSpeed, dt, rigs);
    this.face(p, handler ? handler.position : this.ball!.ballState.position, dt);
    p.stance = p.velocity.lengthSq() > 0.16 ? 'run' : 'idle';
  }

  /** Mark a man: goal-side, always, with the tightness set by the ball. */
  private defend(p: RigView, dt: number, rigs: RigView[], handler: number): void {
    const manIndex = this.mark[p.index];
    const man = manIndex >= 0 ? rigs[manIndex] : null;
    if (!man) {
      this.settle(p, dt, rigs);
      return;
    }

    const onBall = man.index === handler;
    const ballAt = this.ball!.ballState.position;
    goalSidePoint(
      man.position,
      ballAt,
      onBall ? GAP_ON_BALL : GAP_OFF_BALL,
      onBall ? SHADE_ON_BALL : SHADE_OFF_BALL,
      _target,
    );

    this.steer(p, _target, PLAYER.topSpeed, dt, rigs);
    this.face(p, onBall ? ballAt : man.position, dt);

    const close = flatDistance(p.position, man.position) < 3.6;
    p.stance = close ? 'defend' : p.velocity.lengthSq() > 0.16 ? 'run' : 'idle';
  }

  /** Nothing to do: hold the ground and watch the ball. */
  private settle(p: RigView, dt: number, rigs: RigView[]): void {
    this.steer(p, p.position, PLAYER.topSpeed, dt, rigs);
    this.face(p, this.ball!.ballState.position, dt);
    p.stance = 'idle';
  }

  // -------------------------------------------------------------------------
  // Passing
  // -------------------------------------------------------------------------

  /**
   * The most open team-mate, or -1.
   *
   * Openness is the distance from the receiver to his nearest defender, and a
   * body standing in the lane is a **veto**, not a penalty. That distinction is
   * the difference between an offence and a turnover machine: the ball goes
   * loose the instant it leaves the hand and the first man inside arm's reach
   * of it takes it, so a pass threaded past the on-ball defender is not a
   * risky pass, it is a completed pass to the defender. Measured over a
   * possession, penalising blocked lanes rather than refusing them gave the
   * ball away on half of all attempts.
   *
   * `desperate` lifts the veto once a handler has been holding far too long,
   * so a covered offence throws a contested pass instead of dribbling forever.
   */
  private bestPass(from: RigView, rigs: RigView[], desperate: boolean): number {
    let best = -1;
    let bestScore = 1.6;
    for (const p of rigs) {
      if (p.team !== from.team || p.index === from.index) continue;
      const d = flatDistance(from.position, p.position);
      if (d < 2.2 || d > MAX_PASS) continue;
      // You pass to a man who has arrived, not to a man on his way somewhere.
      // Measured over five possessions: every throw to a settled receiver was
      // caught (3-5 m, receiver 0.7-0.8 m off the ball when he took it), and
      // every throw to a moving one was lost (9 m, receiver still 3.6-5.1 m
      // away when a defender collected it). The ball covers 9 m in 0.7 s; a
      // runner heading somewhere else does not, and leading him just moves the
      // catch point to somewhere neither of them is going.
      if (!desperate && p.velocity.lengthSq() > SETTLED_SPEED * SETTLED_SPEED) continue;
      const blocked = !!this.lineBlocker(from.position, p.position, rigs, from.team);
      if (blocked && !desperate) continue;
      let score = 99;
      for (const q of rigs) {
        if (q.team === from.team) continue;
        score = Math.min(score, flatDistance(p.position, q.position));
      }
      if (blocked) score -= 2.2;
      if (score > bestScore) {
        bestScore = score;
        best = p.index;
      }
    }
    return best;
  }

  /**
   * Starts a pass on the animation, not on the decision.
   *
   * Same principle `PlayControl` applies to a shot: the ball leaves on the
   * clip's `release` frame, a quarter of a second after the wind-up starts, so
   * there is a load before the throw instead of a ball that departs from a
   * static pose. If the clip cannot start — an uninterruptible action is
   * already running — the pass is thrown immediately rather than dropped.
   */
  private startPass(from: RigView, to: number, players: PlayersView, engine: Engine): void {
    this.incoming = to;
    this.pendingPass = { from: from.index, to, expires: this.elapsed + 0.8 };
    const target = players.players[to];
    const started = from.animator.trigger('pass', target ? { target: target.position } : undefined);
    if (!started) this.throwPass(players, engine);
  }

  /**
   * Throws the armed pass, or forces one whose clip never reached its release
   * frame. Runs once per frame, before anything else reads `ballHandler`.
   */
  private servicePass(players: PlayersView, engine: Engine): void {
    const pass = this.pendingPass;
    if (!pass) {
      this.releaseArmed = false;
      return;
    }
    if (!this.releaseArmed && this.elapsed <= pass.expires) return;
    this.releaseArmed = false;
    this.throwPass(players, engine);
  }

  /**
   * The ball leaves the hand.
   *
   * The ball is put into free flight and `ballHandler` is cleared, which means
   * a pass resolves through exactly the same code path a rebound does: it is
   * loose, somebody chases it, somebody claims it. One mechanism, so a dropped
   * pass is a live ball rather than a special case.
   */
  private throwPass(players: PlayersView, engine: Engine): void {
    const pass = this.pendingPass;
    this.pendingPass = null;
    if (!pass) return;
    const rigs = players.players;
    const from = rigs[pass.from];
    const to = rigs[pass.to];
    if (!from || !to || players.ballHandler !== pass.from) return;

    players.handPosition(from, _hand);
    if (!Number.isFinite(_hand.x)) _hand.set(from.position.x, from.height * 0.7, from.position.z);

    const g = Math.abs(PHYSICS.gravity);
    // Lead the receiver by the flight time — a pass to where a runner is now
    // arrives behind him — but cap the lead hard. A receiver at top speed for a
    // full second of flight is 7 m of extrapolation, and he is running to a
    // standing spot, not away forever: measured, an uncapped lead threw the ball
    // to a point the receiver never reached and a defender was already standing
    // on, three times out of five.
    let t = clamp(flatDistance(_hand, to.position) / PASS_SPEED, 0.18, 1.0);
    _lead.copy(to.velocity).multiplyScalar(t);
    if (_lead.lengthSq() > 4) _lead.setLength(2);
    _target.set(to.position.x + _lead.x, to.height * 0.62, to.position.z + _lead.z);
    t = clamp(flatDistance(_hand, _target) / PASS_SPEED, 0.18, 1.0);

    const s = this.ball!.ballState;
    s.position.copy(_hand);
    // 6% long: drag is not in this solve and a pass that lands short is a
    // turnover the receiver did not earn.
    s.velocity.set(
      ((_target.x - _hand.x) / t) * 1.06,
      (_target.y - _hand.y) / t + 0.5 * g * t,
      ((_target.z - _hand.z) / t) * 1.06,
    );
    s.spin.set(0, 0, 0);
    s.owner = { kind: 'free' };
    s.resting = false;

    players.ballHandler = -1;
    players.ballPointOverride = null;
    this.passTarget.set(_target.x, 0, _target.z);
    this.passOrigin.set(_hand.x, 0, _hand.z);
    this.logPass(pass.from, pass.to);
    this.holdFor = 0;
    this.incoming = pass.to;
    this.looseFrom = 'pass';
    this.lockout.player = pass.from;
    this.lockout.until = this.elapsed + PASS_LOCKOUT;
    this.protectUntil = this.elapsed + PASS_WINDOW;
    void engine;
  }

  /** Subscribes to each AI rig's release frame once. */
  private wirePass(p: RigView, engine: Engine): void {
    if (this.wired.has(p.index)) return;
    this.wired.add(p.index);
    const previous = p.animator.events.onRelease;
    p.animator.events.onRelease = (kind) => {
      previous?.(kind);
      if (kind !== 'pass' && kind !== 'bouncePass' && kind !== 'overheadPass') return;
      if (this.pendingPass?.from !== p.index) return;
      // Arm only. The throw happens at the top of the next update — see the
      // note there for why doing it here destroys the pass.
      this.releaseArmed = true;
    };
    void engine;
  }

  // -------------------------------------------------------------------------
  // The ball in an AI's hands
  // -------------------------------------------------------------------------

  /**
   * The carry point for an AI handler — the same bounce `PlayControl` gives the
   * human, for the same reason: a ball welded to a static hand while its owner
   * runs is the clearest possible tell that nobody is really dribbling.
   *
   * Safe to write here: `PlayControl` runs first (order 25) and clears the
   * override whenever the handler is not the rig it owns, so this is writing
   * into a field nobody else has an opinion about this frame.
   */
  private carry(p: RigView, dt: number, players: PlayersView, engine: Engine): void {
    if (this.pendingPass) {
      players.ballPointOverride = null;
      return;
    }
    const speed = p.velocity.length();
    if (speed <= 0.35) {
      players.ballPointOverride = null;
      this.dribblePhase = 0;
      return;
    }

    const hz = 1.9 + 1.2 * clamp01(speed / PLAYER.topSpeed);
    const before = this.dribblePhase;
    this.dribblePhase = (this.dribblePhase + hz * dt) % 1;

    const bounce = Math.abs(Math.cos(this.dribblePhase * Math.PI));
    const low = BALL.radius;
    const high = p.height * 0.52;
    const lead = 0.34 + clamp01(speed / PLAYER.topSpeed) * 0.42;
    _carry.set(
      Math.sin(p.facing) * lead,
      low + (high - low) * bounce,
      Math.cos(p.facing) * lead,
    ).add(p.position);
    players.ballPointOverride = _carry;

    if (before > this.dribblePhase) {
      engine.bus.emit('dribble', {
        speed: 4 + speed,
        position: _carry.clone().setY(BALL.radius),
      });
    }
  }

  // -------------------------------------------------------------------------
  // Steering
  // -------------------------------------------------------------------------

  /**
   * Move toward a point.
   *
   * Two properties matter more than the shape of the curve. **It cannot
   * overshoot**: the requested speed is capped at the speed that exactly
   * reaches the target this frame, so a 100 ms frame — which is what a
   * software rasteriser delivers here — cannot fling a defender past his spot
   * and start an oscillation. And **it obeys the same physics the human does**:
   * the approach to the target velocity is bounded by `PLAYER.acceleration` and
   * `PLAYER.deceleration`, so nobody accelerates like a mouse cursor.
   */
  private steer(p: RigView, to: Vector3, top: number, dt: number, rigs: RigView[]): void {
    _to.set(to.x - p.position.x, 0, to.z - p.position.z);
    const dist = _to.length();

    let speed = dist > 1e-4 ? top * clamp01(dist / SLOW_RADIUS) : 0;
    speed = Math.min(speed, dist / Math.max(dt, 1 / 120));
    if (dist < ARRIVE_DEAD_ZONE) speed = 0;
    if (dist > 1e-4) _to.divideScalar(dist);
    _wanted.copy(_to).multiplyScalar(speed);

    // Bodies do not occupy the same square metre.
    _push.set(0, 0, 0);
    for (const q of rigs) {
      if (q.index === p.index) continue;
      const dx = p.position.x - q.position.x;
      const dz = p.position.z - q.position.z;
      const r = Math.hypot(dx, dz);
      if (r > SEPARATION_RADIUS || r < 1e-4) continue;
      const w = ((SEPARATION_RADIUS - r) / SEPARATION_RADIUS) * top * 0.9;
      _push.x += (dx / r) * w;
      _push.z += (dz / r) * w;
    }
    _wanted.add(_push);
    if (_wanted.lengthSq() > top * top) _wanted.setLength(top);

    const rate = _wanted.lengthSq() > 1e-6 ? PLAYER.acceleration : PLAYER.deceleration;
    _diff.copy(_wanted).sub(p.velocity);
    const need = _diff.length();
    const budget = rate * dt;
    if (need > budget) _diff.multiplyScalar(budget / need);
    p.velocity.add(_diff);
    if (p.velocity.lengthSq() < 1e-4) p.velocity.set(0, 0, 0);

    p.position.addScaledVector(p.velocity, dt);
  }

  /**
   * Point the body at something.
   *
   * `PlayerSystem` turns a moving player toward his velocity, and it does that
   * *after* this runs, so what is written here only survives while a player is
   * near-stationary — which is exactly when it should. A defender sliding into
   * position faces where he is going; the moment he arrives he turns and looks
   * at his man. The damp is what keeps that handover from being a snap.
   */
  private face(p: RigView, at: Vector3, dt: number): void {
    const dx = at.x - p.position.x;
    const dz = at.z - p.position.z;
    if (dx * dx + dz * dz < 1e-4) return;
    p.facing = dampAngle(p.facing, Math.atan2(dx, dz), PLAYER.turnRate * 0.55, dt);
  }

  /** The apron is playable; the seating is not. */
  private clampToFloor(p: RigView): void {
    if (Math.abs(p.position.x) > BOUNDS.x) {
      p.position.x = clamp(p.position.x, -BOUNDS.x, BOUNDS.x);
      p.velocity.x = 0;
    }
    if (Math.abs(p.position.z) > BOUNDS.z) {
      p.position.z = clamp(p.position.z, -BOUNDS.z, BOUNDS.z);
      p.velocity.z = 0;
    }
  }

  // -------------------------------------------------------------------------
  // Queries
  // -------------------------------------------------------------------------

  private nearestOpponent(p: RigView, rigs: RigView[]): RigView | null {
    let best: RigView | null = null;
    let bestD = Infinity;
    for (const q of rigs) {
      if (q.team === p.team) continue;
      const d = flatDistance(p.position, q.position);
      if (d < bestD) {
        bestD = d;
        best = q;
      }
    }
    return best;
  }

  /** The first opponent standing on the segment `from`→`to`, if any. */
  private lineBlocker(from: Vector3, to: Vector3, rigs: RigView[], team: number): RigView | null {
    const dx = to.x - from.x;
    const dz = to.z - from.z;
    const len2 = dx * dx + dz * dz;
    if (len2 < 1e-4) return null;
    for (const q of rigs) {
      if (q.team === team) continue;
      const t = ((q.position.x - from.x) * dx + (q.position.z - from.z) * dz) / len2;
      if (t <= 0.12 || t >= 0.94) continue;
      const cx = from.x + dx * t - q.position.x;
      const cz = from.z + dz * t - q.position.z;
      if (Math.hypot(cx, cz) < LANE_BLOCK_RADIUS) return q;
    }
    return null;
  }

  // -------------------------------------------------------------------------
  // Diagnostics — read by the verification probe, not by gameplay.
  // -------------------------------------------------------------------------

  /** Who each defender is currently marking. */
  get assignments(): readonly number[] {
    return this.mark;
  }

  /** Which standing spot each offensive player is holding. */
  get spots(): readonly number[] {
    return this.spot;
  }

  /** Where the AI thinks the offence is attacking. */
  get rim(): Vector3 {
    return RIM;
  }

  /**
   * The last few passes and what became of them.
   *
   * A completion rate is not observable from a still, from a screenshot, or
   * from watching at one frame every few hundred milliseconds — but it is the
   * number that decides whether an AI possession reads as an offence or as a
   * hot potato. Recorded here so the verification probe can read it out.
   */
  readonly passLog: {
    from: number;
    to: number;
    claimedBy: number;
    complete: boolean;
    flight: number;
    /** How far the claim happened from where the ball was aimed. */
    miss: number;
    /** How far the man it was thrown to still was from the ball. */
    receiverGap: number;
    /** Length of the throw. */
    length: number;
  }[] = [];

  private logPass(from: number, to: number): void {
    this.passLog.push({
      from,
      to,
      claimedBy: -1,
      complete: false,
      flight: this.elapsed,
      miss: 0,
      receiverGap: 0,
      length: flatDistance(this.passOrigin, this.passTarget),
    });
    if (this.passLog.length > 24) this.passLog.shift();
  }

  private closePass(by: number, at: Vector3): void {
    const last = this.passLog[this.passLog.length - 1];
    if (!last || last.claimedBy >= 0) return;
    last.claimedBy = by;
    last.complete = by === last.to;
    last.flight = this.elapsed - last.flight;
    last.miss = flatDistance(at, this.passTarget);
    const target = this.players?.players[last.to];
    last.receiverGap = target ? flatDistance(target.position, at) : -1;
  }
}
