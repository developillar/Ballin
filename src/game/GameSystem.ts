/**
 * Match state, rules, possession, scoring and the shot pipeline.
 *
 * Owned by the gameplay agent. Exposes `debugScene(name)` so the screenshot
 * harness can force the game into a specific visual beat.
 */

import { Vector3 } from 'three';
import type { Engine, System } from '../core/Engine';
import { BALL, HOOP, RULES, SHOT, basketX } from '../core/Constants';
import type { BallSystem } from '../physics/BallSystem';
import { clamp01 } from '../core/MathX';

export type Phase = 'tipoff' | 'live' | 'shot' | 'dead' | 'inbound' | 'over';

export interface Score {
  home: number;
  away: number;
}

export class GameSystem implements System {
  readonly name = 'game';
  readonly order = 40;

  score: Score = { home: 0, away: 0 };
  quarter = 1;
  clock: number = RULES.quarterSeconds;
  shotClock: number = RULES.shotClockSeconds;
  possession = 0;
  phase: Phase = 'live';

  /** 0..1 — how well-timed the last release was. */
  lastShotQuality = 0;
  /** Live meter value while the shoot button is held. */
  shotMeter = 0;
  shotCharging = false;

  private ball: BallSystem | null = null;

  /**
   * The shot currently in the air. A basket is worth two or three depending on
   * where it was released from, and nothing downstream can work that out from
   * the ball passing through the ring — so it is recorded at release.
   */
  private liveShot: { team: number; shooter: number; three: boolean; touchedRim: boolean } | null = null;

  /** Last whole second announced, so `clockTick` fires once per second. */
  private lastTick = -1;

  init(engine: Engine): void {
    this.ball = engine.get<BallSystem>('ball') ?? null;

    engine.bus.on('shotReleased', ({ three, shooter }) => {
      this.liveShot = { team: this.possession, shooter, three, touchedRim: false };
    });
    engine.bus.on('rimContact', () => {
      if (this.liveShot) this.liveShot.touchedRim = true;
    });

    engine.bus.on('netSwish', () => this.resolveMake(engine));

    // A shot that comes down and hits the floor without having gone through is
    // a miss. Without this, a missed shot never resolves and the shot that
    // follows is scored against stale state.
    engine.bus.on('floorBounce', () => {
      if (this.phase !== 'shot' || !this.liveShot) return;
      const shot = this.liveShot;
      this.liveShot = null;
      this.phase = 'live';
      engine.bus.emit('missed', { team: shot.team, shooter: shot.shooter, rimContact: shot.touchedRim });
    });
  }

  private resolveMake(engine: Engine): void {
    const shot = this.liveShot;
    // A ball dropped through the ring outside of a live shot still counts —
    // the harness does exactly that to stage the swish frame — so fall back to
    // two for whoever has possession rather than dropping it.
    const team = shot?.team ?? this.possession;
    const points = shot?.three ? 3 : 2;
    const swish = !(shot?.touchedRim ?? false);

    if (team === 0) this.score.home += points;
    else this.score.away += points;

    this.liveShot = null;
    this.phase = 'live';

    engine.bus.emit('scored', {
      points,
      team,
      shooter: shot?.shooter ?? 0,
      swish,
      assisted: null,
    });

    // The scoring team gives the ball up.
    this.changePossession(engine, team === 0 ? 1 : 0);
  }

  private changePossession(engine: Engine, team: number): void {
    if (this.possession === team) return;
    this.possession = team;
    this.shotClock = RULES.shotClockSeconds;
    engine.bus.emit('possessionChanged', { team });
  }

  simulate(step: number, engine: Engine): void {
    if (this.phase === 'over') return;

    this.clock = Math.max(0, this.clock - step);
    this.shotClock = Math.max(0, this.shotClock - step);

    // One tick per whole second, not one per fixed step — this drives the shot
    // clock beep, and at 240 Hz an unguarded emit would fire it 240 times.
    const whole = Math.ceil(this.clock);
    if (whole !== this.lastTick) {
      this.lastTick = whole;
      engine.bus.emit('clockTick', { seconds: whole, shotClock: Math.ceil(this.shotClock) });
    }

    if (this.shotClock <= 0 && this.phase !== 'shot') {
      this.changePossession(engine, this.possession === 0 ? 1 : 0);
      engine.bus.emit('hudToast', { text: 'SHOT CLOCK', kind: 'bad' });
    }

    if (this.clock <= 0) this.endPeriod(engine);
  }

  private endPeriod(engine: Engine): void {
    if (this.quarter >= RULES.quarters) {
      this.phase = 'over';
      engine.bus.emit('gameEnd', { home: this.score.home, away: this.score.away });
      return;
    }
    engine.bus.emit('quarterEnd', { quarter: this.quarter });
    this.quarter++;
    this.clock = RULES.quarterSeconds;
    this.shotClock = RULES.shotClockSeconds;
    this.phase = 'live';
  }

  update(dt: number, _alpha: number, engine: Engine): void {
    const shoot = engine.input.actions.shoot;
    if (shoot.pressed) {
      this.shotCharging = true;
      this.shotMeter = 0;
    }
    if (this.shotCharging) {
      this.shotMeter = clamp01(this.shotMeter + dt / SHOT.chargeSeconds);
    }
    if (shoot.released && this.shotCharging) {
      this.shotCharging = false;
      this.releaseShot(engine, this.shotMeter);
    }
  }

  /** Fires a shot with a physically solved arc toward the target basket. */
  releaseShot(engine: Engine, meter: number): void {
    if (!this.ball) return;
    const s = this.ball.ballState;
    const from = s.position.clone();
    const target = new Vector3(basketX(1), HOOP.rimHeight, 0);

    // Ideal release angle for the distance, then perturb by release timing.
    const flat = new Vector3(target.x - from.x, 0, target.z - from.z);
    const d = flat.length();
    const dy = target.y - from.y;
    const angle = Math.max(0.68, Math.min(1.15, 0.72 + d * 0.012));
    const g = 9.80665;
    const cos = Math.cos(angle);
    const tan = Math.tan(angle);
    const denom = 2 * cos * cos * (d * tan - dy);
    const speed = denom > 0 ? Math.sqrt((g * d * d) / denom) : 8;

    // Release quality: 1.0 anywhere inside the green window the HUD draws,
    // falling off linearly outside it. Scoring against a single point rather
    // than the window would mean the interface promises a band that gameplay
    // does not honour.
    const offset = Math.abs(meter - SHOT.windowCentre);
    const quality = offset <= SHOT.windowHalf ? 1 : 1 - (offset - SHOT.windowHalf) / (1 - SHOT.windowHalf);
    this.lastShotQuality = clamp01(quality);
    const err = (1 - this.lastShotQuality) * 0.09;

    flat.normalize();
    const vel = new Vector3(
      flat.x * Math.cos(angle) * speed,
      Math.sin(angle) * speed,
      flat.z * Math.cos(angle) * speed,
    );
    vel.x *= 1 + (Math.random() - 0.5) * err;
    vel.y *= 1 + (Math.random() - 0.5) * err * 0.6;
    vel.z += (Math.random() - 0.5) * err * 4;

    // Backspin: perpendicular to travel, in the horizontal plane.
    const spin = new Vector3(-flat.z, 0, flat.x).multiplyScalar(-34);
    this.ball.launch(from, vel, spin, 0);
    this.phase = 'shot';

    engine.bus.emit('shotReleased', {
      quality: this.lastShotQuality,
      distance: d,
      three: d > 7.24,
      shooter: 0,
    });
  }

  /** Forces a specific visual beat — used by the screenshot harness. */
  debugScene(name: string): void {
    if (!this.ball) return;
    const s = this.ball.ballState;
    switch (name) {
      case 'shot': {
        s.position.set(basketX(1) - 6.6, 2.35, 1.1);
        s.owner = { kind: 'free' };
        s.resting = false;
        this.releaseShotFromHere();
        break;
      }
      case 'swish': {
        s.position.set(basketX(1), HOOP.rimHeight + 1.1, 0.01);
        s.velocity.set(0, -5.4, 0.1);
        s.spin.set(0, 0, -26);
        s.owner = { kind: 'free' };
        s.resting = false;
        break;
      }
      case 'dunk': {
        s.position.set(basketX(1) - 0.55, HOOP.rimHeight + 0.62, 0);
        s.velocity.set(2.4, -3.1, 0);
        s.spin.set(0, 0, -12);
        s.owner = { kind: 'free' };
        s.resting = false;
        break;
      }
      default: {
        s.position.set(basketX(1) - 7.2, BALL.radius + 1.15, 1.4);
        s.velocity.set(0, 0, 0);
        s.resting = false;
      }
    }
  }

  private releaseShotFromHere(): void {
    const engine = (window as unknown as { __engine?: Engine }).__engine;
    if (engine) this.releaseShot(engine, 0.86);
  }
}
