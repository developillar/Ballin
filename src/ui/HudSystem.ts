/**
 * Portrait HUD: broadcast scorebug under the notch, shot meter, touch control
 * affordances and the annunciator layer.
 *
 * DOM rather than canvas or in-scene geometry, deliberately. Text rendered by
 * the browser is correct at every device pixel ratio for free, picks up the
 * system font stack, and costs nothing in the 3D frame. A canvas HUD has to
 * re-rasterise on every resize and still ends up softer.
 *
 * This reads game state; it never writes it. Input is handled on the canvas by
 * `Input`, so the control graphics here are affordances that mirror what the
 * touch layer is already doing rather than being buttons in their own right.
 * That keeps one input path instead of two that can disagree.
 */

import type { Engine, System } from '../core/Engine';
import { SHOT } from '../core/Constants';
import { clamp01 } from '../core/MathX';
import { HUD_CSS } from './hudStyle';

/**
 * The slice of game state the HUD needs. Declared structurally rather than
 * importing `GameSystem`, so gameplay can be rewritten without touching the UI
 * and the HUD degrades to blanks rather than crashing if a field goes away.
 */
interface GameView {
  score?: { home: number; away: number };
  quarter?: number;
  clock?: number;
  shotClock?: number;
  possession?: number;
  shotMeter?: number;
  shotCharging?: boolean;
  lastShotQuality?: number;
}

const TEAM_COLOR = ['#2f6fd0', '#c8412f'];
const TEAM_TAG = ['HOM', 'AWY'];

export class HudSystem implements System {
  readonly name = 'hud';
  readonly order = 95;

  private root!: HTMLDivElement;
  private style: HTMLStyleElement | null = null;

  private teams: HTMLElement[] = [];
  private pts: HTMLElement[] = [];
  private clockEl!: HTMLElement;
  private qtrEl!: HTMLElement;
  private shotClockEl!: HTMLElement;
  private meter!: HTMLElement;
  private meterFill!: HTMLElement;
  private callout!: HTMLElement;
  private stick!: HTMLElement;
  private knob!: HTMLElement;
  private shootBtn!: HTMLElement;
  private passBtn!: HTMLElement;
  private toast!: HTMLElement;
  private perf!: HTMLElement;

  private toastTimer = 0;
  private calloutTimer = 0;
  /** Cached so the DOM is only touched when a value actually changes. */
  private shown = { home: -1, away: -1, clock: '', shotClock: '', quarter: -1, possession: -1 };

  init(engine: Engine): void {
    this.style = document.createElement('style');
    this.style.textContent = HUD_CSS;
    document.head.appendChild(this.style);

    const root = document.createElement('div');
    root.className = 'hud';
    root.innerHTML = `
      <div class="scrim-t"></div>
      <div class="scrim-b"></div>
      <div class="scorebug">
        <div class="team home" style="--team-color:${TEAM_COLOR[0]}">
          <span class="tag">${TEAM_TAG[0]}</span><span class="pts">0</span>
        </div>
        <div class="mid"><span class="clock">12:00</span><span class="qtr">Q1</span></div>
        <div class="team away" style="--team-color:${TEAM_COLOR[1]}">
          <span class="pts">0</span><span class="tag">${TEAM_TAG[1]}</span>
        </div>
      </div>
      <div class="shotclock">24</div>
      <div class="meter"><i></i><div class="window"></div></div>
      <div class="callout"></div>
      <div class="stick"></div>
      <div class="knob"></div>
      <div class="btns">
        <div class="btn pass">PASS</div>
        <div class="btn shoot">SHOOT</div>
      </div>
      <div class="toast"></div>
      <div class="perf"></div>
    `;
    engine.uiRoot.appendChild(root);
    this.root = root;

    const q = <T extends HTMLElement>(sel: string): T => root.querySelector(sel) as T;
    this.teams = [q('.team.home'), q('.team.away')];
    this.pts = [q('.team.home .pts'), q('.team.away .pts')];
    this.clockEl = q('.clock');
    this.qtrEl = q('.qtr');
    this.shotClockEl = q('.shotclock');
    this.meter = q('.meter');
    this.meterFill = q('.meter i');
    this.callout = q('.callout');
    this.stick = q('.stick');
    this.knob = q('.knob');
    this.shootBtn = q('.btn.shoot');
    this.passBtn = q('.btn.pass');
    this.toast = q('.toast');
    this.perf = q('.perf');

    engine.bus.on('scored', ({ points, swish }) =>
      this.showToast(swish ? 'SWISH' : points >= 3 ? 'THREE' : 'BUCKET', swish ? 'hype' : 'good'),
    );
    engine.bus.on('block', () => this.showToast('BLOCKED', 'hype'));
    engine.bus.on('steal', () => this.showToast('STEAL', 'hype'));
    engine.bus.on('hudToast', ({ text, kind }) => this.showToast(text, kind));
    engine.bus.on('shotReleased', ({ quality }) => this.showRelease(quality));
    engine.bus.on('quarterEnd', ({ quarter }) => this.showToast(`END Q${quarter}`, 'neutral'));
    engine.bus.on('gameEnd', () => this.showToast('FINAL', 'hype'));
  }

  private showToast(text: string, kind: 'good' | 'bad' | 'neutral' | 'hype' = 'neutral'): void {
    this.toast.textContent = text;
    this.toast.className = `toast on ${kind === 'hype' ? 'hype' : kind === 'bad' ? 'bad' : ''}`;
    this.toastTimer = 1.4;
  }

  /**
   * Names the release. The wording matters: a player needs to know *which way*
   * they missed the window, not just that they did, or the feedback teaches
   * them nothing.
   */
  private showRelease(quality: number): void {
    const q = clamp01(quality);
    const label = q >= 0.92 ? 'EXCELLENT' : q >= 0.7 ? 'GOOD' : 'RUSHED';
    this.callout.textContent = label;
    this.callout.className = `callout on ${q >= 0.7 ? '' : 'bad'}`;
    this.calloutTimer = 1.1;
  }

  update(dt: number, _alpha: number, engine: Engine): void {
    const game = engine.get<GameView>('game');

    if (game) {
      const home = game.score?.home ?? 0;
      const away = game.score?.away ?? 0;
      if (home !== this.shown.home) this.pts[0].textContent = String((this.shown.home = home));
      if (away !== this.shown.away) this.pts[1].textContent = String((this.shown.away = away));

      const clock = Math.max(0, game.clock ?? 0);
      // Under a minute, broadcasts switch to tenths. It is a small thing and
      // its absence is one of those details that reads as "not a real game".
      const text =
        clock < 60
          ? clock.toFixed(1)
          : `${Math.floor(clock / 60)}:${String(Math.floor(clock % 60)).padStart(2, '0')}`;
      if (text !== this.shown.clock) this.clockEl.textContent = this.shown.clock = text;

      const quarter = game.quarter ?? 1;
      if (quarter !== this.shown.quarter) {
        this.shown.quarter = quarter;
        this.qtrEl.textContent = quarter > 4 ? `OT${quarter - 4}` : `Q${quarter}`;
      }

      const sc = Math.max(0, game.shotClock ?? 0);
      const scText = sc <= 5 ? sc.toFixed(1) : String(Math.ceil(sc));
      if (scText !== this.shown.shotClock) this.shotClockEl.textContent = this.shown.shotClock = scText;
      this.shotClockEl.classList.toggle('urgent', sc <= 5);

      const poss = game.possession ?? 0;
      if (poss !== this.shown.possession) {
        this.shown.possession = poss;
        this.teams[0].classList.toggle('has-ball', poss === 0);
        this.teams[1].classList.toggle('has-ball', poss === 1);
      }

      const charging = game.shotCharging ?? false;
      const meter = clamp01(game.shotMeter ?? 0);
      this.meter.classList.toggle('on', charging);
      if (charging) {
        this.meterFill.style.height = `${meter * 100}%`;
        this.meter.classList.toggle('perfect', Math.abs(meter - SHOT.windowCentre) <= SHOT.windowHalf);
      }
    }

    const anchor = engine.input.stickAnchor;
    this.stick.classList.toggle('on', anchor.visible);
    this.knob.classList.toggle('on', anchor.visible);
    if (anchor.visible) {
      this.stick.style.left = `${anchor.x}px`;
      this.stick.style.top = `${anchor.y}px`;
      this.knob.style.left = `${engine.input.stickKnob.x}px`;
      this.knob.style.top = `${engine.input.stickKnob.y}px`;
    }

    this.shootBtn.classList.toggle('down', engine.input.actions.shoot.held);
    this.passBtn.classList.toggle('down', engine.input.actions.pass.held);

    if (this.toastTimer > 0) {
      this.toastTimer -= dt;
      if (this.toastTimer <= 0) this.toast.classList.remove('on');
    }
    if (this.calloutTimer > 0) {
      this.calloutTimer -= dt;
      if (this.calloutTimer <= 0) this.callout.classList.remove('on');
    }
  }

  /** Frame-cost overlay. Off by default; the harness turns it on when asked. */
  setDebug(on: boolean, lines: string[] = []): void {
    this.perf.classList.toggle('on', on);
    if (on) this.perf.textContent = lines.join('\n');
  }

  dispose(): void {
    this.root.remove();
    this.style?.remove();
  }
}
