/**
 * Portrait HUD: broadcast scorebug pinned under the notch, shot meter, touch
 * control affordances, and the toast/highlight layer.
 *
 * Owned by the UI agent. DOM-based so text stays crisp at any DPR.
 */

import type { Engine, System } from '../core/Engine';
import type { GameSystem } from '../game/GameSystem';

const CSS = `
.hud { position:absolute; inset:0; pointer-events:none; font-variant-numeric: tabular-nums; }
.scorebug {
  position:absolute; top:calc(var(--safe-t) + 10px); left:50%; transform:translateX(-50%);
  display:flex; align-items:stretch; gap:0; border-radius:12px; overflow:hidden;
  background:linear-gradient(180deg, rgba(14,18,28,.94), rgba(8,10,16,.94));
  border:1px solid rgba(255,255,255,.09);
  box-shadow:0 10px 34px rgba(0,0,0,.55), inset 0 1px 0 rgba(255,255,255,.07);
  backdrop-filter: blur(14px) saturate(1.3);
}
.scorebug .team { display:flex; align-items:center; gap:8px; padding:8px 12px; }
.scorebug .tag { font-size:12px; font-weight:800; letter-spacing:.1em; color:#cdd6e6; }
.scorebug .pts { font-size:21px; font-weight:900; color:#fff; letter-spacing:.01em; }
.scorebug .mid {
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  padding:5px 13px; background:rgba(255,255,255,.045);
  border-left:1px solid rgba(255,255,255,.07); border-right:1px solid rgba(255,255,255,.07);
}
.scorebug .clock { font-size:15px; font-weight:800; color:#fff; }
.scorebug .qtr { font-size:9px; font-weight:700; letter-spacing:.18em; color:#8b97ac; }
.shotclock {
  position:absolute; top:calc(var(--safe-t) + 66px); left:50%; transform:translateX(-50%);
  font-size:13px; font-weight:800; color:#ffb454; letter-spacing:.06em;
  text-shadow:0 0 14px rgba(255,150,40,.5);
}
.meter {
  position:absolute; left:50%; bottom:calc(var(--safe-b) + 132px); transform:translateX(-50%);
  width:132px; height:9px; border-radius:9px; background:rgba(255,255,255,.1);
  border:1px solid rgba(255,255,255,.14); overflow:hidden; opacity:0; transition:opacity .12s;
  box-shadow:0 6px 18px rgba(0,0,0,.45);
}
.meter.on { opacity:1; }
.meter i { display:block; height:100%; width:0%; background:linear-gradient(90deg,#4ea3ff,#7cf5b0); }
.meter .perfect {
  position:absolute; top:0; bottom:0; width:9%; left:81%;
  background:rgba(255,255,255,.42); box-shadow:0 0 10px rgba(255,255,255,.7);
}
.stick { position:absolute; width:124px; height:124px; margin:-62px 0 0 -62px; border-radius:50%;
  border:1.5px solid rgba(255,255,255,.16); background:radial-gradient(circle,rgba(255,255,255,.07),transparent 70%);
  opacity:0; transition:opacity .1s; }
.stick.on { opacity:1; }
.knob { position:absolute; width:52px; height:52px; margin:-26px 0 0 -26px; border-radius:50%;
  background:radial-gradient(circle at 35% 30%, rgba(255,255,255,.85), rgba(190,205,230,.35));
  box-shadow:0 4px 16px rgba(0,0,0,.5); opacity:0; transition:opacity .1s; }
.knob.on { opacity:1; }
.btns { position:absolute; right:calc(var(--safe-r) + 16px); bottom:calc(var(--safe-b) + 26px);
  display:grid; grid-template-columns:repeat(2,64px); gap:12px; }
.btn { width:64px; height:64px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  font-size:10px; font-weight:800; letter-spacing:.1em; color:#e9eefa;
  background:linear-gradient(180deg, rgba(38,48,70,.82), rgba(16,21,33,.86));
  border:1px solid rgba(255,255,255,.14); box-shadow:0 8px 22px rgba(0,0,0,.5), inset 0 1px 0 rgba(255,255,255,.12); }
.btn.shoot { background:linear-gradient(180deg, rgba(255,138,43,.9), rgba(176,72,14,.92)); color:#fff; }
.toast { position:absolute; left:50%; top:38%; transform:translate(-50%,-50%);
  font-size:34px; font-weight:900; letter-spacing:.04em; color:#fff; opacity:0;
  text-shadow:0 6px 26px rgba(0,0,0,.7); transition:opacity .2s, transform .2s; }
.toast.on { opacity:1; transform:translate(-50%,-56%); }
.perf { position:absolute; left:calc(var(--safe-l) + 10px); top:calc(var(--safe-t) + 10px);
  font-size:10px; color:#5c6辐; }
`;

export class HudSystem implements System {
  readonly name = 'hud';
  readonly order = 95;

  private root!: HTMLDivElement;
  private ptsHome!: HTMLElement;
  private ptsAway!: HTMLElement;
  private clockEl!: HTMLElement;
  private shotClockEl!: HTMLElement;
  private meter!: HTMLElement;
  private meterFill!: HTMLElement;
  private stick!: HTMLElement;
  private knob!: HTMLElement;
  private toast!: HTMLElement;
  private toastTimer = 0;

  init(engine: Engine): void {
    const style = document.createElement('style');
    style.textContent = CSS.replace('#5c6辐', '#5c6879');
    document.head.appendChild(style);

    const root = document.createElement('div');
    root.className = 'hud';
    root.innerHTML = `
      <div class="scorebug">
        <div class="team"><span class="tag">HOM</span><span class="pts" id="ptsH">0</span></div>
        <div class="mid"><span class="clock" id="clk">12:00</span><span class="qtr" id="qtr">Q1</span></div>
        <div class="team"><span class="pts" id="ptsA">0</span><span class="tag">AWY</span></div>
      </div>
      <div class="shotclock" id="sc">24</div>
      <div class="meter" id="meter"><i id="meterfill"></i><div class="perfect"></div></div>
      <div class="stick" id="stick"></div>
      <div class="knob" id="knob"></div>
      <div class="btns">
        <div class="btn">PASS</div>
        <div class="btn shoot">SHOOT</div>
      </div>
      <div class="toast" id="toast"></div>
    `;
    engine.uiRoot.appendChild(root);
    this.root = root;

    const q = (id: string) => root.querySelector(`#${id}`) as HTMLElement;
    this.ptsHome = q('ptsH');
    this.ptsAway = q('ptsA');
    this.clockEl = q('clk');
    this.shotClockEl = q('sc');
    this.meter = q('meter');
    this.meterFill = q('meterfill');
    this.stick = q('stick');
    this.knob = q('knob');
    this.toast = q('toast');

    engine.bus.on('scored', ({ swish }) => this.showToast(swish ? 'SWISH!' : 'BUCKET!'));
    engine.bus.on('hudToast', ({ text }) => this.showToast(text));
  }

  private showToast(text: string): void {
    this.toast.textContent = text;
    this.toast.classList.add('on');
    this.toastTimer = 1.4;
  }

  update(dt: number, _alpha: number, engine: Engine): void {
    const game = engine.get<GameSystem>('game');
    if (game) {
      this.ptsHome.textContent = String(game.score.home);
      this.ptsAway.textContent = String(game.score.away);
      const m = Math.floor(game.clock / 60);
      const s = Math.floor(game.clock % 60);
      this.clockEl.textContent = `${m}:${String(s).padStart(2, '0')}`;
      this.shotClockEl.textContent = String(Math.ceil(game.shotClock));

      this.meter.classList.toggle('on', game.shotCharging);
      this.meterFill.style.width = `${game.shotMeter * 100}%`;
    }

    const a = engine.input.stickAnchor;
    this.stick.classList.toggle('on', a.visible);
    this.knob.classList.toggle('on', a.visible);
    if (a.visible) {
      this.stick.style.left = `${a.x}px`;
      this.stick.style.top = `${a.y}px`;
      this.knob.style.left = `${engine.input.stickKnob.x}px`;
      this.knob.style.top = `${engine.input.stickKnob.y}px`;
    }

    if (this.toastTimer > 0) {
      this.toastTimer -= dt;
      if (this.toastTimer <= 0) this.toast.classList.remove('on');
    }
  }

  dispose(): void {
    this.root.remove();
  }
}
