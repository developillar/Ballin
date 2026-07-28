/**
 * HUD stylesheet.
 *
 * Kept out of `HudSystem.ts` so the markup and the behaviour stay readable, and
 * because this is where most of the portrait-specific decisions live.
 *
 * Three rules run through all of it:
 *
 * 1. **Everything scales with the viewport.** A phone HUD sized in fixed pixels
 *    is either lost on a 430 px-wide device or overwhelming on a 320 px one.
 *    Sizes use `clamp(min, vw-relative, max)` so the layout holds from a small
 *    phone to a tablet without a media query.
 * 2. **Nothing touches the edges.** Safe-area variables come from index.html and
 *    every anchored element adds them, so the scorebug clears the notch and the
 *    controls clear the home indicator.
 * 3. **Text is always legible over hardwood.** A bright maple floor under white
 *    type is the single worst case, so the top and bottom carry a scrim
 *    gradient. It is the same trick a broadcast graphics package uses and it
 *    costs one gradient per edge.
 */

export const HUD_CSS = `
.hud {
  position: absolute;
  inset: 0;
  pointer-events: none;
  font-variant-numeric: tabular-nums;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
  --hud-accent: #ff8a2b;
  --hud-ink: #f2f5fb;
  --hud-dim: #8b97ac;
  --hud-panel: linear-gradient(180deg, rgba(15,19,29,.93), rgba(8,10,16,.95));
  --hud-edge: rgba(255,255,255,.10);
}

/* Scrims: keep type readable over bright hardwood without dimming the game. */
.hud .scrim-t, .hud .scrim-b { position:absolute; left:0; right:0; pointer-events:none; }
.hud .scrim-t { top:0; height:clamp(90px, 20vh, 190px);
  background:linear-gradient(180deg, rgba(2,4,8,.55), rgba(2,4,8,0)); }
.hud .scrim-b { bottom:0; height:clamp(120px, 26vh, 250px);
  background:linear-gradient(0deg, rgba(2,4,8,.58), rgba(2,4,8,0)); }

/* --- Scorebug ---------------------------------------------------------- */

.scorebug {
  position: absolute;
  top: calc(var(--safe-t) + clamp(6px, 1.6vh, 14px));
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: stretch;
  border-radius: clamp(9px, 2.6vw, 14px);
  overflow: hidden;
  background: var(--hud-panel);
  border: 1px solid var(--hud-edge);
  box-shadow: 0 10px 34px rgba(0,0,0,.55), inset 0 1px 0 rgba(255,255,255,.07);
  backdrop-filter: blur(14px) saturate(1.25);
  -webkit-backdrop-filter: blur(14px) saturate(1.25);
}

.scorebug .team {
  display: flex;
  align-items: center;
  gap: clamp(5px, 1.6vw, 9px);
  padding: clamp(6px, 1.7vw, 9px) clamp(9px, 3vw, 15px);
  position: relative;
}
/* Team colour reads as a stripe on the outer edge, the way a broadcast bug
   does — a coloured panel behind the score fights the numbers for attention. */
.scorebug .team::before {
  content: ''; position: absolute; top: 0; bottom: 0; width: 3px;
  background: var(--team-color, #4ea3ff);
}
.scorebug .team.home::before { left: 0; }
.scorebug .team.away::before { right: 0; }

.scorebug .tag {
  font-size: clamp(9px, 2.9vw, 12px);
  font-weight: 800; letter-spacing: .11em; color: #cdd6e6;
}
.scorebug .pts {
  font-size: clamp(17px, 5.4vw, 23px);
  font-weight: 900; color: #fff; letter-spacing: .005em; line-height: 1;
}
/* The team in possession is lifted, not boxed. */
.scorebug .team.has-ball .tag { color: #fff; }
.scorebug .team.has-ball::after {
  content: ''; position: absolute; bottom: 3px; left: 50%; transform: translateX(-50%);
  width: 14px; height: 2px; border-radius: 2px; background: var(--hud-accent);
  box-shadow: 0 0 8px rgba(255,138,43,.8);
}

.scorebug .mid {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  padding: clamp(4px, 1.2vw, 6px) clamp(10px, 3.4vw, 16px);
  background: rgba(255,255,255,.045);
  border-left: 1px solid rgba(255,255,255,.07);
  border-right: 1px solid rgba(255,255,255,.07);
}
.scorebug .clock { font-size: clamp(13px, 4.1vw, 16px); font-weight: 800; color: #fff; line-height: 1.15; }
.scorebug .qtr { font-size: clamp(8px, 2.4vw, 10px); font-weight: 700; letter-spacing: .19em; color: var(--hud-dim); }

/* Shot clock sits below the bug, in broadcast amber. It turns red and pulses
   under five seconds, which is the one HUD element a player must never miss. */
.shotclock {
  position: absolute;
  top: calc(var(--safe-t) + clamp(52px, 9.4vh, 78px));
  left: 50%; transform: translateX(-50%);
  font-size: clamp(12px, 3.9vw, 15px); font-weight: 800;
  color: #ffb454; letter-spacing: .06em;
  text-shadow: 0 0 14px rgba(255,150,40,.45);
  transition: color .12s;
}
.shotclock.urgent { color: #ff5c4d; text-shadow: 0 0 18px rgba(255,70,50,.7); animation: sc-pulse .5s steps(1) infinite; }
@keyframes sc-pulse { 50% { opacity: .35; } }

/* --- Shot meter -------------------------------------------------------- */

/* Vertical, on the left of the thumb zone. Vertical rather than horizontal for
   two reasons: it does not fight the horizontal scorebug for the eye, and in
   portrait there is far more spare height than width. */
.meter {
  position: absolute;
  left: calc(var(--safe-l) + clamp(16px, 5vw, 30px));
  bottom: calc(var(--safe-b) + clamp(90px, 17vh, 150px));
  width: clamp(9px, 2.6vw, 13px);
  height: clamp(120px, 22vh, 190px);
  border-radius: 99px;
  background: rgba(6,9,15,.6);
  border: 1px solid rgba(255,255,255,.16);
  box-shadow: 0 6px 20px rgba(0,0,0,.5), inset 0 0 12px rgba(0,0,0,.5);
  opacity: 0;
  transform: translateY(8px) scaleY(.9);
  transform-origin: bottom center;
  transition: opacity .1s ease-out, transform .14s cubic-bezier(.2,.9,.3,1.2);
  overflow: hidden;
}
.meter.on { opacity: 1; transform: translateY(0) scaleY(1); }
.meter i {
  position: absolute; left: 0; right: 0; bottom: 0;
  height: 0%;
  background: linear-gradient(0deg, #2f7fd6, #6fd8ff);
  transition: none;
}
/* The release window. Hitting inside it is the difference between a good shot
   and a great one, so it is the brightest thing on the bar.
   These two numbers are SHOT.windowCentre +/- SHOT.windowHalf from
   src/core/Constants.ts, as percentages. Change them together or the interface
   draws the window somewhere other than where gameplay actually rewards. */
.meter .window {
  position: absolute; left: 0; right: 0;
  bottom: 78%; height: 12%;
  background: rgba(120,255,180,.28);
  border-top: 1px solid rgba(150,255,200,.85);
  border-bottom: 1px solid rgba(150,255,200,.5);
  box-shadow: 0 0 12px rgba(110,255,170,.55);
}
.meter.perfect i { background: linear-gradient(0deg, #1f9d5c, #7cf5b0); }

/* Release feedback pill, right above the meter. */
.callout {
  position: absolute;
  left: calc(var(--safe-l) + clamp(10px, 3.6vw, 22px));
  bottom: calc(var(--safe-b) + clamp(218px, 40vh, 348px));
  padding: 4px 10px; border-radius: 99px;
  font-size: clamp(9px, 2.7vw, 11px); font-weight: 800; letter-spacing: .12em;
  color: #06210f; background: #7cf5b0;
  box-shadow: 0 4px 16px rgba(0,0,0,.45);
  opacity: 0; transform: translateY(6px);
  transition: opacity .12s, transform .18s cubic-bezier(.2,.9,.3,1.2);
}
.callout.on { opacity: 1; transform: translateY(0); }
.callout.early, .callout.late { background: #ffd166; color: #2a1c00; }
.callout.bad { background: #ff6f61; color: #2a0704; }

/* --- Touch controls ---------------------------------------------------- */

.stick {
  position: absolute; width: clamp(96px, 30vw, 138px); height: clamp(96px, 30vw, 138px);
  border-radius: 50%;
  border: 1.5px solid rgba(255,255,255,.15);
  background: radial-gradient(circle, rgba(255,255,255,.07), transparent 70%);
  opacity: 0; transition: opacity .1s;
  translate: -50% -50%;
}
.knob {
  position: absolute; width: clamp(42px, 13vw, 58px); height: clamp(42px, 13vw, 58px);
  border-radius: 50%;
  background: radial-gradient(circle at 34% 28%, rgba(255,255,255,.88), rgba(180,198,228,.32));
  box-shadow: 0 4px 16px rgba(0,0,0,.5), inset 0 -2px 6px rgba(0,0,0,.25);
  opacity: 0; transition: opacity .1s;
  translate: -50% -50%;
}
.stick.on, .knob.on { opacity: 1; }

.btns {
  position: absolute;
  right: calc(var(--safe-r) + clamp(12px, 4vw, 22px));
  bottom: calc(var(--safe-b) + clamp(18px, 4vh, 34px));
  display: grid;
  grid-template-columns: repeat(2, clamp(56px, 17vw, 74px));
  gap: clamp(9px, 3vw, 15px);
}
.btn {
  width: clamp(56px, 17vw, 74px); height: clamp(56px, 17vw, 74px);
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: clamp(9px, 2.6vw, 11px); font-weight: 800; letter-spacing: .1em;
  color: #e9eefa;
  background: linear-gradient(180deg, rgba(40,50,72,.84), rgba(15,20,32,.88));
  border: 1px solid rgba(255,255,255,.14);
  box-shadow: 0 8px 22px rgba(0,0,0,.5), inset 0 1px 0 rgba(255,255,255,.13);
  transition: transform .07s ease-out, filter .07s;
}
.btn.shoot {
  background: linear-gradient(180deg, rgba(255,142,48,.92), rgba(172,68,12,.94));
  color: #fff;
  box-shadow: 0 8px 24px rgba(120,45,0,.45), inset 0 1px 0 rgba(255,255,255,.28);
}
.btn.down { transform: translateY(2px) scale(.96); filter: brightness(1.18); }

/* --- Annunciator ------------------------------------------------------- */

.toast {
  position: absolute; left: 50%; top: 36%;
  transform: translate(-50%, -50%) scale(.92);
  font-size: clamp(26px, 9vw, 42px); font-weight: 900; letter-spacing: .03em;
  color: #fff; opacity: 0; white-space: nowrap;
  text-shadow: 0 6px 26px rgba(0,0,0,.75), 0 0 40px rgba(255,160,60,.25);
  transition: opacity .16s ease-out, transform .28s cubic-bezier(.16,1,.3,1);
}
.toast.on { opacity: 1; transform: translate(-50%, -58%) scale(1); }
.toast.hype { color: #ffd166; }
.toast.bad { color: #ff8f7f; }

.perf {
  position: absolute;
  left: calc(var(--safe-l) + 10px);
  top: calc(var(--safe-t) + 10px);
  font-size: 10px; line-height: 1.45; color: #5c6879;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  white-space: pre; display: none;
}
.perf.on { display: block; }
`;
