import { Engine } from './core/Engine';
import { LightingSystem } from './render/Lighting';
import { PostFXSystem } from './render/PostFX';
import { CourtSystem } from './world/Court';
import { ArenaSystem } from './world/Arena';
import { HoopSystem } from './world/Hoop';
import { BallSystem } from './physics/BallSystem';
import { PlayerSystem } from './entities/PlayerSystem';
import { GameSystem } from './game/GameSystem';
import { PlayControlSystem } from './game/PlayControl';
import { TeamAISystem } from './game/TeamAI';
import { CameraSystem } from './render/CameraSystem';
import { HudSystem } from './ui/HudSystem';
import { AudioSystem } from './audio/AudioSystem';
import { VfxSystem } from './render/VfxSystem';
import { FlatField } from './dev/flatfield';

const canvas = document.getElementById('gl') as HTMLCanvasElement;
const uiRoot = document.getElementById('ui') as HTMLElement;
const boot = document.getElementById('boot') as HTMLElement;
const bootBar = document.getElementById('bootbar') as HTMLElement;
const bootTip = document.getElementById('boottip') as HTMLElement;

const TIPS = [
  'Milling the hardwood…',
  'Stringing the nets…',
  'Lighting the rig…',
  'Filling the lower bowl…',
  'Chalking the lines…',
  'Inflating to 8 psi…',
];

async function main(): Promise<void> {
  const engine = new Engine({ canvas, uiRoot });

  engine
    .add(new LightingSystem())
    .add(new CourtSystem())
    .add(new ArenaSystem())
    .add(new HoopSystem())
    .add(new BallSystem())
    .add(new PlayerSystem())
    .add(new PlayControlSystem())
    .add(new TeamAISystem())
    .add(new GameSystem())
    .add(new CameraSystem())
    .add(new VfxSystem())
    .add(new PostFXSystem())
    .add(new AudioSystem())
    .add(new HudSystem());

  await engine.initSystems((done, total, label) => {
    const pct = Math.round((done / Math.max(1, total)) * 100);
    bootBar.style.width = `${pct}%`;
    bootTip.textContent = TIPS[done % TIPS.length] ?? label;
  });

  engine.start();

  // Give the first few frames a chance to compile shaders before the reveal.
  await new Promise<void>((r) => setTimeout(r, 260));
  boot.classList.add('hidden');

  // Handles for the screenshot / QA harness.
  const flatField = new FlatField();
  Object.assign(window as unknown as Record<string, unknown>, {
    __engine: engine,
    __ready: true,
    /**
     * Replaces the scene with a uniform field so the review harness can measure
     * what the post chain does to an image, separately from scene content.
     */
    __flatfield: (on: boolean) => {
      if (on) flatField.enable(engine.scene, uiRoot);
      else flatField.disable(engine.scene);
    },
  });
}

main().catch((err) => {
  console.error(err);
  bootTip.textContent = 'Failed to start — see console.';
  bootTip.style.color = '#ff6b6b';
});
