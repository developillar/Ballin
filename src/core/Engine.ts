/**
 * Engine — owns the WebGL context, the frame loop, the resize/orientation
 * plumbing and the adaptive quality governor. Everything else in the game is a
 * `System` registered here; systems get a fixed-step `simulate` for anything
 * physical and a variable-step `update` for presentation.
 */

import {
  ACESFilmicToneMapping,
  Clock,
  PCFSoftShadowMap,
  PerspectiveCamera,
  Scene,
  SRGBColorSpace,
  WebGLRenderer,
} from 'three';
import { AdaptiveGovernor, detectTier, tierBudget, type QualityBudget } from './Quality';
import { Input } from './Input';
import { bus } from './Events';
import { PHYSICS } from './Constants';

export interface System {
  readonly name: string;
  /** Ascending order; lower runs first. */
  readonly order?: number;
  init?(engine: Engine): void | Promise<void>;
  /** Fixed timestep — deterministic physics and gameplay rules live here. */
  simulate?(step: number, engine: Engine): void;
  /** Variable timestep — animation, cameras, UI, VFX. */
  update?(dt: number, alpha: number, engine: Engine): void;
  /** Called after the main render pass, for overlays that need the frame. */
  lateUpdate?(dt: number, engine: Engine): void;
  resize?(width: number, height: number, engine: Engine): void;
  dispose?(): void;
}

export interface EngineOptions {
  canvas: HTMLCanvasElement;
  uiRoot: HTMLElement;
}

export class Engine {
  readonly renderer: WebGLRenderer;
  readonly scene = new Scene();
  readonly camera: PerspectiveCamera;
  readonly input: Input;
  readonly uiRoot: HTMLElement;
  readonly bus = bus;

  quality: QualityBudget;
  readonly governor = new AdaptiveGovernor(60);

  /** Drives every render-time effect that wants a global time base. */
  elapsed = 0;
  frame = 0;
  /** Smoothed frame time in ms, for the debug HUD. */
  frameMs = 16.7;
  /** Set by the game when play is paused; systems may skip simulation. */
  paused = false;
  /** Global slow-motion multiplier used by replays and dunk cams. */
  timeScale = 1;

  /** CSS pixel size of the drawing surface. */
  width = 1;
  height = 1;
  /** Backing-store size after pixel ratio and render scale. */
  pixelWidth = 1;
  pixelHeight = 1;

  /** Set when the post-processing stack takes over presentation. */
  renderOverride: ((dt: number) => void) | null = null;

  private systems: System[] = [];
  private clock = new Clock();
  private accumulator = 0;
  private running = false;
  private rafId = 0;
  private resizeQueued = true;

  constructor(opts: EngineOptions) {
    this.uiRoot = opts.uiRoot;

    const renderer = new WebGLRenderer({
      canvas: opts.canvas,
      antialias: false, // handled by the post stack (TAA/FXAA)
      alpha: false,
      stencil: false,
      depth: true,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: true, // the screenshot harness reads the buffer back
      failIfMajorPerformanceCaveat: false,
    });
    renderer.outputColorSpace = SRGBColorSpace;
    renderer.toneMapping = ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = PCFSoftShadowMap;
    renderer.shadowMap.autoUpdate = true;
    renderer.setClearColor(0x05070c, 1);
    this.renderer = renderer;

    this.quality = tierBudget(detectTier(renderer.getContext() as WebGL2RenderingContext));

    this.camera = new PerspectiveCamera(46, 9 / 19.5, 0.08, 260);
    this.camera.position.set(0, 3.2, 12);
    this.scene.add(this.camera);

    this.input = new Input(opts.canvas);

    const onResize = () => {
      this.resizeQueued = true;
    };
    window.addEventListener('resize', onResize);
    window.addEventListener('orientationchange', onResize);
    if (window.visualViewport) window.visualViewport.addEventListener('resize', onResize);

    renderer.domElement.addEventListener('webglcontextlost', (e) => {
      e.preventDefault();
      this.stop();
      console.warn('[engine] WebGL context lost');
    });
    renderer.domElement.addEventListener('webglcontextrestored', () => {
      console.warn('[engine] WebGL context restored');
      this.resizeQueued = true;
      this.start();
    });

    document.addEventListener('visibilitychange', () => {
      if (document.hidden) this.clock.stop();
      else {
        this.clock.start();
        this.accumulator = 0;
      }
    });
  }

  get anisotropy(): number {
    return Math.min(this.quality.anisotropy, this.renderer.capabilities.getMaxAnisotropy());
  }

  add(system: System): this {
    this.systems.push(system);
    this.systems.sort((a, b) => (a.order ?? 100) - (b.order ?? 100));
    return this;
  }

  /**
   * Look a system up by name. The return type is intentionally unconstrained so
   * callers can request just the slice of the interface they depend on, which
   * keeps subsystems from having to import each other wholesale.
   */
  get<T>(name: string): T | undefined {
    return this.systems.find((s) => s.name === name) as T | undefined;
  }

  async initSystems(onProgress?: (done: number, total: number, label: string) => void): Promise<void> {
    const total = this.systems.length;
    for (let i = 0; i < this.systems.length; i++) {
      const s = this.systems[i];
      onProgress?.(i, total, s.name);
      await s.init?.(this);
      // Yield so the boot screen can paint between heavy bakes.
      await new Promise<void>((r) => requestAnimationFrame(() => r()));
    }
    onProgress?.(total, total, 'ready');
    this.applyResize();
  }

  private applyResize(): void {
    const w = Math.max(1, window.innerWidth);
    const h = Math.max(1, window.innerHeight);
    this.width = w;
    this.height = h;

    const dpr = Math.min(window.devicePixelRatio || 1, this.quality.maxPixelRatio);
    const scale = dpr * this.governor.renderScale;
    this.pixelWidth = Math.max(1, Math.round(w * scale));
    this.pixelHeight = Math.max(1, Math.round(h * scale));

    this.renderer.setPixelRatio(1);
    this.renderer.setSize(this.pixelWidth, this.pixelHeight, false);
    this.renderer.domElement.style.width = `${w}px`;
    this.renderer.domElement.style.height = `${h}px`;

    const aspect = w / h;
    this.camera.aspect = aspect;
    // Portrait wants a taller vertical FOV so the hoop and the shooter both fit.
    this.camera.fov = aspect < 0.75 ? 52 : 42;
    this.camera.updateProjectionMatrix();

    for (const s of this.systems) s.resize?.(this.pixelWidth, this.pixelHeight, this);
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.clock.start();
    const tick = () => {
      this.rafId = requestAnimationFrame(tick);
      this.step();
    };
    this.rafId = requestAnimationFrame(tick);
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.rafId);
  }

  private step(): void {
    const t0 = performance.now();

    if (this.resizeQueued) {
      this.resizeQueued = false;
      this.applyResize();
    }

    const raw = Math.min(this.clock.getDelta(), 0.1);
    const dt = raw * this.timeScale;
    this.elapsed += dt;
    this.frame++;

    this.input.update(raw);

    if (!this.paused) {
      this.accumulator += dt;
      const step = PHYSICS.fixedStep;
      let steps = 0;
      while (this.accumulator >= step && steps < PHYSICS.maxSubSteps) {
        for (const s of this.systems) s.simulate?.(step, this);
        this.accumulator -= step;
        steps++;
      }
      // Avoid a death spiral if we cannot keep up.
      if (steps === PHYSICS.maxSubSteps) this.accumulator = 0;
    }

    const alpha = this.accumulator / PHYSICS.fixedStep;
    for (const s of this.systems) s.update?.(dt, alpha, this);

    if (this.renderOverride) this.renderOverride(dt);
    else this.renderer.render(this.scene, this.camera);

    for (const s of this.systems) s.lateUpdate?.(dt, this);

    this.input.postUpdate();

    const t1 = performance.now();
    this.frameMs += ((t1 - t0) - this.frameMs) * 0.08;
    if (this.governor.update(t1 - t0, raw)) {
      this.resizeQueued = true;
      this.bus.emit('qualityChanged', {
        tier: this.quality.tier,
        renderScale: this.governor.renderScale,
      });
    }
  }

  dispose(): void {
    this.stop();
    for (const s of this.systems) s.dispose?.();
    this.systems = [];
    this.input.dispose();
    this.renderer.dispose();
  }
}
