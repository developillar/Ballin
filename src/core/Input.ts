/**
 * Portrait touch input. Left thumb drives a floating virtual stick; the right
 * thumb owns shoot / pass / special via tap, hold and directional flick. Every
 * gesture is also mapped to a keyboard equivalent so the game is testable in a
 * desktop browser and by the screenshot harness.
 */

import { clamp01 } from './MathX';

export interface StickState {
  x: number;
  y: number;
  magnitude: number;
  active: boolean;
  /** Set for one frame when the stick was flicked hard. */
  flick: boolean;
}

export interface ActionState {
  /** Currently held. */
  held: boolean;
  /** Went down this frame. */
  pressed: boolean;
  /** Came up this frame. */
  released: boolean;
  /** Seconds the button has been held. */
  holdTime: number;
  /** Normalised drag from the press origin while held. */
  dragX: number;
  dragY: number;
}

export type ActionName = 'shoot' | 'pass' | 'special' | 'defend';

const DEAD_ZONE = 0.14;
const STICK_RADIUS = 62;
const FLICK_SPEED = 900;

function emptyAction(): ActionState {
  return { held: false, pressed: false, released: false, holdTime: 0, dragX: 0, dragY: 0 };
}

export class Input {
  readonly move: StickState = { x: 0, y: 0, magnitude: 0, active: false, flick: false };
  readonly actions: Record<ActionName, ActionState> = {
    shoot: emptyAction(),
    pass: emptyAction(),
    special: emptyAction(),
    defend: emptyAction(),
  };

  /** Raw swipe on the right half, consumed by gesture recognisers. */
  swipe = { x: 0, y: 0, magnitude: 0, fired: false };

  private movePointer = -1;
  private moveOrigin = { x: 0, y: 0 };
  private moveCurrent = { x: 0, y: 0 };
  private moveVel = { x: 0, y: 0 };
  private lastMoveTime = 0;

  private actionPointer = new Map<number, { name: ActionName; x: number; y: number }>();
  private keys = new Set<string>();
  private keyEdge = new Set<string>();
  private keyRelease = new Set<string>();

  /** Screen-space anchor for the on-screen stick, in CSS pixels. */
  stickAnchor = { x: 0, y: 0, visible: false };
  stickKnob = { x: 0, y: 0 };

  private detachers: Array<() => void> = [];

  constructor(private readonly target: HTMLElement) {
    this.attach();
  }

  private attach(): void {
    const el = this.target;
    const opts: AddEventListenerOptions = { passive: false };

    const down = (e: PointerEvent) => this.onDown(e);
    const move = (e: PointerEvent) => this.onMove(e);
    const up = (e: PointerEvent) => this.onUp(e);
    const kd = (e: KeyboardEvent) => this.onKey(e, true);
    const ku = (e: KeyboardEvent) => this.onKey(e, false);
    const ctx = (e: Event) => e.preventDefault();

    el.addEventListener('pointerdown', down, opts);
    window.addEventListener('pointermove', move, opts);
    window.addEventListener('pointerup', up, opts);
    window.addEventListener('pointercancel', up, opts);
    window.addEventListener('keydown', kd);
    window.addEventListener('keyup', ku);
    el.addEventListener('contextmenu', ctx);

    this.detachers = [
      () => el.removeEventListener('pointerdown', down),
      () => window.removeEventListener('pointermove', move),
      () => window.removeEventListener('pointerup', up),
      () => window.removeEventListener('pointercancel', up),
      () => window.removeEventListener('keydown', kd),
      () => window.removeEventListener('keyup', ku),
      () => el.removeEventListener('contextmenu', ctx),
    ];
  }

  dispose(): void {
    for (const d of this.detachers) d();
    this.detachers = [];
  }

  /** Right-half zones, resolved at press time so the buttons can float. */
  private zoneFor(x: number, y: number): ActionName {
    const h = window.innerHeight;
    const w = window.innerWidth;
    const fromBottom = h - y;
    const fromRight = w - x;
    if (fromBottom > 260) return 'special';
    if (fromRight < 96 && fromBottom < 150) return 'pass';
    if (fromBottom > 165) return 'defend';
    return 'shoot';
  }

  private onDown(e: PointerEvent): void {
    this.target.setPointerCapture?.(e.pointerId);
    e.preventDefault();
    const x = e.clientX;
    const y = e.clientY;
    const leftHalf = x < window.innerWidth * 0.46;

    if (leftHalf && this.movePointer === -1) {
      this.movePointer = e.pointerId;
      this.moveOrigin = { x, y };
      this.moveCurrent = { x, y };
      this.moveVel = { x: 0, y: 0 };
      this.lastMoveTime = performance.now();
      this.stickAnchor = { x, y, visible: true };
      this.stickKnob = { x, y };
      return;
    }

    const name = this.zoneFor(x, y);
    this.actionPointer.set(e.pointerId, { name, x, y });
    const a = this.actions[name];
    a.pressed = true;
    a.held = true;
    a.holdTime = 0;
    a.dragX = 0;
    a.dragY = 0;
  }

  private onMove(e: PointerEvent): void {
    if (e.pointerId === this.movePointer) {
      e.preventDefault();
      const now = performance.now();
      const dt = Math.max(1, now - this.lastMoveTime) / 1000;
      this.moveVel.x = (e.clientX - this.moveCurrent.x) / dt;
      this.moveVel.y = (e.clientY - this.moveCurrent.y) / dt;
      this.lastMoveTime = now;
      this.moveCurrent = { x: e.clientX, y: e.clientY };

      let dx = this.moveCurrent.x - this.moveOrigin.x;
      let dy = this.moveCurrent.y - this.moveOrigin.y;
      const len = Math.hypot(dx, dy);
      // Drag the anchor along so the stick never pins to the screen edge.
      if (len > STICK_RADIUS) {
        const over = len - STICK_RADIUS;
        this.moveOrigin.x += (dx / len) * over;
        this.moveOrigin.y += (dy / len) * over;
        dx = (dx / len) * STICK_RADIUS;
        dy = (dy / len) * STICK_RADIUS;
      }
      this.stickAnchor.x = this.moveOrigin.x;
      this.stickAnchor.y = this.moveOrigin.y;
      this.stickKnob = { x: this.moveOrigin.x + dx, y: this.moveOrigin.y + dy };
      return;
    }

    const rec = this.actionPointer.get(e.pointerId);
    if (rec) {
      e.preventDefault();
      const a = this.actions[rec.name];
      a.dragX = (e.clientX - rec.x) / 110;
      a.dragY = (e.clientY - rec.y) / 110;
    }
  }

  private onUp(e: PointerEvent): void {
    if (e.pointerId === this.movePointer) {
      const speed = Math.hypot(this.moveVel.x, this.moveVel.y);
      if (speed > FLICK_SPEED) this.move.flick = true;
      this.movePointer = -1;
      this.stickAnchor.visible = false;
      this.moveVel = { x: 0, y: 0 };
      return;
    }
    const rec = this.actionPointer.get(e.pointerId);
    if (rec) {
      const a = this.actions[rec.name];
      a.held = false;
      a.released = true;
      this.actionPointer.delete(e.pointerId);
      const mag = Math.hypot(a.dragX, a.dragY);
      if (mag > 0.35) {
        this.swipe = { x: a.dragX / mag, y: a.dragY / mag, magnitude: mag, fired: true };
      }
    }
  }

  private onKey(e: KeyboardEvent, down: boolean): void {
    const k = e.key.toLowerCase();
    if (down) {
      if (!this.keys.has(k)) this.keyEdge.add(k);
      this.keys.add(k);
    } else {
      this.keys.delete(k);
      this.keyRelease.add(k);
    }
    if (['arrowup', 'arrowdown', 'arrowleft', 'arrowright', ' '].includes(k)) e.preventDefault();
  }

  private keyAxis(neg: string[], pos: string[]): number {
    let v = 0;
    if (neg.some((k) => this.keys.has(k))) v -= 1;
    if (pos.some((k) => this.keys.has(k))) v += 1;
    return v;
  }

  private bindKeyAction(name: ActionName, key: string, dt: number): void {
    const a = this.actions[name];
    if (this.keyEdge.has(key)) {
      a.pressed = true;
      a.held = true;
      a.holdTime = 0;
    }
    if (this.keyRelease.has(key) && a.held) {
      a.held = false;
      a.released = true;
    }
    if (this.keys.has(key)) a.held = true;
    if (a.held) a.holdTime += dt;
  }

  /** Call once per frame, before gameplay reads the state. */
  update(dt: number): void {
    // Touch stick.
    if (this.movePointer !== -1) {
      const dx = (this.stickKnob.x - this.stickAnchor.x) / STICK_RADIUS;
      const dy = (this.stickKnob.y - this.stickAnchor.y) / STICK_RADIUS;
      const mag = Math.min(1, Math.hypot(dx, dy));
      if (mag > DEAD_ZONE) {
        const scaled = (mag - DEAD_ZONE) / (1 - DEAD_ZONE);
        const inv = mag > 0 ? 1 / mag : 0;
        this.move.x = dx * inv * scaled;
        this.move.y = dy * inv * scaled;
        this.move.magnitude = scaled;
        this.move.active = true;
      } else {
        this.move.x = 0;
        this.move.y = 0;
        this.move.magnitude = 0;
        this.move.active = true;
      }
    } else {
      // Keyboard fallback.
      const kx = this.keyAxis(['a', 'arrowleft'], ['d', 'arrowright']);
      const ky = this.keyAxis(['w', 'arrowup'], ['s', 'arrowdown']);
      const mag = Math.min(1, Math.hypot(kx, ky));
      this.move.x = mag > 0 ? (kx / Math.max(1e-4, Math.hypot(kx, ky))) * mag : 0;
      this.move.y = mag > 0 ? (ky / Math.max(1e-4, Math.hypot(kx, ky))) * mag : 0;
      this.move.magnitude = mag;
      this.move.active = mag > 0;
    }

    for (const rec of this.actionPointer.values()) {
      this.actions[rec.name].holdTime += dt;
    }

    this.bindKeyAction('shoot', ' ', dt);
    this.bindKeyAction('pass', 'e', dt);
    this.bindKeyAction('special', 'q', dt);
    this.bindKeyAction('defend', 'shift', dt);
  }

  /** Call at the very end of the frame to clear one-shot edges. */
  postUpdate(): void {
    for (const key of Object.keys(this.actions) as ActionName[]) {
      const a = this.actions[key];
      a.pressed = false;
      a.released = false;
      if (!a.held) {
        a.holdTime = 0;
        a.dragX = 0;
        a.dragY = 0;
      }
    }
    this.move.flick = false;
    this.swipe.fired = false;
    this.keyEdge.clear();
    this.keyRelease.clear();
  }

  /** Test hook: the screenshot harness drives the game through this. */
  debugPress(name: ActionName): void {
    const a = this.actions[name];
    a.pressed = true;
    a.held = true;
    a.holdTime = 0;
  }

  debugRelease(name: ActionName): void {
    const a = this.actions[name];
    a.held = false;
    a.released = true;
  }
}
