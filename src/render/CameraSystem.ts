/**
 * Broadcast camera. Portrait framing means the vertical axis is precious: the
 * camera sits lower and closer than a landscape sports game would, tracks the
 * ball with spring damping, and leads the action so the hoop stays in the
 * upper third while the ball-handler holds the lower third.
 *
 * Owned by the camera agent.
 */

import { Vector3 } from 'three';
import type { Engine, System } from '../core/Engine';
import { HOOP, basketX } from '../core/Constants';
import { clamp, damp } from '../core/MathX';

type PoseName = 'play' | 'rim' | 'closeup' | 'arena' | 'floor' | 'replay';

interface Pose {
  position: Vector3;
  look: Vector3;
  fov: number;
}

const POSES: Record<PoseName, Pose> = {
  play: {
    position: new Vector3(basketX(1) - 13.5, 4.6, 6.2),
    look: new Vector3(basketX(1) - 4.5, 2.1, 0),
    fov: 52,
  },
  rim: {
    position: new Vector3(basketX(1) - 3.4, 3.55, 2.5),
    look: new Vector3(basketX(1), HOOP.rimHeight - 0.15, 0),
    fov: 44,
  },
  closeup: {
    position: new Vector3(basketX(1) - 7.4, 1.85, 2.1),
    look: new Vector3(basketX(1) - 8.4, 1.5, 0.4),
    fov: 38,
  },
  arena: {
    position: new Vector3(-6, 12.5, 22),
    look: new Vector3(2, 3.4, 0),
    fov: 58,
  },
  floor: {
    position: new Vector3(-9, 0.42, 5.4),
    look: new Vector3(basketX(1) - 2, 2.2, 0),
    fov: 56,
  },
  replay: {
    position: new Vector3(basketX(1) - 5.5, 2.4, -6.5),
    look: new Vector3(basketX(1) - 1, 2.8, 0),
    fov: 46,
  },
};

export class CameraSystem implements System {
  readonly name = 'camera';
  readonly order = 50;

  private pos = POSES.play.position.clone();
  private look = POSES.play.look.clone();
  private targetPos = POSES.play.position.clone();
  private targetLook = POSES.play.look.clone();
  private targetFov = POSES.play.fov;
  private shake = 0;
  private shakeDecay = 1;
  private forced: PoseName | null = null;

  init(engine: Engine): void {
    engine.camera.position.copy(this.pos);
    engine.camera.lookAt(this.look);
    engine.bus.on('cameraShake', ({ amount, duration }) => {
      this.shake = Math.max(this.shake, amount);
      this.shakeDecay = 1 / Math.max(0.05, duration);
    });
  }

  /** Test hook used by the screenshot harness. */
  debugPose(name: string): void {
    if (name in POSES) {
      this.forced = name as PoseName;
      const p = POSES[this.forced];
      this.pos.copy(p.position);
      this.look.copy(p.look);
      this.targetPos.copy(p.position);
      this.targetLook.copy(p.look);
      this.targetFov = p.fov;
    } else {
      this.forced = null;
    }
  }

  update(dt: number, _alpha: number, engine: Engine): void {
    if (!this.forced) {
      const ball = engine.get<{ ballState?: { position: Vector3 } }>('ball');
      const b = ball?.ballState?.position;
      const hoop = new Vector3(basketX(1), HOOP.rimHeight, 0);
      const focus = b ? b.clone().lerp(hoop, 0.34) : hoop.clone();

      // Sit behind and below, framing the hoop high in a 9:19.5 window.
      const toHoop = hoop.clone().sub(focus).setY(0);
      const dist = clamp(9 + toHoop.length() * 0.55, 9, 17);
      toHoop.normalize();
      this.targetPos.set(
        focus.x - toHoop.x * dist,
        3.0 + clamp(focus.y * 0.32, 0, 2.1),
        focus.z - toHoop.z * dist + 4.6,
      );
      this.targetLook.copy(focus).setY(focus.y * 0.6 + 1.55);
      this.targetFov = engine.width / engine.height < 0.75 ? 52 : 42;
    }

    const rate = 5.2;
    this.pos.set(
      damp(this.pos.x, this.targetPos.x, rate, dt),
      damp(this.pos.y, this.targetPos.y, rate, dt),
      damp(this.pos.z, this.targetPos.z, rate, dt),
    );
    this.look.set(
      damp(this.look.x, this.targetLook.x, rate * 1.3, dt),
      damp(this.look.y, this.targetLook.y, rate * 1.3, dt),
      damp(this.look.z, this.targetLook.z, rate * 1.3, dt),
    );

    engine.camera.position.copy(this.pos);
    if (this.shake > 0.0005) {
      const t = engine.elapsed * 47;
      engine.camera.position.x += Math.sin(t * 1.7) * this.shake * 0.09;
      engine.camera.position.y += Math.sin(t * 2.3 + 1.1) * this.shake * 0.07;
      this.shake = Math.max(0, this.shake - this.shakeDecay * dt);
    }
    engine.camera.lookAt(this.look);
    if (Math.abs(engine.camera.fov - this.targetFov) > 0.01) {
      engine.camera.fov = damp(engine.camera.fov, this.targetFov, 4, dt);
      engine.camera.updateProjectionMatrix();
    }
  }
}
