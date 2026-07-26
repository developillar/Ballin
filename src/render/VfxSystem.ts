/**
 * Gameplay VFX: sweat, floor dust, net ripple particles, swish flash, score
 * bursts and the speed-line treatment on dunks.
 *
 * Owned by the VFX agent.
 */

import { Group } from 'three';
import type { Engine, System } from '../core/Engine';

export class VfxSystem implements System {
  readonly name = 'vfx';
  readonly order = 70;

  group = new Group();

  init(engine: Engine): void {
    this.group.name = 'vfx';
    engine.scene.add(this.group);
  }
}
