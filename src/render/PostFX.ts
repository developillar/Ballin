/**
 * Post-processing stack. Renders the scene to an HDR target and composites
 * bloom, ambient occlusion, motion blur, depth of field, grain, vignette and
 * antialiasing in as few full-screen passes as the tier allows.
 *
 * Owned by the post-processing agent. Until it lands, this system stays
 * pass-through so the game always renders.
 */

import type { Engine, System } from '../core/Engine';

export class PostFXSystem implements System {
  readonly name = 'postfx';
  readonly order = 90;

  enabled = false;

  init(engine: Engine): void {
    // Pass-through until the post agent installs the real composer.
    void engine;
  }

  resize(_w: number, _h: number, _engine: Engine): void {}

  dispose(): void {}
}
