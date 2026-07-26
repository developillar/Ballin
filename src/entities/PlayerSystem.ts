/**
 * Players: procedurally generated, skinned, animated athletes.
 *
 * Owned by the character agent. The contract other systems rely on:
 *   - `players[]` with a stable index used by every gameplay event
 *   - each player exposes `position`, `facing`, `handAnchor`, `state`
 *   - `setPose()` drives the animation blend tree from gameplay intent
 */

import { Group, Object3D, Vector3 } from 'three';
import type { Engine, System } from '../core/Engine';
import { PLAYER } from '../core/Constants';
import { dampAngle } from '../core/MathX';

export type PlayerStance =
  | 'idle'
  | 'run'
  | 'sprint'
  | 'dribble'
  | 'triple-threat'
  | 'shoot'
  | 'jump'
  | 'land'
  | 'dunk'
  | 'layup'
  | 'defend'
  | 'block'
  | 'celebrate';

export interface PlayerRig {
  index: number;
  team: number;
  root: Group;
  position: Vector3;
  velocity: Vector3;
  facing: number;
  stance: PlayerStance;
  /** World-space point where the ball sits when this player holds it. */
  handAnchor: Object3D;
  /** 0..1 how far into the current animation. */
  phase: number;
  height: number;
  /** Vertical offset from a jump. */
  jumpY: number;
}

export class PlayerSystem implements System {
  readonly name = 'players';
  readonly order = 30;

  group = new Group();
  players: PlayerRig[] = [];

  init(engine: Engine): void {
    this.group.name = 'players';
    engine.scene.add(this.group);
    // The character agent builds the real roster here.
  }

  create(team: number, at: Vector3): PlayerRig {
    const root = new Group();
    root.position.copy(at);
    const handAnchor = new Object3D();
    handAnchor.position.set(0.32, PLAYER.height * 0.82, 0.22);
    root.add(handAnchor);
    this.group.add(root);

    const rig: PlayerRig = {
      index: this.players.length,
      team,
      root,
      position: at.clone(),
      velocity: new Vector3(),
      facing: 0,
      stance: 'idle',
      handAnchor,
      phase: 0,
      height: PLAYER.height,
      jumpY: 0,
    };
    this.players.push(rig);
    return rig;
  }

  update(dt: number): void {
    for (const p of this.players) {
      p.phase += dt;
      const target = Math.atan2(p.velocity.x, p.velocity.z);
      if (p.velocity.lengthSq() > 0.04) p.facing = dampAngle(p.facing, target, PLAYER.turnRate, dt);
      p.root.position.set(p.position.x, p.position.y + p.jumpY, p.position.z);
      p.root.rotation.y = p.facing;
    }
  }

  /** World position of a player's ball-carrying hand. */
  handPosition(p: PlayerRig, out = new Vector3()): Vector3 {
    p.handAnchor.updateWorldMatrix(true, false);
    return out.setFromMatrixPosition(p.handAnchor.matrixWorld);
  }
}
