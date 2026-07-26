/**
 * The physical light rig: where the fixtures are.
 *
 * The arena builds the *visible* hardware — catwalks, truss, emissive pods — up
 * in the rafters. This module reproduces that same plot in numbers so the two
 * halves of the lighting agree with it: `envArena.ts` paints each pod as a
 * rectangular emitter (so chrome and glass reflect a recognisable grid of
 * quads), and `Lighting.ts` hangs its analytic banks off the same clusters. The
 * layout lives in `LIGHT_RIG` in Constants, which mirrors the arena's
 * `ARENA.riggingY` / `LIGHT_BANKS`.
 *
 * Owned by the lighting agent.
 */

import { Vector3 } from 'three';
import { LIGHT_RIG } from '../core/Constants';

export interface RigBank {
  /** Pod centres in world space, in order along the run. */
  readonly pods: Vector3[];
  /** Relative output of a pod in this run. The wash banks are much dimmer. */
  readonly gain: number;
  /** Colour temperature bias applied on top of the neutral bank white. */
  readonly tint: readonly [number, number, number];
}

function run(
  from: Vector3,
  to: Vector3,
  count: number,
  gain: number,
  tint: readonly [number, number, number],
): RigBank {
  const pods: Vector3[] = [];
  for (let i = 0; i < count; i++) {
    const t = count === 1 ? 0.5 : i / (count - 1);
    pods.push(new Vector3().lerpVectors(from, to, t));
  }
  return { pods, gain, tint };
}

/** Every fixture run in the rafters, in the order the arena builds them. */
export function rigBanks(): RigBank[] {
  const { bankHeight, sideline, cross, wash } = LIGHT_RIG;
  const sh = sideline.length * 0.5;
  const ch = cross.length * 0.5;
  const wh = wash.length * 0.5;
  return [
    run(
      new Vector3(-sh, bankHeight, sideline.z),
      new Vector3(sh, bankHeight, sideline.z),
      sideline.pods,
      1.0,
      [0.96, 0.975, 1.0],
    ),
    run(
      new Vector3(-sh, bankHeight, -sideline.z),
      new Vector3(sh, bankHeight, -sideline.z),
      sideline.pods,
      0.82,
      [1.0, 0.965, 0.9],
    ),
    run(
      new Vector3(cross.x, bankHeight - cross.drop, -ch),
      new Vector3(cross.x, bankHeight - cross.drop, ch),
      cross.pods,
      0.94,
      [1.0, 0.972, 0.93],
    ),
    run(
      new Vector3(-cross.x, bankHeight - cross.drop, -ch),
      new Vector3(-cross.x, bankHeight - cross.drop, ch),
      cross.pods,
      0.94,
      [1.0, 0.972, 0.93],
    ),
    run(
      new Vector3(-wh, bankHeight + wash.rise, wash.z),
      new Vector3(wh, bankHeight + wash.rise, wash.z),
      wash.pods,
      0.34,
      [0.86, 0.92, 1.0],
    ),
    run(
      new Vector3(-wh, bankHeight + wash.rise, -wash.z),
      new Vector3(wh, bankHeight + wash.rise, -wash.z),
      wash.pods,
      0.34,
      [0.86, 0.92, 1.0],
    ),
  ];
}

/**
 * Positions for the analytic pool spots. Picked off the two sideline runs and
 * spread evenly, because those are the fixtures a real rig aims across the
 * court — the wash banks point at the seats and the cross banks at the glass.
 */
export function rigSpotPositions(count: number): Vector3[] {
  if (count <= 0) return [];
  const banks = rigBanks();
  const out: Vector3[] = [];
  for (let i = 0; i < count; i++) {
    // Alternate sidelines so even two spots cross-light instead of stacking.
    const side = i % 2;
    const along = count <= 2 ? (i === 0 ? 0.32 : 0.68) : (i + 0.5) / count;
    const podsOnSide = banks[side].pods;
    const idx = Math.min(podsOnSide.length - 1, Math.round(along * (podsOnSide.length - 1)));
    out.push(podsOnSide[idx].clone());
  }
  return out;
}
