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
import { COURT, LIGHT_RIG } from '../core/Constants';

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

/** One aimed pool fixture: where it hangs and where its axis points. */
export interface PoolAim {
  readonly from: Vector3;
  readonly at: Vector3;
}

/**
 * Placement for the *pool* — the analytic term that gives the floor its gentle
 * centre-bright falloff.
 *
 * The version this replaces picked six pods straight off `rigSpotPositions`,
 * which walks the full 44 m catwalk run: two of the six ended up at |x| ≈ 19 m,
 * outside the playing surface entirely, and every cone was aimed at a different
 * scattered point. Six 26°-half-angle cones aimed at six scattered points is a
 * recipe for exactly the failure the rubric names in §10 — readable oval pools
 * with edges on them.
 *
 * The rule now: **every pool fixture is aimed at the court's long axis and every
 * cone is wide enough to cover the whole floor**, so the cone edge never lands
 * anywhere near the hardwood and what remains is the smooth `smoothstep`
 * shoulder that three's spot attenuation gives for free at `penumbra = 1`. Their
 * sum is a single soft dome over the middle of the court that fades toward the
 * apron — a falloff, not a pool with a rim.
 *
 * Fixtures still alternate sidelines and still hang at the real catwalk height,
 * so the *direction* the shaping comes from is honest even though the shape
 * itself is deliberately featureless.
 */
export function poolAims(count: number): PoolAim[] {
  if (count <= 0) return [];
  const { bankHeight, sideline } = LIGHT_RIG;
  // Spread across the playing surface, not across the whole catwalk run.
  const span = COURT.halfLength * 0.78;
  const out: PoolAim[] = [];
  for (let i = 0; i < count; i++) {
    const t = count === 1 ? 0.5 : i / (count - 1);
    const x = -span + t * span * 2;
    const z = (i % 2 === 0 ? 1 : -1) * sideline.z;
    out.push({
      from: new Vector3(x, bankHeight, z),
      // Aimed at the long axis, three quarters of the way in from the fixture:
      // near enough to vertical that the cone is centred on the hardwood, tipped
      // just enough that the two sidelines cross-light instead of stacking.
      at: new Vector3(x * 0.55, 0, z * -0.12),
    });
  }
  return out;
}
