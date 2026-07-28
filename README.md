# Ballin

A portrait-orientation mobile basketball game built on Three.js, aiming at the
visual and physical fidelity of a shipped AAA sports title.

Everything you see is generated in code. There are no downloaded models, no
texture packs, no HDR probes and no audio samples — the hardwood grain, the
pebbled leather, the arena environment map, the players, their animation and
the crowd noise are all synthesised at boot. That is a hard constraint of the
project, and it is the reason the interesting work here is procedural craft
rather than asset sourcing.

## Running it

```sh
npm install
npm run dev        # http://localhost:5173
npm run build
npm run preview
```

Open it on a phone, or in a desktop browser with the device toolbar set to a
9:19.5 viewport. Desktop keyboard controls mirror the touch scheme:

| Touch | Keyboard |
|---|---|
| Left-thumb floating stick | `WASD` / arrows |
| Shoot (hold to charge, release on the meter) | `Space` |
| Pass | `E` |
| Special / signature move | `Q` |
| Defend | `Shift` |

Append `?quality=low\|medium\|high\|ultra` to pin a quality tier instead of
letting the device heuristic pick one.

## Architecture

The engine is a small system registry with a split update:

- `simulate(step)` runs at a fixed 240 Hz and is deterministic. Ball dynamics,
  the net solver and the rules clock live here.
- `update(dt, alpha)` runs once per frame and interpolates. Animation, cameras,
  VFX and UI live here.

Systems never reach into each other's internals directly. They look each other
up by name with a narrow structural type (`engine.get<{ ballState }>('ball')`)
and they notify each other over a typed event bus (`src/core/Events.ts`).

Cost is never hard-coded. Every expensive decision — shadow resolution, crowd
count, net solver iterations, whether reflections and volumetrics are on — reads
a budget from the active quality tier in `src/core/Quality.ts`. On top of that,
a closed-loop governor trades render resolution before it drops features, so a
struggling device gets softer pixels rather than a visible downgrade mid-play.

| Path | Concern |
|---|---|
| `src/core/` | Engine, input, quality tiers, regulation constants, math, event bus |
| `src/render/` | Lighting rig and IBL, post-processing, broadcast camera, VFX |
| `src/world/` | Court, arena bowl and crowd, hoop and net |
| `src/textures/` | Procedural texture bakes |
| `src/entities/` | Skeleton, body mesh generation, player rigs |
| `src/anim/` | IK, poses, clips, blend trees |
| `src/physics/` | Ball dynamics and collision |
| `src/game/` | Rules, possession, shooting, AI |
| `src/ui/` | HUD and touch controls |
| `src/audio/` | Procedural synthesis |
| `src/dev/` | Development-only hooks, such as the flat-field test pattern |
| `tools/` | Capture harness, frame analyser, blind comparison sheets |

## The visual review loop

Quality is verified by looking *and* by measuring, in that order.

**Capture.** `tools/capture.mjs` boots the production bundle in a portrait phone
viewport, drives the game into a set of standard framings and writes PNGs:

```sh
npm run build
node tools/capture.mjs --dir shots/round1 --quality high
```

Scenes: `gameplay`, `arena`, `rim`, `closeup`, `floor`, `shot`, `swish`, `dunk`,
and `flatfield`.

**Measure.** `docs/AAA_RUBRIC.md` states its criteria as numbers — luminance
ranges, stop ratios, hue angles, pixel widths at a stated reference resolution —
and `tools/analyze.mjs` reports them:

```sh
node tools/analyze.mjs shots/round1
node tools/analyze.mjs shots/round1/rim.png --region net=0.42,0.30,0.58,0.44
```

Regions and line profiles are given in fractional frame coordinates, so a
criterion written against the reference frame applies at any capture size.
Checks the measurement cannot separate from scene content report `WARN` rather
than `FAIL`; capturing the `flatfield` scene, which runs the post chain over a
uniform field, turns the grade criteria into exact measurements.

**Compare.** `tools/blind.mjs` builds a side-by-side sheet of two frames in a
seeded random order with the answer key written outside the repository, so a
reviewer judging whether an iteration improved on the last one cannot be swayed
by knowing which is newer.

```sh
node tools/blind.mjs shots/round1/rim.png shots/round2/rim.png \
    --out shots/compare/rim.png --key "$SCRATCH/rim.key.json"
```

**Judge.** `docs/CRITIC_PROTOCOL.md` sets out how a frame is scored, what a
passing verdict requires, and what a review has to return. Its scale is anchored
so that a 9 means a knowledgeable observer could not pick our frame out of a
blind pair against the real thing.

Note that the capture harness rasterises in software, so the frame times it
reports are not representative of a real device. Judge pixels from captures and
performance from a real GPU.

## Regulation geometry

Court, hoop and ball dimensions come from `src/core/Constants.ts` and follow
NBA rules: a 94 × 50 ft floor, a 10 ft rim of 18 in inside diameter set 6 in off
a 6 × 3.5 ft board, a 23 ft 9 in three-point arc breaking to 22 ft straightaways
in the corners, and a size-7 ball at 0.1192 m radius and 0.624 kg.
