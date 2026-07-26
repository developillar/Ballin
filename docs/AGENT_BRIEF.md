# Ballin — contributor brief

You are building **Ballin**, a portrait-orientation mobile basketball game on
Three.js whose visual bar is *NBA 2K*. Not "good for a web game." Actually
comparable, frame for frame, to a shipped AAA sports title.

## Non-negotiables

1. **Portrait first.** The target window is 9:19.5 (430×932 CSS px, DPR 2–3).
   Every framing, HUD element and control affordance is designed for a thumb
   reaching from the bottom of a phone. Never assume landscape.
2. **No external assets.** There is no CDN, no model download, no texture pack.
   Every texture, mesh, animation and sound is generated in code. This is a
   hard constraint — it is also why the bar is procedural craft, not sourcing.
3. **Mobile budget.** 60 fps on a 2021 mid-range phone at the `medium` tier.
   Read budgets from `src/core/Quality.ts`; never hard-code a cost.
4. **One source of truth for geometry.** `src/core/Constants.ts`. If you need a
   dimension that is not there, add it there.
5. **Systems, not globals.** Register with the engine (`src/core/Engine.ts`),
   implement `init` / `simulate` / `update` / `resize` / `dispose`. Cross-system
   reads go through `engine.get<T>('name')` with a narrow structural type.
   Cross-system *notifications* go through `engine.bus` (`src/core/Events.ts`).
6. **Fixed step for physics.** `simulate(step)` runs at 240 Hz and must be
   deterministic. Presentation-only work belongs in `update(dt, alpha)`.

## Working rules for parallel agents

- **You own the files listed in your task and nothing else.** Other agents are
  editing other files at the same time. If you genuinely need a change outside
  your files, keep it to an additive, non-breaking edit and say so in your
  report.
- Verify with `npx tsc --noEmit` before you finish. A type error blocks everyone.
- To see your work, build and capture into **your own** directories so you do
  not collide:

  ```sh
  npx vite build --outDir dist-<yourname> --emptyOutDir
  node tools/capture.mjs --dist dist-<yourname> --port <yourport> \
      --dir shots/<yourname> --scenes rim,floor --quality high
  ```

  Then `Read` the PNGs. Rendering here is software-rasterised, so a capture
  takes 1–3 minutes and reported FPS is meaningless — judge *pixels*, not
  frame time. Capture 1–2 scenes at a time.
- Add `dist-*` output to nothing — `.gitignore` already covers `dist*`.

## The visual target, concretely

What makes a 2K frame read as 2K, in rough order of impact:

- **Light has shape.** Broadcast arenas are dark bowls with bright, *directional*
  overhead banks. Players are lit from above and rimmed from behind; the crowd
  falls into near-black. Flat ambient light is the single fastest way to look
  like a web demo.
- **Materials are specific.** Hardwood has anisotropic clear-coat streaks along
  the grain. Jersey mesh has visible weave and sheen falloff. Skin has broad
  subsurface warmth in shadow terminators and sharp sweat specular on top of it.
  Leather has pebbling that catches light at grazing angles. Nothing is a flat
  colour with one roughness value.
- **Contact is believable.** Feet plant and shadows meet them. The net reacts.
  The rim flexes. Nothing floats.
- **Motion has weight.** Bodies anticipate and follow through. Cameras have
  inertia and lead the action. Nothing snaps.
- **The frame is graded.** Highlight bloom, a slight cool shadow / warm highlight
  split, gentle vignette, and a touch of grain. An ungraded render looks like a
  render.

## Repo map

| Path | Owner concern |
|---|---|
| `src/core/` | Engine, input, quality tiers, constants, math, event bus |
| `src/render/` | Lighting, post-processing, camera, VFX |
| `src/world/` | Court, arena bowl and crowd, hoop and net |
| `src/textures/` | Procedural texture bakes shared across systems |
| `src/entities/` | Players: mesh generation, skinning, rigs |
| `src/anim/` | Animation clips, blend trees, IK |
| `src/physics/` | Ball dynamics and collision |
| `src/game/` | Rules, possession, shooting, AI |
| `src/ui/` | HUD and controls |
| `src/audio/` | Procedural sound |
| `tools/` | Screenshot / capture harness |
