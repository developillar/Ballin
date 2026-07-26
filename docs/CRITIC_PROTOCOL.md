# Critic protocol

This is the review half of the loop. `docs/AAA_RUBRIC.md` says what the target
looks like; this says how a reviewer decides whether a frame has reached it, and
what happens when it has not.

A reviewer following this protocol is not a collaborator. They did not write the
code, they do not care how hard it was, and they have no stake in the frame
passing. Their only job is to find the reason it is not good enough. If they
cannot find one, they say so and the item is done.

---

## 1. What is actually being compared

The stated goal for this project is a blind side-by-side against real NBA 2K
frames, picking which looks better without knowing which is which.

**That test cannot be run here, and no reviewer should claim to have run it.**
There are no NBA 2K captures in this environment, and pulling copyrighted game
footage into the repository to serve as reference is not something this project
does. Any review that reports "I compared this to NBA 2K and ours looked better"
is reporting something that did not happen.

Two things stand in for it, and between them they cover most of what the real
test would have caught:

**Absolute grading against the rubric (§3).** `docs/AAA_RUBRIC.md` was written
from measurable properties of real broadcast basketball and of the target's
rendering — light levels in lux, board dimensions, hue relationships, pixel
widths at a stated reference resolution. Grading a frame against those numbers
asks the same question the blind test asks, just through a written intermediary
instead of a side-by-side. The scale is anchored so that **9 means: a
knowledgeable observer shown this frame and a real NBA 2K frame, blind, could
not confidently say which was which.**

**Blind A/B against our own previous best (§5).** `tools/blind.mjs` builds a
composite of two frames in a seeded random order with no labels, and writes the
answer key somewhere the reviewer cannot see. This is a genuine blind
comparison — it just measures progress rather than absolute quality. It exists
because reviewers reliably prefer whichever frame they were told is newer, and
this removes that.

Report both. Never merge them into one number, and never describe the rubric
score as though it were the head-to-head result.

---

## 2. Order of work

Do these in order. Most failures are found in the first two steps, and arguing
about pebble grain while the exposure is wrong wastes everyone's time.

1. **Measure.** Run `node tools/analyze.mjs <frame-or-dir>`. Read the output
   before looking at the image. Note every FAIL. WARN lines are advisory —
   the measurement cannot fully separate them from scene content — so confirm
   or dismiss those by eye rather than quoting them as failures.

   To turn the advisory ones into real measurements, capture the `flatfield`
   scene alongside the rest:

   ```sh
   node tools/capture.mjs --dist <dist> --port <port> --dir <dir> --scenes gameplay,flatfield
   ```

   It replaces the scene with a uniform field and runs the post chain over it,
   so everything that is not flat in the result is something the stack did.
   Vignette, grain and chromatic aberration are then exact, and `analyze.mjs`
   enforces them as hard checks on that frame. Anyone reviewing §8 should
   capture it; without it, a vignette that does not exist reads as a 29% one.
2. **Squint.** Look at the frame at a glance. Is there a readable light shape?
   Is the bowl dark and the floor bright? Does anything separate from its
   background? A frame that fails here fails, whatever its measurements say.
3. **Material.** At full size: does every surface answer "what is this made of"
   with more than one cue — albedo, roughness variation, normal detail, and a
   specular shape that behaves?
4. **Forensic.** Go looking for the specific tells in rubric §10. Do not wait
   for them to announce themselves.

---

## 3. Scoring

Score each of the ten rubric categories 0–10.

| Score | Meaning |
|---|---|
| 0–3 | Missing, broken, or actively wrong |
| 4–5 | Present and recognisable. Reads as a competent web demo |
| 6–7 | Good. Reads as a solid console game from a generation ago |
| 8 | Very good. Would not look out of place next to the target, but a trained eye finds the seams |
| 9 | Blind-indistinguishable from a real NBA 2K frame to a knowledgeable observer |
| 10 | Reserved. Do not award it |

Four rules, and they are the whole point of the protocol:

- **Default down.** If you are between two scores, take the lower one.
- **An 8 or above must cite a measurement.** A number from `analyze.mjs`, a
  pixel width, a luminance ratio, a hue angle. "Looks great" does not support
  an 8. Without a citation the score is capped at 7.
- **Every score below 9 must name a specific, actionable defect** — what is
  wrong, where in the frame, and what it should be instead. A score with no
  defect attached is not a review.
- **The item's score is the minimum across its categories, not the mean.** One
  broken category is what a viewer will notice; averaging it away is how demos
  ship looking like demos.

## 4. Verdict

- **PASS** — every category relevant to this item scores ≥ 8, at least one
  scores ≥ 9, and no hard check in `analyze.mjs` fails.
- **ITERATE** — anything else. Return the ranked defect list; the highest-impact
  defect goes first, and it becomes the next round's first task.

A reviewer who returns PASS on the first round should re-read §3 and check they
have not simply failed to look. It happens, and it is the failure mode this
protocol exists to prevent.

## 5. Blind A/B

Once an item has a previous accepted frame to compare against:

```sh
node tools/blind.mjs <previous>.png <candidate>.png \
  --out shots/compare/<item>.png --key "$SCRATCH/<item>.key.json"
```

Show the reviewer **only** `shots/compare/<item>.png`. Ask: which side is
better, and name the three differences that decided it. The reviewer must not
open the key, the source directories, or git history — knowing which frame is
newer invalidates the result.

Decode afterwards. If the reviewer picked the older frame, the round is a
regression regardless of what the rubric score did, and it gets reverted or
fixed before anything else proceeds.

## 6. Reporting

A review returns:

- `scores` — the ten categories, each with its number and its one-line reason
- `min` — the lowest, which is the item's score
- `verdict` — PASS or ITERATE
- `defects` — ranked, each with location, what is wrong, and the target state
- `measurements` — the numbers cited in support of any score ≥ 8
- `blind` — which side was preferred and why, when a comparison was run

No praise. No summary of what the implementer did well. The implementer is not
the audience; the next round is.
