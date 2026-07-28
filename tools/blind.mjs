/**
 * Builds a blind side-by-side comparison sheet.
 *
 *   node tools/blind.mjs shots/r2/rim.png shots/r3/rim.png \
 *     --out shots/compare/rim.png --key <scratchpad>/rim.key.json
 *
 * The two frames are placed left and right in an order chosen by a seeded
 * shuffle, with no labels drawn on the image. The mapping from side to source
 * is written to `--key`, which is deliberately kept outside the repository so a
 * reviewer working in the tree cannot find it. The reviewer answers "left" or
 * "right"; whoever holds the key decodes that into a winner afterwards.
 *
 * This is the honest version of the comparison this project wants to run.
 * Comparing against real NBA 2K captures is not possible here — there are no
 * such frames in this environment and pulling copyrighted game captures into
 * the repository is not something to do. What this does test is whether an
 * iteration actually improved on the one before it, with the reviewer unable to
 * favour the newer frame simply because it is newer. Absolute quality against
 * the target is judged separately, through the rubric.
 *
 * Options:
 *   --out PATH     composite to write (default shots/compare/blind.png)
 *   --key PATH     where to write the answer key (default alongside --out)
 *   --seed N       shuffle seed; the same seed always produces the same order
 *   --crop x0,y0,x1,y1   compare a fractional detail region instead of the frame
 *   --height N     scale each panel to this height (default 1400)
 */

import { mkdirSync } from 'node:fs';
import { writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { readPng } from './frame/png.mjs';
import { blank, blit, crop, resize, writePng } from './frame/pngwrite.mjs';

const args = process.argv.slice(2);
const positional = args.filter((a) => !a.startsWith('--') && !isFlagValue(a));

function isFlagValue(a) {
  const i = args.indexOf(a);
  return i > 0 && args[i - 1].startsWith('--');
}
function flag(name, fallback) {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
}

if (positional.length !== 2) {
  process.stderr.write('usage: node tools/blind.mjs <frameA.png> <frameB.png> [--out P] [--key P] [--seed N]\n');
  process.exit(2);
}

const [srcA, srcB] = positional;
const out = resolve(flag('out', 'shots/compare/blind.png'));
const keyPath = resolve(flag('key', out.replace(/\.png$/, '.key.json')));
const seed = Number(flag('seed', '1'));
const panelHeight = Number(flag('height', '1400'));
const cropBox = flag('crop', null)
  ?.split(',')
  .map(Number);

/** Deterministic, seeded — so a comparison can be rebuilt exactly. */
function coinFlip(n) {
  let h = (n ^ 0x9e3779b9) >>> 0;
  h ^= h << 13;
  h >>>= 0;
  h ^= h >> 17;
  h ^= h << 5;
  h >>>= 0;
  return (h & 1) === 1;
}

function load(path) {
  let img = readPng(path);
  if (cropBox && cropBox.length === 4) img = crop(img, cropBox);
  const scale = panelHeight / img.height;
  return resize(img, Math.max(1, Math.round(img.width * scale)), panelHeight);
}

const swap = coinFlip(seed);
const leftSrc = swap ? srcB : srcA;
const rightSrc = swap ? srcA : srcB;

const left = load(leftSrc);
const right = load(rightSrc);

const gutter = 24;
const pad = 16;
const sheet = blank(left.width + right.width + gutter + pad * 2, panelHeight + pad * 2, [24, 24, 26]);
blit(sheet, left, pad, pad);
blit(sheet, right, pad + left.width + gutter, pad);

mkdirSync(dirname(out), { recursive: true });
writePng(out, sheet);

mkdirSync(dirname(keyPath), { recursive: true });
writeFileSync(
  keyPath,
  JSON.stringify({ seed, left: leftSrc, right: rightSrc, out, crop: cropBox ?? null }, null, 2),
);

process.stdout.write(
  `wrote ${out} (${sheet.width}x${sheet.height})\n` +
    `key   ${keyPath}\n` +
    'Show only the composite to the reviewer. Ask which side is better and why; decode with the key.\n',
);
