/**
 * Frame analyser for the visual review loop.
 *
 *   node tools/analyze.mjs shots/r2/gameplay.png
 *   node tools/analyze.mjs shots/r2 --json > shots/r2/metrics.json
 *   node tools/analyze.mjs shots/r2/rim.png --region net=0.42,0.30,0.58,0.44
 *   node tools/analyze.mjs shots/r2/floor.png --line shadow=0.40,0.82,0.46,0.86
 *
 * Prints the numbers `docs/AAA_RUBRIC.md` asks for, with the target range and a
 * pass/fail mark beside each one it can judge on its own. Criteria that need a
 * human to say *where* to measure (is that region crowd or is it a tunnel?) are
 * reported as raw measurements with no verdict — the reviewer supplies the
 * region with --region and the rubric supplies the range.
 *
 * Verdicts here are a floor, not a ceiling. Passing every automated check means
 * the frame is not obviously broken; it does not mean it looks like NBA 2K.
 */

import { readdirSync, statSync } from 'node:fs';
import { join, basename } from 'node:path';
import { readPng } from './frame/png.mjs';
import {
  bandProfile,
  bloomFalloff,
  chromaticAberration,
  colorSplit,
  detailOctaves,
  exposureDiscipline,
  flattestWindow,
  grainEstimate,
  histogram,
  lineProfile,
  regionStats,
  stops,
  tileGrid,
  vignette,
} from './frame/metrics.mjs';

const args = process.argv.slice(2);
const asJson = args.includes('--json');
const quiet = args.includes('--quiet');
const target = args.find((a) => !a.startsWith('--')) ?? 'shots';

/** `--region name=x0,y0,x1,y1`, repeatable. */
function parsePairs(flag) {
  const out = {};
  for (let i = 0; i < args.length; i++) {
    if (args[i] !== flag) continue;
    const [name, nums] = (args[i + 1] ?? '').split('=');
    if (!name || !nums) continue;
    const v = nums.split(',').map(Number);
    if (v.length === 4 && v.every((n) => Number.isFinite(n))) out[name] = v;
  }
  return out;
}

const regions = parsePairs('--region');
const lines = parsePairs('--line');

/**
 * Ranges the analyser can check without being told where to look.
 * `[min, max, reference, advisory?]`; null bounds mean unbounded on that side.
 *
 * Advisory checks are ones the measurement cannot fully separate from scene
 * content. They print WARN instead of FAIL and must not be used to hold a frame
 * back on their own — a reviewer has to look. The vignette and grain estimates
 * are both in this category on a rendered arena frame: dark bowl walls at the
 * frame edges read as vignetting, and aliasing on high-contrast geometry reads
 * as grain. Capturing the dedicated `flatfield` scene, where the post chain
 * processes a uniform mid-grey, turns both into exact measurements.
 */
const CHECKS = {
  clippedPct: [null, 1.5, '§1.5'],
  crushedPct: [null, 3.0, '§1.5'],
  flattestStddev: [6, null, '§1.1'],
  // §8.4 quotes the corner (normalised radius 1.0) at 10–22% down with the
  // falloff starting near 55%. The measurement below is taken at the left and
  // right edges at mid-height, which is r ≈ 0.71 — roughly a third of the way
  // through that curve. Hence the scaled band rather than the quoted one.
  edgeDarkerPct: [3, 9, '§8.4 scaled to r≈0.71', true],
  edgeDesatPct: [1, 6, '§8.4 scaled', true],
  grainRms: [1.5, 4.0, '§8.5', true],
  shadowBlueMinusRed: [5, 14, '§8.3'],
  highlightRedMinusBlue: [4, 12, '§8.3'],
  aberrationRefPx: [0, 2.0, '§8.6'],
  histogramPeaks: [null, 1, '§1.5'],
};

/** True once any measurement has failed a non-advisory check. */
let failed = false;

function verdict(key, value) {
  const range = CHECKS[key];
  if (!range || value === null || value === undefined) return '';
  const [lo, hi, ref, advisory] = range;
  const ok = (lo === null || value >= lo) && (hi === null || value <= hi);
  if (!ok && !advisory) failed = true;
  const bound = `${lo === null ? '' : lo}${lo !== null && hi !== null ? '–' : ''}${hi === null ? '+' : hi}`;
  const mark = ok ? 'PASS' : advisory ? 'WARN' : 'FAIL';
  return `${mark}  (target ${bound}, ${ref})`;
}

function analyse(path) {
  const img = readPng(path);
  const whole = regionStats(img, [0, 0, 1, 1]);
  const exposure = exposureDiscipline(img);
  const flat = flattestWindow(img);
  const vig = vignette(img);
  const split = colorSplit(img);
  const grain = grainEstimate(img);
  const ca = chromaticAberration(img);
  const hist = histogram(img);
  const bands = bandProfile(img);
  const bloom = bloomFalloff(img);

  // The squint pass, mechanised: the brightest and darkest horizontal bands and
  // the stop gap between them. In a correct arena frame the bright band is the
  // floor and the dark band is the bowl.
  const sortedBands = [...bands].sort((a, b) => a.mean - b.mean);
  const darkest = sortedBands[0];
  const brightest = sortedBands[sortedBands.length - 1];

  const named = {};
  for (const [name, box] of Object.entries(regions)) named[name] = regionStats(img, box);

  const profiles = {};
  for (const [name, seg] of Object.entries(lines)) {
    profiles[name] = lineProfile(img, [seg[0], seg[1]], [seg[2], seg[3]]);
  }

  return {
    file: basename(path),
    size: [img.width, img.height],
    whole,
    exposure,
    flatField: flat,
    vignette: vig,
    colorSplit: split,
    grain,
    chromaticAberration: ca,
    histogram: hist,
    bands,
    bandExtremes: {
      darkest,
      brightest,
      stopGap: +stops(brightest.mean, Math.max(darkest.mean, 0.5)).toFixed(2),
    },
    bloom,
    detailOctaves: detailOctaves(img),
    tiles: tileGrid(img),
    regions: named,
    lines: profiles,
  };
}

function report(m) {
  const L = [];
  L.push(`\n=== ${m.file}  ${m.size[0]}x${m.size[1]} ===`);
  L.push(
    `frame        mean ${m.whole.mean}  sd ${m.whole.stddev}  p01 ${m.whole.p01}  p50 ${m.whole.p50}  p99 ${m.whole.p99}  max ${m.whole.max}`,
  );
  L.push(`             mean RGB ${m.whole.meanRGB.join('/')}  sat ${m.whole.meanSat}  hue ${m.whole.meanHue}`);
  L.push(`clipped      ${m.exposure.clippedPct}%   ${verdict('clippedPct', m.exposure.clippedPct)}`);
  L.push(`crushed      ${m.exposure.crushedPct}%   ${verdict('crushedPct', m.exposure.crushedPct)}`);
  L.push(
    `flat field   flattest ${m.flatField.windowPx}px window sd ${m.flatField.flattestStddev} at ` +
      `(${m.flatField.flattestAt?.x},${m.flatField.flattestAt?.y}) mean ${m.flatField.flattestAt?.mean}   ` +
      verdict('flattestStddev', m.flatField.flattestStddev),
  );
  L.push(
    `histogram    ${m.histogram.peaks.length} peak(s) at ${m.histogram.peaks.map((p) => p.centre).join(', ') || '—'}   ` +
      verdict('histogramPeaks', m.histogram.peaks.length),
  );
  L.push(
    `squint       brightest band y=${m.bandExtremes.brightest.y} mean ${m.bandExtremes.brightest.mean}  |  ` +
      `darkest band y=${m.bandExtremes.darkest.y} mean ${m.bandExtremes.darkest.mean}  |  gap ${m.bandExtremes.stopGap} stops`,
  );
  L.push(`vignette     edges ${m.vignette.edgeDarkerPct}% darker  ${verdict('edgeDarkerPct', m.vignette.edgeDarkerPct)}`);
  L.push(`             edges ${m.vignette.edgeDesatPct}% desat   ${verdict('edgeDesatPct', m.vignette.edgeDesatPct)}`);
  L.push(`  radial     ` + m.vignette.profile.map((p) => `${p.r}:${p.mean}`).join('  '));
  L.push(
    `grade split  shadows B-R ${m.colorSplit.shadows.blueMinusRed} (hue ${m.colorSplit.shadows.hue}, sat ${m.colorSplit.shadows.meanSat})  ` +
      verdict('shadowBlueMinusRed', m.colorSplit.shadows.blueMinusRed),
  );
  L.push(
    `             highs   R-B ${m.colorSplit.highlights.redMinusBlue} (hue ${m.colorSplit.highlights.hue}, sat ${m.colorSplit.highlights.meanSat})  ` +
      verdict('highlightRedMinusBlue', m.colorSplit.highlights.redMinusBlue),
  );
  L.push(`grain        rms ${m.grain.rms}  chroma ${m.grain.chromaRms}   ${verdict('grainRms', m.grain.rms)}`);
  L.push(
    `aberration   ${m.chromaticAberration.refSeparationPx} ref-px corner separation   ` +
      verdict('aberrationRefPx', m.chromaticAberration.refSeparationPx),
  );
  L.push(
    `bloom        peak ${m.bloom.peak} at (${m.bloom.at.x},${m.bloom.at.y})  falloff ` +
      m.bloom.profile.map((p) => `${p.refPx}:${p.mean}`).join('  '),
  );
  L.push(`detail       ` + m.detailOctaves.map((o) => `${o.scalePx}px:${o.rms}`).join('  '));

  L.push(`bands        ` + m.bands.map((b) => b.mean.toFixed(0).padStart(4)).join(''));
  L.push(`  (stddev)   ` + m.bands.map((b) => b.stddev.toFixed(0).padStart(4)).join(''));

  L.push('tiles (mean luminance, 8 cols x 16 rows, top to bottom)');
  for (const row of m.tiles) L.push('  ' + row.map((t) => String(Math.round(t.mean)).padStart(5)).join(''));

  for (const [name, r] of Object.entries(m.regions)) {
    L.push(
      `region ${name.padEnd(10)} mean ${r.mean}  sd ${r.stddev}  RGB ${r.meanRGB.join('/')}  ` +
        `sat ${r.meanSat}  hue ${r.meanHue}  clip ${r.clippedPct}%  crush ${r.crushedPct}%`,
    );
  }
  const names = Object.keys(m.regions);
  for (let i = 0; i < names.length; i++) {
    for (let j = i + 1; j < names.length; j++) {
      const a = m.regions[names[i]];
      const b = m.regions[names[j]];
      const hi = a.mean >= b.mean ? names[i] : names[j];
      const lo = a.mean >= b.mean ? names[j] : names[i];
      L.push(
        `  ratio ${hi} over ${lo}: ${stops(Math.max(a.mean, b.mean), Math.max(Math.min(a.mean, b.mean), 0.5)).toFixed(2)} stops`,
      );
    }
  }

  for (const [name, p] of Object.entries(m.lines)) {
    L.push(
      `line ${name.padEnd(12)} len ${p.lengthPx}px  min ${p.min}  max ${p.max}  ` +
        `transition ${p.transitionPx}px (${p.transitionRefPx} ref-px)`,
    );
    L.push('  ' + p.values.map((v) => v.toFixed(0).padStart(4)).join(''));
  }

  return L.join('\n');
}

const files = [];
const st = statSync(target);
if (st.isDirectory()) {
  for (const f of readdirSync(target).sort()) if (f.endsWith('.png')) files.push(join(target, f));
} else {
  files.push(target);
}

const results = files.map(analyse);
// Always build the reports: that is what evaluates the checks and sets the exit
// status, so `--json` and `--quiet` still tell a caller whether anything failed.
const text = results.map(report).join('\n');

if (asJson) process.stdout.write(JSON.stringify(results.length === 1 ? results[0] : results, null, 2));
else if (!quiet) process.stdout.write(text + '\n');

if (!asJson && !quiet) {
  process.stdout.write(
    `\n${files.length} frame(s) analysed — ${failed ? 'at least one hard check FAILED' : 'all hard checks passed'}.\n` +
      'Hard checks are a floor. A frame that passes them is not broken; whether it looks\n' +
      'like NBA 2K is a question for the rubric and a reviewer.\n',
  );
}
process.exitCode = failed ? 1 : 0;
