/**
 * Frame measurements for the visual review loop.
 *
 * `docs/AAA_RUBRIC.md` states its criteria as numbers — "court mean 95–140",
 * "corners 10–22% darker", "flattest 200×200 region stddev > 6". This module
 * produces those numbers so a reviewer argues with measurements instead of
 * adjectives. Nothing here decides whether a frame is good; it only reports
 * what is in the pixels.
 *
 * Conventions:
 *  - Luminance is Rec.709 over *display* sRGB values, 0–255, matching how the
 *    rubric quotes every figure.
 *  - Regions are fractional boxes `[x0, y0, x1, y1]` in 0..1 so a criterion
 *    written against the 1080×2340 reference frame applies at any capture size.
 *  - Pixel distances are reported both raw and scaled to the reference frame,
 *    since the rubric's pixel figures assume 2340 px of height.
 */

const REFERENCE_HEIGHT = 2340;

export const luma = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

/** sRGB display value (0–255) to scene-linear 0..1. */
export function toLinear(v) {
  const c = v / 255;
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/** Difference between two display luminances expressed in photographic stops. */
export function stops(brighter, darker) {
  const a = Math.max(toLinear(brighter), 1e-6);
  const b = Math.max(toLinear(darker), 1e-6);
  return Math.log2(a / b);
}

/** HSV hue in degrees and saturation 0..1 from 8-bit RGB. */
export function hueSat(r, g, b) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const d = max - min;
  const sat = max === 0 ? 0 : d / max;
  let hue = 0;
  if (d > 0) {
    if (max === r) hue = 60 * (((g - b) / d) % 6);
    else if (max === g) hue = 60 * ((b - r) / d + 2);
    else hue = 60 * ((r - g) / d + 4);
  }
  return { hue: (hue + 360) % 360, sat };
}

/** Resolves a fractional box to integer pixel bounds, clamped to the frame. */
export function resolveBox(img, box) {
  const [fx0, fy0, fx1, fy1] = box;
  const x0 = Math.max(0, Math.min(img.width - 1, Math.round(fx0 * img.width)));
  const y0 = Math.max(0, Math.min(img.height - 1, Math.round(fy0 * img.height)));
  const x1 = Math.max(x0 + 1, Math.min(img.width, Math.round(fx1 * img.width)));
  const y1 = Math.max(y0 + 1, Math.min(img.height, Math.round(fy1 * img.height)));
  return { x0, y0, x1, y1, w: x1 - x0, h: y1 - y0 };
}

/** Scales a pixel measurement taken at this capture size to the reference frame. */
export const toRef = (img, px) => (px * REFERENCE_HEIGHT) / img.height;

/**
 * Everything worth knowing about one rectangle: exposure, spread, colour and
 * how much of it is pinned against either end of the range.
 */
export function regionStats(img, box) {
  const { x0, y0, x1, y1, w, h } = resolveBox(img, box);
  const n = w * h;
  const lums = new Float32Array(n);

  let sum = 0;
  let sumR = 0;
  let sumG = 0;
  let sumB = 0;
  let sumSat = 0;
  let clipped = 0;
  let crushed = 0;
  let i = 0;
  // Hue averages on the unit circle so 359° and 1° do not average to 180°.
  let hueX = 0;
  let hueY = 0;

  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const p = (y * img.width + x) * 3;
      const r = img.data[p];
      const g = img.data[p + 1];
      const b = img.data[p + 2];
      const l = luma(r, g, b);
      lums[i++] = l;
      sum += l;
      sumR += r;
      sumG += g;
      sumB += b;
      if (r >= 252 && g >= 252 && b >= 252) clipped++;
      if (l <= 4) crushed++;
      const { hue, sat } = hueSat(r, g, b);
      sumSat += sat;
      const rad = (hue * Math.PI) / 180;
      hueX += Math.cos(rad) * sat;
      hueY += Math.sin(rad) * sat;
    }
  }

  const mean = sum / n;
  let variance = 0;
  for (let k = 0; k < n; k++) {
    const d = lums[k] - mean;
    variance += d * d;
  }

  const sorted = Float32Array.from(lums).sort();
  const pct = (q) => sorted[Math.min(n - 1, Math.max(0, Math.round(q * (n - 1))))];

  return {
    box,
    pixels: n,
    mean: +mean.toFixed(2),
    stddev: +Math.sqrt(variance / n).toFixed(2),
    min: +sorted[0].toFixed(1),
    max: +sorted[n - 1].toFixed(1),
    p001: +pct(0.001).toFixed(1),
    p01: +pct(0.01).toFixed(1),
    p05: +pct(0.05).toFixed(1),
    p50: +pct(0.5).toFixed(1),
    p95: +pct(0.95).toFixed(1),
    p99: +pct(0.99).toFixed(1),
    meanRGB: [+(sumR / n).toFixed(1), +(sumG / n).toFixed(1), +(sumB / n).toFixed(1)],
    meanSat: +(sumSat / n).toFixed(3),
    meanHue: +(((Math.atan2(hueY, hueX) * 180) / Math.PI + 360) % 360).toFixed(1),
    clippedPct: +((clipped / n) * 100).toFixed(2),
    crushedPct: +((crushed / n) * 100).toFixed(2),
  };
}

/** Summed-area tables over luminance and luminance², for O(1) window queries. */
function integrals(img) {
  const { width: w, height: h } = img;
  const sum = new Float64Array((w + 1) * (h + 1));
  const sqr = new Float64Array((w + 1) * (h + 1));
  for (let y = 0; y < h; y++) {
    let rowSum = 0;
    let rowSqr = 0;
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 3;
      const l = luma(img.data[p], img.data[p + 1], img.data[p + 2]);
      rowSum += l;
      rowSqr += l * l;
      const o = (y + 1) * (w + 1) + (x + 1);
      sum[o] = sum[o - (w + 1)] + rowSum;
      sqr[o] = sqr[o - (w + 1)] + rowSqr;
    }
  }
  return { sum, sqr, stride: w + 1 };
}

/**
 * The rubric's "no large flat field" test: slides a window over the frame and
 * reports the least-textured placement. A region that samples to a single value
 * is untextured or unlit, and this is what finds it.
 */
export function flattestWindow(img, refWindow = 200, step = 16, bounds = [0, 0, 1, 1]) {
  const win = Math.max(8, Math.round((refWindow * img.height) / REFERENCE_HEIGHT));
  const { sum, sqr, stride } = integrals(img);
  const b = resolveBox(img, bounds);
  const area = win * win;

  let best = Infinity;
  let bestAt = null;
  let worst = -Infinity;
  const query = (t, x0, y0) =>
    t[(y0 + win) * stride + x0 + win] - t[y0 * stride + x0 + win] - t[(y0 + win) * stride + x0] + t[y0 * stride + x0];

  for (let y = b.y0; y + win <= b.y1; y += step) {
    for (let x = b.x0; x + win <= b.x1; x += step) {
      const s = query(sum, x, y);
      const q = query(sqr, x, y);
      const mean = s / area;
      const sd = Math.sqrt(Math.max(0, q / area - mean * mean));
      if (sd < best) {
        best = sd;
        bestAt = { x, y, mean: +mean.toFixed(1) };
      }
      if (sd > worst) worst = sd;
    }
  }

  return {
    windowPx: win,
    flattestStddev: +best.toFixed(2),
    flattestAt: bestAt,
    busiestStddev: +worst.toFixed(2),
  };
}

/**
 * Mean luminance per horizontal band. The fastest way to see whether the bowl
 * is actually held under the floor, or whether the frame is one flat wash.
 */
export function bandProfile(img, count = 24) {
  const bands = [];
  for (let i = 0; i < count; i++) {
    const s = regionStats(img, [0, i / count, 1, (i + 1) / count]);
    bands.push({ band: i, y: +((i + 0.5) / count).toFixed(3), mean: s.mean, stddev: s.stddev });
  }
  return bands;
}

/** Coarse luminance map of the frame — the squint pass, as numbers. */
export function tileGrid(img, cols = 8, rows = 16) {
  const grid = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      const s = regionStats(img, [c / cols, r / rows, (c + 1) / cols, (r + 1) / rows]);
      row.push({ mean: s.mean, sd: s.stddev });
    }
    grid.push(row);
  }
  return grid;
}

/** 32-bin luminance histogram plus a peak count, for the unimodal-with-a-tail test. */
export function histogram(img, bins = 32) {
  const counts = new Array(bins).fill(0);
  const n = img.width * img.height;
  for (let i = 0; i < n; i++) {
    const p = i * 3;
    const l = luma(img.data[p], img.data[p + 1], img.data[p + 2]);
    counts[Math.min(bins - 1, Math.floor((l / 256) * bins))]++;
  }
  const norm = counts.map((c) => +((c / n) * 100).toFixed(2));

  // Smooth before counting peaks so noise does not read as modality.
  const smooth = norm.map((_, i) => {
    const a = norm[Math.max(0, i - 1)];
    const b = norm[i];
    const c = norm[Math.min(bins - 1, i + 1)];
    return (a + b + c) / 3;
  });
  const peaks = [];
  for (let i = 1; i < bins - 1; i++) {
    if (smooth[i] > smooth[i - 1] && smooth[i] >= smooth[i + 1] && smooth[i] > 1.5) {
      peaks.push({ bin: i, centre: Math.round(((i + 0.5) / bins) * 255), pct: +smooth[i].toFixed(2) });
    }
  }
  return { bins: norm, peaks };
}

/**
 * §8.3 — shadows should lean cool and highlights warm, subtly. Measured as the
 * mean B−R in the darkest quartile and R−B in the brightest fifth.
 */
export function colorSplit(img) {
  const n = img.width * img.height;
  const lums = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const p = i * 3;
    lums[i] = luma(img.data[p], img.data[p + 1], img.data[p + 2]);
  }
  const sorted = Float32Array.from(lums).sort();
  const shadowCut = sorted[Math.round(0.25 * (n - 1))];
  const highCut = sorted[Math.round(0.8 * (n - 1))];

  const acc = (test) => {
    let count = 0;
    let r = 0;
    let g = 0;
    let b = 0;
    let sat = 0;
    let hx = 0;
    let hy = 0;
    for (let i = 0; i < n; i++) {
      if (!test(lums[i])) continue;
      const p = i * 3;
      const pr = img.data[p];
      const pg = img.data[p + 1];
      const pb = img.data[p + 2];
      r += pr;
      g += pg;
      b += pb;
      const hs = hueSat(pr, pg, pb);
      sat += hs.sat;
      const rad = (hs.hue * Math.PI) / 180;
      hx += Math.cos(rad) * hs.sat;
      hy += Math.sin(rad) * hs.sat;
      count++;
    }
    if (!count) return null;
    return {
      pixels: count,
      meanRGB: [+(r / count).toFixed(1), +(g / count).toFixed(1), +(b / count).toFixed(1)],
      meanSat: +(sat / count).toFixed(3),
      hue: +(((Math.atan2(hy, hx) * 180) / Math.PI + 360) % 360).toFixed(1),
    };
  };

  const shadows = acc((l) => l <= shadowCut);
  const highs = acc((l) => l >= highCut);
  return {
    shadows: { ...shadows, blueMinusRed: +(shadows.meanRGB[2] - shadows.meanRGB[0]).toFixed(2) },
    highlights: { ...highs, redMinusBlue: +(highs.meanRGB[0] - highs.meanRGB[2]).toFixed(2) },
  };
}

/**
 * §8.4 — vignette strength and shape.
 *
 * Measured horizontally and per row, then reduced with a median. A naive radial
 * average cannot work on an arena frame: the rings mix bright floor with dark
 * bowl, and the result describes the composition rather than the lens. Within a
 * single row the content is far more homogeneous, so the edge-versus-centre
 * ratio in that row is mostly the vignette; taking the median across two
 * thousand rows discards the rows where that assumption fails.
 *
 * The vertical profile is reported too, but it is *not* judged — in a portrait
 * arena frame the top is crowd and the bottom is hardwood, and no amount of
 * averaging separates that from the lens.
 */
export function vignette(img, rings = 8) {
  const cx = img.width / 2;
  const cy = img.height / 2;

  const bandMean = (y, fromF, toF) => {
    const x0 = Math.round(fromF * img.width);
    const x1 = Math.round(toF * img.width);
    let sum = 0;
    let sat = 0;
    for (let x = x0; x < x1; x++) {
      const p = (y * img.width + x) * 3;
      sum += luma(img.data[p], img.data[p + 1], img.data[p + 2]);
      sat += hueSat(img.data[p], img.data[p + 1], img.data[p + 2]).sat;
    }
    const n = Math.max(1, x1 - x0);
    return { lum: sum / n, sat: sat / n };
  };

  const lumRatios = [];
  const satRatios = [];
  for (let y = 0; y < img.height; y += 2) {
    const centre = bandMean(y, 0.42, 0.58);
    const left = bandMean(y, 0.0, 0.08);
    const right = bandMean(y, 0.92, 1.0);
    if (centre.lum < 6) continue; // nothing to be darker *than*
    const edge = (left.lum + right.lum) / 2;
    lumRatios.push(edge / centre.lum);
    if (centre.sat > 0.02) satRatios.push((left.sat + right.sat) / 2 / centre.sat);
  }
  const median = (a) => {
    if (!a.length) return null;
    const s = [...a].sort((p, q) => p - q);
    return s[s.length >> 1];
  };
  const lumRatio = median(lumRatios);
  const satRatio = median(satRatios);

  // Radial profile, reported for shape only — a vignette that has already
  // bottomed out by r=0.6 is a tight circle on a tall frame.
  const sums = new Float64Array(rings);
  const counts = new Float64Array(rings);
  for (let y = 0; y < img.height; y += 2) {
    const dy = (y - cy) / cy;
    for (let x = 0; x < img.width; x += 2) {
      const dx = (x - cx) / cx;
      const rad = Math.min(0.9999, Math.sqrt(dx * dx + dy * dy) / Math.SQRT2);
      const idx = Math.floor(rad * rings);
      const p = (y * img.width + x) * 3;
      sums[idx] += luma(img.data[p], img.data[p + 1], img.data[p + 2]);
      counts[idx]++;
    }
  }
  const profile = [];
  for (let i = 0; i < rings; i++) {
    profile.push({
      r: +((i + 0.5) / rings).toFixed(3),
      mean: counts[i] ? +(sums[i] / counts[i]).toFixed(2) : null,
    });
  }

  // Being precise about what was measured: the sampled edge columns sit at
  // mid-height, which on the normalised ellipse is r ≈ 0.71, not the corner at
  // r = 1.0. So this is edge falloff, and the rubric's corner figure has to be
  // read through a smooth vignette curve to compare — see EDGE_TARGET below.
  const centreRing = profile[0].mean;
  const cornerRing = profile[rings - 1].mean;

  return {
    profile,
    method: 'per-row horizontal edge/centre median, sampled at r≈0.71',
    edgeDarkerPct: lumRatio === null ? null : +((1 - lumRatio) * 100).toFixed(1),
    edgeDesatPct: satRatio === null ? null : +((1 - satRatio) * 100).toFixed(1),
    /**
     * True corner falloff, r = 1.0 against the centre. Only meaningful on a
     * uniform frame: on an arena frame the rings mix bright floor with dark
     * bowl and this number describes the composition, not the lens.
     */
    radialCornerDarkerPct:
      centreRing && cornerRing !== null ? +(((centreRing - cornerRing) / centreRing) * 100).toFixed(1) : null,
  };
}

/**
 * Detail energy per octave: the frame is box-halved repeatedly and the energy
 * lost at each step is the detail living at that scale. A frame whose energy
 * is all in the coarse octaves is smooth geometry with no material on it.
 */
export function detailOctaves(img, box = [0, 0, 1, 1], levels = 6) {
  const b = resolveBox(img, box);
  let w = b.w;
  let h = b.h;
  let buf = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = ((y + b.y0) * img.width + (x + b.x0)) * 3;
      buf[y * w + x] = luma(img.data[p], img.data[p + 1], img.data[p + 2]);
    }
  }

  const out = [];
  for (let level = 0; level < levels && w > 3 && h > 3; level++) {
    const hw = w >> 1;
    const hh = h >> 1;
    const down = new Float32Array(hw * hh);
    for (let y = 0; y < hh; y++) {
      for (let x = 0; x < hw; x++) {
        const a = buf[y * 2 * w + x * 2];
        const c = buf[y * 2 * w + x * 2 + 1];
        const d = buf[(y * 2 + 1) * w + x * 2];
        const e = buf[(y * 2 + 1) * w + x * 2 + 1];
        down[y * hw + x] = (a + c + d + e) / 4;
      }
    }
    // Residual against the upsampled coarse level = energy at this octave.
    let energy = 0;
    let count = 0;
    for (let y = 0; y < hh * 2; y++) {
      for (let x = 0; x < hw * 2; x++) {
        const d = buf[y * w + x] - down[(y >> 1) * hw + (x >> 1)];
        energy += d * d;
        count++;
      }
    }
    out.push({ scalePx: 1 << level, rms: +Math.sqrt(energy / count).toFixed(2) });
    buf = down;
    w = hw;
    h = hh;
  }
  return out;
}

/**
 * §8.5 — grain amplitude, estimated as the residual against a 3×3 box blur
 * restricted to mid-tones, plus how much of that residual is chromatic.
 */
export function grainEstimate(img, loLum = 60, hiLum = 190) {
  const { width: w, height: h } = img;
  let sum = 0;
  let count = 0;
  let chroma = 0;
  const at = (x, y, c) => img.data[(y * w + x) * 3 + c];
  for (let y = 1; y < h - 1; y += 2) {
    for (let x = 1; x < w - 1; x += 2) {
      const p = (y * w + x) * 3;
      const l = luma(img.data[p], img.data[p + 1], img.data[p + 2]);
      if (l < loLum || l > hiLum) continue;
      const perChannel = [0, 0, 0];
      for (let c = 0; c < 3; c++) {
        let acc = 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) acc += at(x + dx, y + dy, c);
        perChannel[c] = img.data[p + c] - acc / 9;
      }
      const lumRes = luma(perChannel[0], perChannel[1], perChannel[2]);
      sum += lumRes * lumRes;
      chroma += (perChannel[0] - perChannel[2]) ** 2;
      count++;
    }
  }
  if (!count) return { samples: 0, rms: 0, chromaRms: 0 };
  return {
    samples: count,
    rms: +Math.sqrt(sum / count).toFixed(2),
    chromaRms: +Math.sqrt(chroma / count).toFixed(2),
  };
}

/**
 * §8.6 — radial R/B separation. Cross-correlates the red and blue high-pass in
 * a corner patch against the same patch at the centre; the difference in best
 * shift is the aberration the grade is adding.
 */
export function chromaticAberration(img) {
  const patch = Math.round(Math.min(img.width, img.height) * 0.12);

  const bestShift = (cx, cy, dirX, dirY) => {
    const x0 = Math.max(1, Math.min(img.width - patch - 2, Math.round(cx - patch / 2)));
    const y0 = Math.max(1, Math.min(img.height - patch - 2, Math.round(cy - patch / 2)));
    let best = 0;
    let bestScore = Infinity;
    const scores = [];
    for (let s = -3; s <= 3; s += 0.5) {
      let acc = 0;
      let n = 0;
      for (let y = y0; y < y0 + patch; y += 2) {
        for (let x = x0; x < x0 + patch; x += 2) {
          const sx = x + dirX * s;
          const sy = y + dirY * s;
          if (sx < 1 || sy < 1 || sx >= img.width - 2 || sy >= img.height - 2) continue;
          const p = (y * img.width + x) * 3;
          // High-pass each channel horizontally so flat areas contribute nothing.
          const rHi = img.data[p] - img.data[p + 3];
          const bx = Math.round(sx);
          const by = Math.round(sy);
          const q = (by * img.width + bx) * 3;
          const bHi = img.data[q + 2] - img.data[q + 5];
          acc += (rHi - bHi) ** 2;
          n++;
        }
      }
      if (!n) continue;
      const score = acc / n;
      scores.push(score);
      if (score < bestScore) {
        bestScore = score;
        best = s;
      }
    }
    return best;
  };

  const cx = img.width / 2;
  const cy = img.height / 2;
  const corners = [
    bestShift(img.width * 0.12, img.height * 0.06, -1, 0),
    bestShift(img.width * 0.88, img.height * 0.06, 1, 0),
    bestShift(img.width * 0.12, img.height * 0.94, -1, 0),
    bestShift(img.width * 0.88, img.height * 0.94, 1, 0),
  ];
  const centre = bestShift(cx, cy, 1, 0);
  const mean = corners.reduce((a, b) => a + b, 0) / corners.length;
  return {
    centreShiftPx: +centre.toFixed(2),
    cornerShiftPx: +mean.toFixed(2),
    radialSeparationPx: +Math.abs(mean - centre).toFixed(2),
    refSeparationPx: +toRef(img, Math.abs(mean - centre)).toFixed(2),
  };
}

/** Bilinear luminance sample. */
export function sampleLum(img, x, y) {
  const x0 = Math.max(0, Math.min(img.width - 1, Math.floor(x)));
  const y0 = Math.max(0, Math.min(img.height - 1, Math.floor(y)));
  const x1 = Math.min(img.width - 1, x0 + 1);
  const y1 = Math.min(img.height - 1, y0 + 1);
  const fx = x - x0;
  const fy = y - y0;
  const g = (px, py) => {
    const p = (py * img.width + px) * 3;
    return luma(img.data[p], img.data[p + 1], img.data[p + 2]);
  };
  return (
    g(x0, y0) * (1 - fx) * (1 - fy) + g(x1, y0) * fx * (1 - fy) + g(x0, y1) * (1 - fx) * fy + g(x1, y1) * fx * fy
  );
}

/**
 * Luminance along a line, in fractional frame coordinates. This is how you
 * measure a penumbra, a rim-light band or a net strand: take a profile across
 * it and read the transition.
 */
export function lineProfile(img, from, to, samples = 128) {
  const x0 = from[0] * img.width;
  const y0 = from[1] * img.height;
  const x1 = to[0] * img.width;
  const y1 = to[1] * img.height;
  const lengthPx = Math.hypot(x1 - x0, y1 - y0);
  const values = [];
  for (let i = 0; i < samples; i++) {
    const t = i / (samples - 1);
    values.push(+sampleLum(img, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t).toFixed(1));
  }
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  // 10–90% transition width, in pixels, for whichever edge is steepest.
  const loMark = lo + (hi - lo) * 0.1;
  const hiMark = lo + (hi - lo) * 0.9;
  let widest = 0;
  let run = null;
  for (let i = 0; i < values.length; i++) {
    const inside = values[i] > loMark && values[i] < hiMark;
    if (inside && run === null) run = i;
    if ((!inside || i === values.length - 1) && run !== null) {
      widest = Math.max(widest, i - run);
      run = null;
    }
  }
  const perSample = lengthPx / (samples - 1);
  return {
    lengthPx: +lengthPx.toFixed(1),
    values,
    min: +lo.toFixed(1),
    max: +hi.toFixed(1),
    transitionPx: +(widest * perSample).toFixed(2),
    transitionRefPx: +toRef(img, widest * perSample).toFixed(2),
  };
}

/**
 * Finds the brightest cluster in the frame and measures how its light falls off
 * with distance — §8.1's bloom test. A halo that is still bright 200 ref-px out
 * is a screen-wide wash.
 */
export function bloomFalloff(img, rings = 10, maxRefRadius = 300) {
  let peak = -1;
  let px = 0;
  let py = 0;
  // Work on a coarse grid so a single hot pixel does not win.
  const step = Math.max(1, Math.round(img.height / 300));
  for (let y = 0; y < img.height; y += step) {
    for (let x = 0; x < img.width; x += step) {
      const l = sampleLum(img, x, y);
      if (l > peak) {
        peak = l;
        px = x;
        py = y;
      }
    }
  }
  const maxR = (maxRefRadius * img.height) / REFERENCE_HEIGHT;
  const sums = new Float64Array(rings);
  const counts = new Float64Array(rings);
  const r0 = Math.ceil(maxR);
  for (let dy = -r0; dy <= r0; dy++) {
    for (let dx = -r0; dx <= r0; dx++) {
      const d = Math.hypot(dx, dy);
      if (d > maxR) continue;
      const x = px + dx;
      const y = py + dy;
      if (x < 0 || y < 0 || x >= img.width || y >= img.height) continue;
      const idx = Math.min(rings - 1, Math.floor((d / maxR) * rings));
      const p = (y * img.width + x) * 3;
      sums[idx] += luma(img.data[p], img.data[p + 1], img.data[p + 2]);
      counts[idx]++;
    }
  }
  const profile = [];
  for (let i = 0; i < rings; i++) {
    profile.push({
      refPx: Math.round(((i + 0.5) / rings) * maxRefRadius),
      mean: counts[i] ? +(sums[i] / counts[i]).toFixed(1) : null,
    });
  }
  return { peak: +peak.toFixed(1), at: { x: px, y: py }, profile };
}

/** Fraction of the frame pinned at either end of the range. */
export function exposureDiscipline(img) {
  const n = img.width * img.height;
  let clipped = 0;
  let crushed = 0;
  for (let i = 0; i < n; i++) {
    const p = i * 3;
    const r = img.data[p];
    const g = img.data[p + 1];
    const b = img.data[p + 2];
    if (r >= 252 && g >= 252 && b >= 252) clipped++;
    if (luma(r, g, b) <= 4) crushed++;
  }
  return {
    clippedPct: +((clipped / n) * 100).toFixed(2),
    crushedPct: +((crushed / n) * 100).toFixed(2),
  };
}

export { REFERENCE_HEIGHT };
