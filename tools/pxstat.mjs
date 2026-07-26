#!/usr/bin/env node
/** Tiny PNG region sampler for review work: node tools/pxstat.mjs file.png x0 y0 x1 y1 [...] */
import { readFileSync } from 'node:fs';
import { inflateSync } from 'node:zlib';

function decodePNG(buf) {
  let p = 8;
  let w = 0, h = 0, bitDepth = 0, colorType = 0;
  const idat = [];
  while (p < buf.length) {
    const len = buf.readUInt32BE(p);
    const type = buf.toString('ascii', p + 4, p + 8);
    const data = buf.subarray(p + 8, p + 8 + len);
    if (type === 'IHDR') {
      w = data.readUInt32BE(0); h = data.readUInt32BE(4);
      bitDepth = data[8]; colorType = data[9];
    } else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
    p += 12 + len;
  }
  if (bitDepth !== 8) throw new Error('only 8-bit');
  const ch = colorType === 6 ? 4 : colorType === 2 ? 3 : colorType === 0 ? 1 : 4;
  const raw = inflateSync(Buffer.concat(idat));
  const stride = w * ch;
  const out = Buffer.alloc(h * stride);
  let q = 0;
  for (let y = 0; y < h; y++) {
    const f = raw[q++];
    const line = raw.subarray(q, q + stride); q += stride;
    const cur = out.subarray(y * stride, (y + 1) * stride);
    const prev = y > 0 ? out.subarray((y - 1) * stride, y * stride) : null;
    for (let i = 0; i < stride; i++) {
      const a = i >= ch ? cur[i - ch] : 0;
      const b = prev ? prev[i] : 0;
      const c = prev && i >= ch ? prev[i - ch] : 0;
      let v = line[i];
      if (f === 1) v += a;
      else if (f === 2) v += b;
      else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) {
        const pp = a + b - c, pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c);
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      cur[i] = v & 255;
    }
  }
  return { w, h, ch, data: out };
}

const [file, ...rest] = process.argv.slice(2);
const img = decodePNG(readFileSync(file));
console.log(`${file}  ${img.w}x${img.h}`);

function stat(x0, y0, x1, y1) {
  let n = 0, sr = 0, sg = 0, sb = 0, sl = 0, sl2 = 0, clip = 0, crush = 0;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
    const i = (y * img.w + x) * img.ch;
    const r = img.data[i], g = img.data[i + 1], b = img.data[i + 2];
    const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    sr += r; sg += g; sb += b; sl += l; sl2 += l * l; n++;
    if (r >= 252 && g >= 252 && b >= 252) clip++;
    if (l <= 4) crush++;
  }
  const mean = sl / n;
  return {
    L: mean.toFixed(1), sd: Math.sqrt(sl2 / n - mean * mean).toFixed(1),
    R: (sr / n).toFixed(0), G: (sg / n).toFixed(0), B: (sb / n).toFixed(0),
    'B-R': (sb / n - sr / n).toFixed(1),
    clip: ((clip / n) * 100).toFixed(2) + '%', crush: ((crush / n) * 100).toFixed(2) + '%',
  };
}

if (rest.length === 0) {
  console.log('whole', stat(0, 0, img.w, img.h));
} else {
  for (let i = 0; i + 3 < rest.length; i += 5) {
    const [name, x0, y0, x1, y1] = rest.slice(i, i + 5);
    console.log(name.padEnd(14), stat(+x0, +y0, +x1, +y1));
  }
}

if (process.env.GRID) {
  const rows = 24, cols = 6;
  console.log('luma grid (rows top→bottom):');
  for (let r = 0; r < rows; r++) {
    const line = [];
    for (let c = 0; c < cols; c++) {
      const s = stat(
        Math.floor((c * img.w) / cols), Math.floor((r * img.h) / rows),
        Math.floor(((c + 1) * img.w) / cols), Math.floor(((r + 1) * img.h) / rows),
      );
      line.push(String(Math.round(+s.L)).padStart(4));
    }
    console.log(String(Math.floor((r * img.h) / rows)).padStart(5) + ' |' + line.join(''));
  }
}
