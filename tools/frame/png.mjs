/**
 * A minimal PNG reader.
 *
 * The project has a hard no-runtime-dependency rule and the review harness is
 * held to the same standard, so rather than pull in a decoder we read the
 * subset of the format Playwright actually writes: 8-bit, non-interlaced,
 * truecolour with or without alpha, plus palette for good measure.
 *
 * Everything downstream wants straight RGB in a flat Uint8Array, so that is
 * what comes out.
 */

import { inflateSync } from 'node:zlib';
import { readFileSync } from 'node:fs';

const SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

/** Paeth predictor, straight from the spec. */
function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

/**
 * Reverses the per-scanline filters in place and returns the packed samples
 * with the filter bytes stripped out.
 */
function unfilter(raw, width, height, bytesPerPixel) {
  const stride = width * bytesPerPixel;
  const out = Buffer.allocUnsafe(stride * height);
  let src = 0;
  for (let y = 0; y < height; y++) {
    const filter = raw[src++];
    const line = y * stride;
    const prev = line - stride;
    for (let x = 0; x < stride; x++) {
      const value = raw[src + x];
      const a = x >= bytesPerPixel ? out[line + x - bytesPerPixel] : 0;
      const b = y > 0 ? out[prev + x] : 0;
      const c = x >= bytesPerPixel && y > 0 ? out[prev + x - bytesPerPixel] : 0;
      let recon;
      switch (filter) {
        case 0: recon = value; break;
        case 1: recon = value + a; break;
        case 2: recon = value + b; break;
        case 3: recon = value + ((a + b) >> 1); break;
        case 4: recon = value + paeth(a, b, c); break;
        default: throw new Error(`unsupported PNG filter ${filter} on row ${y}`);
      }
      out[line + x] = recon & 0xff;
    }
    src += stride;
  }
  return out;
}

/**
 * Decodes a PNG file to `{ width, height, data }` where `data` is RGB triples.
 * Alpha is dropped — every frame the harness captures is opaque.
 */
export function readPng(path) {
  const buf = readFileSync(path);
  for (let i = 0; i < SIGNATURE.length; i++) {
    if (buf[i] !== SIGNATURE[i]) throw new Error(`${path} is not a PNG`);
  }

  let width = 0;
  let height = 0;
  let depth = 0;
  let colorType = 0;
  let palette = null;
  const idat = [];

  let at = 8;
  while (at < buf.length) {
    const length = buf.readUInt32BE(at);
    const type = buf.toString('ascii', at + 4, at + 8);
    const body = buf.subarray(at + 8, at + 8 + length);
    at += 12 + length; // length + type + data + crc

    if (type === 'IHDR') {
      width = body.readUInt32BE(0);
      height = body.readUInt32BE(4);
      depth = body[8];
      colorType = body[9];
      if (body[12] !== 0) throw new Error('interlaced PNGs are not supported');
      if (depth !== 8) throw new Error(`only 8-bit PNGs are supported, got ${depth}`);
    } else if (type === 'PLTE') {
      palette = Buffer.from(body);
    } else if (type === 'IDAT') {
      idat.push(Buffer.from(body));
    } else if (type === 'IEND') {
      break;
    }
  }

  if (!width || !height) throw new Error(`${path} has no IHDR`);

  const channels = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[colorType];
  if (!channels) throw new Error(`unsupported PNG colour type ${colorType}`);

  const raw = inflateSync(Buffer.concat(idat));
  const packed = unfilter(raw, width, height, channels);

  const data = new Uint8Array(width * height * 3);
  const n = width * height;
  for (let i = 0; i < n; i++) {
    const s = i * channels;
    const d = i * 3;
    if (colorType === 3) {
      const p = packed[s] * 3;
      data[d] = palette[p];
      data[d + 1] = palette[p + 1];
      data[d + 2] = palette[p + 2];
    } else if (colorType === 0 || colorType === 4) {
      data[d] = data[d + 1] = data[d + 2] = packed[s];
    } else {
      data[d] = packed[s];
      data[d + 1] = packed[s + 1];
      data[d + 2] = packed[s + 2];
    }
  }

  return { width, height, data };
}
