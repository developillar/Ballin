/**
 * A minimal PNG writer — the counterpart to png.mjs.
 *
 * Only what the review harness needs: 8-bit RGB, no interlacing, filter 0 on
 * every scanline. Compression is left to zlib at its default level; these are
 * intermediate review images, not shipped assets.
 */

import { deflateSync } from 'node:zlib';
import { writeFileSync } from 'node:fs';

const SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

const CRC_TABLE = (() => {
  const table = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c;
  }
  return table;
})();

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, body) {
  const head = Buffer.allocUnsafe(8);
  head.writeUInt32BE(body.length, 0);
  head.write(type, 4, 'ascii');
  const crc = Buffer.allocUnsafe(4);
  crc.writeUInt32BE(crc32(Buffer.concat([head.subarray(4), body])), 0);
  return Buffer.concat([head, body, crc]);
}

/** Writes `{ width, height, data }` (RGB triples) out as a PNG. */
export function writePng(path, img) {
  const { width, height, data } = img;
  const stride = width * 3;
  const raw = Buffer.allocUnsafe((stride + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (stride + 1)] = 0; // filter: none
    Buffer.from(data.buffer, data.byteOffset + y * stride, stride).copy(raw, y * (stride + 1) + 1);
  }

  const ihdr = Buffer.allocUnsafe(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // truecolour
  ihdr[10] = 0; // deflate
  ihdr[11] = 0; // adaptive filtering
  ihdr[12] = 0; // no interlace

  writeFileSync(
    path,
    Buffer.concat([SIGNATURE, chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]),
  );
}

/** Allocates a blank image filled with one RGB colour. */
export function blank(width, height, rgb = [0, 0, 0]) {
  const data = new Uint8Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    data[i * 3] = rgb[0];
    data[i * 3 + 1] = rgb[1];
    data[i * 3 + 2] = rgb[2];
  }
  return { width, height, data };
}

/** Copies `src` into `dst` at (dx, dy), clipping at the destination bounds. */
export function blit(dst, src, dx, dy) {
  for (let y = 0; y < src.height; y++) {
    const ty = dy + y;
    if (ty < 0 || ty >= dst.height) continue;
    for (let x = 0; x < src.width; x++) {
      const tx = dx + x;
      if (tx < 0 || tx >= dst.width) continue;
      const s = (y * src.width + x) * 3;
      const d = (ty * dst.width + tx) * 3;
      dst.data[d] = src.data[s];
      dst.data[d + 1] = src.data[s + 1];
      dst.data[d + 2] = src.data[s + 2];
    }
  }
}

/** Nearest-neighbour resize. Good enough for laying frames out side by side. */
export function resize(src, width, height) {
  const out = { width, height, data: new Uint8Array(width * height * 3) };
  for (let y = 0; y < height; y++) {
    const sy = Math.min(src.height - 1, Math.floor((y * src.height) / height));
    for (let x = 0; x < width; x++) {
      const sx = Math.min(src.width - 1, Math.floor((x * src.width) / width));
      const s = (sy * src.width + sx) * 3;
      const d = (y * width + x) * 3;
      out.data[d] = src.data[s];
      out.data[d + 1] = src.data[s + 1];
      out.data[d + 2] = src.data[s + 2];
    }
  }
  return out;
}

/** Crops a fractional box out of an image. */
export function crop(src, box) {
  const x0 = Math.max(0, Math.round(box[0] * src.width));
  const y0 = Math.max(0, Math.round(box[1] * src.height));
  const x1 = Math.min(src.width, Math.round(box[2] * src.width));
  const y1 = Math.min(src.height, Math.round(box[3] * src.height));
  const width = Math.max(1, x1 - x0);
  const height = Math.max(1, y1 - y0);
  const out = { width, height, data: new Uint8Array(width * height * 3) };
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const s = ((y + y0) * src.width + (x + x0)) * 3;
      const d = (y * width + x) * 3;
      out.data[d] = src.data[s];
      out.data[d + 1] = src.data[s + 1];
      out.data[d + 2] = src.data[s + 2];
    }
  }
  return out;
}
