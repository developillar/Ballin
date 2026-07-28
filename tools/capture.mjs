#!/usr/bin/env node
/**
 * Review-sheet capture. Boots the built app once and grabs every framing in a
 * single browser session — far cheaper than one launch per shot, which matters
 * because CI here rasterises in software.
 *
 * Usage:
 *   node tools/capture.mjs                       # all scenes → shots/
 *   node tools/capture.mjs --scenes rim,swish    # a subset
 *   node tools/capture.mjs --dir shots/round3    # into a round folder
 */

import { chromium } from 'playwright';
import { mkdir, writeFile, readFile, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, join, resolve } from 'node:path';

const ROOT = resolve(import.meta.dirname, '..');
const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.png': 'image/png',
  '.jpg': 'image/jpeg', '.svg': 'image/svg+xml', '.webp': 'image/webp',
  '.wasm': 'application/wasm', '.bin': 'application/octet-stream',
};

function args() {
  const out = {};
  const a = process.argv.slice(2);
  for (let i = 0; i < a.length; i++) {
    if (!a[i].startsWith('--')) continue;
    const k = a[i].slice(2);
    const n = a[i + 1];
    if (n && !n.startsWith('--')) { out[k] = n; i++; } else { out[k] = true; }
  }
  return out;
}

async function serveDist(port, distDir) {
  const base = join(ROOT, distDir);
  const server = createServer(async (req, res) => {
    try {
      let p = decodeURIComponent((req.url || '/').split('?')[0]);
      if (p.endsWith('/')) p += 'index.html';
      let file = join(base, p);
      try { if ((await stat(file)).isDirectory()) file = join(file, 'index.html'); }
      catch { file = join(base, 'index.html'); }
      const body = await readFile(file);
      res.writeHead(200, { 'content-type': MIME[extname(file)] || 'application/octet-stream', 'cache-control': 'no-store' });
      res.end(body);
    } catch { res.writeHead(404); res.end('nf'); }
  });
  await new Promise((r) => server.listen(port, r));
  return server;
}

/** Each entry: [name, async setup(page)] — run in order in one session. */
const SCENES = {
  gameplay: async () => {},
  arena:   async (p) => { await p.evaluate(() => window.__engine?.get('camera')?.debugPose?.('arena')); },
  rim:     async (p) => { await p.evaluate(() => window.__engine?.get('camera')?.debugPose?.('rim')); },
  closeup: async (p) => { await p.evaluate(() => window.__engine?.get('camera')?.debugPose?.('closeup')); },
  floor:   async (p) => { await p.evaluate(() => window.__engine?.get('camera')?.debugPose?.('floor')); },
  shot:    async (p) => {
    await p.evaluate(() => { window.__engine?.get('camera')?.debugPose?.(''); window.__engine?.get('game')?.debugScene?.('shot'); });
    await p.waitForTimeout(700);
  },
  swish:   async (p) => {
    await p.evaluate(() => { window.__engine?.get('camera')?.debugPose?.('rim'); window.__engine?.get('game')?.debugScene?.('swish'); });
    await p.waitForTimeout(500);
  },
  dunk:    async (p) => {
    await p.evaluate(() => { window.__engine?.get('camera')?.debugPose?.('rim'); window.__engine?.get('game')?.debugScene?.('dunk'); });
    await p.waitForTimeout(400);
  },
  // A uniform field through the post chain. Everything that is not flat in the
  // result is something the stack did, which is what makes vignette, grain and
  // chromatic aberration exactly measurable instead of merely advisory.
  // Capture it LAST -- it hides the scene, and restoring is only best-effort.
  flatfield: async (p) => {
    await p.evaluate(() => window.__flatfield?.(true));
    await p.waitForTimeout(500);
  },
};

async function main() {
  const o = args();
  const port = Number(o.port || 4193);
  const dir = resolve(ROOT, String(o.dir || 'shots'));
  const width = Number(o.width || 430);
  const height = Number(o.height || 932);
  const dpr = Number(o.dpr || 2);
  const quality = String(o.quality || 'high');
  const settle = Number(o.settle || 900);
  const names = o.scenes ? String(o.scenes).split(',') : Object.keys(SCENES);
  // Parallel agents each build into their own outDir and serve their own port.
  const distDir = String(o.dist || 'dist');

  await mkdir(dir, { recursive: true });
  const server = await serveDist(port, distDir);

  const browser = await chromium.launch({
    args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
           '--ignore-gpu-blocklist', '--disable-dev-shm-usage'],
  });
  const ctx = await browser.newContext({
    viewport: { width, height }, deviceScaleFactor: dpr, isMobile: true, hasTouch: true,
    userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1',
  });
  const page = await ctx.newPage();
  const logs = [];
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') logs.push(`[${m.type()}] ${m.text()}`); });
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));

  await page.goto(`http://localhost:${port}/?quality=${quality}&test=1`, { waitUntil: 'load', timeout: 90000 });
  try { await page.waitForFunction(() => window.__ready === true, { timeout: 90000 }); }
  catch { logs.push('[capture] engine never reported ready'); }
  await page.waitForTimeout(2500);

  const written = [];
  for (const name of names) {
    const fn = SCENES[name];
    if (!fn) { logs.push(`[capture] unknown scene ${name}`); continue; }
    // The flat field hides the whole scene, so make sure it is off for anything
    // that is not itself the flat field, whatever order scenes were asked for.
    if (name !== 'flatfield') await page.evaluate(() => window.__flatfield?.(false));
    await fn(page);
    await page.waitForTimeout(settle);
    const out = join(dir, `${name}.png`);
    await writeFile(out, await page.screenshot({ type: 'png', timeout: 240000 }));
    written.push(out);
  }

  const stats = await page.evaluate(() => {
    const e = window.__engine; if (!e) return null;
    const i = e.renderer.info;
    return {
      fps: Math.round(1000 / Math.max(1, e.frameMs || 16)),
      frameMs: Number((e.frameMs || 0).toFixed(1)),
      tier: e.quality?.tier, renderScale: Number((e.governor?.renderScale ?? 1).toFixed(2)),
      drawCalls: i.render.calls, triangles: i.render.triangles,
      programs: i.programs?.length ?? 0, textures: i.memory.textures, geometries: i.memory.geometries,
    };
  });

  await writeFile(join(dir, 'meta.json'), JSON.stringify({ stats, logs: logs.slice(-80), written }, null, 2));
  console.log(JSON.stringify({ dir, written, stats, errors: logs.filter((l) => l.includes('error')).slice(0, 20) }, null, 2));

  await browser.close();
  server.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
