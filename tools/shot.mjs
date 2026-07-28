#!/usr/bin/env node
/**
 * Screenshot harness. Boots the built (or dev) app in a portrait phone viewport,
 * waits for the engine to report ready, optionally drives the game through a
 * scripted beat, and writes a PNG plus the console log.
 *
 * Usage:
 *   node tools/shot.mjs --out shots/wide.png --scene idle --wait 2500
 *   node tools/shot.mjs --all            # capture the standard review sheet
 */

import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join } from 'node:path';

const ROOT = resolve(import.meta.dirname, '..');

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.png': 'image/png',
  '.jpg': 'image/jpeg', '.svg': 'image/svg+xml', '.webp': 'image/webp',
  '.wasm': 'application/wasm', '.ktx2': 'application/octet-stream',
  '.hdr': 'application/octet-stream', '.bin': 'application/octet-stream',
};

function args() {
  const out = {};
  const a = process.argv.slice(2);
  for (let i = 0; i < a.length; i++) {
    if (!a[i].startsWith('--')) continue;
    const key = a[i].slice(2);
    const next = a[i + 1];
    if (next && !next.startsWith('--')) { out[key] = next; i++; } else { out[key] = true; }
  }
  return out;
}

/** Serve dist/ statically so screenshots exercise the production bundle. */
async function serveDist(port) {
  const base = join(ROOT, 'dist');
  const server = createServer(async (req, res) => {
    try {
      let p = decodeURIComponent((req.url || '/').split('?')[0]);
      if (p.endsWith('/')) p += 'index.html';
      let file = join(base, p);
      try { if ((await stat(file)).isDirectory()) file = join(file, 'index.html'); }
      catch { file = join(base, 'index.html'); }
      const body = await readFile(file);
      res.writeHead(200, {
        'content-type': MIME[extname(file)] || 'application/octet-stream',
        'cache-control': 'no-store',
      });
      res.end(body);
    } catch {
      res.writeHead(404); res.end('not found');
    }
  });
  await new Promise((r) => server.listen(port, r));
  return server;
}

const SCENES = {
  /** Default gameplay framing, ball in the shooter's hands. */
  idle: async () => {},
  /** Mid-flight jumper, ball at the apex. */
  shot: async (page) => {
    await page.evaluate(() => window.__engine?.get('game')?.debugScene?.('shot'));
    await page.waitForTimeout(900);
  },
  /** Ball dropping through the net. */
  swish: async (page) => {
    await page.evaluate(() => window.__engine?.get('game')?.debugScene?.('swish'));
    await page.waitForTimeout(1500);
  },
  /** Rim-level hero framing on the hoop. */
  rim: async (page) => {
    await page.evaluate(() => window.__engine?.get('camera')?.debugPose?.('rim'));
    await page.waitForTimeout(700);
  },
  /** Close-up on the ball-handler for material and skin review. */
  closeup: async (page) => {
    await page.evaluate(() => window.__engine?.get('camera')?.debugPose?.('closeup'));
    await page.waitForTimeout(700);
  },
  /** Wide arena establishing shot: crowd, jumbotron, rafters. */
  arena: async (page) => {
    await page.evaluate(() => window.__engine?.get('camera')?.debugPose?.('arena'));
    await page.waitForTimeout(700);
  },
  /** Court-level look down the hardwood for reflection / texture review. */
  floor: async (page) => {
    await page.evaluate(() => window.__engine?.get('camera')?.debugPose?.('floor'));
    await page.waitForTimeout(700);
  },
  /** Dunk animation peak. */
  dunk: async (page) => {
    await page.evaluate(() => window.__engine?.get('game')?.debugScene?.('dunk'));
    await page.waitForTimeout(1400);
  },
};

async function main() {
  const o = args();
  const port = Number(o.port || 4181);
  const scene = String(o.scene || 'idle');
  const wait = Number(o.wait || 2600);
  const width = Number(o.width || 430);
  const height = Number(o.height || 932);
  const dpr = Number(o.dpr || 3);
  const quality = String(o.quality || 'ultra');

  let server = null;
  let devProc = null;
  let url;

  if (o.dev) {
    devProc = spawn('npx', ['vite', '--port', String(port), '--strictPort'], {
      cwd: ROOT, stdio: 'ignore', detached: false,
    });
    await new Promise((r) => setTimeout(r, 3500));
    url = `http://localhost:${port}/`;
  } else {
    server = await serveDist(port);
    url = `http://localhost:${port}/`;
  }

  const browser = await chromium.launch({
    args: [
      '--use-gl=angle',
      '--use-angle=swiftshader',
      '--enable-unsafe-swiftshader',
      '--ignore-gpu-blocklist',
      '--enable-gpu-rasterization',
      '--disable-dev-shm-usage',
    ],
  });
  const ctx = await browser.newContext({
    viewport: { width, height },
    deviceScaleFactor: dpr,
    isMobile: true,
    hasTouch: true,
    userAgent:
      'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1',
  });
  const page = await ctx.newPage();
  const logs = [];
  page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}\n${e.stack ?? ''}`));

  await page.goto(`${url}?quality=${quality}&test=1`, { waitUntil: 'load', timeout: 60000 });

  try {
    await page.waitForFunction(() => window.__ready === true, { timeout: 55000 });
  } catch {
    logs.push('[harness] engine never reported ready');
  }

  await page.waitForTimeout(wait);
  await (SCENES[scene] ?? SCENES.idle)(page);
  await page.waitForTimeout(450);

  const stats = await page.evaluate(() => {
    const e = window.__engine;
    if (!e) return null;
    const info = e.renderer.info;
    return {
      frameMs: Number(e.frameMs?.toFixed?.(2) ?? 0),
      fps: Math.round(1000 / Math.max(1, e.frameMs || 16)),
      tier: e.quality?.tier,
      renderScale: Number(e.governor?.renderScale?.toFixed?.(2) ?? 1),
      drawCalls: info.render.calls,
      triangles: info.render.triangles,
      programs: info.programs?.length ?? 0,
      textures: info.memory.textures,
      geometries: info.memory.geometries,
      pixel: [e.pixelWidth, e.pixelHeight],
    };
  });

  const out = resolve(ROOT, String(o.out || `shots/${scene}.png`));
  await mkdir(dirname(out), { recursive: true });
  // Software rasterisation makes a full-resolution grab slow; give it room.
  const buf = await page.screenshot({ type: 'png', timeout: 180000, animations: 'allow' });
  await writeFile(out, buf);

  const meta = { scene, url, stats, logs: logs.slice(-60) };
  await writeFile(out.replace(/\.png$/, '.json'), JSON.stringify(meta, null, 2));

  console.log(JSON.stringify({ out, ...meta }, null, 2));

  await browser.close();
  server?.close();
  if (devProc) devProc.kill();
}

main().catch((e) => { console.error(e); process.exit(1); });
