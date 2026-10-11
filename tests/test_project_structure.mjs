import assert from "node:assert/strict";
import { accessSync, existsSync, readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

const staticFiles = [
  "public/index.html",
  "public/_headers",
  "public/_redirects",
  "public/_routes.json",
  "public/shared/base.css",
  "public/shared/fonts/fonts.css",
  "public/shared/leaderboard.js",
  "public/shared/onnx-worker-runtime.js",
  "public/gomoku/index.html",
  "public/gomoku/style.css",
  "public/gomoku/tree/index.html",
  "public/gomoku/models/manifest.json",
  "public/2048/index.html",
  "public/hex/index.html",
  "public/hex/models/manifest.json",
  "public/2048/models/value.onnx",
  "public/channel-dodge/index.html",
  "public/scotlandyard/index.html",
  "public/scotlandyard/models.json",
];

test("deployable files live under the public boundary", () => {
  for (const path of staticFiles) {
    assert.doesNotThrow(() => accessSync(join(root, path)));
  }
  const wrangler = readFileSync(join(root, "wrangler.toml"), "utf8");
  assert.match(wrangler, /pages_build_output_dir\s*=\s*"\.\/public"/);
});

test("stable routes target the domain entry points", () => {
  const redirects = readFileSync(join(root, "public/_redirects"), "utf8");
  assert.match(redirects, /^\/gomoku\s+\/gomoku\/index\.html\s+200$/m);
  const manifest = JSON.parse(readFileSync(join(root, "public/gomoku/models/manifest.json"), "utf8"));
  for (const model of manifest.models) {
    assert.ok(redirects.split("\n").some(line => {
      const [source, target, status] = line.trim().split(/\s+/);
      return source === `/gomoku/${model.id}` && target === "/gomoku/index.html" && status === "200";
    }), `missing model route: ${model.id}`);
  }
  assert.match(redirects, /^\/2048\s+\/2048\/index\.html\s+200$/m);
  assert.match(redirects, /^\/hex\s+\/hex\/index\.html\s+200$/m);
  assert.match(redirects, /^\/channel-dodge\s+\/channel-dodge\/index\.html\s+200$/m);
  assert.match(redirects, /^\/mcts-tree\s+\/gomoku\/tree\/index\.html\s+200$/m);
});

test("Pages rewrites do not intercept static assets", () => {
  const redirects = readFileSync(join(root, "public/_redirects"), "utf8")
    .split("\n").map(line => line.trim()).filter(line => line && !line.startsWith("#"))
    .map(line => line.split(/\s+/));
  const assets = readdirSync(join(root, "public"), { recursive: true })
    .filter(path => /\.(css|js|json|onnx|woff2|png|svg|wasm)$/.test(path));
  for (const [source, target] of redirects) {
    const pattern = new RegExp("^" + source.split("/").map(segment =>
      segment.startsWith(":") ? "[^/]+" : segment === "*" ? ".*"
        : segment.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
    ).join("/") + "$");
    for (const asset of assets) {
      assert.ok(!pattern.test(`/${asset}`), `${source} intercepts /${asset} and serves ${target}`);
    }
  }
});

test("source-only concerns stay outside the deployable tree", () => {
  for (const path of [
    "functions/api/leaderboard.js",
    "database/schema.sql",
    "docs/README.md",
    "scripts/serve.py",
    "tests/test_gomoku.mjs",
    "tools/export_v719_onnx.py",
  ]) {
    assert.doesNotThrow(() => accessSync(join(root, path)));
  }
  for (const path of [
    "index.html",
    "models",
    "scotlandyard",
    "style.css",
    "worker.js",
    "public/database",
    "public/docs",
    "public/functions",
    "public/tests",
    "public/tools",
  ]) {
    assert.equal(existsSync(join(root, path)), false, path);
  }
});
