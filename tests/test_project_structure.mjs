import assert from "node:assert/strict";
import { accessSync, existsSync, readFileSync } from "node:fs";
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
  assert.match(redirects, /^\/gomoku\/:model\s+\/gomoku\/index\.html\s+200$/m);
  assert.match(redirects, /^\/2048\s+\/2048\/index\.html\s+200$/m);
  assert.match(redirects, /^\/channel-dodge\s+\/channel-dodge\/index\.html\s+200$/m);
  assert.match(redirects, /^\/mcts-tree\s+\/gomoku\/tree\/index\.html\s+200$/m);
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
