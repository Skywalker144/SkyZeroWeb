// Cross-checks mcts2048.js (browser afterstate Stochastic Gumbel AlphaZero MCTS)
// against a fixture generated from SkyZero_2048/python/mcts.py. Both sides run
// the SAME deterministic synthetic evaluator (gumbel_noise OFF), so the whole
// search must agree: a real porting bug shows up as a gross mismatch, not a
// float ulp. Regenerate the fixture with tests/gen_mcts2048_fixture.py if
// python/mcts.py changes.
import { test } from "node:test";
import assert from "node:assert";
import { createRequire } from "module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const require = createRequire(import.meta.url);
const AI = require("../ai2048.js");
const MCTS = require("../mcts2048.js");
const here = dirname(fileURLToPath(import.meta.url));
const cases = JSON.parse(readFileSync(join(here, "mcts2048_fixture.json"), "utf8"));
const puctCases = JSON.parse(readFileSync(join(here, "mcts2048_puct_fixture.json"), "utf8"));

const CFG = { num_simulations: 64, gamma: 0.999, c_puct: 1.25,
              gumbel_c_visit: 50.0, gumbel_c_scale: 1.0, gumbel_noise: false };
const PUCT_CFG = { ...CFG, root_algo: "puct" };

// Same synthetic evaluator as gen_mcts2048_fixture.py:
//   value = sum 2**exp ; logits[a] = applyMove(state,a).reward * 0.01.
function runNet(flat, B) {
  const P = AI.NUM_PLANES * AI.AREA;
  const logits = new Float32Array(B * 4);
  const values = new Float32Array(B);
  for (let s = 0; s < B; s++) {
    // Decode the one-hot planes back to an exponent board.
    const st = new Array(AI.AREA).fill(0);
    let sum = 0;
    for (let loc = 0; loc < AI.AREA; loc++) {
      for (let p = 0; p < AI.NUM_PLANES; p++) {
        if (flat[s * P + p * AI.AREA + loc] !== 0) { st[loc] = p; if (p > 0) sum += (1 << p); }
      }
    }
    values[s] = sum;
    for (let a = 0; a < 4; a++) logits[s * 4 + a] = AI.applyMove(st, a).reward * 0.01;
  }
  return Promise.resolve({ logits, values });
}

test("mcts2048 matches python/mcts.py on best action (deterministic, no noise)", async () => {
  for (const rec of cases) {
    const res = await MCTS.chooseMoveMCTS(rec.state, runNet, CFG);
    assert.strictEqual(res.action, rec.best_action,
      `best_action mismatch for ${rec.state}: js=${res.action} py=${rec.best_action}`);
  }
});

test("mcts2048 root value + improved policy match python to tolerance", async () => {
  for (const rec of cases) {
    const res = await MCTS.chooseMoveMCTS(rec.state, runNet, CFG);
    const relV = Math.abs(res.value - rec.root_value) / Math.max(1, Math.abs(rec.root_value));
    assert.ok(relV < 1e-4,
      `root_value mismatch for ${rec.state}: js=${res.value} py=${rec.root_value}`);
    for (let a = 0; a < 4; a++) {
      assert.ok(Math.abs(res.qs[a] - rec.improved_policy[a]) < 1e-4,
        `improved_policy[${a}] mismatch for ${rec.state}: js=${res.qs[a]} py=${rec.improved_policy[a]}`);
    }
  }
});

test("mcts2048 visit counts match python (sum = sims, per-action equal)", async () => {
  for (const rec of cases) {
    const res = await MCTS.chooseMoveMCTS(rec.state, runNet, CFG);
    const sum = res.visits.reduce((x, y) => x + y, 0);
    assert.strictEqual(sum, CFG.num_simulations, `visit sum != sims for ${rec.state}`);
    assert.deepStrictEqual(res.visits, rec.visits,
      `visit counts mismatch for ${rec.state}: js=${res.visits} py=${rec.visits}`);
  }
});

// --- PUCT root (classic AlphaZero): cross-check against the V2 python puct path.
test("mcts2048 PUCT root matches python on best action + visits (deterministic)", async () => {
  for (const rec of puctCases) {
    const res = await MCTS.chooseMoveMCTS(rec.state, runNet, PUCT_CFG);
    assert.strictEqual(res.action, rec.best_action,
      `puct best_action mismatch for ${rec.state}: js=${res.action} py=${rec.best_action}`);
    const sum = res.visits.reduce((x, y) => x + y, 0);
    assert.strictEqual(sum, PUCT_CFG.num_simulations, `puct visit sum != sims for ${rec.state}`);
    assert.deepStrictEqual(res.visits, rec.visits,
      `puct visit counts mismatch for ${rec.state}: js=${res.visits} py=${rec.visits}`);
  }
});

test("mcts2048 PUCT root value + improved policy match python to tolerance", async () => {
  for (const rec of puctCases) {
    const res = await MCTS.chooseMoveMCTS(rec.state, runNet, PUCT_CFG);
    const relV = Math.abs(res.value - rec.root_value) / Math.max(1, Math.abs(rec.root_value));
    assert.ok(relV < 1e-4,
      `puct root_value mismatch for ${rec.state}: js=${res.value} py=${rec.root_value}`);
    for (let a = 0; a < 4; a++) {
      assert.ok(Math.abs(res.qs[a] - rec.improved_policy[a]) < 1e-4,
        `puct improved_policy[${a}] mismatch for ${rec.state}: js=${res.qs[a]} py=${rec.improved_policy[a]}`);
    }
  }
});

// --- Tree reuse (no python reference): structural correctness + safe fallback.
test("mcts2048 tree reuse warm-starts from the spawned child node", async () => {
  let tested = 0;
  for (const rec of puctCases.slice(0, 25)) {
    const res = await MCTS.chooseMoveMCTS(rec.state, runNet, PUCT_CFG);
    if (res.action < 0) continue;
    const chance = res.search.root.children[res.action];
    const edge = chance.edges.find((e) => e.child !== null);  // an explored spawn
    if (!edge) continue;
    const newState = chance.afterstate.slice();
    newState[edge.cell] = edge.exp;
    const reuse = MCTS.reuseFrom(res.search, res.action, newState);
    assert.ok(reuse, `reuse should hit for an explored spawn (state ${rec.state})`);
    assert.strictEqual(reuse.root, edge.child, "reused root must be the carried child node");
    assert.deepStrictEqual(reuse.root.state, newState, "reused root state must equal observed board");
    // Warm-started search must run without error and pick a legal move.
    const res2 = await MCTS.chooseMoveMCTS(newState, runNet, PUCT_CFG, reuse);
    if (res2.action >= 0) {
      assert.strictEqual(AI.legalActions(newState)[res2.action], 1, "reused search must pick a legal move");
    }
    tested++;
  }
  assert.ok(tested > 0, "expected at least one reuse scenario to be exercised");
});

test("mcts2048 tree reuse rejects non-single-spawn boards (safe fallback)", async () => {
  const rec = puctCases.find((c) => true);
  const res = await MCTS.chooseMoveMCTS(rec.state, runNet, PUCT_CFG);
  assert.ok(res.action >= 0);
  const after = res.search.root.children[res.action].afterstate;
  // Two spawned tiles => not a single-spawn successor => no reuse.
  const twoSpawn = after.slice();
  let changed = 0;
  for (let i = 0; i < twoSpawn.length && changed < 2; i++) if (twoSpawn[i] === 0) { twoSpawn[i] = 1; changed++; }
  if (changed === 2) assert.strictEqual(MCTS.reuseFrom(res.search, res.action, twoSpawn), null);
  // An existing tile vanished (board diverged) => no reuse.
  assert.strictEqual(MCTS.reuseFrom(res.search, res.action, after.map(() => 0)), null);
  // Missing prior search / action => no reuse.
  assert.strictEqual(MCTS.reuseFrom(null, res.action, rec.state), null);
  assert.strictEqual(MCTS.reuseFrom(res.search, -1, rec.state), null);
});
