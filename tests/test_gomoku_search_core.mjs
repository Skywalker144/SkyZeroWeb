import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const { runSimulation } = require("../public/gomoku/workers/search-core.js");

test("shared Gomoku simulation expands and backs up an evaluated leaf", async () => {
  const root = {
    state: new Int8Array(4),
    toPlay: 1,
    actionTaken: null,
    isExpanded: () => false,
  };
  const calls = [];
  const mcts = {
    select: () => null,
    expand: (...args) => calls.push(["expand", ...args]),
    backpropagate: (...args) => calls.push(["backup", ...args]),
  };
  const game = { getWinner: () => null };
  const evaluation = {
    policyPriorSoft: new Float32Array([1, 0, 0, 0]),
    policyPriorMaskedLogits: new Float32Array(4),
    wdl: new Float32Array([0.6, 0.3, 0.1]),
    stError: 0.2,
  };

  const result = await runSimulation({
    root,
    mcts,
    game,
    evaluate: async () => evaluation,
  });

  assert.equal(result.leaf, root);
  assert.equal(result.expanded, true);
  assert.deepEqual(result.nodes, [root]);
  assert.equal(result.value, evaluation.wdl);
  assert.equal(calls[0][0], "expand");
  assert.deepEqual(calls[1], ["backup", [root], evaluation.wdl, false]);
});

test("shared Gomoku simulation returns terminal WDL without inference", async () => {
  const child = {
    state: new Int8Array([1]),
    toPlay: -1,
    actionTaken: 0,
    isExpanded: () => false,
  };
  const root = {
    state: new Int8Array(1),
    toPlay: 1,
    actionTaken: null,
    isExpanded: () => true,
  };
  let evaluated = false;
  let backup;
  const result = await runSimulation({
    root,
    mcts: {
      select: () => child,
      expand: () => assert.fail("terminal nodes are not expanded"),
      backpropagate: (...args) => { backup = args; },
    },
    game: { getWinner: () => 1 },
    evaluate: async () => { evaluated = true; },
  });

  assert.equal(evaluated, false);
  assert.equal(result.expanded, false);
  assert.equal(result.terminalResult, -1);
  assert.deepEqual(Array.from(result.value), [0, 0, 1]);
  assert.deepEqual(backup, [[root, child], result.value, true]);
});
