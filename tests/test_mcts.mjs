import { test } from "node:test";
import assert from "node:assert";
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const {
    Node, MCTS, softmax, uncertaintyWeight, valueWeightCdf,
    searchFactorWhenWinning, boardSymmetries, rootSymmetryMask,
} = require("../mcts.js");

const stubGame = {
    boardSize: 3,
    getInitialState: () => new Int8Array(9),
    getLegalActions(state) {
        return Uint8Array.from(state, x => x === 0 ? 1 : 0);
    },
    getNextState(state, action, toPlay) {
        const next = new Int8Array(state);
        next[action] = toPlay;
        return next;
    },
};

test("softmax masks infinities and normalizes", () => {
    const p = softmax(new Float64Array([1, -Infinity, 3]));
    assert.strictEqual(p[1], 0);
    assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12);
});

test("V7.19 uncertainty weight uses the active play.cfg constants", () => {
    const args = {
        use_uncertainty: true, uncertainty_coeff: 0.25,
        uncertainty_exponent: 1, uncertainty_max_weight: 8,
    };
    assert.strictEqual(uncertaintyWeight(-1, args), 1);
    assert.strictEqual(uncertaintyWeight(0, args), 8);
    assert.ok(Math.abs(uncertaintyWeight(0.25, args)
        - 0.25 / (0.25 + 0.25 / 8)) < 1e-12);
});

test("three-degree Student-t CDF is symmetric and monotonic", () => {
    assert.ok(Math.abs(valueWeightCdf(0) - 0.5) < 1e-12);
    assert.ok(Math.abs(valueWeightCdf(-2) + valueWeightCdf(2) - 1) < 1e-12);
    assert.ok(valueWeightCdf(-1) < valueWeightCdf(0));
    assert.ok(valueWeightCdf(0) < valueWeightCdf(1));
});

test("KataGo winning search factor requires three sustained high values", () => {
    assert.strictEqual(searchFactorWhenWinning([1, 1]), 1);
    assert.strictEqual(searchFactorWhenWinning([1, 0.94, 1]), 1);
    assert.ok(Math.abs(
        searchFactorWhenWinning([0.975, 0.98, 0.99]) - 0.65) < 1e-12);
    assert.ok(Math.abs(
        searchFactorWhenWinning([1, 1, 1]) - 0.3) < 1e-12);
});

test("empty root keeps only the center and never applies RNNM", () => {
    const mcts = new MCTS(stubGame, {});
    const root = new Node(stubGame.getInitialState(), 1);
    const policy = new Float32Array(9).fill(1 / 9);
    const logits = new Float32Array(9);
    mcts.expand(root, policy, [0.4, 0.2, 0.4], logits, 0.1, true);
    assert.deepStrictEqual(root.children.map(c => c.actionTaken), [4]);

    const occupied = new Node(stubGame.getNextState(
        stubGame.getInitialState(), 4, 1), -1);
    mcts.expand(occupied, policy, [0.4, 0.2, 0.4], logits, 0.1, true);
    // Symmetry pruning remains active, but no distance/non-neighbour mask exists:
    // at least one corner at Chebyshev distance 1 survives.
    assert.ok(occupied.children.some(c => [0, 2, 6, 8].includes(c.actionTaken)));
    assert.strictEqual("root_nonneighbour_mask_radius" in mcts.args, false);
});

test("root symmetry mask contains one representative per orbit", () => {
    const state = new Int8Array(9);
    assert.strictEqual(boardSymmetries(state, 3).length, 8);
    const mask = rootSymmetryMask(state, 3, 1);
    assert.strictEqual(mask.reduce((a, b) => a + b, 0), 3);
});

test("weighted recompute and backup flip WDL between player frames", () => {
    const mcts = new MCTS(stubGame, {
        root_symmetry_pruning: false,
        subtree_value_bias_factor: 0,
        value_weight_exponent: 0,
        use_uncertainty: false,
    });
    const root = new Node(stubGame.getInitialState(), 1);
    const policy = new Float32Array(9);
    policy[0] = 0.5; policy[1] = 0.5;
    const logits = new Float32Array(9);
    mcts.expand(root, policy, [0.5, 0, 0.5], logits, -1, false);
    const child = root.children[0];
    mcts.expand(child, new Float32Array(9),
        [0.7, 0, 0.3], logits, -1, false);
    mcts.backpropagate([root, child], [0.7, 0, 0.3], false);
    assert.ok(child.weightSum > 0);
    assert.ok(root.weightSum > child.weightSum);
    assert.ok(root.v[2] > root.v[0]);
    assert.strictEqual(root.n, 2);
});

test("terminal backup contributes an uncertainty weight of eight", () => {
    const mcts = new MCTS(stubGame, {});
    const leaf = new Node(stubGame.getInitialState(), 1);
    mcts.backpropagate([leaf], new Float64Array([1, 0, 0]), true);
    assert.strictEqual(leaf.n, 1);
    assert.strictEqual(leaf.weightSum, 8);
    assert.deepStrictEqual(Array.from(leaf.v), [8, 0, 0]);
});

test("root play selection returns normalized V7.19 policy and LCB data", () => {
    const mcts = new MCTS(stubGame, {
        root_symmetry_pruning: false,
        subtree_value_bias_factor: 0,
        value_weight_exponent: 0,
        use_uncertainty: false,
    });
    const root = new Node(stubGame.getInitialState(), 1);
    const policy = new Float32Array(9);
    policy[0] = 0.7; policy[1] = 0.3;
    mcts.expand(root, policy, [0.5, 0, 0.5], new Float32Array(9), -1, false);
    root.children[0].updateTerminal([0.2, 0, 0.8], 1);
    root.children[1].updateTerminal([0.8, 0, 0.2], 1);
    mcts.recomputeNode(root);
    const selection = mcts.rootPlaySelection(root);
    assert.ok([0, 1].includes(selection.action));
    const dist = mcts.getPlaySelectionPolicy(root);
    assert.ok(Math.abs(dist.reduce((a, b) => a + b, 0) - 1) < 1e-6);
    assert.strictEqual(mcts.getLcb(root).lcb.length, 9);
});
