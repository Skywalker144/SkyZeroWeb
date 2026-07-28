import { test } from "node:test";
import assert from "node:assert";
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const {
    Gomoku, ForbiddenPointFinder,
    MIN_BOARD_SIZE, MAX_BOARD_SIZE,
} = require("../gomoku.js");

test("V7.19 accepts exactly board sizes 11 through 15", () => {
    assert.strictEqual(MIN_BOARD_SIZE, 11);
    assert.strictEqual(MAX_BOARD_SIZE, 15);
    for (let n = 11; n <= 15; n++) {
        assert.strictEqual(new Gomoku(n).getInitialState().length, n * n);
    }
    assert.throws(() => new Gomoku(10));
    assert.throws(() => new Gomoku(16));
});

test("Renju forbidden black move stays legal and is a white-win terminal", () => {
    const g = new Gomoku(15, "renju");
    let s = g.getInitialState();
    for (const [r, c] of [[7,5], [7,6], [5,7], [6,7]]) {
        s[r * 15 + c] = 1;
    }
    const action = 7 * 15 + 7;
    assert.strictEqual(g.getLegalActions(s, 1)[action], 1);
    s = g.getNextState(s, action, 1);
    assert.strictEqual(g.getWinner(s, action, 1), -1);
});

test("Forbidden-point finder recognizes 3-3, 4-4, and overline", () => {
    const cases = [
        [[7,5], [7,6], [5,7], [6,7]],
        [[7,4], [7,5], [7,6], [4,7], [5,7], [6,7]],
        [[7,4], [7,5], [7,6], [7,8], [7,9]],
    ];
    for (const stones of cases) {
        const fpf = new ForbiddenPointFinder(15);
        for (const [r, c] of stones) fpf.setStone(r, c, 1);
        assert.strictEqual(fpf.isForbidden(7, 7), true);
    }
});

test("Rule win semantics match V7.19", () => {
    const row = (rule, player, count) => {
        const g = new Gomoku(15, rule);
        const s = g.getInitialState();
        for (let c = 3; c < 3 + count; c++) s[7 * 15 + c] = player;
        return g.getWinner(s, 7 * 15 + 3 + count - 1, player);
    };
    for (const rule of ["renju", "standard", "freestyle"]) {
        assert.strictEqual(row(rule, 1, 5), 1);
        assert.strictEqual(row(rule, -1, 5), -1);
    }
    assert.strictEqual(row("renju", 1, 6), -1);
    assert.strictEqual(row("renju", -1, 6), -1);
    assert.strictEqual(row("standard", 1, 6), null);
    assert.strictEqual(row("standard", -1, 6), null);
    assert.strictEqual(row("freestyle", 1, 6), 1);
    assert.strictEqual(row("freestyle", -1, 6), -1);
});

test("5 spatial planes pad every runtime size to the fixed 15 canvas", () => {
    for (let n = 11; n <= 15; n++) {
        const g = new Gomoku(n, "freestyle");
        const enc = g.encodeState(g.getInitialState(), 1);
        assert.strictEqual(enc.length, 5 * 15 * 15);
        for (let r = 0; r < 15; r++) {
            for (let c = 0; c < 15; c++) {
                assert.strictEqual(enc[r * 15 + c],
                    r < n && c < n ? 1 : 0);
            }
        }
    }
});

test("own and opponent planes are side-to-move relative", () => {
    const g = new Gomoku(15);
    const s = g.getInitialState();
    s[7 * 15 + 7] = 1;
    const b = g.encodeState(s, 1);
    const w = g.encodeState(s, -1);
    assert.strictEqual(b[225 + 7 * 15 + 7], 1);
    assert.strictEqual(w[2 * 225 + 7 * 15 + 7], 1);
});

test("forbidden planes are Renju-only and use the side-specific slot", () => {
    for (const rule of ["renju", "standard", "freestyle"]) {
        const g = new Gomoku(15, rule);
        const s = g.getInitialState();
        for (const [r, c] of [[7,5], [7,6], [5,7], [6,7]]) {
            s[r * 15 + c] = 1;
        }
        const enc = g.encodeState(s, 1);
        assert.strictEqual(enc[3 * 225 + 7 * 15 + 7],
            rule === "renju" ? 1 : 0);
    }
});

test("7 global features match the V7.19 protocol", () => {
    assert.deepStrictEqual(
        Array.from(new Gomoku(15, "freestyle").computeGlobalFeatures(99, 1)),
        [0, 0, 0, 0, 0, 0, 0]);
    assert.deepStrictEqual(
        Array.from(new Gomoku(15, "standard").computeGlobalFeatures(0, 1)),
        [1, 0, 0, 0, 0, 0, 0]);
    assert.deepStrictEqual(
        Array.from(new Gomoku(15, "renju").computeGlobalFeatures(0, 1)),
        [0, 1, -1, 1, 0, 0, 0]);
    assert.deepStrictEqual(
        Array.from(new Gomoku(15, "renju").computeGlobalFeatures(0, -1, 0.2, 2, -1)),
        [0, 1, 1, 1, 0.20000000298023224, 1, 1]);
});

test("unknown rules are rejected", () => {
    assert.throws(() => new Gomoku(15, "tic-tac-toe"));
});
