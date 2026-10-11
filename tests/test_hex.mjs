import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import test from 'node:test';
import { createHash } from 'node:crypto';

const require = createRequire(import.meta.url);
const { HexGame, encode, softmax } = require('../public/hex/engine.js');
const { search } = require('../public/hex/search.js');

test('six-neighbor connections follow the two player goals', () => {
    const game = new HexGame(5);
    const board = new Int8Array(25);
    for (const action of [4, 8, 12, 16, 20]) board[action] = 1;
    assert.equal(game.winner(board), 1);
    board.fill(0);
    for (const action of [20, 16, 12, 8, 4]) board[action] = -1;
    assert.equal(game.winner(board), -1);
    board.fill(0);
    for (const action of [0, 5, 11, 16, 21]) board[action] = 1;
    assert.equal(game.winner(board), 0);
});

test('rules and canonical inputs match native V1.4 Game', () => {
    const fixture = JSON.parse(readFileSync(new URL('hex-rules.json', import.meta.url)));
    for (const item of fixture.cases) {
        const game = new HexGame(item.size);
        let board = new Int8Array(item.size ** 2);
        for (let ply = 0; ply < item.moves.length; ply++) {
            board = game.play(board, item.moves[ply], ply % 2 ? -1 : 1);
            assert.equal(game.winner(board), item.winners[ply]);
        }
        assert.deepEqual(Array.from(board), item.board);
        const features = encode(board, item.player, item.size, fixture.canvas, 0);
        assert.deepEqual(Array.from(features.spatial), item.spatial);
        assert.deepEqual(Array.from(features.globals), item.globals);
    }
});

test('white transpose and 180-degree rotation restore policy coordinates', () => {
    for (const size of [9, 15]) for (const player of [1, -1]) for (const rotation of [0, 1]) {
        const board = new Int8Array(size ** 2);
        board[2 * size + 3] = player;
        const features = encode(board, player, size, 15, rotation);
        assert.equal(features.spatial[225 + features.indices[2 * size + 3]], 1);
        assert.equal(new Set(features.indices).size, size ** 2);
        for (const index of features.indices) assert.equal(features.spatial[index], 1);
        assert.equal(features.globals[7], player === -1 ? 1 : 0);
    }
});

test('illegal moves fail explicitly', () => {
    const game = new HexGame(5);
    const board = game.play(new Int8Array(25), 0, 1);
    assert.throws(() => game.play(board, 0, -1));
    assert.throws(() => game.play(board, 25, -1));
});

test('PUCT finds an immediate connection and backs up opponent perspective', async () => {
    const game = new HexGame(5);
    const board = new Int8Array(25);
    for (const action of [0, 5, 10, 15]) board[action] = 1;
    const result = await search({game, board, player: 1, maxVisits: 128, thinkMs: Infinity,
        evaluate: async state => ({policy: softmax(Float32Array.from(state, cell => cell ? -Infinity : 0)), value: 0})});
    assert.equal(result.action, 20);
    assert.ok(result.value > 0);
    assert.equal(result.visits, 128);
});

test('search cancellation cannot publish a stale move', async () => {
    let current = true;
    const result = await search({game: new HexGame(5), board: new Int8Array(25), player: 1,
        maxVisits: 8, thinkMs: Infinity, shouldContinue: () => current,
        evaluate: async () => { current = false; return {policy: new Float32Array(25).fill(1 / 25), value: 0}; }});
    assert.equal(result, null);
});

test('white PUCT uses the white value frame', async () => {
    const game = new HexGame(5);
    const board = new Int8Array(25);
    for (const action of [0, 1, 2, 3]) board[action] = -1;
    const result = await search({game, board, player: -1, maxVisits: 128, thinkMs: Infinity,
        evaluate: async state => ({policy: softmax(Float32Array.from(state, cell => cell ? -Infinity : 0)), value: 0})});
    assert.equal(result.action, 4);
    assert.ok(result.value > 0);
});

test('released ONNX models match their manifest and browser fixtures', () => {
    const manifest = JSON.parse(readFileSync(new URL('../public/hex/models/manifest.json', import.meta.url)));
    assert.ok(manifest.models.some(model => model.id === manifest.default));
    for (const model of manifest.models) {
        const bytes = readFileSync(new URL(`../public/hex/models/${model.file}`, import.meta.url));
        assert.equal(bytes.length, model.bytes);
        assert.equal(createHash('sha256').update(bytes).digest('hex'), model.sha256);
        const fixture = JSON.parse(readFileSync(new URL(`hex-${model.id}-inference.json`, import.meta.url)));
        assert.equal(fixture.model, model.file);
        assert.equal(fixture.canvas, manifest.network.canvas);
    }
});


test('Hex opening acceptance matches native hand-calculated cases', () => {
    const { acceptanceRate } = require('../public/hex/opening.js');
    assert.equal(acceptanceRate(0.5, 20, 0.001), 1);
    assert.equal(acceptanceRate(0.75, 2, 0.001), 0.5625);
    assert.equal(acceptanceRate(0, 20, 0.001), 0.001);
    assert.equal(acceptanceRate(1, 20, 0.001), 0.001);
    for (const values of [[NaN, 20, 0.001], [1.1, 20, 0.001], [0.5, 20, 0], [0.5, -1, 0.001]]) {
        assert.throws(() => acceptanceRate(...values));
    }
});

test('balanced opening evaluates White W and resamples after rejection', async () => {
    const { generateOpening } = require('../public/hex/opening.js');
    const draws = [0.04, 0.5, 0.84, 0.95];
    const evaluated = [];
    const result = await generateOpening({size: 5, config: {balance_exponent: 2, min_accept_rate: 0.001},
        random: () => draws.shift(), evaluate: async (board, player) => {
            assert.equal(player, -1);
            assert.equal(board.filter(value => value === 1).length, 1);
            evaluated.push(board.indexOf(1));
            return {wdl: evaluated.length === 1 ? [0.9, 0.1, 0] : [0.5, 0.4, 0.1]};
        }});
    assert.deepEqual(evaluated, [1, 21]);
    assert.equal(result.action, 21);
    assert.equal(result.attempts, 2);
    assert.equal(result.whiteWinProbability, 0.5);
    assert.equal(result.whiteWinLoss, 0.4);
    assert.equal(result.acceptanceRate, 1);
});

test('opening cancellation discards an in-flight evaluation and never publishes', async () => {
    const { generateOpening } = require('../public/hex/opening.js');
    let current = true;
    const result = await generateOpening({size: 5, config: {balance_exponent: 20, min_accept_rate: 0.001},
        shouldContinue: () => current, evaluate: async () => { current = false; return {wdl: [0.5, 0, 0.5]}; }});
    assert.equal(result, null);
});

test('invalid opening WDL fails explicitly', async () => {
    const { generateOpening } = require('../public/hex/opening.js');
    await assert.rejects(generateOpening({size: 5, config: {balance_exponent: 20, min_accept_rate: 0.001},
        evaluate: async () => ({wdl: [NaN, 0, 0]})}));
});
