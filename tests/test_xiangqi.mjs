import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import test from 'node:test';

const require = createRequire(import.meta.url);
const root = new URL('../', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('public/xiangqi/models/manifest.json', root)));
const createRules = require(new URL(`public/xiangqi/${manifest.rules.loader}`, root).pathname);
const wasmBinary = readFileSync(new URL(`public/xiangqi/${manifest.rules.wasm}`, root));
const native = await createRules({wasmBinary});
const {XiangqiGame} = require('../public/xiangqi/engine.js');
const fixture = JSON.parse(readFileSync(new URL('tests/xiangqi-rules.json', root)));

test('Xiangqi model, WASM and source snapshot retain their published identities', () => {
    const digest = bytes => createHash('sha256').update(bytes).digest('hex');
    const bytes = readFileSync(new URL(`public/xiangqi/models/${manifest.model.file}`, root));
    assert.equal(digest(bytes), manifest.model.sha256);
    assert.equal(bytes.length, manifest.model.bytes);
    for (const key of ['loader', 'wasm']) {
        assert.equal(digest(readFileSync(new URL(`public/xiangqi/${manifest.rules[key]}`, root))), manifest.rules[key + '_sha256']);
    }
    const provenance = JSON.parse(readFileSync(new URL('tools/xiangqi/source.json', root)));
    for (const [path, hash] of Object.entries(provenance.files)) {
        assert.equal(digest(readFileSync(new URL(`tools/xiangqi/source/${path}`, root))), hash, path);
    }
    const inference = JSON.parse(readFileSync(new URL('tests/xiangqi-inference.json', root)));
    assert.equal(inference.model, manifest.model.file);
    assert.equal(manifest.model.source.weights, 'swa');
});

test('original Xiangqi rules and complete observations match native trajectories', () => {
    for (const item of fixture.cases) {
        const game = new XiangqiGame(native, item.fen);
        try {
            for (let turn = 0; turn < item.states.length; turn++) {
                const expected = item.states[turn];
                const actual = game.snapshot();
                for (const key of ['board', 'player', 'finished', 'winner', 'reason', 'legal']) {
                    assert.deepEqual(actual[key], expected[key], `${item.name} turn ${turn}: ${key}`);
                }
                const observation = game.observation();
                assert.deepEqual(Array.from(observation.spatial.keys()).filter(i => observation.spatial[i]), expected.spatial);
                assert.deepEqual(Array.from(observation.globals), Array.from(new Float32Array(expected.globals)));
                if (turn < item.actions.length) game.play(item.actions[turn]);
            }
        } finally { game.dispose(); }
    }
});

test('Xiangqi initial perft, independent clones and illegal moves', () => {
    const game = new XiangqiGame(native);
    try {
        assert.equal(game.snapshot().legal.length, 44);
        let count = 0;
        for (const action of game.snapshot().legal) {
            const next = game.clone();
            try { next.play(action); count += next.snapshot().legal.length; }
            finally { next.dispose(); }
        }
        assert.equal(count, 1920);
        assert.equal(game.snapshot().player, 1);
        assert.throws(() => game.play(-1), /Illegal/);
    } finally { game.dispose(); }
});

test('PUCT selects a forced win and releases every search state', async () => {
    const {search} = require('../public/shared/puct-search.js');
    let alive = 0;
    const progress = [];
    const result = await search({state: {ply: 0, player: 1},
        legal: state => state.ply ? [] : [0, 1],
        play: (state, action) => { alive++; return {ply: 1, player: -1, action}; },
        terminalValue: state => state.ply ? (state.action === 1 ? -1 : 1) : null,
        dispose: () => alive--,
        evaluate: async () => ({policy: [0.5, 0.5], value: 0}), maxVisits: 30, thinkMs: 10000,
        onProgress: update => progress.push(update)});
    assert.ok(progress.length > 0);
    assert.ok(progress.every(update => Number.isFinite(update.elapsedMs) && update.elapsedMs >= 0));
    assert.equal(result.action, 1);
    assert.equal(result.candidates[0].action, 1);
    assert.equal(result.candidates[0].value, 1);
    assert.equal(result.candidates.reduce((sum, item) => sum + item.visits, 0), result.visits - 1);
    assert.equal(alive, 0);
});

test('PUCT cancellation and evaluation failures release search states', async () => {
    const {search} = require('../public/shared/puct-search.js');
    for (const fail of [false, true]) {
        let alive = 0, calls = 0;
        const promise = search({state: 0, legal: () => [0], play: state => { alive++; return state + 1; },
            terminalValue: () => null, dispose: () => alive--,
            evaluate: async () => { if (++calls > 1 && fail) throw new Error('probe'); return {policy: [1], value: 0}; },
            shouldContinue: () => calls < 2, maxVisits: 20});
        if (fail) await assert.rejects(promise, /probe/);
        else assert.equal(await promise, null);
        assert.equal(alive, 0);
    }
});
