import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
const require = createRequire(import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('../public/go/models/manifest.json',import.meta.url)));
const fixture = JSON.parse(readFileSync(new URL('./go-rules.json',import.meta.url)));
const {GoGame} = require('../public/go/engine.js');
const {GoInference} = require('../public/go/inference.js');
const factory = require('../public/go/'+manifest.rules.loader);
const native = await factory({wasmBinary:readFileSync(new URL('../public/go/'+manifest.rules.wasm,import.meta.url))});

test('Go rules and full history features match native EtaZero V4 on all board sizes', () => {
    for (const row of fixture.cases) {
        const game = new GoGame(native,row.size,7.5);
        try {
            for (const action of row.actions) game.play(action);
            for (const key of ['player','finished','winner','board','legal','score']) assert.deepEqual(game.snapshot()[key],row[key],`${row.size} ${row.actions.length} ${key}`);
            const observation = game.observation();
            const expected = new Float32Array(22*361);
            for (const [i,v] of row.spatial) expected[i]=v;
            assert.deepEqual(observation.spatial,expected);
            assert.deepEqual(Array.from(observation.globals),row.globals);
            const copy=game.clone();
            if (!row.finished) {
                copy.play(copy.snapshot().legal[0]);
                assert.deepEqual(game.snapshot().board,row.board);
            }
            copy.dispose();
        } finally { game.dispose(); }
    }
});

test('Go rejects occupied cells, suicide, out-of-range actions and play after two passes', () => {
    const game=new GoGame(native,9,7.5);
    for (const a of [1,0,9,81,10]) game.play(a);
    assert.equal(game.snapshot().board[0],0);
    assert.throws(()=>game.play(0));
    assert.throws(()=>game.play(1));
    assert.throws(()=>game.play(-1));
    assert.throws(()=>game.play(82));
    game.play(81); game.play(81);
    assert.equal(game.snapshot().finished,true);
    assert.throws(()=>game.play(2));
    game.dispose();
});

test('Go inference maps pass and smaller boards to the 19x19 canvas and masks illegal moves', async () => {
    const game=new GoGame(native,9,7.5);
    game.play(0);
    class Tensor { constructor(type,data,dims) { Object.assign(this,{type,data,dims}); } }
    const inference=new GoInference({Tensor},{run:async inputs=> {
        assert.deepEqual(inputs.input_spatial.dims,[1,22,19,19]);
        assert.deepEqual(inputs.input_global.dims,[1,19]);
        const policy=new Float32Array(362);
        policy[361]=5;
        return {policy_logits:{data:policy},wdl_logits:{data:new Float32Array([2,0,-2])}};
    }});
    const result=await inference.evaluate(game);
    assert.equal(result.policy[0],0);
    assert.ok(result.policy[81]>result.policy[1]);
    assert.ok(Math.abs(result.policy.reduce((a,b)=>a+b,0)-1)<1e-6);
    assert.ok(result.value>0);
    game.dispose();
});

test('Go simple ko forbids immediate recapture and a pass releases the ko', () => {
    const game=new GoGame(native,9,7.5);
    for (const action of [9,10,1,18,11,20,80,28,19]) game.play(action);
    assert.equal(game.snapshot().board[10],0);
    assert.equal(game.snapshot().legal.includes(10),false);
    assert.throws(()=>game.play(10));
    game.play(81);game.play(79);
    assert.equal(game.snapshot().legal.includes(10),true);
    game.play(10);
    assert.equal(game.snapshot().board[19],0);
    game.dispose();
});

test('Go release artifacts retain their manifest identities and native inference fixture', async () => {
    const {createHash}=await import('node:crypto');
    const digest=path=>createHash('sha256').update(readFileSync(new URL(path,import.meta.url))).digest('hex');
    assert.equal(digest('../public/go/models/'+manifest.model.file),manifest.model.sha256);
    assert.equal(digest('../public/go/'+manifest.rules.loader),manifest.rules.loader_sha256);
    assert.equal(digest('../public/go/'+manifest.rules.wasm),manifest.rules.wasm_sha256);
    const inference=JSON.parse(readFileSync(new URL('./go-inference.json',import.meta.url)));
    assert.equal(inference.model,manifest.model.file);
    const provenance=JSON.parse(readFileSync(new URL('../tools/go/source.json',import.meta.url)));
    for (const [name,sha256] of Object.entries(provenance.files)) assert.equal(digest('../tools/go/source/'+name),sha256,name);
});
