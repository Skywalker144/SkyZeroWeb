import { test } from "node:test";
import assert from "node:assert";
import { createRequire } from "module";

const require = createRequire(import.meta.url);
const { softmax, transformCoord } = require("../public/gomoku/engine/mcts.js");

test("shared inference runtime owns reversible D4 transforms", () => {
    const { transformSpatial, undoSpatial } = require("../public/gomoku/engine/inference.js");
    const input = Float32Array.from({ length: 18 }, (_, index) => index + 1);
    for (let rotation = 0; rotation < 4; rotation++) {
        for (const flip of [false, true]) {
            const transformed = transformSpatial(
                input, 2, 3, rotation, flip, transformCoord);
            assert.deepStrictEqual(
                Array.from(undoSpatial(
                    transformed, 2, 3, rotation, flip, transformCoord)),
                Array.from(input));
        }
    }
});

test("shared inference runtime exposes every policy head from one forward pass", async () => {
    const { GomokuInference } = require("../public/gomoku/engine/inference.js");
    const policy = new Float32Array(20);
    policy.set([0, 1, 2, 3], 0);
    policy.set([3, 2, 1, 0], 4);
    policy.set([1, 3, 0, 2], 16);
    const session = {
        async run(feeds) {
            assert.deepStrictEqual(feeds.input_spatial.dims, [1, 5, 2, 2]);
            assert.deepStrictEqual(feeds.input_global.dims, [1, 7]);
            return {
                policy_logits: { data: policy },
                value_wdl_logits: { data: Float32Array.of(2, 1, 0) },
                value_st_error_sq: { data: Float32Array.of(0.25) },
            };
        },
    };
    const ort = {
        Tensor: class {
            constructor(type, data, dims) {
                this.type = type;
                this.data = data;
                this.dims = dims;
            }
        },
    };
    const game = {
        encodeState: () => new Float32Array(20),
        computeGlobalFeatures: () => new Float32Array(7),
        getLegalActions: () => Uint8Array.of(1, 0, 1, 1),
    };
    const runtime = new GomokuInference({
        ort, softmax, transformCoord, maxBoardSize: 2,
        boardSize: 2, game, session, random: () => 0,
    });
    const result = await runtime.evaluate(Int8Array.of(1, 0, 0, -1), -1, {
        isRoot: true,
        policyOptimism: 0.5,
    });
    assert.equal(result.stError, 0.5);
    assert.equal(result.policyMainSoft[1], 0);
    assert.equal(result.policyOppSoft[1], 0);
    assert.equal(result.policyOptSoft[1], 0);
    assert.equal(result.policyPriorSoft[1], 0);
    assert.ok(Math.abs(result.wdl.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
    assert.ok(result.policyPriorSoft[3] > result.policyPriorSoft[0]);
});
