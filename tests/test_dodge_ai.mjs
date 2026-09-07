import { test } from "node:test";
import assert from "node:assert";
import { createRequire } from "module";

const require = createRequire(import.meta.url);

const emptyState = () => ({
    W: 450,
    H: 600,
    player: { x: 225, y: 300, hp: 50, invuln: 0.3 },
    projectiles: [],
    lasers: [],
    bombs: [],
    pickups: [],
});

test("Dodge SAC observation keeps the training layout and normalization", () => {
    const { buildObservation, OBSERVATION_SIZE } = require("../dodge-ai.js");
    const observation = buildObservation(emptyState(), {});
    assert.equal(OBSERVATION_SIZE, 271);
    assert.equal(observation.length, 271);
    assert.deepStrictEqual(Array.from(observation.slice(263)), [
        0.5, 0.5, 0.5, 0.5,
        0.8999999761581421, 0.8999999761581421, 1, 1,
    ]);
    assert.ok(Array.from(observation.slice(0, 263)).every(value => value === 0));
});

test("Dodge policy controller accepts only continuous SAC output", () => {
    const { createPolicyController } = require("../dodge-ai.js");
    const weights = new Float32Array(271 * 2 + 2);
    weights[weights.length - 2] = Math.atanh(0.5);
    weights[weights.length - 1] = Math.atanh(-0.25);
    const policy = {
        version: "test",
        obsDim: 271,
        continuous: true,
        actDim: 2,
        arch: [[271, 2, "linear"]],
        weightsB64: Buffer.from(weights.buffer).toString("base64"),
    };
    const controller = createPolicyController({
        policy,
        getState: emptyState,
        types: {},
    });
    assert.equal(controller.load(), true);
    controller.tick(1 / 30);
    assert.ok(Math.abs(controller.move.x - 0.5) < 1e-6);
    assert.ok(Math.abs(controller.move.y + 0.25) < 1e-6);
    const legacy = createPolicyController({
        policy: { ...policy, continuous: false },
        getState: emptyState,
        types: {},
    });
    assert.equal(legacy.load(), false);
});
