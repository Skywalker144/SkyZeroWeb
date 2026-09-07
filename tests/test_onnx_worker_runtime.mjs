import { test } from "node:test";
import assert from "node:assert";
import { createRequire } from "module";

const require = createRequire(import.meta.url);

test("worker runtime keeps the ONNX version and WASM setup in one module", () => {
    const { ORT_BASE_URL, loadOnnxRuntime } = require("../onnx-worker-runtime.js");
    const loaded = [];
    const worker = {
        importScripts(url) {
            loaded.push(url);
            this.ort = { env: { wasm: {} } };
        },
    };
    const ort = loadOnnxRuntime(worker);
    assert.deepStrictEqual(loaded, [`${ORT_BASE_URL}ort.min.js`]);
    assert.equal(ort.env.wasm.wasmPaths, ORT_BASE_URL);
    assert.equal(ort.env.wasm.numThreads, 1);
    loadOnnxRuntime(worker);
    assert.equal(loaded.length, 1);
});
