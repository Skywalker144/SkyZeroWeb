(function (scope) {
    const ORT_VERSION = "1.17.0";
    const ORT_BASE_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

    function loadOnnxRuntime(target = scope) {
        if (!target.ort) target.importScripts(`${ORT_BASE_URL}ort.min.js`);
        target.ort.env.wasm.wasmPaths = ORT_BASE_URL;
        target.ort.env.wasm.numThreads = 1;
        return target.ort;
    }

    const api = { ORT_VERSION, ORT_BASE_URL, loadOnnxRuntime };
    scope.SkyZeroOnnxRuntime = api;
    if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : globalThis);
