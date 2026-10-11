importScripts('/shared/onnx-worker-runtime.js?v=20261009', 'engine.js?v=20261009',
    'search.js?v=20261009', 'inference.js?v=20261009', 'opening.js?v=20261009');

let inference = null;
let generation = 0;
self.onmessage = async ({data}) => {
    const {id, type} = data;
    if (type === 'cancel') { generation++; return; }
    try {
        if (type === 'init') {
            const ort = SkyZeroOnnxRuntime.loadOnnxRuntime();
            const response = await fetch(data.url);
            if (!response.ok) throw new Error(`Model download failed: HTTP ${response.status}`);
            const bytes = await response.arrayBuffer();
            self.postMessage({id, type: 'loading'});
            const session = await ort.InferenceSession.create(bytes, {executionProviders: ['wasm']});
            inference = new HexInference(ort, session, data.canvas);
            self.postMessage({id, type: 'ready'});
        } else if (type === 'evaluate') {
            const result = await inference.evaluate(new Int8Array(data.board), data.player, data.size, data.rotation);
            self.postMessage({id, type: 'evaluation', result});
        } else if (type === 'opening') {
            const token = ++generation;
            const result = await HexOpening.generateOpening({size: data.size, config: data.config,
                shouldContinue: () => token === generation,
                evaluate: (board, player) => inference.evaluate(board, player, data.size),
                onProgress: result => self.postMessage({id, type: 'opening-progress', result})});
            if (result && token === generation) self.postMessage({id, type: 'opening', result});
        } else if (type === 'search') {
            const token = ++generation;
            const result = await HexSearch.search({game: new HexEngine.HexGame(data.size),
                board: new Int8Array(data.board), player: data.player, maxVisits: data.maxVisits, thinkMs: data.thinkMs,
                shouldContinue: () => token === generation,
                evaluate: (board, player) => inference.evaluate(board, player, data.size),
                onProgress: result => self.postMessage({id, type: 'progress', result})});
            if (result && token === generation) self.postMessage({id, type: 'move', result});
        }
    } catch (error) { self.postMessage({id, type: 'error', message: error.message}); }
};
