importScripts('/shared/onnx-worker-runtime.js?v=20261009', '/shared/puct-search.js?v=20261023',
    'engine.js?v=20261023', 'inference.js?v=20261023');

let native, inference, moves;
let generation = 0;

self.onmessage = async ({data}) => {
    const {id, type} = data;
    if (type === 'cancel') { generation++; return; }
    try {
        if (type === 'init') {
            const {manifest} = data;
            importScripts(manifest.rules.loader);
            native = await createXiangqiRules({locateFile: () => manifest.rules.wasm});
            const index = native._xq_moves() >>> 2;
            moves = Array.from({length: manifest.model.source.action_count}, (_, action) =>
                Array.from(native.HEAP32.subarray(index + action * 4, index + action * 4 + 4)));
            self.postMessage({id, type: 'loading'});
            const ort = SkyZeroOnnxRuntime.loadOnnxRuntime();
            const response = await fetch('models/' + manifest.model.file);
            if (!response.ok) throw new Error(`Model download failed: HTTP ${response.status}`);
            const session = await ort.InferenceSession.create(await response.arrayBuffer(), {executionProviders: ['wasm']});
            inference = new XiangqiInference(ort, session, moves);
            self.postMessage({id, type: 'ready', moves});
            return;
        }
        const game = new XiangqiEngine.XiangqiGame(native, data.fen || '');
        try {
            for (const action of data.actions) game.play(action);
            if (type === 'position') {
                self.postMessage({id, type: 'position', state: {...game.snapshot(), check: !!game.observation().globals[6]}});
            } else if (type === 'evaluate') {
                const result = await inference.evaluate(game, data.mirror, true);
                self.postMessage({id, type: 'evaluation', result});
            } else if (type === 'search') {
                const token = ++generation;
                const result = await SkyZeroPUCT.search({state: game, maxVisits: 512, thinkMs: data.thinkMs,
                    legal: position => position.snapshot().legal,
                    play: (position, action) => { const next = position.clone(); next.play(action); return next; },
                    terminalValue: position => { const state = position.snapshot(); return state.finished ? state.winner * state.player : null; },
                    evaluate: position => inference.evaluate(position), dispose: position => position.dispose(),
                    shouldContinue: () => token === generation,
                    onProgress: result => self.postMessage({id, type: 'progress', result})});
                if (result && token === generation) self.postMessage({id, type: 'move', result});
            }
        } finally { game.dispose(); }
    } catch (error) { self.postMessage({id, type: 'error', message: error.message || String(error)}); }
};
