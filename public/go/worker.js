importScripts('/shared/onnx-worker-runtime.js?v=20261009','/shared/puct-search.js?v=20261102',
    'engine.js?v=20261105','inference.js?v=20261105');
let native, inference, generation=0;
self.onmessage=async ({data})=> {
    const {id,type}=data;
    if (type==='cancel') { generation++; return; }
    try {
        if (type==='init') {
            importScripts(data.manifest.rules.loader);
            native=await createGoRules({locateFile:()=>data.manifest.rules.wasm});
            const ort=SkyZeroOnnxRuntime.loadOnnxRuntime();
            const response=await fetch('models/'+data.manifest.model.file);
            if (!response.ok) throw new Error(`Model HTTP ${response.status}`);
            self.postMessage({id,type:'loading'});
            const session=await ort.InferenceSession.create(await response.arrayBuffer(),{executionProviders:['wasm']});
            inference=new GoInference(ort,session);
            self.postMessage({id,type:'ready'});
            return;
        }
        const game=new GoEngine.GoGame(native,data.size ?? 19,data.komi ?? 7.5);
        try {
            for (const action of data.actions) game.play(action);
            if (type==='position') self.postMessage({id,type,state:game.snapshot()});
            else if (type==='evaluate') self.postMessage({id,type:'evaluation',result:await inference.evaluate(game)});
            else if (type==='search') {
                const token=++generation;
                const result=await SkyZeroPUCT.search({state:game,maxVisits:512,thinkMs:data.thinkMs,
                    legal:position=>position.snapshot().legal,
                    play:(position,action)=>{const next=position.clone();next.play(action);return next;},
                    terminalValue:position=>{const s=position.snapshot();return s.finished ? s.winner*s.player : null;},
                    evaluate:position=>inference.evaluate(position),dispose:position=>position.dispose(),
                    shouldContinue:()=>token===generation,
                    onProgress:result=>self.postMessage({id,type:'progress',result})});
                if (result && token===generation) self.postMessage({id,type:'move',result});
            }
        } finally { game.dispose(); }
    } catch (error) { self.postMessage({id,type:'error',message:error.message || String(error)}); }
};
