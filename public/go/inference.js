(function (scope) {
    function softmax(values) {
        const max = Math.max(...values);
        const result = Float32Array.from(values,value=>Math.exp(value-max));
        const total = result.reduce((a,b)=>a+b,0);
        return result.map(value=>value/total);
    }
    class GoInference {
        constructor(ort, session) { this.ort=ort; this.session=session; }
        async evaluate(game) {
            const state=game.snapshot(), input=game.observation();
            const outputs=await this.session.run({
                input_spatial:new this.ort.Tensor('float32',input.spatial,[1,22,19,19]),
                input_global:new this.ort.Tensor('float32',input.globals,[1,19])
            });
            const raw=Object.fromEntries(Object.entries(outputs).map(([name,value])=>[name,Array.from(value.data)]));
            if (Object.values(raw).some(values=>values.some(value=>!Number.isFinite(value)))) throw new Error('Non-finite Go inference');
            const logits=new Float32Array(state.size**2+1).fill(-Infinity);
            for (const a of state.legal) logits[a]=outputs.policy_logits.data[a===state.size**2 ? 361 : Math.floor(a/state.size)*19+a%state.size];
            const wdl=softmax(outputs.wdl_logits.data);
            return {policy:softmax(logits),wdl,value:wdl[0]-wdl[2],raw};
        }
    }
    scope.GoInference=GoInference;
    if (typeof module !== 'undefined' && module.exports) module.exports={GoInference};
})(typeof self !== 'undefined' ? self : globalThis);
