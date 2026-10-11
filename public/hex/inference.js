(function (scope) {
    class HexInference {
        constructor(ort, session, canvas) { this.ort = ort; this.session = session; this.canvas = canvas; }

        async evaluate(board, player, size, rotation = Math.floor(Math.random() * 2)) {
            const { encode, softmax } = scope.HexEngine;
            const input = encode(board, player, size, this.canvas, rotation);
            const outputs = await this.session.run({
                input_spatial: new this.ort.Tensor('float32', input.spatial, [1, 5, this.canvas, this.canvas]),
                input_global: new this.ort.Tensor('float32', input.globals, [1, 8]),
            });
            const logits = new Float32Array(board.length).fill(-Infinity);
            for (let action = 0; action < board.length; action++) {
                if (!board[action]) logits[action] = outputs.policy_logits.data[input.indices[action]];
            }
            const wdl = softmax(outputs.wdl_logits.data);
            const raw = Object.fromEntries(Object.entries(outputs).map(([name, value]) => [name, Array.from(value.data)]));
            for (const values of Object.values(raw)) if (values.some(value => !Number.isFinite(value))) throw new Error('Non-finite Hex inference');
            return {policy: softmax(logits), value: wdl[0] - wdl[2], wdl, raw};
        }
    }
    scope.HexInference = HexInference;
    if (typeof module !== 'undefined' && module.exports) module.exports = { HexInference };
})(typeof self !== 'undefined' ? self : globalThis);
