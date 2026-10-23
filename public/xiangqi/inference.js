(function (scope) {
    function softmax(values) {
        const maximum = Math.max(...values);
        const result = Float32Array.from(values, value => Math.exp(value - maximum));
        const total = result.reduce((sum, value) => sum + value, 0);
        return result.map(value => value / total);
    }

    class XiangqiInference {
        constructor(ort, session, moves) {
            this.ort = ort;
            this.session = session;
            const indices = new Map(moves.map((move, action) => [move.join(','), action]));
            this.spatialMaps = [];
            this.actionMaps = [];
            for (let choice = 0; choice < 4; choice++) {
                const transform = (x, y) => {
                    if (choice & 2) { x = 8 - x; y = 9 - y; }
                    if (choice & 1) x = 8 - x;
                    return [x, y];
                };
                const spatial = new Int32Array(100);
                for (let y = 0; y < 10; y++) for (let x = 0; x < 9; x++) {
                    const [u, v] = transform(x, y);
                    spatial[y * 10 + x] = v * 10 + u;
                }
                this.spatialMaps.push(spatial);
                this.actionMaps.push(moves.map(([x, y, u, v]) => indices.get([...transform(x, y), ...transform(u, v)].join(','))));
            }
        }

        async evaluate(game, mirror = Math.floor(Math.random() * 2), includeRaw = false) {
            const state = game.snapshot();
            const observation = game.observation();
            const choice = mirror + 2 * (state.player === -1);
            const mapping = this.spatialMaps[choice];
            const spatial = new Float32Array(11300);
            for (let plane = 0; plane < 113; plane++) {
                for (let y = 0; y < 10; y++) for (let x = 0; x < 9; x++) {
                    const index = y * 10 + x;
                    spatial[plane * 100 + mapping[index]] = observation.spatial[plane * 100 + index];
                }
            }
            const output = await this.session.run({
                input_spatial: new this.ort.Tensor('float32', spatial, [1, 113, 10, 10]),
                input_global: new this.ort.Tensor('float32', observation.globals, [1, 8]),
            });
            for (const tensor of Object.values(output)) {
                if (tensor.data.some(value => !Number.isFinite(value))) throw new Error('Non-finite Xiangqi inference');
            }
            const logits = state.legal.map(action => output.policy_logits.data[this.actionMaps[choice][action]]);
            const probabilities = softmax(logits);
            const policy = new Float32Array(this.actionMaps[choice].length);
            state.legal.forEach((action, index) => { policy[action] = probabilities[index]; });
            const wdl = softmax(output.wdl_logits.data);
            return {policy, value: wdl[0] - wdl[2], wdl,
                ...(includeRaw ? {raw: Object.fromEntries(Object.entries(output).map(([name, tensor]) => [name, Array.from(tensor.data)]))} : {})};
        }
    }

    scope.XiangqiInference = XiangqiInference;
    if (typeof module !== 'undefined' && module.exports) module.exports = {XiangqiInference};
})(typeof self !== 'undefined' ? self : globalThis);
