(function (scope) {
    function transformSpatial(input, channels, size, rotation, flip, transform) {
        const area = size * size;
        const output = new Float32Array(input.length);
        for (let channel = 0; channel < channels; channel++) {
            for (let row = 0; row < size; row++) {
                for (let col = 0; col < size; col++) {
                    const [targetRow, targetCol] = transform(
                        row, col, size, rotation, flip);
                    output[channel * area + targetRow * size + targetCol] =
                        input[channel * area + row * size + col];
                }
            }
        }
        return output;
    }

    function undoSpatial(input, channels, size, rotation, flip, transform) {
        const area = size * size;
        const output = new Float32Array(input.length);
        for (let channel = 0; channel < channels; channel++) {
            for (let row = 0; row < size; row++) {
                for (let col = 0; col < size; col++) {
                    const [targetRow, targetCol] = transform(
                        row, col, size, rotation, flip);
                    output[channel * area + row * size + col] =
                        input[channel * area + targetRow * size + targetCol];
                }
            }
        }
        return output;
    }

    function concatChunks(chunks, total) {
        const result = new Uint8Array(total);
        let offset = 0;
        for (const chunk of chunks) {
            result.set(chunk, offset);
            offset += chunk.length;
        }
        return result;
    }

    async function fetchModel(url, onProgress = () => {}) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`fetch ${url} → ${response.status}`);
        const expected = Number(response.headers.get("Content-Length")) || 0;
        if (!response.body) {
            const bytes = new Uint8Array(await response.arrayBuffer());
            onProgress({ percent: 100, loaded: bytes.byteLength, total: bytes.byteLength });
            return bytes;
        }
        const reader = response.body.getReader();
        const chunks = [];
        let loaded = 0;
        if (expected > 0) onProgress({ percent: 0, loaded: 0, total: expected });
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            loaded += value.length;
            onProgress({
                percent: expected > 0 ? loaded / expected * 100 : null,
                loaded,
                total: expected || null,
            });
        }
        onProgress({ percent: 100, loaded, total: expected || loaded });
        return concatChunks(chunks, expected || loaded);
    }

    class GomokuInference {
        constructor({
            ort,
            softmax,
            transformCoord,
            maxBoardSize = 15,
            boardSize = 15,
            game = null,
            session = null,
            cacheSize = 0,
            random = Math.random,
        }) {
            this.ort = ort;
            this.softmax = softmax;
            this.transformCoord = transformCoord;
            this.maxBoardSize = maxBoardSize;
            this.boardSize = boardSize;
            this.game = game;
            this.session = session;
            this.cacheSize = cacheSize;
            this.random = random;
            this.pda = 0;
            this.pdaPla = 0;
            this.cache = new Map();
        }

        async load(modelUrl, onProgress) {
            const bytes = await fetchModel(modelUrl, onProgress);
            this.session = await this.ort.InferenceSession.create(bytes, {
                executionProviders: ["wasm"],
                intraOpNumThreads: 1,
                interOpNumThreads: 1,
            });
            this.clear();
        }

        setGame(game, boardSize) {
            this.game = game;
            this.boardSize = boardSize;
            this.clear();
        }

        setConditioning(pda, pdaPla) {
            if (this.pda === pda && this.pdaPla === pdaPla) return;
            this.pda = pda;
            this.pdaPla = pdaPla;
            this.clear();
        }

        clear() {
            this.cache.clear();
        }

        cacheKey(state, toPlay) {
            return `${this.boardSize}|${this.game.rule}|${toPlay}`
                + `|pda=${this.pda}|pdaPla=${this.pdaPla}|`
                + Array.from(state).join("");
        }

        cachePut(key, value) {
            if (this.cacheSize <= 0) return;
            if (this.cache.has(key)) this.cache.delete(key);
            this.cache.set(key, value);
            if (this.cache.size > this.cacheSize) {
                this.cache.delete(this.cache.keys().next().value);
            }
        }

        async evaluate(
            state,
            toPlay,
            { isRoot = false, policyOptimism = isRoot ? 0 : 1 } = {},
        ) {
            if (!this.session) throw new Error("session not ready");
            if (!this.game) throw new Error("game not ready");
            const modelSize = this.maxBoardSize;
            const modelArea = modelSize * modelSize;
            const boardArea = this.boardSize * this.boardSize;
            let ply = 0;
            for (const value of state) if (value !== 0) ply++;
            const spatial = this.game.encodeState(state, toPlay);
            const globalFeatures = this.game.computeGlobalFeatures(
                ply, toPlay, 0, this.pda, this.pdaPla);
            const key = this.cacheKey(state, toPlay);
            let raw = this.cache.get(key);
            if (raw) {
                this.cache.delete(key);
                this.cache.set(key, raw);
            } else {
                const transformType = Math.floor(this.random() * 8);
                const rotation = transformType % 4;
                const flip = transformType >= 4;
                const output = await this.session.run({
                    input_spatial: new this.ort.Tensor(
                        "float32",
                        transformSpatial(
                            spatial, 5, modelSize, rotation, flip,
                            this.transformCoord),
                        [1, 5, modelSize, modelSize],
                    ),
                    input_global: new this.ort.Tensor(
                        "float32", globalFeatures, [1, 7]),
                });
                raw = {
                    policy: undoSpatial(
                        output.policy_logits.data, 5, modelSize,
                        rotation, flip, this.transformCoord),
                    valueLogits: new Float32Array(output.value_wdl_logits.data),
                    stErrorSquared: output.value_st_error_sq.data[0],
                };
                this.cachePut(key, raw);
            }
            const channel = index => {
                const values = new Float32Array(boardArea);
                for (let row = 0; row < this.boardSize; row++) {
                    for (let col = 0; col < this.boardSize; col++) {
                        values[row * this.boardSize + col] = raw.policy[
                            index * modelArea + row * modelSize + col];
                    }
                }
                return values;
            };
            const main = channel(0);
            const opponent = channel(1);
            const optimistic = channel(4);
            const legal = this.game.getLegalActions(state, toPlay);
            const blend = Math.max(0, Math.min(1, policyOptimism));
            const priorMasked = new Float32Array(boardArea);
            const mainMasked = new Float32Array(boardArea);
            const opponentMasked = new Float32Array(boardArea);
            const optimisticMasked = new Float32Array(boardArea);
            for (let index = 0; index < boardArea; index++) {
                if (!legal[index]) {
                    priorMasked[index] = -Infinity;
                    mainMasked[index] = -Infinity;
                    opponentMasked[index] = -Infinity;
                    optimisticMasked[index] = -Infinity;
                    continue;
                }
                priorMasked[index] = main[index]
                    + (optimistic[index] - main[index]) * blend;
                mainMasked[index] = main[index];
                opponentMasked[index] = opponent[index];
                optimisticMasked[index] = optimistic[index];
            }
            const wdl = this.softmax(new Float64Array(raw.valueLogits));
            return {
                policyPriorSoft: new Float32Array(this.softmax(priorMasked)),
                policyPriorMaskedLogits: priorMasked,
                policyMainSoft: new Float32Array(this.softmax(mainMasked)),
                policyMainMaskedLogits: mainMasked,
                policyOppSoft: new Float32Array(this.softmax(opponentMasked)),
                policyOptSoft: new Float32Array(this.softmax(optimisticMasked)),
                wdl: new Float64Array(wdl),
                stError: Math.sqrt(Math.max(0, raw.stErrorSquared)),
            };
        }
    }

    const api = {
        GomokuInference,
        fetchModel,
        transformSpatial,
        undoSpatial,
    };
    scope.GomokuInferenceRuntime = api;
    if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : globalThis);
