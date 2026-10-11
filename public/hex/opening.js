(function (scope) {
    function acceptanceRate(whiteWinProbability, exponent, minimum) {
        if (!Number.isFinite(whiteWinProbability) || whiteWinProbability < 0 || whiteWinProbability > 1
            || !Number.isFinite(exponent) || exponent < 0 || exponent > 100
            || !Number.isFinite(minimum) || minimum <= 0 || minimum > 1) {
            throw new Error('Invalid Hex opening probability/exponent/minimum');
        }
        const bias = 2 * whiteWinProbability - 1;
        return Math.max((1 - bias * bias) ** exponent, minimum);
    }

    async function generateOpening({size, config, evaluate, random = Math.random,
        shouldContinue = () => true, onProgress = () => {}}) {
        new scope.HexEngine.HexGame(size);
        acceptanceRate(0.5, config.balance_exponent, config.min_accept_rate);
        for (let attempts = 1; ; attempts++) {
            if (!shouldContinue()) return null;
            const action = Math.floor(random() * size * size);
            const board = new Int8Array(size * size);
            board[action] = 1;
            const {wdl} = await evaluate(board, -1);
            if (!shouldContinue()) return null;
            if (!wdl || wdl.length !== 3 || Array.from(wdl).some(value => !Number.isFinite(value) || value < 0 || value > 1)
                || Math.abs(wdl[0] + wdl[1] + wdl[2] - 1) > 1e-5) throw new Error('Invalid Hex opening WDL');
            const whiteWinProbability = wdl[0];
            const rate = acceptanceRate(whiteWinProbability, config.balance_exponent, config.min_accept_rate);
            if (attempts === 1 || attempts % 8 === 0) onProgress({attempts});
            if (random() >= rate) continue;
            return {action, attempts, whiteWinProbability, whiteWinLoss: wdl[0] - wdl[2], acceptanceRate: rate};
        }
    }

    const api = { acceptanceRate, generateOpening };
    scope.HexOpening = api;
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof self !== 'undefined' ? self : globalThis);
