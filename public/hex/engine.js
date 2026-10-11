(function (scope) {
    const NEIGHBORS = [[0, -1], [1, -1], [1, 0], [0, 1], [-1, 1], [-1, 0]];

    class HexGame {
        constructor(size) {
            if (!Number.isInteger(size) || size < 5 || size > 15) throw new Error('Invalid Hex size');
            this.size = size;
        }

        connection(board, player) {
            const seen = new Uint8Array(board.length);
            const parent = new Int16Array(board.length).fill(-1);
            const queue = [];
            for (let i = 0; i < this.size; i++) {
                const action = player === 1 ? i : i * this.size;
                if (board[action] === player) { queue.push(action); seen[action] = 1; }
            }
            for (let head = 0; head < queue.length; head++) {
                const action = queue[head];
                const row = Math.floor(action / this.size), col = action % this.size;
                if ((player === 1 ? row : col) === this.size - 1) {
                    const path = [];
                    for (let point = action; point !== -1; point = parent[point]) path.push(point);
                    return path;
                }
                for (const [dr, dc] of NEIGHBORS) {
                    const r = row + dr, c = col + dc, next = r * this.size + c;
                    if (r < 0 || c < 0 || r >= this.size || c >= this.size || seen[next] || board[next] !== player) continue;
                    seen[next] = 1;
                    parent[next] = action;
                    queue.push(next);
                }
            }
            return [];
        }

        winner(board) {
            for (const player of [1, -1]) if (this.connection(board, player).length) return player;
            if (!board.includes(0)) throw new Error('Full Hex board has no connection');
            return 0;
        }

        play(board, action, player) {
            if (board.length !== this.size ** 2 || !Number.isInteger(action) || action < 0 || action >= board.length
                || board[action] !== 0 || ![1, -1].includes(player) || this.winner(board)) throw new Error('Illegal Hex move');
            const next = new Int8Array(board);
            next[action] = player;
            return next;
        }
    }

    function encode(board, player, size, canvas, rotation = 0) {
        const area = canvas ** 2;
        const spatial = new Float32Array(5 * area);
        const globals = Float32Array.of(0, 0, 0, 0, 0, 0, 1, player === -1 ? 1 : 0);
        const indices = new Int32Array(board.length);
        for (let action = 0; action < board.length; action++) {
            let row = Math.floor(action / size), col = action % size;
            if (rotation) { row = canvas - 1 - row; col = canvas - 1 - col; }
            if (player === -1) [row, col] = [col, row];
            const index = row * canvas + col;
            indices[action] = index;
            spatial[index] = 1;
            spatial[area + index] = board[action] === player ? 1 : 0;
            spatial[2 * area + index] = board[action] === -player ? 1 : 0;
        }
        return { spatial, globals, indices };
    }

    function softmax(logits) {
        let maximum = -Infinity;
        for (const value of logits) maximum = Math.max(maximum, value);
        const result = new Float32Array(logits.length);
        if (maximum === -Infinity) return result;
        let total = 0;
        for (let i = 0; i < logits.length; i++) { result[i] = Math.exp(logits[i] - maximum); total += result[i]; }
        for (let i = 0; i < logits.length; i++) result[i] /= total;
        return result;
    }

    const api = { HexGame, encode, softmax };
    scope.HexEngine = api;
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof self !== 'undefined' ? self : globalThis);
