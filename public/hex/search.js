(function (scope) {
    async function search({game, board, player, evaluate, maxVisits = 256, thinkMs = 3000,
        shouldContinue = () => true, onProgress = () => {}}) {
        const root = {board, player, visits: 0, sum: 0, children: null};
        const started = performance.now();
        for (let simulation = 0; simulation < maxVisits; simulation++) {
            if (!shouldContinue()) return null;
            let node = root;
            const path = [node];
            while (node.children?.length) {
                let best = null, bestScore = -Infinity;
                for (const child of node.children) {
                    const q = child.visits ? -child.sum / child.visits : 0;
                    const score = q + 1.5 * child.prior * Math.sqrt(node.visits + 1) / (child.visits + 1);
                    if (score > bestScore) { best = child; bestScore = score; }
                }
                node = best;
                if (!node.board) {
                    node.board = new Int8Array(path[path.length - 1].board);
                    node.board[node.action] = -node.player;
                }
                path.push(node);
            }
            const winner = game.winner(node.board);
            let value;
            if (winner) value = winner * node.player;
            else {
                const evaluation = await evaluate(node.board, node.player);
                if (!shouldContinue()) return null;
                value = evaluation.value;
                if (!Number.isFinite(value)) throw new Error('Non-finite Hex value');
                node.children = [];
                for (let action = 0; action < node.board.length; action++) {
                    if (node.board[action]) continue;
                    node.children.push({action, prior: evaluation.policy[action], player: -node.player,
                        board: null, visits: 0, sum: 0, children: null});
                }
            }
            for (let i = path.length - 1; i >= 0; i--) {
                path[i].visits++;
                path[i].sum += value;
                value = -value;
            }
            if (simulation % 8 === 0) onProgress({visits: root.visits, value: root.sum / root.visits});
            if (root.visits >= 2 && performance.now() - started >= thinkMs) break;
        }
        if (!shouldContinue()) return null;
        const selected = root.children?.reduce((best, child) => !best || child.visits > best.visits
            || (child.visits === best.visits && child.prior > best.prior) ? child : best, null);
        if (!selected) throw new Error('No legal Hex move');
        return {action: selected.action, visits: root.visits, value: root.sum / root.visits,
            elapsedMs: performance.now() - started};
    }

    scope.HexSearch = { search };
    if (typeof module !== 'undefined' && module.exports) module.exports = { search };
})(typeof self !== 'undefined' ? self : globalThis);
