(function (scope) {
    async function search({state, legal, play, terminalValue, evaluate, dispose = () => {},
        maxVisits = 256, thinkMs = 3000, shouldContinue = () => true, onProgress = () => {}}) {
        const root = {state, visits: 0, sum: 0, children: null};
        const allocated = [];
        const started = performance.now();
        try {
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
                    if (best.state === null) { best.state = play(node.state, best.action); allocated.push(best.state); }
                    node = best;
                    path.push(node);
                }
                let value = terminalValue(node.state);
                if (value === null) {
                    const evaluation = await evaluate(node.state);
                    if (!shouldContinue()) return null;
                    value = evaluation.value;
                    if (!Number.isFinite(value)) throw new Error('Non-finite search value');
                    node.children = legal(node.state).map(action => ({action, prior: evaluation.policy[action],
                        state: null, visits: 0, sum: 0, children: null}));
                }
                for (let i = path.length - 1; i >= 0; i--) {
                    path[i].visits++;
                    path[i].sum += value;
                    value = -value;
                }
                if (simulation % 8 === 0) {
                    onProgress({visits: root.visits, value: root.sum / root.visits});
                    await new Promise(resolve => setTimeout(resolve, 0));
                }
                if (root.visits >= 2 && performance.now() - started >= thinkMs) break;
            }
            if (!shouldContinue()) return null;
            const selected = root.children?.reduce((best, child) => !best || child.visits > best.visits
                || (child.visits === best.visits && child.prior > best.prior) ? child : best, null);
            if (!selected) throw new Error('No legal search move');
            return {action: selected.action, visits: root.visits, value: root.sum / root.visits,
                elapsedMs: performance.now() - started};
        } finally { for (const position of allocated) dispose(position); }
    }

    scope.SkyZeroPUCT = {search};
    if (typeof module !== 'undefined' && module.exports) module.exports = scope.SkyZeroPUCT;
})(typeof self !== 'undefined' ? self : globalThis);
