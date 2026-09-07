(function (target, factory) {
    const api = factory();
    target.GomokuSearchCore = api;
    if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : globalThis, function () {
    async function runSimulation({
        root, mcts, game, evaluate, shouldContinue = () => true,
    }) {
        let node = root;
        const nodes = [node];
        while (node.isExpanded()) {
            const next = mcts.select(node);
            if (!next) break;
            node = next;
            nodes.push(node);
        }

        const winner = node.actionTaken == null
            ? null
            : game.getWinner(node.state, node.actionTaken, -node.toPlay);
        let value;
        let expanded = false;
        let terminalResult = null;
        if (winner !== null) {
            terminalResult = winner * node.toPlay;
            value = terminalResult === 1
                ? new Float64Array([1, 0, 0])
                : terminalResult === -1
                    ? new Float64Array([0, 0, 1])
                    : new Float64Array([0, 1, 0]);
        } else {
            const evaluation = await evaluate(node.state, node.toPlay);
            if (!shouldContinue()) return null;
            mcts.expand(
                node,
                evaluation.policyPriorSoft,
                evaluation.wdl,
                evaluation.policyPriorMaskedLogits,
                evaluation.stError,
                false);
            value = evaluation.wdl;
            expanded = true;
        }
        mcts.backpropagate(nodes, value, !expanded);
        return { nodes, leaf: node, value, expanded, terminalResult };
    }

    return { runSimulation };
});
