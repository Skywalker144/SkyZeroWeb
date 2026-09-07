const IS_WORKER = (typeof self !== "undefined" && typeof importScripts === "function");
if (IS_WORKER) {
    const _qs = self.location.search || ("?v=" + Date.now());
    importScripts("../../shared/onnx-worker-runtime.js" + _qs);
    SkyZeroOnnxRuntime.loadOnnxRuntime(self);
    importScripts("../engine/game.js" + _qs);
    importScripts("../engine/mcts.js" + _qs);
    importScripts("../engine/inference.js" + _qs);
    importScripts("search-core.js" + _qs);
}
const SearchCore = IS_WORKER
    ? GomokuSearchCore
    : require("./search-core.js");

let inferenceRuntime = null;
let game = null;
let mcts = null;
let root = null;
let boardSize = 15;
let rule = "renju";
let nodeIdCounter = 0;
let seedPosition = null;   // optional {board, toPlay, lastMove} to seed the root from a live game

function nid(node) {
    if (node._id == null) node._id = nodeIdCounter++;
    return node._id;
}

async function inference(state, toPlay, isRoot = false) {
    return inferenceRuntime.evaluate(state, toPlay, {
        isRoot,
        policyOptimism: isRoot ? 0 : 1,
    });
}

function newRoot() {
    // Seed from a live game position when one was handed over (analyze-current-board
    // from the gomoku page); otherwise start from the empty board, black to move.
    if (seedPosition && Array.isArray(seedPosition.board) && seedPosition.board.length === boardSize * boardSize) {
        root = new Node(Int8Array.from(seedPosition.board), seedPosition.toPlay === -1 ? -1 : 1, 0, null,
            (typeof seedPosition.lastMove === "number" && seedPosition.lastMove >= 0) ? seedPosition.lastMove : null);
    } else {
        root = new Node(game.getInitialState(), 1);   // empty board, black (1) to move
    }
}

// Expand the root up front (like worker.js) so the very first user "step" is a
// real select→leaf→backup, and the root already shows the network's priors.
async function expandRoot(inferFn) {
    // Root policy optimism is 0, so use channel 0 for the first expansion.
    const rootInf = await inferFn(root.state, root.toPlay, true);
    mcts.expand(root, rootInf.policyPriorSoft, rootInf.wdl,
        rootInf.policyPriorMaskedLogits, rootInf.stError, true);
}

// Run exactly one simulation. Returns the selection path (node ids), the leaf,
// and what happened there — enough for the UI to animate this step.
async function oneSimulation(inferFn) {
    const result = await SearchCore.runSimulation({
        root, mcts, game, evaluate: inferFn,
    });
    if (!result.expanded) {
        result.leaf._terminal = true;
        result.leaf._term = result.terminalResult;
    }
    return {
        path: result.nodes.map(nid),
        leaf: nid(result.leaf),
        expanded: result.expanded,
        value: Array.from(result.value),
    };
}

// Serialize visited subtree (n >= 1 nodes only — unvisited priors would explode
// the fan-out). For each expanded node we recompute its select params so every
// child carries the Q / U / Q+U it was last scored with; pruning to top-k/top-p
// happens in the UI, not here.
function serializeTree() {
    const out = [];

    function emit(node, parentId, depth, parentSp) {
        const id = nid(node);
        const n = node.n;
        let winrate = null, wdl = null, q = null;
        if (node.weightSum > 0) {
            wdl = [
                node.v[0] / node.weightSum,
                node.v[1] / node.weightSum,
                node.v[2] / node.weightSum,
            ];
            // winrate from node.toPlay's POV — the side to move *next* at this node.
            winrate = ((node.v[0] - node.v[2]) / node.weightSum + 1) / 2;
            q = node.v[2] / node.weightSum - node.v[0] / node.weightSum;
        }
        let U = null, score = null, qUsed = q;
        if (parentSp) {
            qUsed = node.weightSum > 0
                ? (node.v[2] - node.v[0]) / node.weightSum
                : parentSp.fpuValue;
            U = parentSp.exploreScaling * node.prior
                / (1 + node.weightSum);
            score = qUsed + U;
        }
        out.push({
            id, parentId, depth,
            action: node.actionTaken,
            ar: node.actionTaken != null ? (node.actionTaken / boardSize | 0) : null,
            ac: node.actionTaken != null ? (node.actionTaken % boardSize) : null,
            toPlay: node.toPlay,
            n, prior: node.prior,
            winrate, wdl, q: qUsed, U, score,
            isTerminal: !!node._terminal,
            term: node._term != null ? node._term : null,
            board: Array.from(node.state),
        });

        if (node.isExpanded()) {
            let visitedMass = 0;
            for (const ch of node.children) if (ch.n > 0) visitedMass += ch.prior;
            const sp = mcts.computeSelectParams(node, node.parent === null);
            const kids = node.children.filter(c => c.n > 0).sort((a, b) => b.n - a.n);
            for (const ch of kids) emit(ch, id, depth + 1, sp);
        }
    }

    emit(root, null, 0, null);
    let totalRootVisits = 0;
    for (const ch of root.children) totalRootVisits += ch.n;
    return { nodes: out, boardSize, rule, totalSims: root.n, rootVisits: totalRootVisits };
}

async function init(modelUrl, bs, rl, seed) {
    boardSize = bs || 15;
    rule = rl || "renju";
    seedPosition = seed || null;
    nodeIdCounter = 0;
    game = new Gomoku(boardSize, rule);
    mcts = new MCTS(game, {});   // defaults = KataGo-PUCT (see mcts.js constructor)
    newRoot();
    inferenceRuntime = new GomokuInferenceRuntime.GomokuInference({
        ort,
        softmax,
        transformCoord,
        maxBoardSize: MAX_BOARD_SIZE,
        boardSize,
        game,
    });
    await inferenceRuntime.load(modelUrl);
    await expandRoot(inference);
    postMessage({ type: "ready", tree: serializeTree() });
}

async function reset(bs, rl) {
    if (bs != null) boardSize = bs;
    if (rl != null) rule = rl;
    seedPosition = null;   // reset always rebuilds from the empty board
    nodeIdCounter = 0;
    if (root) mcts.clear(root);
    game = new Gomoku(boardSize, rule);
    mcts.game = game;
    inferenceRuntime.setGame(game, boardSize);
    newRoot();
    await expandRoot(inference);
    postMessage({ type: "ready", tree: serializeTree() });
}

onmessage = async (e) => {
    const d = e.data;
    try {
        if (d.type === "init") {
            await init(d.modelUrl, d.boardSize, d.rule, d.seed);
        } else if (d.type === "reset") {
            await reset(d.boardSize, d.rule);
        } else if (d.type === "step") {
            const trace = await oneSimulation(inference);
            postMessage({ type: "step", trace, tree: serializeTree() });
        } else if (d.type === "run") {
            const n = d.n || 50;
            for (let i = 0; i < n; i++) {
                await oneSimulation(inference);
                if ((i & 7) === 7) postMessage({ type: "progress", done: i + 1, total: n });
            }
            postMessage({ type: "done", tree: serializeTree() });
        }
    } catch (err) {
        postMessage({ type: "error", message: err && err.message ? err.message : String(err) });
    }
};

// Node-only test hooks (browser worker ignores this). Lets tests/mcts_tree.test.js
// drive the real search functions with a fake evaluator — no ONNX needed.
if (typeof module !== "undefined" && module.exports) {
    module.exports = {
        serializeTree, oneSimulation, expandRoot, newRoot,
        __test: {
            setGame(g, m, bs, rl) { game = g; mcts = m; boardSize = bs; rule = rl; },
            getRoot: () => root,
        },
    };
}
