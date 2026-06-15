// MCTS tree-visualization worker.
//
// Reuses the real AlphaZero stack from the gomoku play page — ONNX inference,
// gomoku.js game logic, and mcts.js's KataGo-PUCT — but exposes the search one
// simulation at a time and serializes the whole tree after each step so the UI
// can draw it. Nothing here is shared with worker.js; it only importScripts the
// same pure helpers, so the play page is untouched.

const IS_WORKER = (typeof self !== "undefined" && typeof importScripts === "function");
if (IS_WORKER) {
    const _qs = self.location.search || ("?v=" + Date.now());
    importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js");
    importScripts("gomoku.js" + _qs);
    importScripts("mcts.js" + _qs);
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/";
    ort.env.wasm.numThreads = 1;
}

let session = null;
let game = null;
let mcts = null;
let root = null;
let boardSize = 9;
let rule = "renju";
let nodeIdCounter = 0;
let seedPosition = null;   // optional {board, toPlay, lastMove} to seed the root from a live game

function nid(node) {
    if (node._id == null) node._id = nodeIdCounter++;
    return node._id;
}

// One ONNX forward pass — trimmed copy of worker.js::inference, keeping only
// what MCTS expansion needs (masked-softmax prior + WDL value + masked logits).
async function inference(state, toPlay) {
    const M = MAX_BOARD_SIZE, A = M * M;
    const N = boardSize, NA = N * N;

    let ply = 0;
    for (let i = 0; i < state.length; i++) if (state[i] !== 0) ply++;
    const spatial = game.encodeState(state, toPlay);
    const globalF = game.computeGlobalFeatures(ply, toPlay);
    const out = await session.run({
        input_spatial: new ort.Tensor("float32", spatial, [1, 5, M, M]),
        input_global: new ort.Tensor("float32", globalF, [1, 12]),
    });

    const policyAll = out.policy_logits.data;     // (1, 4, A)
    const wdlLogits = out.value_wdl_logits.data;  // (1, 3)

    const policyMainRaw = new Float32Array(NA);
    for (let r = 0; r < N; r++)
        for (let c = 0; c < N; c++)
            policyMainRaw[r * N + c] = policyAll[r * M + c];   // channel 0

    const legal = game.getLegalActions(state, toPlay);
    const masked = new Float32Array(NA);
    for (let i = 0; i < NA; i++) masked[i] = legal[i] ? policyMainRaw[i] : -1e9;
    const policyMainSoft = new Float32Array(softmax(masked));

    const wdl = softmax(new Float64Array([wdlLogits[0], wdlLogits[1], wdlLogits[2]]));
    return {
        policyMainSoft,
        policyMainMaskedLogits: masked,
        wdl: new Float64Array([wdl[0], wdl[1], wdl[2]]),
    };
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
    const inf = await inferFn(root.state, root.toPlay);
    mcts.expand(root, inf.policyMainSoft, inf.wdl, inf.policyMainMaskedLogits);
    mcts.backpropagate(root, inf.wdl);
}

// Run exactly one simulation. Returns the selection path (node ids), the leaf,
// and what happened there — enough for the UI to animate this step.
async function oneSimulation(inferFn) {
    let node = root;
    const path = [nid(node)];
    while (node.isExpanded()) {
        const nx = mcts.select(node);
        if (!nx) break;
        node = nx;
        path.push(nid(node));
    }

    // node.toPlay is who moves next; the actor of node.actionTaken is -node.toPlay.
    const winner = game.getWinner(node.state, node.actionTaken, -node.toPlay);
    let value, expanded = false;
    if (node.actionTaken != null && winner !== null) {
        const result = winner * node.toPlay;   // from node.toPlay's POV
        value = result === 1 ? new Float64Array([1, 0, 0])
              : result === -1 ? new Float64Array([0, 0, 1])
              : new Float64Array([0, 1, 0]);
        node._terminal = true;
        node._term = result;
    } else {
        const inf = await inferFn(node.state, node.toPlay);
        mcts.expand(node, inf.policyMainSoft, inf.wdl, inf.policyMainMaskedLogits);
        value = inf.wdl;
        expanded = true;
    }
    mcts.backpropagate(node, value);
    return { path, leaf: nid(node), expanded, value: Array.from(value) };
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
        if (n > 0) {
            wdl = [node.v[0] / n, node.v[1] / n, node.v[2] / n];     // node.toPlay frame
            // winrate from node.toPlay's POV — the side to move *next* at this node.
            winrate = ((node.v[0] - node.v[2]) / n + 1) / 2;
            q = node.v[2] / n - node.v[0] / n;                       // parent-perspective utility
        }
        let U = null, score = null, qUsed = q;
        if (parentSp) {
            qUsed = n > 0 ? (node.v[2] / n - node.v[0] / n) : parentSp.fpuValue;
            U = parentSp.exploreScaling * node.prior / (1 + n);
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
            const sp = mcts.computeSelectParams(node, node.n, visitedMass);
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
    boardSize = bs || 9;
    rule = rl || "renju";
    seedPosition = seed || null;
    nodeIdCounter = 0;
    game = new Gomoku(boardSize, rule);
    mcts = new MCTS(game, {});   // defaults = KataGo-PUCT (see mcts.js constructor)
    newRoot();
    const resp = await fetch(modelUrl);
    if (!resp.ok) throw new Error(`fetch ${modelUrl} -> ${resp.status}`);
    const bytes = new Uint8Array(await resp.arrayBuffer());
    session = await ort.InferenceSession.create(bytes, {
        executionProviders: ["wasm"], intraOpNumThreads: 1, interOpNumThreads: 1,
    });
    await expandRoot(inference);
    postMessage({ type: "ready", tree: serializeTree() });
}

async function reset(bs, rl) {
    if (bs != null) boardSize = bs;
    if (rl != null) rule = rl;
    seedPosition = null;   // reset always rebuilds from the empty board
    nodeIdCounter = 0;
    game = new Gomoku(boardSize, rule);
    mcts.game = game;
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
