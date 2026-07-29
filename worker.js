// Propagate main.js's `worker.js?v=<ts>` cache-bust to importScripts: without
// it, _headers' max-age=3600 makes the browser keep serving stale gomoku.js /
// mcts.js even when the worker itself is refreshed.
const _qs = self.location.search || ("?v=" + Date.now());
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js");
importScripts("gomoku.js" + _qs);
importScripts("mcts.js" + _qs);

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/";
ort.env.wasm.numThreads = 1;   // SharedArrayBuffer cross-origin fragility; force single-thread

// --- module-level state ---
let session = null;
let game = null;
let mcts = null;
let root = null;
let currentBoardSize = 15;
let currentRule = "renju";
let latestSearchId = 0;
const NN_CACHE_LIMIT = 8192;
const nnCache = new Map();
let currentPda = 0.5;
const SEARCH_FACTOR_WHEN_WINNING_THRESHOLD = 0.95;
const SEARCH_FACTOR_WHEN_WINNING = 0.30;
// KataGo's normal GTP/search defaults use a small optimistic-policy blend at
// the root and the full optimistic policy below it. V7.19 searched play keeps
// its configured root value of 0, while the explicit NN-only mode uses the
// KataGo-style 0.2 root blend.
const SEARCH_ROOT_POLICY_OPTIMISM = 0.0;
const POLICY_ONLY_ROOT_OPTIMISM = 0.2;
const TREE_POLICY_OPTIMISM = 1.0;
let currentPdaPla = 0;
let recentAiWinLossValues = [];
// Hard cap on the root's cumulative visits for play-mode (time-budgeted) search.
// Tree reuse accumulates visits across moves, so near the endgame the root is
// often already well-searched — once it hits this we stop and play, instead of
// spending the full thinking time re-deepening a settled tree. Kept equal to the
// ponder cap (main.js ANALYSIS_CAP_MIN) so the AI's move-search and the
// human's-turn analysis settle at the same depth.
const SEARCH_VISIT_CAP = 2000;

// --- helpers ---

function concatChunks(chunks, total) {
    const result = new Uint8Array(total);
    let offset = 0;
    for (const c of chunks) { result.set(c, offset); offset += c.length; }
    return result;
}

async function fetchModelWithProgress(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`fetch ${url} → ${response.status}`);
    const total = Number(response.headers.get("Content-Length")) || 0;
    if (!response.body) {
        const buf = await response.arrayBuffer();
        postMessage({ type: "model-progress", percent: 100, loaded: buf.byteLength, total: buf.byteLength });
        return new Uint8Array(buf);
    }
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;
    if (total > 0) postMessage({ type: "model-progress", percent: 0, loaded: 0, total });
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        const percent = total > 0 ? (loaded / total) * 100 : null;
        postMessage({ type: "model-progress", percent, loaded, total: total || null });
    }
    postMessage({ type: "model-progress", percent: 100, loaded, total: total || loaded });
    return concatChunks(chunks, total || loaded);
}

/**
 * Run one ONNX forward pass for `state` (board_size × board_size) with
 * current player `toPlay`. Returns the un-padded heatmap arrays the UI
 * wants, plus raw masked logits for MCTS.
 */
function transformSpatial(input, channels, size, rotation, flip) {
    const area = size * size;
    const output = new Float32Array(input.length);
    for (let channel = 0; channel < channels; channel++) {
        for (let row = 0; row < size; row++) {
            for (let col = 0; col < size; col++) {
                const [tr, tc] = transformCoord(
                    row, col, size, rotation, flip);
                output[channel * area + tr * size + tc] =
                    input[channel * area + row * size + col];
            }
        }
    }
    return output;
}

function undoSpatial(input, channels, size, rotation, flip) {
    const area = size * size;
    const output = new Float32Array(input.length);
    for (let channel = 0; channel < channels; channel++) {
        for (let row = 0; row < size; row++) {
            for (let col = 0; col < size; col++) {
                const [tr, tc] = transformCoord(
                    row, col, size, rotation, flip);
                output[channel * area + row * size + col] =
                    input[channel * area + tr * size + tc];
            }
        }
    }
    return output;
}

function cacheKey(state, toPlay, isRoot) {
    return `${currentBoardSize}|${currentRule}|${toPlay}`
        + `|pda=${currentPda}|pdaPla=${currentPdaPla}`
        + `|${isRoot ? "root" : "child"}|`
        + Array.from(state).join("");
}

function cachePut(key, value) {
    if (nnCache.has(key)) nnCache.delete(key);
    nnCache.set(key, value);
    if (nnCache.size > NN_CACHE_LIMIT) {
        nnCache.delete(nnCache.keys().next().value);
    }
}

async function inference(
    state, toPlay, isRoot = false,
    policyOptimism = isRoot
        ? SEARCH_ROOT_POLICY_OPTIMISM : TREE_POLICY_OPTIMISM
) {
    if (!session) throw new Error("session not ready");
    const M = MAX_BOARD_SIZE, A = M * M;         // padded canvas (from gomoku.js)
    const N = currentBoardSize, NA = N * N;      // game canvas

    let ply = 0;
    for (let i = 0; i < state.length; i++) if (state[i] !== 0) ply++;

    const spatial = game.encodeState(state, toPlay);
    const globalF = game.computeGlobalFeatures(
        ply, toPlay, 0, currentPda, currentPdaPla);
    const key = cacheKey(state, toPlay, isRoot);
    let raw = nnCache.get(key);
    if (raw) {
        nnCache.delete(key);
        nnCache.set(key, raw);
    } else {
        const transformType = Math.floor(Math.random() * 8);
        const rotation = transformType % 4;
        const flip = transformType >= 4;
        const transformed = transformSpatial(
            spatial, 5, M, rotation, flip);
        const feeds = {
            input_spatial: new ort.Tensor(
                "float32", transformed, [1, 5, M, M]),
            input_global: new ort.Tensor(
                "float32", globalF, [1, 7]),
        };
        const out = await session.run(feeds);
        raw = {
            policy: undoSpatial(
                out.policy_logits.data, 5, M, rotation, flip),
            valueLogits: new Float32Array(
                out.value_wdl_logits.data),
            stErrorSquared: out.value_st_error_sq.data[0],
        };
        cachePut(key, raw);
    }

    const policyAll = raw.policy;              // (5, 225)
    const wdlLogits = raw.valueLogits;         // (3)

    // --- crop policy channels 0 (main) and 1 (opp) from padded → board area ---
    function cropChannel(channelIdx) {
        const cropped = new Float32Array(NA);
        for (let r = 0; r < N; r++) {
            for (let c = 0; c < N; c++) {
                cropped[r * N + c] = policyAll[channelIdx * A + r * M + c];
            }
        }
        return cropped;
    }
    const policyMainRaw = cropChannel(0);
    const policyOppRaw  = cropChannel(1);
    const policyOptRaw  = cropChannel(4);

    // --- mask illegal + softmax ---
    const legal = game.getLegalActions(state, toPlay);
    const optimism = Math.max(0, Math.min(1, policyOptimism));
    const policyPriorMasked = new Float32Array(NA);
    const policyMainMasked = new Float32Array(NA);
    for (let i = 0; i < NA; i++) {
        const blended = policyMainRaw[i]
            + (policyOptRaw[i] - policyMainRaw[i]) * optimism;
        policyPriorMasked[i] = legal[i] ? blended : -Infinity;
        policyMainMasked[i] = legal[i] ? policyMainRaw[i] : -Infinity;
    }
    const policyPriorSoft = new Float32Array(
        softmax(policyPriorMasked));
    const policyMainSoft = new Float32Array(softmax(policyMainMasked));

    const policyOppMasked = new Float32Array(NA);
    for (let i = 0; i < NA; i++) {
        policyOppMasked[i] = legal[i] ? policyOppRaw[i] : -Infinity;
    }
    const policyOppSoft = new Float32Array(
        softmax(policyOppMasked));
    const policyOptMasked = new Float32Array(NA);
    for (let i = 0; i < NA; i++) {
        policyOptMasked[i] = legal[i] ? policyOptRaw[i] : -Infinity;
    }
    const policyOptSoft = new Float32Array(
        softmax(policyOptMasked));

    // --- value WDL: softmax 3 logits ---
    const wdl = softmax(new Float64Array([wdlLogits[0], wdlLogits[1], wdlLogits[2]]));
    const wdlF64 = new Float64Array([wdl[0], wdl[1], wdl[2]]);

    return {
        policyPriorSoft,                   // effective root/tree prior
        policyPriorMaskedLogits: policyPriorMasked,
        policyMainSoft,                    // unblended main-policy UI heatmap
        policyOppSoft,                     // UI heatmap
        policyOptSoft,                     // optimistic-policy UI heatmap
        wdl: wdlF64,                       // root nn value
        stError: Math.sqrt(Math.max(0, raw.stErrorSquared)),
    };
}

async function initSession(modelUrl, boardSize, rule) {
    currentBoardSize = boardSize;
    currentRule = rule || "renju";
    game = new Gomoku(boardSize, currentRule);
    mcts = new MCTS(game, {
        c_puct: 1.1,
        c_puct_log: 0.45,
        c_puct_base: 500,
        fpu_reduction_max: 0.16,
        root_fpu_reduction_max: 0.1,
        fpu_pow: 2.0,
        fpu_loss_prop: 0.0,
        cpuct_utility_stdev_prior: 0.40,
        cpuct_utility_stdev_prior_weight: 2.0,
        cpuct_utility_stdev_scale: 0.85,
        use_uncertainty: true,
        uncertainty_coeff: 0.25,
        uncertainty_exponent: 1.0,
        uncertainty_max_weight: 8.0,
        value_weight_exponent: 0.5,
        subtree_value_bias_factor: 0.45,
        subtree_value_bias_weight_exponent: 0.8,
        subtree_value_bias_free_prop: 0.8,
        root_lcb_selection: true,
        lcb_stdevs: 5.0,
        min_visit_prop_for_lcb: 0.15,
        root_symmetry_pruning: true,
    });
    root = null;
    nnCache.clear();
    const bytes = await fetchModelWithProgress(modelUrl);
    session = await ort.InferenceSession.create(bytes, {
        executionProviders: ["wasm"],
        intraOpNumThreads: 1,
        interOpNumThreads: 1,
    });
    postMessage({ type: "ready" });
}

function resetGame(boardSize, rule) {
    const sizeChanged = boardSize !== undefined && boardSize !== currentBoardSize;
    const ruleChanged = rule !== undefined && rule !== currentRule;
    if (sizeChanged) currentBoardSize = boardSize;
    if (ruleChanged) currentRule = rule;
    if (sizeChanged || ruleChanged) {
        game = new Gomoku(currentBoardSize, currentRule);
        if (mcts) mcts.game = game;   // keep MCTS bound to the live game
    }
    if (root && mcts) mcts.clear(root);
    root = null;
    nnCache.clear();
    currentPdaPla = 0;
    recentAiWinLossValues = [];
}

function setPdaReference(pdaPla) {
    if (pdaPla !== 1 && pdaPla !== -1) {
        throw new Error(
            "search requires pdaPla to be black (+1) or white (-1)");
    }
    if (currentPdaPla === pdaPla) return;
    if (root && mcts) mcts.clear(root);
    root = null;
    nnCache.clear();
    currentPdaPla = pdaPla;
    recentAiWinLossValues = [];
}

function setPdaMagnitude(pda) {
    if (!Number.isFinite(pda) || pda < -1 || pda > 1) {
        throw new Error("browser search requires pda in [-1, 1]");
    }
    if (currentPda === pda) return;
    if (root && mcts) mcts.clear(root);
    root = null;
    nnCache.clear();
    currentPda = pda;
    recentAiWinLossValues = [];
}

function applyMove(action, nextState, nextToPlay) {
    if (root && root.children.length > 0) {
        const child = root.children.find(c => c.actionTaken === action);
        if (child) {
            for (const sibling of root.children) {
                if (sibling !== child) mcts.releaseTree(sibling);
            }
            mcts.releaseBias(root);
            root = child;
            root.parent = null;   // detach for GC
            root.rootPolicyApplied = false;
            return;
        }
    }
    if (root) mcts.releaseTree(root);
    root = new Node(nextState, nextToPlay);
}

async function runSearch(
    state, toPlay, sims, gen, externalSearchId, analyze, timeMs, pda, pdaPla,
    policyOnly
) {
    setPdaMagnitude(pda);
    setPdaReference(pdaPla);
    if (!root) root = new Node(state, toPlay);
    const rootPolicyOptimism = policyOnly
        ? POLICY_ONLY_ROOT_OPTIMISM : SEARCH_ROOT_POLICY_OPTIMISM;

    // Root inference if not already expanded.
    let networkPolicy, rootPriorPolicy, oppPolicy, optimisticPolicy;
    let nnValueWDL;
    if (!root.isExpanded()) {
        const inf = await inference(
            root.state, root.toPlay, true, rootPolicyOptimism);
        if (latestSearchId !== gen) return;
        mcts.expand(
            root,
            inf.policyPriorSoft,
            inf.wdl,
            inf.policyPriorMaskedLogits,
            inf.stError,
            true);
        root.rootPolicyOptimismApplied = rootPolicyOptimism;
        nnValueWDL = inf.wdl;
        networkPolicy = inf.policyMainSoft;
        rootPriorPolicy = inf.policyPriorSoft;
        oppPolicy = inf.policyOppSoft;
        optimisticPolicy = inf.policyOptSoft;
    } else {
        // A promoted child was expanded with non-root policyOptimism=1. Refresh
        // its root prior with the active root optimism (searched or policy-only)
        // and root symmetry pruning while preserving visited descendants.
        const inf = await inference(
            root.state, root.toPlay, true, rootPolicyOptimism);
        if (latestSearchId !== gen) return;
        if (!root.rootPolicyApplied
            || root.rootPolicyOptimismApplied !== rootPolicyOptimism) {
            mcts.refreshRoot(
                root,
                inf.policyPriorSoft,
                inf.wdl,
                inf.policyPriorMaskedLogits,
                inf.stError);
            root.rootPolicyOptimismApplied = rootPolicyOptimism;
        }
        nnValueWDL = root.nnValue;   // cached
        networkPolicy = inf.policyMainSoft;
        rootPriorPolicy = inf.policyPriorSoft;
        oppPolicy = inf.policyOppSoft;
        optimisticPolicy = inf.policyOptSoft;
    }

    // The NN-only heatmaps (network policy / opp policy / future positions) are
    // pure root outputs — ready now, before any simulation. Push them immediately
    // so the analysis panel fills in on the human's turn too, and survives an early
    // abort if the player moves before this chunk finishes.
    postMessage({
        type: "progress",
        progress: 0,
        searchId: externalSearchId,
        policyOnly,
        policyPrior:   Array.from(rootPriorPolicy),
        nnPolicy:      Array.from(networkPolicy),
        nnOptimisticPolicy: Array.from(optimisticPolicy),
        nnOppPolicy:   Array.from(oppPolicy),
    });

    let totalSims = 0;
    const searchStart = performance.now();
    let lastProgress = searchStart;
    // Pure node-search throughput for this chunk (sims / wall-time), reported to
    // the UI. searchStart is taken after root (re)inference, so it measures the
    // simulation loop's speed, not the per-chunk inference overhead.
    const npsNow = (now) => Math.round(totalSims / Math.max(0.001, (now - searchStart) / 1000));

    let selectedAction, rootValue, phases;
    let appliedSearchFactor = 1;
    let effectiveTimeMs = timeMs;
    let effectiveVisitCap = SEARCH_VISIT_CAP;
    // One plain-PUCT simulation from the root (no Gumbel / no Dirichlet noise).
    // Returns false if the search was superseded mid-inference (caller bails).
    const puctStep = async () => {
        let node = root;
        const path = [root];
        while (node.isExpanded()) {
            const nx = mcts.select(node);
            if (!nx) break;
            node = nx;
            path.push(node);
        }
        const winner = game.getWinner(node.state, node.actionTaken, -node.toPlay);
        let value;
        let terminal = winner !== null;
        if (winner !== null) {
            const result = winner * node.toPlay;
            if      (result === 1)  value = new Float64Array([1, 0, 0]);
            else if (result === -1) value = new Float64Array([0, 0, 1]);
            else                    value = new Float64Array([0, 1, 0]);
        } else {
            const inf = await inference(
                node.state, node.toPlay, false, TREE_POLICY_OPTIMISM);
            if (latestSearchId !== gen) return false;
            mcts.expand(
                node,
                inf.policyPriorSoft,
                inf.wdl,
                inf.policyPriorMaskedLogits,
                inf.stError,
                false);
            value = inf.wdl;
        }
        mcts.backpropagate(path, value, terminal);
        totalSims++;
        return true;
    };
    // Live candidate snapshot (visits / win rate / cumulative depth) streamed so
    // the analysis UI fills in per-move data during the search, not only at the end.
    const streamCandidates = (now, progress) => {
        let liveVisits = 0;
        for (const ch of root.children) liveVisits += ch.n;
        postMessage({
            type: "progress",
            progress: Math.min(100, progress),
            searchId: externalSearchId,
            mctsVisits:  Array.from(mcts.getMCTSPolicy(root)),
            mctsPlaySelection:
                Array.from(mcts.getPlaySelectionPolicy(root)),
            mctsWinrate: Array.from(mcts.getMCTSWinrate(root)),
            searchSims:  liveVisits,
            nps:         npsNow(now),
        });
    };
    // V7.19 play finalization: raw recomputed root value for reporting and
    // LCB-adjusted play-selection value for the actual move.
    const finishPuct = () => {
        rootValue = root.weightSum > 0
            ? new Float64Array([
                root.vRaw[0] / root.weightSum,
                root.vRaw[1] / root.weightSum,
                root.vRaw[2] / root.weightSum,
            ])
            : new Float64Array(nnValueWDL);
        selectedAction = mcts.rootPlaySelection(root).action;
        phases = [];
    };

    if (timeMs > 0) {
        appliedSearchFactor = searchFactorWhenWinning(
            recentAiWinLossValues,
            SEARCH_FACTOR_WHEN_WINNING_THRESHOLD,
            SEARCH_FACTOR_WHEN_WINNING);
        effectiveTimeMs = timeMs * appliedSearchFactor;
        effectiveVisitCap = Math.max(
            1, Math.ceil(SEARCH_VISIT_CAP * appliedSearchFactor));
        // Anytime PUCT — run until the time budget elapses OR the root's cumulative
        // visits reach the KataGo-style winning-factor-adjusted cap. Tree reuse
        // carries visits across moves, so a settled root may already exceed it.
        let rootVisits = 0;
        for (const ch of root.children) rootVisits += ch.n;
        while (performance.now() - searchStart < effectiveTimeMs
               && rootVisits < effectiveVisitCap) {
            if (!await puctStep()) return;
            rootVisits++;
            const now = performance.now();
            if (now - lastProgress > 60) {
                lastProgress = now;
                streamCandidates(
                    now, ((now - searchStart) / effectiveTimeMs) * 100);
            }
        }
        if (latestSearchId !== gen) return;
        finishPuct();
        recentAiWinLossValues.push(rootValue[0] - rootValue[2]);
        if (recentAiWinLossValues.length > 3) {
            recentAiWinLossValues.shift();
        }
    } else if (analyze) {
        // Fixed-sims PUCT — the analysis board deepens in chunks (no time budget).
        for (let i = 0; i < sims; i++) {
            if (!await puctStep()) return;
            const now = performance.now();
            if (now - lastProgress > 60) {
                lastProgress = now;
                streamCandidates(now, (totalSims / Math.max(1, sims)) * 100);
            }
        }
        if (latestSearchId !== gen) return;
        finishPuct();
    } else {
        // Zero-search mode performs exactly one root NN evaluation and chooses
        // the strongest legal effective prior. It intentionally ignores any
        // stale visits retained for tree reuse, so disabling thinking always
        // means pure network play.
        selectedAction = -1;
        let bestPrior = -Infinity;
        for (let action = 0; action < rootPriorPolicy.length; action++) {
            if (rootPriorPolicy[action] > bestPrior) {
                bestPrior = rootPriorPolicy[action];
                selectedAction = action;
            }
        }
        rootValue = new Float64Array(nnValueWDL);
        phases = [];
    }

    postMessage({ type: "progress", progress: 100, searchId: externalSearchId });

    // Raw visits remain a diagnostic. V7.19 move choice uses the separate,
    // LCB-adjusted play-selection distribution.
    const visitDist = mcts.getMCTSPolicy(root);
    const playSelectionDist = mcts.getPlaySelectionPolicy(root);
    // Per-move win rate (root player's view) for the candidate list. NaN where
    // unvisited; structured clone preserves NaN so main.js can tell "no data".
    const winrateDist = mcts.getMCTSWinrate(root);
    const lcbData = mcts.getLcb(root);
    // Cumulative root visits across tree-reuse chunks — the analysis ponder's
    // depth so far. main.js drives the continuous search off this.
    let rootVisits = 0;
    for (const ch of root.children) rootVisits += ch.n;

    postMessage({
        type: "result",
        searchId: externalSearchId,
        policyOnly,
        selectedAction,
        gumbelAction: selectedAction, // compatibility with cached older main.js
        rootValueWDL: rootValue,
        nnValueWDL,                           // [W, D, L] root NN
        mctsVisits:    Array.from(visitDist),
        mctsPlaySelection: Array.from(playSelectionDist),
        mctsWinrate:   Array.from(winrateDist),     // per-move win rate for the candidate list
        mctsLcb:       Array.from(lcbData.lcb),
        mctsLcbRadius: Array.from(lcbData.radius),
        policyPrior:   Array.from(rootPriorPolicy),
        nnPolicy:      Array.from(networkPolicy),
        nnOptimisticPolicy: Array.from(optimisticPolicy),
        nnOppPolicy:   Array.from(oppPolicy),
        gumbelPhases:  phases,
        iterations:    totalSims,
        searchSims:    rootVisits,   // cumulative root visits (analysis ponder depth)
        nps:           npsNow(performance.now()),
        pda:           currentPda,
        pdaPla:        currentPdaPla,
        searchFactor:  appliedSearchFactor,
        effectiveTimeMs,
        effectiveVisitCap,
    });
}

onmessage = async (e) => {
    const data = e.data;
    try {
        if (data.type === "init") {
            await initSession(data.modelUrl, data.boardSize, data.rule);
        } else if (data.type === "reset") {
            latestSearchId++;
            resetGame(data.boardSize, data.rule);
        } else if (data.type === "move") {
            applyMove(data.action, data.nextState, data.nextToPlay);
        } else if (data.type === "search") {
            latestSearchId++;
            const gen = latestSearchId;
            await runSearch(data.state, data.toPlay, data.sims, gen,
                data.searchId, data.analyze, data.timeMs, data.pda, data.pdaPla,
                data.policyOnly === true);
        } else if (data.type === "set-pda") {
            latestSearchId++;
            setPdaMagnitude(data.pda);
        } else if (data.type === "swap-model") {
            latestSearchId++;
            session = null;
            if (root && mcts) mcts.clear(root);
            root = null;
            nnCache.clear();
            recentAiWinLossValues = [];
            const bytes = await fetchModelWithProgress(data.modelUrl);
            session = await ort.InferenceSession.create(bytes, {
                executionProviders: ["wasm"],
                intraOpNumThreads: 1,
                interOpNumThreads: 1,
            });
            postMessage({ type: "ready" });
        }
    } catch (err) {
        postMessage({ type: "error", message: err && err.message ? err.message : String(err) });
    }
};
