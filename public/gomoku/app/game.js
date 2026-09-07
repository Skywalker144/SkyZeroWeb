const EDIT_TOOLS = ["alternate", "black", "white", "erase"];

function refreshEditUndoBtn() {
    const btn = document.getElementById("edit_undo_btn");
    if (btn) btn.disabled = editUndoStack.length === 0;
}

const worker = new Worker("/gomoku/workers/search.js?v=" + Date.now());
worker.onerror = (e) => {
    stopInitializationFeedback();
    const where = e.filename ? ` (${e.filename}:${e.lineno})` : "";
    const msg = t("err_worker_failed", e.message || t("err_unknown"), where);
    setStatusRaw(msg, "error");
    const tEl = document.getElementById("loading_text");
    if (tEl) tEl.textContent = msg;
    console.error("[worker.onerror]", e);
};
worker.onmessageerror = (e) => {
    setStatus("err_worker_msg", "error");
    console.error("[worker.onmessageerror]", e);
};

let lastStatus = { key: "status_idle", args: [], variant: "idle", raw: null };
function setStatus(key, variant, ...args) {
    lastStatus = { key, args, variant, raw: null };
    document.getElementById("status").textContent = t(key, ...args);
    document.getElementById("status_pill").dataset.variant = variant || "idle";
}
function setStatusRaw(text, variant) {
    lastStatus = { key: null, args: [], variant, raw: text };
    document.getElementById("status").textContent = text;
    document.getElementById("status_pill").dataset.variant = variant || "idle";
}
function rerenderStatus() {
    if (lastStatus.raw != null) return;
    if (!lastStatus.key) return;
    document.getElementById("status").textContent = t(lastStatus.key, ...lastStatus.args);
}

function board1Dto2D(b1d) {
    const b = [];
    for (let r = 0; r < N; r++) {
        b.push([]);
        for (let c = 0; c < N; c++) b[r].push(b1d[r * N + c]);
    }
    return b;
}

function publishStateForDrawing(extras = {}) {
    state = {
        get board() { return board1Dto2D(boardState); },
        last_move: lastMove ? [lastMove.r, lastMove.c] : null,
        board_size: N,
        // Heat data (set by handleResult; null otherwise — drawHeat handles nulls).
        mcts_visits:         extras.mcts_visits         || null,
        mcts_play_selection: extras.mcts_play_selection || null,
        mcts_lcb:            extras.mcts_lcb            || null,
        mcts_lcb_radius:     extras.mcts_lcb_radius     || null,
        mcts_winrate:        extras.mcts_winrate        || null,
        policy_only:         extras.policy_only === true,
        policy_prior:        extras.policy_prior        || null,
        nn_policy:            extras.nn_policy            || null,
        nn_optimistic_policy: extras.nn_optimistic_policy || null,
        nn_opp_policy:        extras.nn_opp_policy        || null,
        pda: Number.isFinite(extras.pda) ? extras.pda : null,
    };
}

function repaintAllHeatmaps() {
    drawHeat("h_nn_policy", state ? state.nn_policy : null);
    drawHeat("h_nn_optimistic_policy",
        state ? state.nn_optimistic_policy : null);
    drawHeat("h_nn_opp_policy", state ? state.nn_opp_policy : null);
    drawHeat("h_mcts_play_selection",
        state ? state.mcts_play_selection : null);
    paintHeatModal();
}

function drawAll() {
    draw();
    drawValueChart();
    repaintAllHeatmaps();
    renderCandidates();
}

// Convert flat Float32Array(N*N) → 2D N×N for heatmap render.
function flatToGrid(flat) {
    const g = [];
    for (let r = 0; r < N; r++) {
        g.push([]);
        for (let c = 0; c < N; c++) g[r].push(flat[r * N + c]);
    }
    return g;
}

// Live (mid-search) candidate refresh. The worker streams partial visit / win-rate
// snapshots during a PUCT analysis; here we patch just the candidate fields of
// `state` and repaint the list + board overlay, so the per-move win% and sim
// counts fill in as it thinks. Heatmaps and the win-rate chart stay untouched —
// those still update only on the final `result`.
function applyLiveCandidates(visitsFlat, playSelectionFlat, winrateFlat, rootVisits) {
    if (!state) return;
    state.mcts_visits  = flatToGrid(visitsFlat);
    state.mcts_play_selection = playSelectionFlat
        ? flatToGrid(playSelectionFlat) : state.mcts_play_selection;
    state.mcts_winrate = winrateFlat ? flatToGrid(winrateFlat) : state.mcts_winrate;
    if (rootVisits > 0) searchSimsTotal = rootVisits;
    drawHeatById("h_mcts_play_selection",
        state.mcts_play_selection);
    paintHeatModal();
    renderCandidates();
    draw();
}

// NN-only heatmaps stream in the instant
// the worker (re)expands the root — before the search runs. Patch them into
// `state` and repaint just those canvases so the panel populates on your turn too,
// not only on a move's final `result`.
function applyLiveNNHeatmaps(data) {
    if (!state) return;
    state.policy_only = data.policyOnly === true;
    if (data.policyPrior) {
        state.policy_prior = flatToGrid(data.policyPrior);
    }
    state.nn_policy = flatToGrid(data.nnPolicy);
    state.nn_optimistic_policy =
        flatToGrid(data.nnOptimisticPolicy);
    state.nn_opp_policy = flatToGrid(data.nnOppPolicy);
    drawHeatById("h_nn_policy", state.nn_policy);
    drawHeatById("h_nn_optimistic_policy",
        state.nn_optimistic_policy);
    drawHeatById("h_nn_opp_policy", state.nn_opp_policy);
    paintHeatModal();
    renderCandidates();
    draw();
}

function newGame() {
    commitSkillLosses(humanMoveLosses().map(l => l.drop));   // bank the finished game into the skill window
    game = new Gomoku(N, currentRule);
    boardState = game.getInitialState();
    toPlay = 1;
    lastMove = null;
    ply = 0;
    gameOver = false;
    history = [];
    valueHistory = [];
    moveTimes = [];
    lastMoveAt = Date.now();
    gumbelPhases = null;
    winLine = null;
    reviewIndex = null;
    setCurrentValues(null, null);
    candSig = "";
    searchSimsTotal = 0;
    publishStateForDrawing();
    drawAll();
    updateReviewBar();
    worker.postMessage({ type: "reset", boardSize: N, ply: 0, rule: currentRule });
    // The engine ponders the human's opening (or analysis board); on the AI's
    // opening (human plays white) it searches and plays.
    triggerAISearch();
}

// True when the engine should *analyze* (ponder) the side-to-move rather than
// pick a move for it: always on the analysis board, and always on the human's
// turn in play mode — so the left-column stats are always live and the human can
// optionally see the analysis. Only the AI's own turn in play mode is a
// move-search.
function isPonderTurn() {
    if (gameOver || !boardState) return false;
    if (currentMode === "analysis") return true;
    return toPlay === humanSide;
}

function triggerAISearch() {
    if (gameOver) return;
    aiThinking = true;
    searchId++;
    const ponder = isPonderTurn();
    if (ponder && currentMode === "analysis") setStatus("status_analyzing", "thinking");
    else if (ponder) setStatus("status_your_turn", "active");   // play mode: human's turn, analysis runs quietly
    else setStatus("status_ai_thinking", "thinking");
    let sims = 0, timeMs = 0;
    const policyOnly = currentMode === "play" && !searchEnabled;
    if (ponder && !policyOnly) {
        // A ponder (analysis board, or the human's turn in play mode) deepens in
        // fixed-size PUCT chunks, re-triggered after each result until the player
        // moves or the depth cap is hit (see the result handler).
        sims = ANALYSIS_CHUNK;
    } else if (!policyOnly) {
        // Play mode, the AI's own turn: a single anytime-PUCT move-search for the
        // configured thinking time, then it plays.
        timeMs = thinkMs;
    }
    worker.postMessage({
        type: "search",
        state: boardState,
        toPlay: toPlay,
        ply: ply,
        sims: sims,
        timeMs: timeMs,
        searchId: searchId,
        // Play mode keeps the AI color as the PDA reference for the whole game;
        // free analysis keeps Black as the stable reference side.
        pda,
        pdaPla: currentMode === "analysis" ? 1 : -humanSide,
        policyOnly,
        // timeMs > 0 → anytime PUCT (the AI's move-search); else fixed-sims PUCT (any ponder).
        analyze: ponder && !policyOnly,
    });
}

function applyMoveLocal(action) {
    reviewIndex = null;   // a real move advances the line → snap back to live
    history.push({
        board: new Int8Array(boardState),
        toPlay,
        lastMove: lastMove ? { ...lastMove } : null,
        ply,
        gumbelPhases,
        rootWDL: lastRootWDL,
        nnWDL: lastNNWDL,
    });
    const r = (action / N) | 0, c = action % N;
    boardState = game.getNextState(boardState, action, toPlay);
    lastMove = { r, c };
    const winner = game.getWinner(boardState, action, toPlay);
    winLine = (winner === 1 || winner === -1) ? game.getWinLine(boardState, winner) : null;
    const movedBy = toPlay;
    toPlay = -toPlay;
    ply++;
    // Record this ply's wall-clock (time since the previous ply landed = how long
    // the mover took). Trim entries a prior undo left beyond `ply` before adding,
    // so replaying after an undo overwrites cleanly. (Blunder size is derived from
    // the win-rate history in renderGameStats, not captured here.)
    const nowTs = Date.now();
    if (lastMoveAt) {
        while (moveTimes.length && moveTimes[moveTimes.length - 1].step >= ply) moveTimes.pop();
        moveTimes.push({ step: ply, ms: nowTs - lastMoveAt, movedBy });
    }
    lastMoveAt = nowTs;
    if (winner !== null) {
        gameOver = true;
        let key;
        if      (winner === 1)  key = "status_black_wins";
        else if (winner === -1) key = "status_white_wins";
        else                    key = "status_draw";
        setStatus(key, "done");
    }
    // Clear the previous position's analysis overlay (candidate discs / heatmaps)
    // right away so the new position paints clean instead of freezing on stale
    // data until the next search result lands.
    publishStateForDrawing();
    drawAll();
    worker.postMessage({
        type: "move",
        action,
        nextState: boardState,
        nextToPlay: toPlay,
        ply,
    });
    return { winner, movedBy };
}

// Analysis mode: place the side-to-move's stone on an empty cell, then
// re-analyze the new position (the result handler won't play a move back).
function placeAnalysisStone(idx) {
    if (!boardState || boardState[idx] !== 0) return;
    searchSimsTotal = 0;
    const { winner } = applyMoveLocal(idx);
    if (winner === null) triggerAISearch();
}
// Play a move from the candidate list, honoring the current mode.
function playCandidate(r, c) {
    if (reviewIndex !== null) { setReview(null); return; }   // leave review first
    if (editMode || gameOver || !boardState) return;
    const idx = r * N + c;
    if (boardState[idx] !== 0) return;
    if (currentMode === "analysis") { placeAnalysisStone(idx); return; }
    if (toPlay !== humanSide) return;   // AI's turn; the human-turn ponder stays clickable
    const legal = game.getLegalActions(boardState, toPlay);
    if (!legal[idx]) return;
    const { winner } = applyMoveLocal(idx);
    if (winner === null) triggerAISearch();
}

// --- Board click ---
cv.addEventListener("click", (ev) => {
    if (editMode) { handleEditClick(ev); return; }
    if (reviewIndex !== null) { setReview(null); return; }   // a board click leaves review
    if (gameOver) return;
    const rect = cv.getBoundingClientRect();
    const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
    const c = Math.round((x - MARGIN) / CELL), r = Math.round((y - MARGIN) / CELL);
    if (r < 0 || r >= N || c < 0 || c >= N) return;
    const idx = r * N + c;
    if (boardState[idx] !== 0) return;
    // Analysis mode: free placement (both colors) the engine re-analyzes.
    if (currentMode === "analysis") { placeAnalysisStone(idx); return; }
    // Play mode: only the human's turn, only legal cells.
    if (toPlay !== humanSide) return;   // AI's turn; the human-turn ponder stays clickable
    const legal = game.getLegalActions(boardState, toPlay);
    if (!legal[idx]) return;   // Renju forbidden for black, etc.
    const { winner } = applyMoveLocal(idx);
    if (winner === null) triggerAISearch();
});

// Ghost-stone preview: track the hovered empty intersection and repaint only when
// it changes (mousemove fires a lot). Skipped while editing or reviewing; touch
// devices don't fire mousemove, so they just tap directly with no preview.
cv.addEventListener("mousemove", (ev) => {
    if (editMode || reviewIndex !== null) return;
    const rect = cv.getBoundingClientRect();
    const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
    const c = Math.round((x - MARGIN) / CELL), r = Math.round((y - MARGIN) / CELL);
    const cell = (r >= 0 && r < N && c >= 0 && c < N && boardState && boardState[r * N + c] === 0)
        ? { r, c } : null;
    if ((!cell && !hoverCell) || (cell && hoverCell && cell.r === hoverCell.r && cell.c === hoverCell.c)) return;
    hoverCell = cell;
    draw();
});
cv.addEventListener("mouseleave", () => {
    if (hoverCell) { hoverCell = null; draw(); }
});

function handleEditClick(ev) {
    const rect = cv.getBoundingClientRect();
    const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
    const c = Math.round((x - MARGIN) / CELL), r = Math.round((y - MARGIN) / CELL);
    if (r < 0 || r >= N || c < 0 || c >= N) return;
    const idx = r * N + c;
    let target;
    if (editTool === "alternate") {
        // Alternate: only fill empty cells (skip occupied so a misclick doesn't
        // overwrite). Color follows normal alternation rules — black if
        // counts are equal, otherwise the side with fewer stones plays.
        if (boardState[idx] !== 0) return;
        let nB = 0, nW = 0;
        for (let i = 0; i < boardState.length; i++) {
            if (boardState[i] === 1) nB++;
            else if (boardState[i] === -1) nW++;
        }
        target = (nB <= nW) ? 1 : -1;
    } else {
        target = (editTool === "black") ? 1 : (editTool === "white") ? -1 : 0;
    }
    if (boardState[idx] === target) return;
    editUndoStack.push({ idx, prev: boardState[idx] });
    refreshEditUndoBtn();
    boardState[idx] = target;
    publishStateForDrawing();
    draw();
    // Clear any prior "invalid count" warning so the user sees they're free
    // to keep editing once the position has changed.
    setStatus("status_editing", "info");
}

function undoEditStep() {
    if (!editMode || editUndoStack.length === 0) return;
    const { idx, prev } = editUndoStack.pop();
    boardState[idx] = prev;
    refreshEditUndoBtn();
    publishStateForDrawing();
    draw();
    setStatus("status_editing", "info");
}

// --- Worker message router ---
worker.onmessage = (e) => {
    const data = e.data;
    if (data.type === "model-progress") {
        if (Number.isFinite(data.percent)) setLoadingProgress(data.percent, data.loaded, data.total);
        // Download done — the worker is now building the inference session; tell
        // the user and run simulated progress until "ready" arrives.
        if (data.percent >= 100) {
            const m = modelById(currentModelId);
            if (m) startInitializationFeedback(m.id.toUpperCase(), m.label);
            else startInitializationFeedback("", "");
        }
        return;
    }
    if (data.type === "ready") {
        hideLoadingOverlay();
        // First-ready means model is loaded; if it's a swap, resume the search
        // the model switch aborted (ponder or the AI's own move-search). Don't
        // resume while editing a position: enterEditMode clears gameOver so the
        // `!gameOver` test alone would start a search on the half-built board.
        if (!boardState) newGame();
        else if (!gameOver && !editMode) triggerAISearch();
        return;
    }
    if (data.type === "error") {
        stopInitializationFeedback();
        const msg = t("err_prefix", data.message);
        setStatusRaw(msg, "error");
        aiThinking = false;
        // The loading overlay covers the status pill during first-model setup.
        // Surface worker/session failures in the overlay too; otherwise an
        // InferenceSession error looks like an endless initialization.
        const text = document.getElementById("loading_text");
        const progress = document.getElementById("loading_pct");
        if (text) text.textContent = msg;
        if (progress) progress.textContent = t("err_loading_retry");
        return;
    }
    if (data.type === "progress") {
        if (data.searchId !== searchId) return;
        // Mid-search snapshot: refresh the candidate list / board overlay / heatmaps
        // live as the search deepens — including the human's-turn ponder, so the
        // analysis fills in on your turn too, not only after a move's final result.
        if (data.nnPolicy) applyLiveNNHeatmaps(data);
        if (data.mctsVisits) {
            applyLiveCandidates(
                data.mctsVisits, data.mctsPlaySelection,
                data.mctsWinrate, data.searchSims);
        }
        return;
    }
    if (data.type === "result") {
        if (data.searchId !== searchId) return;
        aiThinking = false;
        searchSimsTotal = data.searchSims || 0;
        // Update gumbel overlay + heatmaps + candidate data.
        gumbelPhases = data.gumbelPhases;
        publishStateForDrawing({
            policy_only:   data.policyOnly === true,
            policy_prior:  flatToGrid(data.policyPrior),
            mcts_visits:    flatToGrid(data.mctsVisits),
            mcts_play_selection: flatToGrid(data.mctsPlaySelection),
            mcts_lcb:       flatToGrid(data.mctsLcb),
            mcts_lcb_radius:flatToGrid(data.mctsLcbRadius),
            mcts_winrate:   flatToGrid(data.mctsWinrate),
            nn_policy:      flatToGrid(data.nnPolicy),
            nn_optimistic_policy:
                flatToGrid(data.nnOptimisticPolicy),
            nn_opp_policy:  flatToGrid(data.nnOppPolicy),
            pda: data.pda,
        });
        // Values are from the searched side-to-move (= toPlay here, before any
        // move is applied). Convert to Black's frame for the chart + legend.
        const persp = toPlay;
        const rootRaw = data.rootValueWDL ? { w: data.rootValueWDL[0], d: data.rootValueWDL[1], l: data.rootValueWDL[2] } : null;
        const nnRaw   = data.nnValueWDL   ? { w: data.nnValueWDL[0],   d: data.nnValueWDL[1],   l: data.nnValueWDL[2]   } : null;
        const rootBlack = wdlToBlack(rootRaw, persp);
        const nnBlack   = wdlToBlack(nnRaw, persp);
        setCurrentValues(rootBlack, nnBlack);
        recordValues(rootBlack, nnBlack, state.board);
        drawAll();

        // Was this a ponder (analysis board, or the human's turn in play mode) or
        // the AI's move-search? At result time toPlay is the searched side.
        const wasPonder = currentMode === "analysis" || toPlay === humanSide;
        if (wasPonder) {
            // Both the analysis board and the human's turn keep deepening by
            // reusing the tree, up to a fixed memory-bounded depth — the human's
            // ponder runs until they move (which aborts via searchId) or the cap.
            const policyOnly = currentMode === "play" && !searchEnabled;
            const ponderOn = !policyOnly && !gameOver && searchSimsTotal < ANALYSIS_CAP_MIN;
            if (currentMode === "analysis") {
                setStatus(ponderOn ? "status_analyzing" : "status_analysis_ready",
                          ponderOn ? "thinking" : "info");
            } else {
                setStatus("status_your_turn", "active");   // play mode: quiet "your turn"
            }
            if (ponderOn) triggerAISearch();   // next chunk goes deeper (tree reuse)
            return;
        }
        // Play mode, AI's turn: play the chosen move, then ponder the human's reply.
        // Guard against a late ponder result landing after the game already ended:
        // a human's *winning* move skips triggerAISearch() (winner !== null), so
        // searchId isn't bumped to discard the in-flight ponder chunk — without
        // this check its result would drop a ghost AI stone on the won board.
        if (gameOver) return;
        const { winner } = applyMoveLocal(data.selectedAction);
        if (winner === null) triggerAISearch();
    }
};
(function initCandidateOverlayToggles() {
    const bindings = [
        ["show_analysis_human_input", "skz_show_analysis_human",
            () => showAnalysisOnHumanTurn,
            value => { showAnalysisOnHumanTurn = value; }],
        ["show_analysis_ai_input", "skz_show_analysis_ai",
            () => showAnalysisOnAiTurn,
            value => { showAnalysisOnAiTurn = value; }],
    ];
    for (const [id, storageKey, readValue, writeValue] of bindings) {
        const input = document.getElementById(id);
        if (!input) continue;
        input.checked = readValue();
        input.addEventListener("change", () => {
            writeValue(input.checked);
            try {
                localStorage.setItem(storageKey, input.checked ? "1" : "0");
            } catch (_) {}
            draw();
        });
    }
})();
// Candidate marker palette picker (Settings): fill each swatch with its scheme
// color, persist the choice, and redraw the board overlay on change.
(function initPalettePicker() {
    const seg = document.getElementById("palette_seg");
    if (!seg) return;
    const btns = seg.querySelectorAll(".swatch-btn[data-palette]");
    for (const b of btns) {
        const pal = CAND_PALETTES[b.dataset.palette];
        if (pal) b.style.background = pal.best.fill;
        b.setAttribute("aria-pressed", b.dataset.palette === candPalette ? "true" : "false");
        b.addEventListener("click", () => {
            const key = b.dataset.palette;
            if (!PALETTE_KEYS.includes(key) || key === candPalette) return;
            candPalette = key;
            try { localStorage.setItem("skz_cand_palette", key); } catch (_) {}
            for (const o of btns) o.setAttribute("aria-pressed", o.dataset.palette === key ? "true" : "false");
            if (typeof draw === "function") draw();
        });
    }
})();
// Thinking-time slider (toolbar): a stepped range over THINK_MS_OPTIONS. The
// options aren't evenly spaced (0.5/1/2/3/5/10s), so the slider walks their
// index rather than the raw ms; the live label shows the current value and it
// persists while dragging.
(function initThinkTime() {
    const range = document.getElementById("think_time_range");
    const valEl = document.getElementById("think_time_val");
    const trigEl = document.getElementById("think_trigger_val");
    if (!range) return;
    const fmt = (ms) => (ms / 1000) + (getLang() === "zh" ? "秒" : "s");
    range.max = String(THINK_MS_OPTIONS.length - 1);
    function reflect(idx) {
        const ms = THINK_MS_OPTIONS[idx];
        const txt = fmt(ms);
        if (valEl) valEl.textContent = txt;       // in-popover readout
        if (trigEl) trigEl.textContent = searchEnabled ? txt : t("thinking_off");
        range.setAttribute("aria-valuetext", txt);
        return ms;
    }
    let idx = THINK_MS_OPTIONS.indexOf(thinkMs);
    if (idx < 0) idx = THINK_MS_OPTIONS.indexOf(1000);
    range.value = String(idx);
    reflect(idx);
    range.addEventListener("input", () => {
        const i = parseInt(range.value, 10);
        if (!(i >= 0 && i < THINK_MS_OPTIONS.length)) return;
        thinkMs = reflect(i);
        try { localStorage.setItem("skz_think_ms", String(thinkMs)); } catch (_) {}
    });
    const toggle = document.getElementById("search_toggle");
    const toggleLabel = document.getElementById("thinking_toggle_label");
    function reflectSearch() {
        if (toggle) toggle.checked = searchEnabled;
        if (range) {
            range.disabled = !searchEnabled;
            range.setAttribute("aria-disabled", searchEnabled ? "false" : "true");
        }
        if (toggleLabel) toggleLabel.textContent = searchEnabled
            ? t("enable_thinking") : t("thinking_off");
        if (trigEl) trigEl.textContent = searchEnabled
            ? fmt(thinkMs) : t("thinking_off");
    }
    reflectSearch();
    if (toggle) toggle.addEventListener("change", () => {
        searchEnabled = toggle.checked;
        try { localStorage.setItem("skz_search_enabled", searchEnabled ? "1" : "0"); } catch (_) {}
        searchId++;
        aiThinking = false;
        if (boardState && !gameOver && !editMode) triggerAISearch();
    });
    // A label-wrapped checkbox can dispatch its synthetic click/change in a
    // browser-specific order. Re-read the final checked state after the click
    // so the collapsed trigger cannot be overwritten with the old time value.
    if (toggle) toggle.addEventListener("click", () => {
        setTimeout(() => {
            searchEnabled = toggle.checked;
            reflectSearch();
        }, 0);
    });
    registerI18nCallback(() => {
        reflect(idx);
        reflectSearch();
    });
})();

// PDA style slider: this is NN conditioning only and does not alter the search
// budget. Negative values present a more conservative style and positive values
// a more aggressive one. Stop the old search during a drag, then restart once
// the chosen value is committed.
(function initPdaStyle() {
    const range = document.getElementById("pda_range");
    if (!range) return;
    const snapToBalance = (value) => Math.abs(value) <= 0.15 ? 0 : value;
    function reflect(value) {
        const rounded = Math.round(value * 10) / 10;
        const normalized = Object.is(rounded, -0) ? 0 : rounded;
        const numericText = normalized.toFixed(1);
        range.value = String(normalized);
        range.setAttribute("aria-valuenow", String(normalized));
        range.setAttribute("aria-valuetext",
            normalized > 0 ? `+${numericText}` : numericText);
    }
    reflect(pda);
    range.addEventListener("input", () => {
        const value = Number(range.value);
        if (!Number.isFinite(value) || value < -1 || value > 1) return;
        pda = Math.round(snapToBalance(value) * 10) / 10;
        reflect(pda);
        try { localStorage.setItem("skz_pda", String(pda)); } catch (_) {}
        searchId++;
        aiThinking = false;
        worker.postMessage({ type: "set-pda", pda });
    });
    range.addEventListener("change", () => {
        if (boardState && !gameOver && !editMode) triggerAISearch();
    });
})();

// Think-time popover: the trigger (current seconds) toggles the slider panel.
// Mirrors the model dropdown / settings dialog (outside-click + Escape close),
// but flips a .open class on the wrapper instead of the .hidden toggle. On phones
// the panel is position:fixed (CSS) to escape #play_row's overflow:hidden, so it's
// anchored under the trigger here (clamped to the viewport); on desktop it's a
// plain CSS-positioned dropdown and these inline styles stay cleared.
(function initThinkPopover() {
    const sel = document.getElementById("think_select");
    const trigger = document.getElementById("think_trigger");
    const pop = document.getElementById("think_pop");
    if (!sel || !trigger || !pop) return;
    const mq = window.matchMedia("(max-width: 720px)");
    function anchorFixed() {
        const r = trigger.getBoundingClientRect();
        const margin = 8;
        const centeredLeft = r.left + r.width / 2 - pop.offsetWidth / 2;
        const left = Math.max(margin,
            Math.min(centeredLeft, window.innerWidth - pop.offsetWidth - margin));
        pop.style.left = left + "px";
        pop.style.top = (r.bottom + 6) + "px";
    }
    function setOpen(open) {
        if (open) {
            const modelMenu = document.getElementById("model_menu");
            const modelTrigger = document.getElementById("model_trigger");
            if (modelMenu) modelMenu.classList.add("hidden");
            if (modelTrigger) modelTrigger.setAttribute("aria-expanded", "false");
        }
        sel.classList.toggle("open", open);
        trigger.setAttribute("aria-expanded", open ? "true" : "false");
        if (open && mq.matches) anchorFixed();          // measured after .open shows it
        else { pop.style.left = ""; pop.style.top = ""; }
    }
    trigger.addEventListener("click", (ev) => {
        ev.stopPropagation();
        setOpen(!sel.classList.contains("open"));
    });
    pop.addEventListener("click", (ev) => ev.stopPropagation());
    document.addEventListener("click", () => setOpen(false));
    document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") setOpen(false); });
})();

// --- Buttons ---
document.getElementById("new_btn").addEventListener("click", newGame);

document.getElementById("undo_btn").addEventListener("click", () => {
    if (history.length === 0) return;
    reviewIndex = null;   // undo acts on the live game, not the reviewed ply
    // Analysis mode: undo a single placement. Play mode: undo back to the
    // human's turn (one ply if the AI is to move, otherwise the pair).
    let target = (currentMode === "analysis")
        ? history.length - 1
        : (toPlay !== humanSide ? history.length - 1 : history.length - 2);
    target = Math.max(0, target);
    if (target === history.length) return;
    let restoredRootWDL = null, restoredNNWDL = null;
    while (history.length > target) {
        const prev = history.pop();
        boardState = prev.board;
        toPlay = prev.toPlay;
        lastMove = prev.lastMove;
        ply = prev.ply;
        gumbelPhases = prev.gumbelPhases;
        restoredRootWDL = prev.rootWDL || null;
        restoredNNWDL = prev.nnWDL || null;
    }
    gameOver = false;
    winLine = null;
    aiThinking = false;
    searchId++;   // abort any in-flight search
    const sc = stoneCount(board1Dto2D(boardState));
    while (valueHistory.length && valueHistory[valueHistory.length - 1].step > sc) {
        valueHistory.pop();
    }
    while (moveTimes.length && moveTimes[moveTimes.length - 1].step > sc) moveTimes.pop();
    lastMoveAt = Date.now();   // restart the clock for the side about to move
    setCurrentValues(restoredRootWDL, restoredNNWDL);
    candSig = "";
    searchSimsTotal = 0;
    publishStateForDrawing();
    drawAll();
    updateReviewBar();
    worker.postMessage({ type: "reset", boardSize: N, ply, rule: currentRule });
    if (!gameOver) triggerAISearch();   // ponder the human's turn / search the AI's
});

// --- Edit-position mode ---
// On enter we snapshot the live game state; cancel restores it byte-for-byte
// without needing to talk to the worker (worker still mirrors the snapshot
// since we don't send it any move/reset during the edit).
//
// On commit we derive `toPlay` from stone counts (B==W → black to move,
// B==W+1 → white to move; anything else is a setup error and the user is
// kept in edit mode until they fix it). History/undo and the value chart
// reset because the new starting position has no recorded prior plies.
function snapshotForEdit() {
    return {
        boardState: new Int8Array(boardState),
        toPlay,
        lastMove: lastMove ? { ...lastMove } : null,
        ply, gameOver,
        gumbelPhases,
        history: history.slice(),
        valueHistory: valueHistory.slice(),
        moveTimes: moveTimes.slice(),
        lastRootWDL, lastNNWDL,
        lastStatus: { ...lastStatus, args: lastStatus.args.slice() },
        aiThinking,
    };
}

function setEditTool(tool) {
    if (!EDIT_TOOLS.includes(tool)) return;
    editTool = tool;
    for (const b of document.querySelectorAll(".seg-btn[data-edit-tool]")) {
        b.setAttribute("aria-pressed", b.dataset.editTool === tool ? "true" : "false");
    }
}

function enterEditMode() {
    if (editMode) return;
    // Close the settings dialog the edit button lives in.
    const pop = document.getElementById("settings_pop");
    if (pop) pop.classList.add("hidden");
    document.body.classList.remove("settings-open");
    document.getElementById("settings_btn")?.setAttribute("aria-expanded", "false");
    editMode = true;
    editSnapshot = snapshotForEdit();
    editUndoStack = [];
    refreshEditUndoBtn();
    searchId++;            // abort any in-flight search
    aiThinking = false;
    gameOver = false;      // edits are unrestricted; we recheck on commit
    gumbelPhases = null;   // hide gumbel overlay during setup
    winLine = null;
    reviewIndex = null;
    document.body.classList.add("editing");
    setEditTool(editTool);
    setStatus("status_editing", "info");
    publishStateForDrawing();
    updateReviewBar();
    syncBoardSize();   // toolbar height differs from play actions
    draw();
}

function exitEditMode(commit) {
    if (!editMode) return;

    if (!commit) {
        const s = editSnapshot;
        boardState = new Int8Array(s.boardState);
        toPlay = s.toPlay;
        lastMove = s.lastMove ? { ...s.lastMove } : null;
        ply = s.ply;
        gameOver = s.gameOver;
        const cw = s.gameOver ? game.getWinner(boardState, null, null) : null;
        winLine = (cw === 1 || cw === -1) ? game.getWinLine(boardState, cw) : null;
        gumbelPhases = s.gumbelPhases;
        history = s.history.slice();
        valueHistory = s.valueHistory.slice();
        moveTimes = s.moveTimes.slice();
        lastMoveAt = Date.now();
        setCurrentValues(s.lastRootWDL, s.lastNNWDL);
        candSig = "";
        editMode = false;
        editSnapshot = null;
        editUndoStack = [];
        refreshEditUndoBtn();
        document.body.classList.remove("editing");
        publishStateForDrawing();
        syncBoardSize();
        drawAll();
        // Restore exact prior status text and resume search if AI was thinking.
        const ls = s.lastStatus;
        if (ls.raw != null) setStatusRaw(ls.raw, ls.variant);
        else if (ls.key) setStatus(ls.key, ls.variant, ...ls.args);
        if (s.aiThinking) triggerAISearch();
        return;
    }

    // Commit: validate stone counts → derive toPlay.
    let nB = 0, nW = 0;
    for (let i = 0; i < boardState.length; i++) {
        if (boardState[i] === 1) nB++;
        else if (boardState[i] === -1) nW++;
    }
    let derived;
    if (nB === nW) derived = 1;            // black to play
    else if (nB === nW + 1) derived = -1;  // white to play
    else {
        setStatus("status_edit_invalid", "warn", nB, nW);
        return;   // stay in edit mode
    }

    toPlay = derived;
    ply = nB + nW;
    lastMove = null;
    history = [];
    valueHistory = [];
    moveTimes = [];
    lastMoveAt = Date.now();
    gumbelPhases = null;
    setCurrentValues(null, null);
    candSig = "";
    gameOver = false;

    // If the setup is already a finished position (5-in-a-row), surface that.
    // We pass null for lastAction so the renju forbidden-detect path is
    // skipped — there is no "last move" in a setup.
    const winner = game.getWinner(boardState, null, null);
    winLine = (winner === 1 || winner === -1) ? game.getWinLine(boardState, winner) : null;
    if (winner !== null) {
        gameOver = true;
        const key = winner === 1 ? "status_black_wins"
                  : winner === -1 ? "status_white_wins"
                  : "status_draw";
        setStatus(key, "done");
    }

    editMode = false;
    editSnapshot = null;
    editUndoStack = [];
    refreshEditUndoBtn();
    document.body.classList.remove("editing");
    publishStateForDrawing();
    syncBoardSize();
    drawAll();

    // Wipe the worker's MCTS root and let it rebuild from the new position
    // on the next search.
    worker.postMessage({ type: "reset", boardSize: N, ply, rule: currentRule });

    if (gameOver) return;
    if (toPlay === humanSide) setStatus("status_your_turn", "active");
    else triggerAISearch();
}

document.getElementById("edit_btn").addEventListener("click", enterEditMode);
document.getElementById("edit_done_btn").addEventListener("click", () => exitEditMode(true));
document.getElementById("edit_cancel_btn").addEventListener("click", () => exitEditMode(false));
document.getElementById("edit_undo_btn").addEventListener("click", undoEditStep);
for (const b of document.querySelectorAll(".seg-btn[data-edit-tool]")) {
    b.addEventListener("click", () => setEditTool(b.dataset.editTool));
}
document.addEventListener("keydown", (ev) => {
    if (!editMode) return;
    const tag = ev.target && ev.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (ev.key === "Escape") { ev.preventDefault(); exitEditMode(false); }
    else if (ev.key === "Enter") { ev.preventDefault(); exitEditMode(true); }
    else if ((ev.ctrlKey || ev.metaKey) && (ev.key === "z" || ev.key === "Z")) {
        ev.preventDefault(); undoEditStep();
    }
    else if (ev.key === "a" || ev.key === "A") setEditTool("alternate");
    else if (ev.key === "b" || ev.key === "B") setEditTool("black");
    else if (ev.key === "w" || ev.key === "W") setEditTool("white");
    else if (ev.key === "e" || ev.key === "E") setEditTool("erase");
});

// --- Move review / navigation --------------------------------------------
// Non-destructive scrubbing through the played move line: reviewIndex = null is
// the live tip; otherwise it's a history index (0 = empty opening) shown
// read-only while the live game, worker, candidate list and heatmaps keep
// running. Driven by ←/→/Home/End, the on-board review bar, and the win-rate
// chart. Stepping forward past the tip (→ / End / Live) returns to play.
function navStepAt(i) {
    return stoneCount1D(i < history.length ? history[i].board : boardState);
}
function navIndexForStep(step) {
    for (let i = 0; i < history.length; i++) if (navStepAt(i) === step) return i;
    return null;   // not a recorded past ply → the live tip
}
function updateReviewBar() {
    const bar = document.getElementById("review_bar");
    if (!bar) return;
    const active = reviewIndex !== null && reviewIndex < history.length;
    const wasActive = !bar.classList.contains("hidden");
    if (active) {
        const label = document.getElementById("review_label");
        if (label) label.textContent = t("review_label", reviewIndex, history.length);
        const prev = document.getElementById("review_prev");
        const first = document.getElementById("review_first");
        if (prev) prev.disabled = reviewIndex <= 0;
        if (first) first.disabled = reviewIndex <= 0;
        bar.classList.remove("hidden");
    } else {
        bar.classList.add("hidden");
    }
    if (active !== wasActive) syncBoardSize();   // reserve / free the bar's row
}
function setReview(i) {
    if (i === null || i >= history.length) i = null;
    else if (i < 0) i = 0;
    if (i === reviewIndex) { updateReviewBar(); return; }
    reviewIndex = i;
    hoverCell = null;          // a stale ghost would belong to the live position
    updateReviewBar();
    renderValuePanel();        // legend → reviewed ply + chart cursor
    draw();
}
function reviewStep(delta) {
    if (history.length === 0) return;
    if (reviewIndex === null) { if (delta < 0) setReview(history.length - 1); return; }
    setReview(reviewIndex + delta);
}
for (const [id, fn] of [
    ["review_first", () => setReview(0)],
    ["review_prev",  () => reviewStep(-1)],
    ["review_next",  () => reviewStep(1)],
    ["review_live",  () => setReview(null)],
]) {
    const el = document.getElementById(id);
    if (el) el.addEventListener("click", fn);
}

// Play-mode keyboard: ←/→/Home/End scrub the move line, Esc returns to live,
// u / Ctrl+Z take back a move, n starts a new game. Edit mode and open modals
// own the keyboard, and typing in a field is never intercepted.
document.addEventListener("keydown", (ev) => {
    if (editMode || expandedSourceId !== null || pendingSizeChange != null) return;
    const tag = ev.target && ev.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (ev.ctrlKey || ev.metaKey) {
        if (ev.key === "z" || ev.key === "Z") { ev.preventDefault(); document.getElementById("undo_btn").click(); }
        return;
    }
    if (ev.altKey) return;
    switch (ev.key) {
        case "ArrowLeft":  ev.preventDefault(); reviewStep(-1); break;
        case "ArrowRight": ev.preventDefault(); reviewStep(1); break;
        case "Home":       ev.preventDefault(); if (history.length) setReview(0); break;
        case "End":        ev.preventDefault(); setReview(null); break;
        case "Escape":     if (reviewIndex !== null) { ev.preventDefault(); setReview(null); } break;
        case "u": case "U": ev.preventDefault(); document.getElementById("undo_btn").click(); break;
        case "n": case "N": ev.preventDefault(); newGame(); break;
    }
});

// Side toggle buttons. Non-destructive: just swaps who plays which color
// without resetting the position. After an edit-mode setup the user may
// switch sides to ask the AI to play the side it shows as next-to-move; the
// position must survive that switch. The MCTS root tree is keyed on
// (state, toPlay) — neither changes here — so it's safe to keep.
function setSide(side) {
    if (side !== 1 && side !== -1) return;
    if (humanSide === side) return;
    humanSide = side;
    document.getElementById("side_black").setAttribute("aria-pressed", side === 1 ? "true" : "false");
    document.getElementById("side_white").setAttribute("aria-pressed", side === -1 ? "true" : "false");
    if (!boardState) return;            // pre-bootstrap: nothing else to do
    searchId++;                          // abort any in-flight search
    aiThinking = false;
    searchSimsTotal = 0;
    if (!gameOver) triggerAISearch();   // ponder the human's turn / search the AI's
}
document.getElementById("side_black").addEventListener("click", () => setSide(1));
document.getElementById("side_white").addEventListener("click", () => setSide(-1));

// Rule toggle buttons (renju / freestyle).
function setRule(rl) {
    if (!["renju", "standard", "freestyle"].includes(rl)) return;
    if (rl === currentRule) return;
    currentRule = rl;
    for (const b of document.querySelectorAll(".seg-btn[data-rule]")) {
        b.setAttribute("aria-pressed", b.dataset.rule === rl ? "true" : "false");
    }
    newGame();
}
for (const b of document.querySelectorAll(".seg-btn[data-rule]")) {
    b.addEventListener("click", () => setRule(b.dataset.rule));
}

// Board size slider (input range 11..15). `input` updates the display label
// continuously during drag; `change` (release) is what actually rebuilds the
// game so we don't fire newGame() once per intermediate value.
//
// Mid-game resize: keep every stone at its original (r, c) cell index. New
// offset = 0 keeps stones at the same (r,c) — bottom-right-only expansion.
// offset ≠ 0 shifts every stone by offset in both axes, used when both srcN and
// dstN are odd so new rows/columns are distributed symmetrically on all 4 sides.
function tryFitSameIndex(srcBoard, srcN, dstN, offset) {
    const dst = new Int8Array(dstN * dstN);
    for (let r = 0; r < srcN; r++) {
        for (let c = 0; c < srcN; c++) {
            const stone = srcBoard[r * srcN + c];
            if (stone === 0) continue;
            const nr = r + offset;
            const nc = c + offset;
            if (nr < 0 || nr >= dstN || nc < 0 || nc >= dstN) return null;
            dst[nr * dstN + nc] = stone;
        }
    }
    let translatedLast = null;
    if (lastMove) {
        const nr = lastMove.r + offset;
        const nc = lastMove.c + offset;
        if (nr >= 0 && nr < dstN && nc >= 0 && nc < dstN) {
            translatedLast = { r: nr, c: nc };
        }
    }
    return { board: dst, lastMove: translatedLast };
}

function migrateBoardSize(target, fitted) {
    N = target;
    game = new Gomoku(N, currentRule);
    boardState = fitted.board;
    lastMove = fitted.lastMove;
    gumbelPhases = null;
    winLine = null;
    reviewIndex = null;
    aiThinking = false;
    searchId++;   // abort any in-flight search
    history = [];   // old snapshots are at the previous size — undo can't replay across sizes
    moveTimes = [];
    lastMoveAt = Date.now();
    // publish before syncBoardSize: if the canvas resizes, syncBoardSize calls
    // draw(), which would otherwise iterate 0..N over a stale state.board.
    publishStateForDrawing();
    syncBoardSize();
    drawAll();
    updateReviewBar();
    worker.postMessage({ type: "reset", boardSize: N, ply, rule: currentRule });
    if (!gameOver && toPlay !== humanSide) {
        triggerAISearch();
    } else if (!gameOver) {
        setStatus("status_your_turn", "active");
    }
}

function setSize(sz) {
    if (!Number.isFinite(sz) || !BOARD_SIZES.includes(sz)) return;
    if (sz === N) return;
    if (!boardState) {
        // Pre-bootstrap (worker still loading the first model). The ready
        // handler will create the game once via newGame(); just update sizes.
        N = sz;
        syncBoardSize();
        return;
    }
    const offset = Math.floor((sz - 1) / 2) - Math.floor((N - 1) / 2);
    const fitted = tryFitSameIndex(boardState, N, sz, offset);
    if (fitted) {
        migrateBoardSize(sz, fitted);
    } else {
        showSizeConfirmModal(sz);
    }
}

// --- Board-size confirm modal ---
let pendingSizeChange = null;   // target size while modal is open
function syncSizeSlider() {
    const sizeInput   = document.getElementById("size_input");
    const sizeValueEl = document.getElementById("size_value");
    if (sizeInput)   sizeInput.value = String(N);
    if (sizeValueEl) sizeValueEl.textContent = String(N);
}
function renderSizeConfirmBody() {
    if (pendingSizeChange == null) return;
    const el = document.getElementById("size_confirm_body");
    if (el) el.textContent = t("size_confirm_body", pendingSizeChange);
}
function showSizeConfirmModal(target) {
    pendingSizeChange = target;
    renderSizeConfirmBody();
    document.getElementById("size_confirm_modal").classList.remove("hidden");
}
function closeSizeConfirmModal(commit) {
    if (pendingSizeChange == null) return;
    const target = pendingSizeChange;
    pendingSizeChange = null;
    document.getElementById("size_confirm_modal").classList.add("hidden");
    if (commit) {
        N = target;
        syncBoardSize();
        newGame();
    } else {
        syncSizeSlider();
    }
}
{
    const sizeInput   = document.getElementById("size_input");
    const sizeValueEl = document.getElementById("size_value");
    sizeInput.addEventListener("input",  () => { sizeValueEl.textContent = sizeInput.value; });
    sizeInput.addEventListener("change", () => { setSize(parseInt(sizeInput.value, 10)); });

    document.getElementById("size_confirm_ok").addEventListener("click", () => closeSizeConfirmModal(true));
    document.getElementById("size_confirm_cancel").addEventListener("click", () => closeSizeConfirmModal(false));
    document.getElementById("size_confirm_modal").addEventListener("click", (ev) => {
        if (ev.target === ev.currentTarget) closeSizeConfirmModal(false);
    });
    document.addEventListener("keydown", (ev) => {
        if (ev.key === "Escape" && pendingSizeChange != null) closeSizeConfirmModal(false);
    });
}
