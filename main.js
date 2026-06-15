// Board geometry — N is mutable (board size dropdown can change it).
let N = 15;
const MARGIN_DESKTOP = 28;   // leaves room for top + left coordinate labels
const MARGIN_COMPACT = 10;   // mobile / portrait — labels hidden, just enough for stones
let MARGIN = MARGIN_DESKTOP;
let CELL = 36;
let BOARD_LOGICAL = MARGIN * 2 + CELL * (N - 1);
const MONO_FONT = '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "DejaVu Sans Mono", monospace';
const HEAT_LOGICAL = 240;
const DPR = window.devicePixelRatio || 1;

function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function setupCanvas(canvas, logicalW, logicalH, setStyle = true) {
    canvas.width = Math.round(logicalW * DPR);
    canvas.height = Math.round(logicalH * DPR);
    if (setStyle) {
        canvas.style.width = logicalW + "px";
        canvas.style.height = logicalH + "px";
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx._logicalW = logicalW;
    ctx._logicalH = logicalH;
    return ctx;
}
function clearLogical(ctx) { ctx.clearRect(0, 0, ctx._logicalW, ctx._logicalH); }

// Board canvas + 6 heat canvases set up later in Tasks 20-21.
const cv = document.getElementById("board");
const ctx = setupCanvas(cv, BOARD_LOGICAL, BOARD_LOGICAL);

// Theme controller (tri-state segmented buttons).
const THEME_MODES = ["auto", "light", "dark"];
function resolveTheme(mode) {
    if (mode === "light" || mode === "dark") return mode;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function updateThemeSegPressed(mode) {
    const seg = document.getElementById("theme_seg");
    if (!seg) return;
    for (const b of seg.querySelectorAll(".seg-btn[data-theme]")) {
        b.setAttribute("aria-pressed", b.dataset.theme === mode ? "true" : "false");
    }
}
function applyTheme(mode) {
    if (!THEME_MODES.includes(mode)) mode = "auto";
    document.documentElement.dataset.theme = resolveTheme(mode);
    document.documentElement.dataset.themeMode = mode;
    updateThemeSegPressed(mode);
    if (typeof drawAll === "function") drawAll();
}
updateThemeSegPressed(document.documentElement.dataset.themeMode || "auto");
document.addEventListener("DOMContentLoaded", () => {
    const seg = document.getElementById("theme_seg");
    if (!seg) return;
    for (const b of seg.querySelectorAll(".seg-btn[data-theme]")) {
        b.addEventListener("click", () => {
            const next = b.dataset.theme;
            if (!THEME_MODES.includes(next)) return;
            try {
                if (next === "auto") localStorage.removeItem("skz_theme");
                else localStorage.setItem("skz_theme", next);
            } catch (_) {}
            applyTheme(next);
        });
    }
    updateThemeSegPressed(document.documentElement.dataset.themeMode || "auto");
});
try {
    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    const onSysChange = () => {
        if ((document.documentElement.dataset.themeMode || "auto") !== "auto") return;
        applyTheme("auto");
    };
    if (mql.addEventListener) mql.addEventListener("change", onSysChange);
    else if (mql.addListener) mql.addListener(onSysChange);
} catch (_) {}

// Play / Analysis mode (port of V7.5's 对弈 / 分析 toggle). Both modes show the
// full analysis surfaces; the toggle changes interaction only:
//   play     — human vs AI; the side picker chooses the human's color.
//   analysis — a free-placement board (both colors, alternating) the engine
//              re-analyzes after every stone; no human side, AI never plays.
// Persisted in localStorage("skz_mode"); a pre-paint inline script in
// gomoku.html applies body.mode-analysis before first paint.
let currentMode = (function () {
    try { return localStorage.getItem("skz_mode") === "analysis" ? "analysis" : "play"; }
    catch (_) { return "play"; }
})();
function updateModeButtons() {
    const p = document.getElementById("mode_play");
    const a = document.getElementById("mode_analysis");
    if (p) p.setAttribute("aria-pressed", currentMode === "play" ? "true" : "false");
    if (a) a.setAttribute("aria-pressed", currentMode === "analysis" ? "true" : "false");
}
function setMode(m) {
    if (m !== "play" && m !== "analysis") return;
    if (currentMode === m || editMode) return;
    currentMode = m;
    document.body.classList.toggle("mode-analysis", m === "analysis");
    try { localStorage.setItem("skz_mode", m); } catch (_) {}
    updateModeButtons();
    searchId++;            // abort any in-flight search
    aiThinking = false;
    if (typeof syncBoardSize === "function") syncBoardSize();
    if (!boardState) return;   // pre-bootstrap; newGame() will settle it
    candSig = "";              // mode change flips the candidate list ↔ analyze button
    if (!gameOver) triggerAISearch();   // ponder / play for the new mode
    renderCandidates();        // settle the list/button for the new mode
}

// "Analyze my turn" (play mode only): whether the engine ponders the human's own
// turn — filling the candidate list with live win% / visits — or sits idle until
// the human plays. Off by default for a distraction-free game; the header toggle
// or the centered "开启我的回合分析" button flips it on. Analysis mode ignores it (it
// always analyzes). Persisted in localStorage("skz_analyze_my_turn").
let analyzeMyTurn = (function () {
    try { return localStorage.getItem("skz_analyze_my_turn") === "1"; }
    catch (_) { return false; }
})();
function setAnalyzeMyTurn(on) {
    analyzeMyTurn = !!on;
    const cb = document.getElementById("analyze_turn_input");
    if (cb) cb.checked = analyzeMyTurn;
    try { localStorage.setItem("skz_analyze_my_turn", analyzeMyTurn ? "1" : "0"); } catch (_) {}
    candSig = "";   // flip the list ↔ analyze button
    if (analyzeMyTurn) {
        // Turned on: analyze the current position now if it's your turn,
        // otherwise the setting just takes effect on your next turn.
        if (!editMode && !aiThinking && !gameOver && isPonderTurn()) triggerAISearch();
        else renderCandidates();
    } else if (currentMode !== "analysis" && toPlay === humanSide) {
        // Turned off on your turn: abort the in-flight ponder and clear the
        // analysis back to the "开启我的回合分析" button.
        searchId++;
        aiThinking = false;
        searchSimsTotal = 0;
        if (state) { state.mcts_visits = null; state.mcts_winrate = null; }
        if (!gameOver) setStatus("status_your_turn", "active");
        renderCandidates();
        draw();
    } else {
        renderCandidates();   // off during the AI's turn: just refresh the toggle
    }
}

document.addEventListener("DOMContentLoaded", () => {
    updateModeButtons();
    const p = document.getElementById("mode_play");
    const a = document.getElementById("mode_analysis");
    if (p) p.addEventListener("click", () => setMode("play"));
    if (a) a.addEventListener("click", () => setMode("analysis"));
});

// --- Board sizing --------------------------------------------------------
// The three-column grid (320 · 1fr · 340) is fixed in CSS, so JS only fits the
// square board canvas into the centre column's width and the viewport height.
const boardCard = document.querySelector(".board-card");
const mainEl = document.querySelector(".main");
const topbarEl = document.querySelector(".topbar");
const appEl = document.querySelector(".app");
const boardColEl = document.querySelector(".board-col");

function syncBoardSize() {
    const compact = window.matchMedia("(max-width: 1180px)").matches;
    MARGIN = compact ? MARGIN_COMPACT : MARGIN_DESKTOP;

    const cardCS = getComputedStyle(boardCard);
    const cardPadX = parseFloat(cardCS.paddingLeft) + parseFloat(cardCS.paddingRight);
    const cardPadY = parseFloat(cardCS.paddingTop)  + parseFloat(cardCS.paddingBottom);
    const cardBorderX = parseFloat(cardCS.borderLeftWidth) + parseFloat(cardCS.borderRightWidth);
    const cardBorderY = parseFloat(cardCS.borderTopWidth)  + parseFloat(cardCS.borderBottomWidth);
    const availW = Math.max(0, boardColEl.clientWidth - cardPadX - cardBorderX);

    // Vertical budget: viewport minus app padding, toolbar, card chrome and
    // (while editing) the edit toolbar that sits under the board.
    const topbarCS = getComputedStyle(topbarEl);
    const topbarH = topbarEl.offsetHeight + parseFloat(topbarCS.marginBottom || 0);
    const appCS = getComputedStyle(appEl);
    const appPadY = parseFloat(appCS.paddingTop) + parseFloat(appCS.paddingBottom);
    const colCS = getComputedStyle(boardColEl);
    const colGap = parseFloat(colCS.rowGap || colCS.gap) || 0;
    const editTb = document.getElementById("edit_toolbar");
    const editH = (document.body.classList.contains("editing") && editTb)
        ? editTb.offsetHeight + colGap : 0;
    const reserved = appPadY + topbarH + cardPadY + cardBorderY + editH + 12;
    const availH = window.innerHeight - reserved;

    const cap = compact ? 560 : 720;
    const size = Math.max(compact ? 220 : 300, Math.min(availW, availH, cap));
    const minCell = compact ? 12 : 18;
    CELL = Math.max(minCell, Math.floor((size - 2 * MARGIN) / (N - 1)));
    BOARD_LOGICAL = MARGIN * 2 + CELL * (N - 1);
    if (cv.width !== Math.round(BOARD_LOGICAL * DPR)) {
        setupCanvas(cv, BOARD_LOGICAL, BOARD_LOGICAL);
        ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
        if (typeof draw === "function") draw();
    }
    // Pin the side columns to the board's height: the win-rate chart fills it,
    // and the analysis panel scrolls internally so expanding the heatmap drawer
    // shrinks the candidate list instead of stretching the page.
    const colH = compact ? "" : boardCard.offsetHeight + "px";
    const wcol = document.getElementById("winrate_col");
    const acol = document.getElementById("right_col");
    if (wcol) wcol.style.height = colH;
    if (acol) acol.style.height = colH;
}
new ResizeObserver(syncBoardSize).observe(mainEl);
new ResizeObserver(syncBoardSize).observe(topbarEl);
window.addEventListener("resize", syncBoardSize);

// Module-level game-display state. Updated by handlers in Task 24.
let state = null;        // { board: 2D N×N int, last_move: [r,c]|null, board_size: N }
let showGumbel = false;
let gumbelPhases = null; // last search's gumbel phases [[r,c]...] per phase
let boardOverlayHeatId = null; // heat canvas id currently mirrored on the board
let hoverCand = null;          // {r,c} candidate row under the pointer → board ring

// Run once synchronously so the first paint is already correctly sized,
// instead of flashing the 560px default on small viewports. Must run after
// the `state` / `gumbelPhases` lets above, since syncBoardSize may invoke
// draw() which reads them (TDZ otherwise).
syncBoardSize();

function draw() {
    clearLogical(ctx);
    const boardLine = cssVar("--board-line") || "#6b5a3a";
    const boardStar = cssVar("--board-star") || "#3a2e1a";
    const stoneB0 = cssVar("--stone-black-0");
    const stoneB1 = cssVar("--stone-black-1");
    const stoneW0 = cssVar("--stone-white-0");
    const stoneW1 = cssVar("--stone-white-1");
    const stoneOutline = cssVar("--stone-outline");
    const stoneShadow = cssVar("--stone-shadow") || "rgba(0,0,0,0.18)";

    ctx.strokeStyle = boardLine; ctx.lineWidth = 1;
    for (let i = 0; i < N; i++) {
        const p = Math.round(MARGIN + i * CELL) + 0.5;
        ctx.beginPath(); ctx.moveTo(MARGIN, p); ctx.lineTo(MARGIN + CELL * (N - 1), p); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(p, MARGIN); ctx.lineTo(p, MARGIN + CELL * (N - 1)); ctx.stroke();
    }
    ctx.fillStyle = boardStar;
    if (N >= 7) {
        const off = (N >= 13) ? 3 : 2;
        const pts = [[off, off], [off, N-1-off], [N-1-off, off], [N-1-off, N-1-off]];
        if (N % 2 === 1) pts.push([(N-1)/2, (N-1)/2]);
        for (const [r, c] of pts) {
            ctx.beginPath();
            ctx.arc(MARGIN + c * CELL, MARGIN + r * CELL, 3.5, 0, Math.PI * 2);
            ctx.fill();
        }
    }
    // Coordinate labels need ~14px of margin space; skip them in compact mode
    // so the board can use the saved real estate for larger cells.
    if (MARGIN >= 16) {
        ctx.fillStyle = boardLine;
        ctx.font = `11px ${MONO_FONT}`;
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        for (let i = 0; i < N; i++) {
            ctx.fillText(i, MARGIN + i * CELL, 12);
            ctx.fillText(i, 10, MARGIN + i * CELL);
        }
    }
    if (!state) return;

    const stoneR    = Math.max(6, Math.round(CELL * 0.39));
    const gumbelR   = Math.max(6, Math.round(CELL * 0.34));
    const lastDotR  = Math.max(2, Math.round(CELL * 0.11));
    const shadowDx  = Math.max(0, Math.round(CELL * 0.015));
    const shadowDy  = Math.max(1, Math.round(CELL * 0.045));
    const gradInner = Math.max(1, Math.round(CELL * 0.11));
    const gumbelFontPx = Math.max(8, Math.round(CELL * 0.28));

    if (showGumbel && gumbelPhases && gumbelPhases.length > 0) {
        const COLORS = ["#9ca3af","#3b82f6","#10b981","#f59e0b","#ef4444"];
        const LABELS = ["16","8","4","2","1"];
        const deepest = new Map();
        for (let i = 0; i < gumbelPhases.length; i++) {
            for (const rc of gumbelPhases[i]) {
                const key = rc[0] * N + rc[1];
                if (!deepest.has(key) || deepest.get(key) < i) deepest.set(key, i);
            }
        }
        for (const [key, idx] of deepest) {
            const r = (key / N) | 0, c = key % N;
            const sizeLabel = String(gumbelPhases[idx].length);
            let bucket = LABELS.indexOf(sizeLabel);
            if (bucket < 0) bucket = Math.min(idx, COLORS.length - 1);
            const x = MARGIN + c * CELL, y = MARGIN + r * CELL;
            ctx.beginPath(); ctx.arc(x, y, gumbelR, 0, Math.PI * 2);
            ctx.lineWidth = 2.5;
            ctx.strokeStyle = COLORS[bucket];
            ctx.stroke();
            ctx.lineWidth = 1;
            if (state.board[r][c] === 0) {
                ctx.fillStyle = COLORS[bucket];
                ctx.font = `bold ${gumbelFontPx}px ${MONO_FONT}`;
                ctx.textAlign = "center"; ctx.textBaseline = "middle";
                ctx.fillText(sizeLabel, x, y);
            }
        }
    }

    const b = state.board, lm = state.last_move;
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
        const v = b[r][c]; if (!v) continue;
        const x = MARGIN + c * CELL, y = MARGIN + r * CELL;
        ctx.beginPath(); ctx.arc(x + shadowDx, y + shadowDy, stoneR, 0, Math.PI * 2);
        ctx.fillStyle = stoneShadow; ctx.fill();
        ctx.beginPath(); ctx.arc(x, y, stoneR, 0, Math.PI * 2);
        if (v === 1) {
            const grad = ctx.createRadialGradient(x - gradInner, y - gradInner, 2, x, y, stoneR);
            grad.addColorStop(0, stoneB0); grad.addColorStop(1, stoneB1);
            ctx.fillStyle = grad;
        } else {
            const grad = ctx.createRadialGradient(x - gradInner, y - gradInner, 2, x, y, stoneR);
            grad.addColorStop(0, stoneW0); grad.addColorStop(1, stoneW1);
            ctx.fillStyle = grad;
        }
        ctx.fill();
        ctx.strokeStyle = stoneOutline; ctx.lineWidth = 1; ctx.stroke();
        if (lm && lm[0] === r && lm[1] === c) {
            ctx.beginPath(); ctx.arc(x, y, lastDotR, 0, Math.PI * 2);
            ctx.fillStyle = "#ef4444"; ctx.fill();
        }
    }

    if (boardOverlayHeatId) drawBoardHeatOverlay(boardOverlayHeatId);
    else drawCandidateOverlay();
}

// Lizzie-style candidate discs on the board (shown when no heatmap is pinned):
// best move in the selected palette's accent, others fade along its ramp by
// visit share; big win%, small visit count. Mirrors V7.5's board overlay.
// Palette is user-selectable in Settings (candColor). Data: computeCandidates().
function drawCandidateOverlay() {
    if (!state) return;
    const cands = computeCandidates();
    if (cands.length === 0) return;
    const candR     = Math.max(7, Math.round(CELL * 0.40));
    const wrFontPx  = Math.max(9, Math.round(CELL * 0.30));
    const visFontPx = Math.max(7, Math.round(CELL * 0.21));
    for (const o of cands) {
        const x = MARGIN + o.c * CELL, y = MARGIN + o.r * CELL;
        const col = candColor(o.frac, o.best);
        ctx.globalAlpha = o.best ? 0.92 : (0.45 + 0.45 * o.frac);
        ctx.beginPath(); ctx.arc(x, y, candR, 0, Math.PI * 2);
        ctx.fillStyle = col.fill; ctx.fill();
        ctx.globalAlpha = 1;
        if (o.best) {
            ctx.beginPath(); ctx.arc(x, y, candR, 0, Math.PI * 2);
            ctx.lineWidth = 2; ctx.strokeStyle = col.stroke; ctx.stroke(); ctx.lineWidth = 1;
        }
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillStyle = col.text;
        const hasWr = o.wr != null, hasVis = o.vf > 0;
        if (hasWr && hasVis) {
            ctx.font = `bold ${wrFontPx}px ${MONO_FONT}`;
            ctx.fillText(Math.round(o.wr * 100), x, y - wrFontPx * 0.42);
            ctx.font = `${visFontPx}px ${MONO_FONT}`;
            ctx.globalAlpha = 0.85;
            ctx.fillText(fmtVisits(o.vf), x, y + visFontPx * 0.75);
            ctx.globalAlpha = 1;
        } else {
            ctx.font = `bold ${wrFontPx}px ${MONO_FONT}`;
            ctx.fillText(hasWr ? Math.round(o.wr * 100) : fmtVisits(o.vf), x, y);
        }
    }
    // Hover highlight driven by the candidate list on the right.
    if (hoverCand && cands.some(o => o.r === hoverCand.r && o.c === hoverCand.c)) {
        const x = MARGIN + hoverCand.c * CELL, y = MARGIN + hoverCand.r * CELL;
        ctx.beginPath(); ctx.arc(x, y, candR + 2, 0, Math.PI * 2);
        ctx.lineWidth = 2.5; ctx.strokeStyle = cssVar("--accent") || "#0969da";
        ctx.stroke(); ctx.lineWidth = 1;
    }
}

// Tint cells centered on intersections; skip labels on occupied cells so
// numbers don't pile up on the stones already drawn underneath.
function drawBoardHeatOverlay(canvasId) {
    if (!state) return;
    const gridKey = HEAT_GRID_KEYS[canvasId];
    if (!gridKey) return;
    const grid = state[gridKey];
    if (!grid) return;
    const signed = SIGNED_HEAT_IDS.has(canvasId);
    const heatText = cssVar("--heat-text") || "#111";

    let maxV = 0;
    if (!signed) {
        for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) {
            if (grid[r][k] > maxV) maxV = grid[r][k];
        }
    }

    const tileX0 = MARGIN - CELL / 2;
    const tileSpan = CELL * N;
    ctx.save();
    ctx.beginPath();
    ctx.rect(tileX0, tileX0, tileSpan, tileSpan);
    ctx.clip();

    const fontPx = Math.max(8, Math.floor(CELL * (signed ? 0.32 : 0.38)));
    ctx.font = `${fontPx}px ${MONO_FONT}`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";

    for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) {
        const v = grid[r][k];
        const cx = MARGIN + k * CELL;
        const cy = MARGIN + r * CELL;
        const x = cx - CELL / 2;
        const y = cy - CELL / 2;

        let alpha;
        if (signed) {
            alpha = Math.min(1, Math.abs(v));
            if (v > 0) ctx.fillStyle = `rgba(9,105,218,${(alpha * 0.7).toFixed(3)})`;
            else if (v < 0) ctx.fillStyle = `rgba(207,34,46,${(alpha * 0.7).toFixed(3)})`;
            else continue;
        } else {
            alpha = (maxV > 0 && v > 0) ? Math.min(1, v / maxV) : 0;
            if (alpha === 0) continue;
            ctx.fillStyle = `rgba(220,38,38,${(alpha * 0.7).toFixed(3)})`;
        }
        ctx.fillRect(x, y, CELL, CELL);

        if (state.board[r][k]) continue;
        const showThreshold = signed ? 0.05 : 0.01;
        if (Math.abs(v) < showThreshold) continue;
        ctx.fillStyle = alpha > 0.55 ? "#fff" : heatText;
        const label = signed
            ? (v >= 0 ? "+" : "") + (v * 100).toFixed(0)
            : (v * 100).toFixed(0);
        ctx.fillText(label, cx, cy);
    }
    ctx.restore();
}

// --- Six heatmap canvases ---
const heatCtxs = {
    h_mcts_policy:    setupCanvas(document.getElementById("h_mcts_policy"),    HEAT_LOGICAL, HEAT_LOGICAL, false),
    h_mcts_visits:    setupCanvas(document.getElementById("h_mcts_visits"),    HEAT_LOGICAL, HEAT_LOGICAL, false),
    h_nn_policy:      setupCanvas(document.getElementById("h_nn_policy"),      HEAT_LOGICAL, HEAT_LOGICAL, false),
    h_nn_opp_policy:  setupCanvas(document.getElementById("h_nn_opp_policy"),  HEAT_LOGICAL, HEAT_LOGICAL, false),
    h_nn_futurepos_8: setupCanvas(document.getElementById("h_nn_futurepos_8"), HEAT_LOGICAL, HEAT_LOGICAL, false),
    h_nn_futurepos_32:setupCanvas(document.getElementById("h_nn_futurepos_32"),HEAT_LOGICAL, HEAT_LOGICAL, false),
};
const HEAT_GRID_KEYS = {
    h_mcts_policy:    "mcts_policy",
    h_mcts_visits:    "mcts_visits",
    h_nn_policy:      "nn_policy",
    h_nn_opp_policy:  "nn_opp_policy",
    h_nn_futurepos_8: "nn_futurepos_8",
    h_nn_futurepos_32:"nn_futurepos_32",
};
const SIGNED_HEAT_IDS = new Set(["h_nn_futurepos_8", "h_nn_futurepos_32"]);

// Pinned heatmap (radio: at most one). When set, the board mirrors it for the
// rest of the session; hover still previews other heatmaps temporarily and
// falls back to the pinned id on mouseleave. Not persisted across loads: every
// load starts with no board overlay (pinning is opt-in, off by default).
let pinnedHeatId = null;
function syncPinButtonsUI() {
    for (const btn of document.querySelectorAll(".pin-btn")) {
        const on = btn.dataset.target === pinnedHeatId;
        btn.classList.toggle("active", on);
        btn.setAttribute("aria-pressed", on ? "true" : "false");
    }
}
function setPinnedHeatId(id) {
    pinnedHeatId = (id && HEAT_GRID_KEYS[id]) ? id : null;
    syncPinButtonsUI();
    boardOverlayHeatId = pinnedHeatId;
    draw();
}

function fitHeatCanvas(canvasId) {
    const c = document.getElementById(canvasId);
    const card = c.parentElement;
    const cardCS = getComputedStyle(card);
    const padX = parseFloat(cardCS.paddingLeft) + parseFloat(cardCS.paddingRight);
    const padY = parseFloat(cardCS.paddingTop)  + parseFloat(cardCS.paddingBottom);
    const title = card.querySelector(".grid-title");
    let titleH = 0;
    if (title) {
        const tCS = getComputedStyle(title);
        titleH = title.offsetHeight + parseFloat(tCS.marginTop || 0) + parseFloat(tCS.marginBottom || 0);
    }
    const availW = card.clientWidth - padX;
    const availH = card.clientHeight - padY - titleH;
    const size = Math.max(60, Math.floor(Math.min(availW, availH > 0 ? availH : availW)));
    c.style.width = size + "px";
    c.style.height = size + "px";
    if (heatCtxs[canvasId]._logicalW === size) return false;
    heatCtxs[canvasId] = setupCanvas(c, size, size, false);
    return true;
}
for (const id of Object.keys(heatCtxs)) {
    const c = document.getElementById(id);
    new ResizeObserver(() => {
        if (!fitHeatCanvas(id)) return;
        const grid = state ? state[HEAT_GRID_KEYS[id]] : null;
        drawHeatById(id, grid);
    }).observe(c.parentElement);
    fitHeatCanvas(id);
    const card = c.parentElement;
    card.addEventListener("mouseenter", () => {
        if (!window.matchMedia("(hover: hover) and (pointer: fine)").matches) return;
        if (expandedSourceId !== null) return;
        boardOverlayHeatId = id;
        draw();
    });
    card.addEventListener("mouseleave", () => {
        if (boardOverlayHeatId === id) {
            boardOverlayHeatId = pinnedHeatId;
            draw();
        }
    });
    const pinBtn = card.querySelector(".pin-btn");
    if (pinBtn) {
        pinBtn.addEventListener("click", (ev) => {
            ev.stopPropagation();
            setPinnedHeatId(pinnedHeatId === id ? null : id);
        });
    }
}
syncPinButtonsUI();

function drawHeatById(id, grid) {
    if (SIGNED_HEAT_IDS.has(id)) drawHeatSigned(id, grid);
    else drawHeat(id, grid);
}

function drawHeat(canvasId, grid) {
    const g = heatCtxs[canvasId];
    clearLogical(g);
    const W = g._logicalW;
    const cell = W / N;
    const gridCol = cssVar("--heat-grid") || "#e5e7eb";
    let maxV = 0;
    if (grid) for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) if (grid[r][k] > maxV) maxV = grid[r][k];
    for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) {
        const x = k * cell, y = r * cell;
        const v = grid ? grid[r][k] : 0;
        const a = (maxV > 0 && v > 0) ? Math.min(1, v / maxV) : 0;
        g.fillStyle = `rgba(220,38,38,${a.toFixed(3)})`;
        g.fillRect(x, y, cell, cell);
        g.strokeStyle = gridCol;
        g.strokeRect(x + 0.5, y + 0.5, cell, cell);
        if (v >= 0.01) {
            g.fillStyle = a > 0.5 ? "#fff" : (cssVar("--heat-text") || "#111");
            g.font = `${Math.floor(cell * 0.38)}px ${MONO_FONT}`;
            g.textAlign = "center"; g.textBaseline = "middle";
            g.fillText((v * 100).toFixed(0), x + cell / 2, y + cell / 2);
        }
    }
    if (state && state.board) overlayStones(g, cell);
}

function drawHeatSigned(canvasId, grid) {
    const g = heatCtxs[canvasId];
    clearLogical(g);
    const W = g._logicalW;
    const cell = W / N;
    const gridCol = cssVar("--heat-grid") || "#e5e7eb";
    const heatText = cssVar("--heat-text") || "#111";
    for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) {
        const x = k * cell, y = r * cell;
        const v = grid ? grid[r][k] : 0;
        const a = Math.min(1, Math.abs(v));
        if (v > 0) g.fillStyle = `rgba(9,105,218,${a.toFixed(3)})`;
        else if (v < 0) g.fillStyle = `rgba(207,34,46,${a.toFixed(3)})`;
        else g.fillStyle = "rgba(0,0,0,0)";
        g.fillRect(x, y, cell, cell);
        g.strokeStyle = gridCol;
        g.strokeRect(x + 0.5, y + 0.5, cell, cell);
        if (Math.abs(v) >= 0.05) {
            g.fillStyle = a > 0.5 ? "#fff" : heatText;
            g.font = `${Math.floor(cell * 0.32)}px ${MONO_FONT}`;
            g.textAlign = "center"; g.textBaseline = "middle";
            const label = (v >= 0 ? "+" : "") + (v * 100).toFixed(0);
            g.fillText(label, x + cell / 2, y + cell / 2);
        }
    }
    if (state && state.board) overlayStones(g, cell);
}

function overlayStones(g, cell) {
    const r0 = cell * 0.32;
    for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) {
        const sv = state.board[r][k]; if (!sv) continue;
        const cx = k * cell + cell / 2, cy = r * cell + cell / 2;
        g.beginPath(); g.arc(cx, cy, r0, 0, Math.PI * 2);
        if (sv === 1) {
            g.fillStyle = "rgba(0,0,0,0.7)"; g.fill();
            g.lineWidth = 1; g.strokeStyle = "rgba(255,255,255,0.6)"; g.stroke();
        } else {
            g.fillStyle = "rgba(255,255,255,0.85)"; g.fill();
            g.lineWidth = 1; g.strokeStyle = "rgba(0,0,0,0.5)"; g.stroke();
        }
    }
}

// --- Heat modal (fullscreen single heatmap) ---
let expandedSourceId = null;
function setupModalCanvas() {
    const canvas = document.getElementById("heat_modal_canvas");
    const card = canvas.parentElement;
    const cardCS = getComputedStyle(card);
    const padX = parseFloat(cardCS.paddingLeft) + parseFloat(cardCS.paddingRight);
    const padY = parseFloat(cardCS.paddingTop)  + parseFloat(cardCS.paddingBottom);
    const header = card.querySelector(".heat-modal-header");
    const headerH = header ? header.offsetHeight + 12 : 0;
    const availW = window.innerWidth  * 0.95 - padX;
    const availH = window.innerHeight * 0.95 - padY - headerH;
    const sz = Math.max(240, Math.floor(Math.min(availW, availH)));
    canvas.style.width  = sz + "px";
    canvas.style.height = sz + "px";
    heatCtxs.h_modal = setupCanvas(canvas, sz, sz, false);
}
function paintHeatModal() {
    if (!expandedSourceId) return;
    const grid = state ? state[HEAT_GRID_KEYS[expandedSourceId]] : null;
    if (SIGNED_HEAT_IDS.has(expandedSourceId)) drawHeatSigned("h_modal", grid);
    else drawHeat("h_modal", grid);
}
function openHeatModal(sourceId) {
    if (!HEAT_GRID_KEYS[sourceId]) return;
    expandedSourceId = sourceId;
    const card = document.getElementById(sourceId).parentElement;
    const titleEl = card.querySelector(".grid-title-text");
    document.getElementById("heat_modal_title").textContent = titleEl ? titleEl.textContent : t("heatmap_default_title");
    document.getElementById("heat_modal").classList.remove("hidden");
    setupModalCanvas();
    paintHeatModal();
}
function closeHeatModal() {
    if (expandedSourceId === null) return;
    expandedSourceId = null;
    document.getElementById("heat_modal").classList.add("hidden");
}
for (const btn of document.querySelectorAll(".expand-btn")) {
    btn.addEventListener("click", () => openHeatModal(btn.dataset.target));
}
document.getElementById("heat_modal_close").addEventListener("click", closeHeatModal);
document.getElementById("heat_modal").addEventListener("click", (ev) => {
    if (ev.target === ev.currentTarget) closeHeatModal();
});
document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && expandedSourceId !== null) closeHeatModal();
});
window.addEventListener("resize", () => {
    if (expandedSourceId !== null) {
        setupModalCanvas();
        paintHeatModal();
    }
});

// --- Value chart ---
// Sizing is handled by CSS (#value_chart in style.css) so that mobile media
// queries can shrink the canvas — JS just keeps the bitmap in sync via the
// ResizeObserver below.
const vcCanvas = document.getElementById("value_chart");
let vctx = setupCanvas(vcCanvas, 280, 160, false);
function resizeValueChart() {
    const rect = vcCanvas.getBoundingClientRect();
    const w = Math.max(120, Math.floor(rect.width));
    const h = Math.max(120, Math.floor(rect.height));
    if (vctx._logicalW === w && vctx._logicalH === h) return;
    // setStyle=false: leave the displayed size to CSS so the canvas can
    // shrink back when toggling out of simple mode (where flex-grow lets
    // it expand to fill the right column). An inline style.height set
    // here would otherwise persist and keep the chart tall in analysis
    // mode.
    vctx = setupCanvas(vcCanvas, w, h, false);
    drawValueChart();
}
new ResizeObserver(resizeValueChart).observe(vcCanvas);

// valueHistory keeps one Black-frame WDL per ply for each evaluation:
//   [{step, root:{w,d,l}|null, nn:{w,d,l}|null}]   w = Black win, l = White win.
let valueHistory = [];
let valueTab = "root";    // which evaluation the chart shows: "root" | "nn"
let lastRootWDL = null;   // current ply's root/nn value, already in Black's frame
let lastNNWDL = null;

function stoneCount(board2d) {
    let n = 0;
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) if (board2d[r][c]) n++;
    return n;
}
// Re-express a side-to-move {w,d,l} in Black's frame given the searcher's side
// (persp: +1 = Black to move, -1 = White to move). Returns normalized {w,d,l}.
function wdlToBlack(v, persp) {
    if (!v) return null;
    const s = v.w + v.d + v.l;
    if (s <= 1e-4) return null;
    const w = v.w / s, d = v.d / s, l = v.l / s;
    return persp === -1 ? { w: l, d, l: w } : { w, d, l };
}
// Append the current ply's Black-frame root/nn WDL (one point per stone count).
function recordValues(rootBlack, nnBlack, board2d) {
    if (!board2d) return;
    const step = stoneCount(board2d);
    while (valueHistory.length && valueHistory[valueHistory.length - 1].step > step) {
        valueHistory.pop();
    }
    const last = valueHistory[valueHistory.length - 1];
    if (!rootBlack && !nnBlack) {
        // No fresh eval (e.g. after undo): carry the last point forward so the
        // chart keeps one point per ply.
        if (last && step > last.step) valueHistory.push({ step, root: last.root, nn: last.nn });
        return;
    }
    if (last && last.step === step) {
        if (rootBlack) last.root = rootBlack;
        if (nnBlack) last.nn = nnBlack;
    } else if (!last || step > last.step) {
        valueHistory.push({ step, root: rootBlack, nn: nnBlack });
    }
}

// Stacked area over moves for the active tab: white (bottom) / draw / black (top).
function drawValueChart() {
    clearLogical(vctx);
    const W = vctx._logicalW, H = vctx._logicalH;
    const padL = 26, padR = 6, padT = 6, padB = 14;
    const innerW = W - padL - padR, innerH = H - padT - padB;
    const axis = cssVar("--border") || "#d8dee4";
    const grid = cssVar("--heat-grid") || "#e5e7eb";
    const subtle = cssVar("--fg-subtle") || "#8b949e";
    const muted = cssVar("--fg-muted") || "#59636e";
    const yOf = (f) => padT + (1 - f) * innerH;
    vctx.strokeStyle = grid; vctx.lineWidth = 1;
    for (const f of [0, 0.25, 0.5, 0.75, 1]) {
        const y = yOf(f) + 0.5;
        vctx.beginPath(); vctx.moveTo(padL, y); vctx.lineTo(W - padR, y); vctx.stroke();
    }
    vctx.fillStyle = subtle; vctx.font = `10px ${MONO_FONT}`;
    vctx.textAlign = "right"; vctx.textBaseline = "middle";
    for (const f of [1, 0.5, 0]) vctx.fillText(Math.round(f * 100) + "%", padL - 4, yOf(f));
    vctx.strokeStyle = axis;
    vctx.beginPath();
    vctx.moveTo(padL + 0.5, padT); vctx.lineTo(padL + 0.5, H - padB);
    vctx.lineTo(W - padR, H - padB); vctx.stroke();

    const pts = valueHistory.filter(p => p[valueTab]).map(p => ({ step: p.step, ...p[valueTab] }));
    if (pts.length === 0) {
        vctx.fillStyle = subtle; vctx.font = `11px ${MONO_FONT}`;
        vctx.textAlign = "center"; vctx.textBaseline = "middle";
        vctx.fillText(t("chart_no_data"), padL + innerW / 2, padT + innerH / 2);
        return;
    }
    const n = pts.length;
    // x = data-point index, so the first eval sits at the left edge.
    const xAt = (i) => n <= 1 ? padL : padL + (i / (n - 1)) * innerW;
    vctx.fillStyle = muted; vctx.textAlign = "center"; vctx.textBaseline = "top";
    vctx.fillText(String(pts[0].step), xAt(0), H - padB + 2);
    if (n > 1) vctx.fillText(String(pts[n - 1].step), xAt(n - 1), H - padB + 2);

    // Fill one stacked band between two cumulative-fraction accessors.
    function band(lowerFn, upperFn, color) {
        vctx.fillStyle = color;
        vctx.beginPath();
        pts.forEach((p, i) => { const x = xAt(i), y = yOf(upperFn(p)); i === 0 ? vctx.moveTo(x, y) : vctx.lineTo(x, y); });
        for (let i = n - 1; i >= 0; i--) vctx.lineTo(xAt(i), yOf(lowerFn(pts[i])));
        vctx.closePath(); vctx.fill();
    }
    const colBlk = cssVar("--stone-black-1") || "#000";
    const colWht = cssVar("--stone-white-0") || "#fff";
    const colDrw = cssVar("--fg-subtle") || "#8b949e";
    vctx.globalAlpha = 0.18;
    band(() => 0,        p => p.l,       colWht);  // white win — bottom
    band(p => p.l,       p => p.l + p.d, colDrw);  // draw — middle
    band(p => p.l + p.d, () => 1,        colBlk);  // black win — top
    vctx.globalAlpha = 1;
    // The two band boundaries (white/draw and draw/black).
    function line(fn) {
        vctx.beginPath();
        pts.forEach((p, i) => { const x = xAt(i), y = yOf(fn(p)); i === 0 ? vctx.moveTo(x, y) : vctx.lineTo(x, y); });
        vctx.stroke();
    }
    vctx.strokeStyle = muted; vctx.lineWidth = 1.5;
    line(p => p.l);
    line(p => p.l + p.d);
}

// --- Win-rate legend + tabs (active tab, current ply, Black's frame) ---
function normalizeVal(v) {
    if (!v) return null;
    const s = v.w + v.d + v.l;
    if (s > 1e-4) return { w: v.w / s * 100, d: v.d / s * 100, l: v.l / s * 100 };
    return { w: v.w * 100, d: v.d * 100, l: v.l * 100 };
}
function renderWinLegend(v) {
    const bEl = document.getElementById("vc_w_black");
    const dEl = document.getElementById("vc_w_draw");
    const wEl = document.getElementById("vc_w_white");
    if (!bEl || !dEl || !wEl) return;
    const n = normalizeVal(v);
    if (!n) { bEl.textContent = dEl.textContent = wEl.textContent = t("wdl_dash"); return; }
    bEl.textContent = n.w.toFixed(1) + "%";   // black win
    dEl.textContent = n.d.toFixed(1) + "%";   // draw
    wEl.textContent = n.l.toFixed(1) + "%";   // white win
}
function renderValuePanel() {
    renderWinLegend(valueTab === "root" ? lastRootWDL : lastNNWDL);
    drawValueChart();
}
function setValueTab(tab) {
    if (tab !== "root" && tab !== "nn") return;
    valueTab = tab;
    const r = document.getElementById("vc_tab_root");
    const nEl = document.getElementById("vc_tab_nn");
    if (r) r.setAttribute("aria-pressed", tab === "root" ? "true" : "false");
    if (nEl) nEl.setAttribute("aria-pressed", tab === "nn" ? "true" : "false");
    renderValuePanel();
}
// Set the current ply's root/nn values (already in Black's frame) + repaint.
function setCurrentValues(rootBlack, nnBlack) {
    lastRootWDL = rootBlack || null;
    lastNNWDL = nnBlack || null;
    renderValuePanel();
}

// --- Candidate move list (right column) ----------------------------------
const candListEl = document.getElementById("cand_list");
let candSig = "";
const MAX_CANDS = 12;
let searchSimsTotal = 0;        // cumulative root visits reported by the worker
const ANALYSIS_CHUNK = 96;      // PUCT sims per continuous-ponder chunk
const ANALYSIS_CAP_MIN = 2000;  // analysis board deepens at least this many root visits
// Per-move thinking time for play mode (AI move + "my-turn" analysis), in ms;
// picked from the toolbar dropdown. Persisted in localStorage("skz_think_ms").
const THINK_MS_OPTIONS = [500, 1000, 2000, 3000, 5000, 10000];
let thinkMs = (function () {
    try {
        const v = parseInt(localStorage.getItem("skz_think_ms"), 10);
        return THINK_MS_OPTIONS.includes(v) ? v : 3000;
    } catch (_) { return 3000; }
})();
function colLetter(c) { return String.fromCharCode(65 + c); }
// Board notation: columns A.. left→right, rows N..1 top→bottom (H8 = center of 15).
function coordLabel(r, c) { return colLetter(c) + (N - r); }
// Rank the side-to-move's candidates from the engine's visit distribution
// (mcts_visits) and per-move win rate (mcts_winrate, side-to-move view ∈ [0,1]).
// Sorted by visits desc; capped at MAX_CANDS.
function computeCandidates() {
    if (!state || !state.board) return [];
    const vis = state.mcts_visits, wrG = state.mcts_winrate;
    if (!vis && !wrG) return [];
    const out = [];
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
        if (state.board[r][c] !== 0) continue;
        const vf = (vis && vis[r]) ? (vis[r][c] || 0) : 0;
        const wRaw = (wrG && wrG[r]) ? wrG[r][c] : null;
        const wr = (wRaw != null && Number.isFinite(wRaw)) ? wRaw : null;
        if (vf > 0 || (wr != null && wr > 0)) out.push({ r, c, vf, wr });
    }
    if (out.length === 0) return [];
    out.sort((a, b) => (b.vf - a.vf) || ((b.wr ?? -1) - (a.wr ?? -1)));
    const maxV = out[0].vf || 0;
    out.forEach((o, i) => { o.frac = maxV > 0 ? o.vf / maxV : 1; o.best = (i === 0); });
    return out.slice(0, MAX_CANDS);
}
// Total root visits → turns a visit fraction back into a count for display,
// using the worker-reported cumulative root visits (0 when search is off).
function totalVisits() {
    return searchSimsTotal > 0 ? searchSimsTotal : 0;
}
function fmtVisits(vf) {
    const tot = totalVisits();
    if (tot > 0) {
        const n = Math.round(vf * tot);
        return n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k" : String(n);
    }
    return Math.round(vf * 100) + "%";
}
// Candidate marker palette (user-selectable in Settings; default violet). Each
// scheme: best fill + stroke, plus a low→high RGB ramp the other moves lerp
// across by visit share. Persisted in localStorage("skz_cand_palette").
const CAND_PALETTES = {
    violet: { best: { fill: "#7c3aed", stroke: "#5b21b6" }, lo: [237, 233, 254], hi: [124, 58, 237] },
    amber:  { best: { fill: "#f59e0b", stroke: "#b45309" }, lo: [253, 230, 138], hi: [217, 119, 6]  },
    blue:   { best: { fill: "#2563eb", stroke: "#1d4ed8" }, lo: [219, 234, 254], hi: [14, 165, 233] },
    teal:   { best: { fill: "#0d9488", stroke: "#115e59" }, lo: [204, 251, 241], hi: [13, 148, 136] },
    rose:   { best: { fill: "#e11d48", stroke: "#9f1239" }, lo: [255, 228, 230], hi: [225, 29, 72]  },
};
const PALETTE_KEYS = ["violet", "amber", "blue", "teal", "rose"];
let candPalette = (function () {
    try {
        const v = localStorage.getItem("skz_cand_palette");
        return PALETTE_KEYS.includes(v) ? v : "violet";
    } catch (_) { return "violet"; }
})();
// White text on dark fills, dark text on light ones (perceived luminance).
function candTextOn(rgb) {
    return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) > 150 ? "#1f2328" : "#ffffff";
}
function hexToRgb(hex) {
    const h = hex.replace("#", "");
    return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16));
}
// Visit-rank color: best move gets the palette's accent (with a darker ring);
// the others lerp the palette's low → high ramp by visit share.
function candColor(frac, best) {
    const pal = CAND_PALETTES[candPalette] || CAND_PALETTES.violet;
    if (best) return { fill: pal.best.fill, stroke: pal.best.stroke, text: candTextOn(hexToRgb(pal.best.fill)) };
    const tt = Math.max(0, Math.min(1, frac));
    const c = pal.lo.map((a, i) => Math.round(a + (pal.hi[i] - a) * tt));
    return { fill: `rgb(${c[0]},${c[1]},${c[2]})`, text: candTextOn(c) };
}
// The centered "开启我的回合分析" button replaces the empty list exactly when the
// engine is idling on your move because per-move analysis is switched off.
function shouldShowAnalyzeButton() {
    return currentMode === "play" && !editMode && !gameOver && !!boardState
        && toPlay === humanSide && !analyzeMyTurn;
}
function renderCandidates() {
    const cands = state ? computeCandidates() : [];
    const showBtn = cands.length === 0 && shouldShowAnalyzeButton();
    // Only rebuild when the data changed, so a row's :hover stays put.
    const sig = (showBtn ? "BTN|" : "") + cands.map(o =>
        o.r + "," + o.c + ":" + Math.round((o.wr ?? -1) * 1000) + ":" + Math.round(o.vf * 1000)).join("|");
    if (sig === candSig) return;
    candSig = sig;
    if (cands.length === 0) {
        candListEl.innerHTML = showBtn
            ? '<button type="button" class="cand-analyze-btn" id="cand_analyze_btn">' +
              '<svg viewBox="0 0 16 16" width="11" height="11" fill="currentColor" aria-hidden="true"><path d="M4.5 3l8 5-8 5z"/></svg>' +
              t("cand_analyze_on") + "</button>"
            : '<div class="cand-empty">' + t("cand_empty") + "</div>";
        return;
    }
    candListEl.innerHTML = cands.map((o, i) => {
        const rank = String.fromCharCode(65 + i);            // A, B, C, …
        const wr = o.wr != null ? Math.round(o.wr * 100) + "%" : t("wdl_dash");
        const barW = o.wr != null ? Math.round(o.wr * 100) : 0;
        return '<div class="cand-row' + (o.best ? " best" : "") +
               '" data-r="' + o.r + '" data-c="' + o.c + '">' +
               '<span class="cand-rank">' + rank + "</span>" +
               '<span class="cand-coord">' + coordLabel(o.r, o.c) + "</span>" +
               '<span class="cand-wr">' + wr + "</span>" +
               '<span class="cand-track"><span style="width:' + barW + '%"></span></span>' +
               '<span class="cand-visits">' + fmtVisits(o.vf) + "</span>" +
               "</div>";
    }).join("");
}
candListEl.addEventListener("mouseover", (ev) => {
    const row = ev.target.closest(".cand-row"); if (!row) return;
    hoverCand = { r: +row.dataset.r, c: +row.dataset.c };
    draw();
});
candListEl.addEventListener("mouseout", (ev) => {
    if (!ev.target.closest(".cand-row")) return;
    if (hoverCand) { hoverCand = null; draw(); }
});
candListEl.addEventListener("click", (ev) => {
    if (ev.target.closest("#cand_analyze_btn")) { setAnalyzeMyTurn(true); return; }
    const row = ev.target.closest(".cand-row"); if (!row) return;
    playCandidate(+row.dataset.r, +row.dataset.c);
});

// --- Model dropdown (loads from models/manifest.json) ---
let manifest = { default: null, models: [] };
let currentModelId = null;

async function loadManifest() {
    const r = await fetch("models/manifest.json", { cache: "no-cache" });
    if (!r.ok) throw new Error("manifest.json fetch failed: " + r.status);
    manifest = await r.json();
    const sel = document.getElementById("model_select");
    sel.innerHTML = "";
    // Sort by ELO ascending so "新手" sits at the top.
    const items = manifest.models.slice().sort((a, b) => a.elo - b.elo);
    for (const m of items) {
        const o = document.createElement("option");
        o.value = m.id;
        const eloStr = (m.elo >= 0 ? "+" : "") + m.elo;
        o.textContent = `${m.id.toUpperCase()} ${m.label} · ELO ${eloStr}`;
        sel.appendChild(o);
    }
    currentModelId = manifest.default || items[0].id;
    sel.value = currentModelId;
}

function modelById(id) {
    return manifest.models.find(m => m.id === id);
}

// Build the model URL with a cache-busting query string derived from the
// model's `params` field (e.g., "b10c128 iter44"). _headers marks .onnx as
// immutable for a year, so without this, replacing a model file leaves
// browsers serving the stale cached blob. Same params → same URL → cache
// hit; new iter → new URL → fresh fetch.
function modelUrl(m) {
    const v = encodeURIComponent(m.params || m.file);
    return `models/${m.file}?v=${v}`;
}

// --- Board size buttons (hard-coded 9-19) ---
const BOARD_SIZES = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19];

// --- Loading overlay (only shown for first model load) ---
function fmtMB(bytes) { return (Number(bytes || 0) / 1048576).toFixed(1); }
function setLoadingProgress(pct, loaded, total) {
    const fill = document.getElementById("loading_fill");
    const text = document.getElementById("loading_pct");
    if (!fill || !text) return;
    if (Number.isFinite(pct)) {
        fill.style.width = Math.max(0, Math.min(100, pct)) + "%";
        let s = Math.round(pct) + "%";
        if (total) s += "  ·  " + fmtMB(loaded) + "/" + fmtMB(total) + " MB";
        else if (loaded) s += "  ·  " + fmtMB(loaded) + " MB";
        text.textContent = s;
    }
}
function hideLoadingOverlay() {
    const o = document.getElementById("loading_overlay");
    if (o) o.style.display = "none";
    lastLoadingMsg = null;
}
let lastLoadingMsg = null; // { key, args }
function showLoadingOverlay(key, ...args) {
    const o = document.getElementById("loading_overlay");
    if (!o) return;
    o.style.display = "";
    lastLoadingMsg = { key, args };
    const tEl = document.getElementById("loading_text");
    if (tEl) tEl.textContent = t(key, ...args);
    setLoadingProgress(0);
}
function rerenderLoadingOverlay() {
    if (!lastLoadingMsg) return;
    const tEl = document.getElementById("loading_text");
    if (tEl) tEl.textContent = t(lastLoadingMsg.key, ...lastLoadingMsg.args);
}

// --- Game state (mirrors V5 gomoku semantics, in-browser) ---
let game = null;          // Gomoku instance (rebuilt on size change)
let boardState = null;    // Int8Array(N*N), +1 / -1 / 0
let toPlay = 1;           // 1 = black, -1 = white
let humanSide = 1;        // 1 = human is black; -1 = human is white
let currentRule = "freestyle"; // "renju" | "freestyle"
let lastMove = null;      // { r, c }
let ply = 0;              // half-move counter
let gameOver = false;
let aiThinking = false;
let searchId = 0;
let history = [];         // [{ board: Int8Array, toPlay, lastMove, ply, gumbelPhases }]

// Position-edit mode (manual setup). When active, board clicks place / erase
// stones according to editTool instead of playing a move; the AI is paused
// and a snapshot is held so cancel can fully revert.
let editMode = false;
let editTool = "alternate";   // "alternate" | "black" | "white" | "erase"
let editSnapshot = null;
let editUndoStack = [];       // stack of { idx, prev } for in-edit undo
const EDIT_TOOLS = ["alternate", "black", "white", "erase"];

function refreshEditUndoBtn() {
    const btn = document.getElementById("edit_undo_btn");
    if (btn) btn.disabled = editUndoStack.length === 0;
}

const worker = new Worker("worker.js?v=" + Date.now());
worker.onerror = (e) => {
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
        board: board1Dto2D(boardState),
        last_move: lastMove ? [lastMove.r, lastMove.c] : null,
        board_size: N,
        // Heat data (set by handleResult; null otherwise — drawHeat handles nulls).
        mcts_policy:    extras.mcts_policy    || null,
        mcts_visits:    extras.mcts_visits    || null,
        mcts_winrate:   extras.mcts_winrate   || null,
        nn_policy:      extras.nn_policy      || null,
        nn_opp_policy:  extras.nn_opp_policy  || null,
        nn_futurepos_8: extras.nn_futurepos_8 || null,
        nn_futurepos_32:extras.nn_futurepos_32|| null,
    };
}

function repaintAllHeatmaps() {
    drawHeat("h_mcts_policy",    state ? state.mcts_policy    : null);
    drawHeat("h_mcts_visits",    state ? state.mcts_visits    : null);
    drawHeat("h_nn_policy",      state ? state.nn_policy      : null);
    drawHeat("h_nn_opp_policy",  state ? state.nn_opp_policy  : null);
    drawHeatSigned("h_nn_futurepos_8",  state ? state.nn_futurepos_8  : null);
    drawHeatSigned("h_nn_futurepos_32", state ? state.nn_futurepos_32 : null);
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
function applyLiveCandidates(visitsFlat, winrateFlat, rootVisits) {
    if (!state) return;
    state.mcts_visits  = flatToGrid(visitsFlat);
    state.mcts_winrate = winrateFlat ? flatToGrid(winrateFlat) : state.mcts_winrate;
    if (rootVisits > 0) searchSimsTotal = rootVisits;
    renderCandidates();
    draw();
}

function newGame() {
    game = new Gomoku(N, currentRule);
    boardState = game.getInitialState();
    toPlay = 1;
    lastMove = null;
    ply = 0;
    gameOver = false;
    history = [];
    valueHistory = [];
    gumbelPhases = null;
    setCurrentValues(null, null);
    candSig = "";
    searchSimsTotal = 0;
    publishStateForDrawing();
    drawAll();
    worker.postMessage({ type: "reset", boardSize: N, ply: 0, rule: currentRule });
    // The engine ponders the human's opening (or analysis board); on the AI's
    // opening (human plays white) it searches and plays.
    triggerAISearch();
}

// True when the engine should *analyze* (ponder) the side-to-move rather than
// pick a move for it: always on the analysis board, and on the human's turn in
// play mode *when "analyze my move" is on* — so the player sees live analysis of
// their own position. Only the AI's turn in play mode is a move-search.
function isPonderTurn() {
    if (gameOver || !boardState) return false;
    if (currentMode === "analysis") return true;
    return toPlay === humanSide && analyzeMyTurn;
}

function triggerAISearch() {
    if (gameOver) return;
    // Play mode, your turn, per-move analysis off: don't ponder and don't move
    // for you — just wait. The candidate area shows the "开启我的回合分析" button.
    if (currentMode !== "analysis" && toPlay === humanSide && !analyzeMyTurn) {
        aiThinking = false;
        setStatus("status_your_turn", "active");
        renderCandidates();
        return;
    }
    aiThinking = true;
    searchId++;
    const ponder = isPonderTurn();
    if (ponder && currentMode === "analysis") setStatus("status_analyzing", "thinking");
    else if (ponder) setStatus("status_your_turn", "active");   // play mode: human's turn, analysis runs quietly
    else setStatus("status_ai_thinking", "thinking");
    let sims = 0, timeMs = 0;
    if (currentMode === "analysis") {
        sims = ANALYSIS_CHUNK;   // analysis board deepens in fixed-size PUCT chunks
    } else {
        // Play mode: the AI's move-search AND the "my-turn" analysis both run
        // anytime PUCT for the configured thinking time.
        timeMs = thinkMs;
    }
    worker.postMessage({
        type: "search",
        state: boardState,
        toPlay: toPlay,
        ply: ply,
        sims: sims,
        timeMs: timeMs,
        gumbel_m: 16,
        searchId: searchId,
        // timeMs > 0 → anytime PUCT (play mode); else fixed-sims PUCT (analysis board).
        analyze: ponder,
    });
}

function applyMoveLocal(action) {
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
    const movedBy = toPlay;
    toPlay = -toPlay;
    ply++;
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
        return;
    }
    if (data.type === "ready") {
        hideLoadingOverlay();
        // First-ready means model is loaded; if it's a swap, just resume.
        if (!boardState) newGame();
        return;
    }
    if (data.type === "error") {
        setStatusRaw(t("err_prefix", data.message), "error");
        aiThinking = false;
        return;
    }
    if (data.type === "progress") {
        if (data.searchId !== searchId) return;
        // Mid-search snapshot: refresh the candidate list / board overlay live as
        // the search deepens — including the AI's move-search in play mode, so you
        // see per-point win% / visits as it thinks.
        if (data.mctsVisits) applyLiveCandidates(data.mctsVisits, data.mctsWinrate, data.searchSims);
        return;
    }
    if (data.type === "result") {
        if (data.searchId !== searchId) return;
        aiThinking = false;
        searchSimsTotal = data.searchSims || 0;
        // Update gumbel overlay + heatmaps + candidate data.
        gumbelPhases = data.gumbelPhases;
        publishStateForDrawing({
            mcts_policy:    flatToGrid(data.mctsPolicy),
            mcts_visits:    flatToGrid(data.mctsVisits),
            mcts_winrate:   flatToGrid(data.mctsWinrate),
            nn_policy:      flatToGrid(data.nnPolicy),
            nn_opp_policy:  flatToGrid(data.nnOppPolicy),
            nn_futurepos_8: flatToGrid(data.nnFuturepos8),
            nn_futurepos_32:flatToGrid(data.nnFuturepos32),
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
            if (currentMode === "analysis") {
                // Analysis mode keeps deepening ("一直算") by reusing the tree, up to a
                // fixed memory-bounded depth (ANALYSIS_CAP_MIN root visits).
                const cap = ANALYSIS_CAP_MIN;
                const ponderOn = !gameOver && searchSimsTotal < cap;
                setStatus(ponderOn ? "status_analyzing" : "status_analysis_ready",
                          ponderOn ? "thinking" : "info");
                if (ponderOn) triggerAISearch();   // next chunk goes deeper (tree reuse)
            } else {
                // Play mode, human's turn: a single time-budgeted analysis (no
                // continuous deepening) — it already ran for the full thinking time.
                setStatus("status_your_turn", "active");
            }
            return;
        }
        // Play mode, AI's turn: play the chosen move, then ponder the human's reply.
        const { winner } = applyMoveLocal(data.gumbelAction);
        if (winner === null) triggerAISearch();
    }
};
(function initAnalyzeToggle() {
    const cb = document.getElementById("analyze_turn_input");
    if (!cb) return;
    cb.checked = analyzeMyTurn;
    cb.addEventListener("change", () => setAnalyzeMyTurn(cb.checked));
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
// Thinking-time dropdown (toolbar): apply the saved value, persist on change.
(function initThinkTime() {
    const sel = document.getElementById("think_time_select");
    if (!sel) return;
    sel.value = String(thinkMs);
    sel.addEventListener("change", () => {
        const v = parseInt(sel.value, 10);
        if (!THINK_MS_OPTIONS.includes(v)) return;
        thinkMs = v;
        try { localStorage.setItem("skz_think_ms", String(v)); } catch (_) {}
    });
})();

// --- Buttons ---
document.getElementById("new_btn").addEventListener("click", newGame);

document.getElementById("undo_btn").addEventListener("click", () => {
    if (history.length === 0) return;
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
    aiThinking = false;
    searchId++;   // abort any in-flight search
    while (valueHistory.length && valueHistory[valueHistory.length - 1].step > stoneCount(board1Dto2D(boardState))) {
        valueHistory.pop();
    }
    setCurrentValues(restoredRootWDL, restoredNNWDL);
    candSig = "";
    searchSimsTotal = 0;
    publishStateForDrawing();
    drawAll();
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
    // Close the settings popover the edit button lives in.
    const pop = document.getElementById("settings_pop");
    if (pop) pop.classList.add("hidden");
    document.getElementById("settings_btn")?.setAttribute("aria-expanded", "false");
    editMode = true;
    editSnapshot = snapshotForEdit();
    editUndoStack = [];
    refreshEditUndoBtn();
    searchId++;            // abort any in-flight search
    aiThinking = false;
    gameOver = false;      // edits are unrestricted; we recheck on commit
    gumbelPhases = null;   // hide gumbel overlay during setup
    document.body.classList.add("editing");
    setEditTool(editTool);
    setStatus("status_editing", "info");
    publishStateForDrawing();
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
        gumbelPhases = s.gumbelPhases;
        history = s.history.slice();
        valueHistory = s.valueHistory.slice();
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
    gumbelPhases = null;
    setCurrentValues(null, null);
    candSig = "";
    gameOver = false;

    // If the setup is already a finished position (5-in-a-row), surface that.
    // We pass null for lastAction so the renju forbidden-detect path is
    // skipped — there is no "last move" in a setup.
    const winner = game.getWinner(boardState, null, null);
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
    if (rl !== "renju" && rl !== "freestyle") return;
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

// Board size slider (input range 9..19). `input` updates the display label
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
    aiThinking = false;
    searchId++;   // abort any in-flight search
    history = [];   // old snapshots are at the previous size — undo can't replay across sizes
    // publish before syncBoardSize: if the canvas resizes, syncBoardSize calls
    // draw(), which would otherwise iterate 0..N over a stale state.board.
    publishStateForDrawing();
    syncBoardSize();
    drawAll();
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

// Model dropdown.
document.getElementById("model_select").addEventListener("change", (ev) => {
    const id = ev.target.value;
    const m = modelById(id);
    if (!m) return;
    currentModelId = id;
    showLoadingOverlay("loading_model", m.label);
    searchId++;
    aiThinking = false;
    worker.postMessage({ type: "swap-model", modelUrl: modelUrl(m) });
});

// Re-render dynamic strings (status, loading overlay, modal title,
// value-chart 'no data') when the user switches language.
registerI18nCallback(() => {
    rerenderStatus();
    rerenderLoadingOverlay();
    if (expandedSourceId) {
        const card = document.getElementById(expandedSourceId).parentElement;
        const titleEl = card.querySelector(".grid-title-text");
        document.getElementById("heat_modal_title").textContent =
            titleEl ? titleEl.textContent : t("heatmap_default_title");
    }
    renderValuePanel();
    candSig = ""; renderCandidates();
    renderSizeConfirmBody();
});

// --- Settings popover (model / rule / size / palette / setup) ---
(function initSettingsPopover() {
    const btn = document.getElementById("settings_btn");
    const pop = document.getElementById("settings_pop");
    if (!btn || !pop) return;
    function setOpen(open) {
        pop.classList.toggle("hidden", !open);
        btn.setAttribute("aria-expanded", open ? "true" : "false");
    }
    btn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        setOpen(pop.classList.contains("hidden"));
    });
    pop.addEventListener("click", (ev) => ev.stopPropagation());
    document.addEventListener("click", () => setOpen(false));
    document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") setOpen(false); });
})();

// On narrow viewports the Settings control moves out of the toolbar into a
// bottom bar (keeping the top tidy); on wide ones it returns to the toolbar.
// The single .settings-wrap node is relocated, so the popover wiring above and
// its element IDs stay intact.
(function initSettingsPlacement() {
    const wrap = document.querySelector(".settings-wrap");
    const topHome = document.querySelector(".topbar-right");
    const bottomHome = document.getElementById("mobile_actionbar");
    if (!wrap || !topHome || !bottomHome) return;
    const mq = window.matchMedia("(max-width: 720px)");
    function place() {
        const dest = mq.matches ? bottomHome : topHome;
        if (wrap.parentElement !== dest) dest.appendChild(wrap);
    }
    place();
    if (mq.addEventListener) mq.addEventListener("change", place);
    else if (mq.addListener) mq.addListener(place);
})();

// --- Heatmap drawer (collapsed by default; canvases measure 0 while hidden) ---
(function initHeatDrawer() {
    const btn = document.getElementById("heat_drawer_btn");
    const body = document.getElementById("heat_drawer_body");
    if (!btn || !body) return;
    btn.addEventListener("click", () => {
        const open = body.classList.toggle("hidden") === false;
        btn.setAttribute("aria-expanded", open ? "true" : "false");
        if (open) {
            for (const id of Object.keys(heatCtxs)) {
                fitHeatCanvas(id);
                drawHeatById(id, state ? state[HEAT_GRID_KEYS[id]] : null);
            }
        }
    });
})();

// --- "Analyze in search tree" button: hand the live position to mcts-tree.html ---
// Stashes the current board / side / settings in localStorage (a one-shot key the
// tree page reads and clears), then navigates there to seed and auto-run the search.
(function initTreeAnalyzeButton() {
    const btn = document.getElementById("tree_analyze_btn");
    if (!btn) return;
    btn.addEventListener("click", () => {
        if (!game || !boardState) return;
        const size = game.boardSize;
        const md = modelById(currentModelId);
        const payload = {
            board: Array.from(boardState),                 // Int8Array N*N, +1/-1/0, row-major
            toPlay,
            size,
            rule: currentRule,
            model: md ? md.file : null,                    // tree page keys its model dropdown on file
            lastMove: lastMove ? (lastMove.r * size + lastMove.c) : null,
        };
        try { localStorage.setItem("skz_tree_pos", JSON.stringify(payload)); } catch (_) {}
        location.href = "mcts-tree.html";
    });
})();

// --- Bootstrap on load ---
(async function bootstrap() {
    showLoadingOverlay("loading_manifest");
    try {
        await loadManifest();
    } catch (err) {
        setStatusRaw(t("err_manifest_load", err.message), "error");
        return;
    }
    const startModel = modelById(currentModelId) || manifest.models[0];
    if (!startModel) {
        setStatus("err_manifest_empty", "error");
        return;
    }
    showLoadingOverlay("loading_model", startModel.label);
    worker.postMessage({
        type: "init",
        modelUrl: modelUrl(startModel),
        boardSize: N,
        rule: currentRule,
    });
    syncBoardSize();
})();
