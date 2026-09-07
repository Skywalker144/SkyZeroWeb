// Board geometry — N is mutable (board size dropdown can change it).
let N = 15;
const MARGIN_DESKTOP = 28;   // leaves room for top + left coordinate labels
const MARGIN_COMPACT = 10;   // mobile / portrait — labels hidden, just enough for stones
const MARGIN_PHONE = 8;      // reclaim phone width while keeping edge stones clear
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
// index.html applies body.mode-analysis before first paint.
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
    candSig = "";              // force the candidate list to rebuild for the new mode
    if (!gameOver) triggerAISearch();   // ponder / play for the new mode
    renderCandidates();        // settle the list for the new mode
}

// Candidate overlays in play mode have independent visibility for the human
// turn and the AI's own search. Analysis mode always shows them. Migrate the
// former single human-turn toggle while keeping the new defaults: human hidden,
// AI shown.
let showAnalysisOnHumanTurn = (function () {
    try {
        const saved = localStorage.getItem("skz_show_analysis_human");
        const legacy = localStorage.getItem("skz_show_analysis_board");
        return (saved ?? legacy) === "1";
    } catch (_) { return false; }
})();
let showAnalysisOnAiTurn = (function () {
    try { return localStorage.getItem("skz_show_analysis_ai") !== "0"; }
    catch (_) { return true; }
})();

document.addEventListener("DOMContentLoaded", () => {
    updateModeButtons();
    const p = document.getElementById("mode_play");
    const a = document.getElementById("mode_analysis");
    if (p) p.addEventListener("click", () => setMode("play"));
    if (a) a.addEventListener("click", () => setMode("analysis"));
    // Enable the mobile #play_row fold transition only after the first paint, so
    // the play-only row doesn't animate in on initial load — only on a later
    // mode toggle (mirrors seg-slide's -ready gate).
    requestAnimationFrame(() => requestAnimationFrame(() =>
        document.body.classList.add("tb-anim-ready")));
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
    const phone = window.matchMedia("(max-width: 720px)").matches;
    MARGIN = phone ? MARGIN_PHONE : (compact ? MARGIN_COMPACT : MARGIN_DESKTOP);

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
    const reviewTb = document.getElementById("review_bar");
    const reviewH = (reviewTb && !reviewTb.classList.contains("hidden"))
        ? reviewTb.offsetHeight + colGap : 0;
    const reserved = appPadY + topbarH + cardPadY + cardBorderY + editH + reviewH + 12;
    const availH = window.innerHeight - reserved;

    const cap = compact ? 560 : 900;
    const size = Math.max(compact ? 220 : 300, Math.min(availW, availH, cap));
    const minCell = compact ? 12 : 18;
    CELL = Math.max(minCell, Math.floor((size - 2 * MARGIN) / (N - 1)));
    // Integer cell widths can leave several unused pixels on wider phones
    // (notably 430px iPhones). Absorb that remainder into the canvas margin so
    // the canvas fills the card content box and all four CSS insets match.
    if (phone) MARGIN = (size - CELL * (N - 1)) / 2;
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

let state = null;        // { board: 2D N×N int, last_move: [r,c]|null, board_size: N }
let showGumbel = false;
let gumbelPhases = null; // last search's gumbel phases [[r,c]...] per phase
let boardOverlayHeatId = null; // heat canvas id currently mirrored on the board
let hoverCand = null;          // {r,c} candidate row under the pointer → board ring
let hoverCell = null;          // {r,c} empty intersection under the pointer → ghost stone
let winLine = null;            // [[r,c]...] winning run to highlight, or null
let reviewIndex = null;        // null = live tip; else a history index being reviewed (read-only)
let chartHover = null;         // win-rate chart: hovered ply step for the crosshair/tooltip

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
            ctx.fillText(colLetter(i), MARGIN + i * CELL, 12);   // top axis: columns A,B,C…
            ctx.fillText(rowNumber(i), 10, MARGIN + i * CELL);   // left axis: rows 1…N
        }
    }
    if (!state) return;

    // When reviewing a past ply, render that snapshot read-only; the live game,
    // worker, candidate list and heatmaps are untouched. board1Dto2D + the {r,c}→
    // [r,c] last-move shim keep the snapshot in the shape draw() expects.
    const reviewing = reviewIndex !== null && reviewIndex < history.length;
    const dispBoard = reviewing ? board1Dto2D(history[reviewIndex].board) : state.board;
    const rl = reviewing ? history[reviewIndex].lastMove : null;
    const dispLast = reviewing ? (rl ? [rl.r, rl.c] : null) : state.last_move;

    const stoneR    = Math.max(6, Math.round(CELL * 0.39));
    const gumbelR   = Math.max(6, Math.round(CELL * 0.34));
    const lastDotR  = Math.max(2, Math.round(CELL * 0.11));
    const shadowDx  = Math.max(0, Math.round(CELL * 0.015));
    const shadowDy  = Math.max(1, Math.round(CELL * 0.045));
    const gradInner = Math.max(1, Math.round(CELL * 0.11));
    const gumbelFontPx = Math.max(8, Math.round(CELL * 0.28));

    if (showGumbel && gumbelPhases && gumbelPhases.length > 0 && !reviewing) {
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

    const b = dispBoard, lm = dispLast;
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

    // Reviewing a past ply is read-only: just the position, no live overlays,
    // ghost or win line (those belong to the current position only).
    if (reviewing) return;

    // Candidate-disc overlay rules. A pinned heatmap is an explicit opt-in and
    // overrides everything. Otherwise: the analysis board always shows discs; in
    // play mode uses independent settings for the AI's own move-search and the
    // human-turn ponder. The ponder still runs silently for the left-column
    // stats and right-side list when its board overlay is hidden.
    const showOverlay = currentMode === "analysis"
        || (toPlay !== humanSide
            ? showAnalysisOnAiTurn
            : showAnalysisOnHumanTurn);
    if (boardOverlayHeatId) drawBoardHeatOverlay(boardOverlayHeatId);
    else if (showOverlay) drawCandidateOverlay();
    else drawCandidateHover(computeCandidates());   // your turn, overlay off: keep the list's hover-peek
    drawGhostStone();
    if (winLine) drawWinLine();   // on a finished game, drawn last so it sits over the stones
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
    drawCandidateHover(cands);
}

// The accent ring previewing the candidate row hovered in the right-hand list.
// Split out of the overlay so it still works when the disc overlay is hidden —
// the list stays linked to the board even with "show on board" off.
function drawCandidateHover(cands) {
    if (!hoverCand || !cands.some(o => o.r === hoverCand.r && o.c === hoverCand.c)) return;
    const candR = Math.max(7, Math.round(CELL * 0.40));
    const x = MARGIN + hoverCand.c * CELL, y = MARGIN + hoverCand.r * CELL;
    ctx.beginPath(); ctx.arc(x, y, candR + 2, 0, Math.PI * 2);
    ctx.lineWidth = 2.5; ctx.strokeStyle = cssVar("--accent") || "#0969da";
    ctx.stroke(); ctx.lineWidth = 1;
}

// Translucent preview of the stone that would land on the hovered empty cell,
// plus its coordinate, so dense boards are easier to aim on. Only on a turn you
// can actually place: analysis (side-to-move) or play mode on the human's turn.
function drawGhostStone() {
    if (!hoverCell || !state || gameOver || editMode) return;
    const { r, c } = hoverCell;
    if (r < 0 || r >= N || c < 0 || c >= N || state.board[r][c] !== 0) return;
    let side;
    if (currentMode === "analysis") side = toPlay;
    else { if (toPlay !== humanSide) return; side = humanSide; }
    const x = MARGIN + c * CELL, y = MARGIN + r * CELL;
    const stoneR = Math.max(6, Math.round(CELL * 0.39));
    const gradInner = Math.max(1, Math.round(CELL * 0.11));
    const g0 = side === 1 ? cssVar("--stone-black-0") : cssVar("--stone-white-0");
    const g1 = side === 1 ? cssVar("--stone-black-1") : cssVar("--stone-white-1");
    ctx.save();
    ctx.globalAlpha = 0.42;
    ctx.beginPath(); ctx.arc(x, y, stoneR, 0, Math.PI * 2);
    const grad = ctx.createRadialGradient(x - gradInner, y - gradInner, 2, x, y, stoneR);
    grad.addColorStop(0, g0); grad.addColorStop(1, g1);
    ctx.fillStyle = grad; ctx.fill();
    ctx.globalAlpha = 0.7;
    ctx.lineWidth = 1; ctx.strokeStyle = cssVar("--stone-outline"); ctx.stroke();
    // Coordinate readout just above the ghost (flips below for the top row).
    const fontPx = Math.max(9, Math.round(CELL * 0.30));
    const above = y - stoneR - 3 >= fontPx;
    ctx.globalAlpha = 0.92;
    ctx.fillStyle = cssVar("--accent") || "#0969da";
    ctx.font = `bold ${fontPx}px ${MONO_FONT}`;
    ctx.textAlign = "center"; ctx.textBaseline = above ? "bottom" : "top";
    ctx.fillText(coordLabel(r, c), x, above ? y - stoneR - 3 : y + stoneR + 3);
    ctx.restore();
}

// Highlight the winning five (or longer run) with a gold line through the run's
// endpoints — amber reads on black and white stones in both themes.
function drawWinLine() {
    if (!winLine || winLine.length < 2) return;
    const a = winLine[0], z = winLine[winLine.length - 1];
    const x1 = MARGIN + a[1] * CELL, y1 = MARGIN + a[0] * CELL;
    const x2 = MARGIN + z[1] * CELL, y2 = MARGIN + z[0] * CELL;
    ctx.save();
    ctx.lineCap = "round";
    ctx.strokeStyle = "rgba(251,191,36,0.35)";
    ctx.lineWidth = Math.max(8, Math.round(CELL * 0.42));
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = Math.max(2.5, Math.round(CELL * 0.10));
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.restore();
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

// --- Four heatmap canvases ---
const heatCtxs = {
    h_nn_policy:             setupCanvas(document.getElementById("h_nn_policy"),             HEAT_LOGICAL, HEAT_LOGICAL, false),
    h_nn_optimistic_policy:  setupCanvas(document.getElementById("h_nn_optimistic_policy"),  HEAT_LOGICAL, HEAT_LOGICAL, false),
    h_nn_opp_policy:         setupCanvas(document.getElementById("h_nn_opp_policy"),         HEAT_LOGICAL, HEAT_LOGICAL, false),
    h_mcts_play_selection:   setupCanvas(document.getElementById("h_mcts_play_selection"),   HEAT_LOGICAL, HEAT_LOGICAL, false),
};
const HEAT_GRID_KEYS = {
    h_nn_policy:             "nn_policy",
    h_nn_optimistic_policy:  "nn_optimistic_policy",
    h_nn_opp_policy:         "nn_opp_policy",
    h_mcts_play_selection:   "mcts_play_selection",
};
const SIGNED_HEAT_IDS = new Set();

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
    // The heatmap fills the card's content width as a square; the card height grows
    // to fit it (no fixed card height), so there's no leftover gap around the image.
    const size = Math.max(60, Math.floor(card.clientWidth - padX));
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
// Sizing is handled by CSS (#value_chart in ../style.css) so that mobile media
// queries can shrink the canvas — JS just keeps the bitmap in sync via the
// ResizeObserver below.
const vcCanvas = document.getElementById("value_chart");
let vctx = setupCanvas(vcCanvas, 280, 160, false);
// Chart inner padding, shared by drawValueChart and the hover/click hit-testing
// below so the x↔ply mapping can't drift between drawing and picking.
const VC_PAD = { l: 26, r: 6, t: 6, b: 14 };
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

// Hover the chart to read the exact WDL at any ply (crosshair + tooltip); click
// to jump the board to that ply (read-only review). Uses the same VC_PAD geometry
// as drawValueChart so the picked point matches what's drawn.
function chartStepAtX(clientX) {
    const pts = valueHistory.filter(p => p[valueTab]);
    if (!pts.length) return null;
    const rect = vcCanvas.getBoundingClientRect();
    if (rect.width <= 0) return null;
    const lx = (clientX - rect.left) * (vctx._logicalW / rect.width);
    const innerW = vctx._logicalW - VC_PAD.l - VC_PAD.r;
    const n = pts.length;
    const frac = n <= 1 ? 0 : Math.max(0, Math.min(1, (lx - VC_PAD.l) / innerW));
    return pts[Math.round(frac * (n - 1))].step;
}
vcCanvas.style.cursor = "crosshair";
vcCanvas.addEventListener("mousemove", (ev) => {
    const step = chartStepAtX(ev.clientX);
    if (step === chartHover) return;
    chartHover = step;
    drawValueChart();
});
vcCanvas.addEventListener("mouseleave", () => {
    if (chartHover !== null) { chartHover = null; drawValueChart(); }
});
vcCanvas.addEventListener("click", (ev) => {
    const step = chartStepAtX(ev.clientX);
    if (step === null) return;
    setReview(navIndexForStep(step));   // null → live tip; else the matching ply
});

// valueHistory keeps one Black-frame WDL per ply for each evaluation:
//   [{step, root:{w,d,l}|null, nn:{w,d,l}|null}]   w = Black win, l = White win.
let valueHistory = [];
let valueTab = "root";    // which evaluation the chart shows: "root" | "nn"
let lastRootWDL = null;   // current ply's root/nn value, already in Black's frame
let lastNNWDL = null;

// Per-move wall-clock for the "avg time / move" stat. moveTimes mirrors the ply
// timeline ([{step, ms, movedBy}]); applyMoveLocal appends and undo trims it,
// just like valueHistory. lastMoveAt is the timestamp the previous ply landed,
// i.e. the moment the side-to-move's clock started.
let moveTimes = [];
let lastMoveAt = 0;

function stoneCount(board2d) {
    let n = 0;
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) if (board2d[r][c]) n++;
    return n;
}
function stoneCount1D(b1d) {
    let n = 0;
    for (let i = 0; i < b1d.length; i++) if (b1d[i]) n++;
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
    renderGameStats();   // keep the stats panel in lock-step (runs even on no-data)
    clearLogical(vctx);
    const W = vctx._logicalW, H = vctx._logicalH;
    const padL = VC_PAD.l, padR = VC_PAD.r, padT = VC_PAD.t, padB = VC_PAD.b;
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

    // With a single eval there's no curve yet — render it as a flat full-width
    // band (the point duplicated at both edges) so the chart appears on the very
    // first point (the opening eval, or at the latest the user's first move)
    // instead of staying blank until a second one lands.
    const rpts = n === 1 ? [pts[0], pts[0]] : pts;
    const rx   = n === 1 ? [padL, W - padR] : pts.map((_, i) => xAt(i));
    // Fill one stacked band between two cumulative-fraction accessors.
    function band(lowerFn, upperFn, color) {
        vctx.fillStyle = color;
        vctx.beginPath();
        rpts.forEach((p, i) => { const x = rx[i], y = yOf(upperFn(p)); i === 0 ? vctx.moveTo(x, y) : vctx.lineTo(x, y); });
        for (let i = rpts.length - 1; i >= 0; i--) vctx.lineTo(rx[i], yOf(lowerFn(rpts[i])));
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
        rpts.forEach((p, i) => { const x = rx[i], y = yOf(fn(p)); i === 0 ? vctx.moveTo(x, y) : vctx.lineTo(x, y); });
        vctx.stroke();
    }
    vctx.strokeStyle = muted; vctx.lineWidth = 1.5;
    line(p => p.l);
    line(p => p.l + p.d);

    // Overlays: a persistent cursor at the reviewed ply + a transient hover
    // crosshair/tooltip. Both map a ply step → x via the plotted points.
    const xForStep = (step) => {
        const j = pts.findIndex(p => p.step === step);
        return j < 0 ? null : (n === 1 ? rx[0] : xAt(j));
    };
    const markStep = (reviewIndex !== null && reviewIndex < history.length)
        ? stoneCount1D(history[reviewIndex].board) : null;
    if (markStep !== null) {
        const x = xForStep(markStep);
        if (x !== null) {
            vctx.strokeStyle = cssVar("--accent") || "#0969da"; vctx.lineWidth = 1.5;
            vctx.beginPath(); vctx.moveTo(x + 0.5, padT); vctx.lineTo(x + 0.5, H - padB); vctx.stroke();
        }
    }
    if (chartHover != null) {
        const x = xForStep(chartHover);
        const p = pts.find(pp => pp.step === chartHover);
        if (x !== null && p) {
            vctx.strokeStyle = subtle; vctx.lineWidth = 1;
            vctx.beginPath(); vctx.moveTo(x + 0.5, padT); vctx.lineTo(x + 0.5, H - padB); vctx.stroke();
            const txt = `#${p.step}  B${Math.round(p.w * 100)} W${Math.round(p.l * 100)}`;
            vctx.font = `10px ${MONO_FONT}`;
            const tw = vctx.measureText(txt).width + 10;
            let bx = x + 6;
            if (bx + tw > W - padR) bx = x - 6 - tw;
            bx = Math.max(padL, Math.min(bx, W - padR - tw));
            const by = padT + 1, bh = 15;
            vctx.fillStyle = cssVar("--surface") || "#fff";
            vctx.fillRect(bx, by, tw, bh);
            vctx.strokeStyle = cssVar("--border") || "#d8dee4";
            vctx.strokeRect(bx + 0.5, by + 0.5, tw, bh);
            vctx.fillStyle = cssVar("--fg") || "#1f2328";
            vctx.textAlign = "left"; vctx.textBaseline = "middle";
            vctx.fillText(txt, bx + 5, by + bh / 2 + 0.5);
        }
    }
}

// --- Per-game stats panel (left column, below the chart) -----------------
// All numbers derive from valueHistory (active tab, Black's frame), the live
// board, and moveTimes. Skill / worst-move / avg-time need a defined human, so
// they are play-mode only; skill additionally persists across games (see
// humanMoveLosses / skillWindow), the rest describe only the current game.
function fmtDuration(ms) {
    if (ms == null) return t("wdl_dash");
    if (ms < 60000) return (ms / 1000).toFixed(1) + "s";
    return Math.floor(ms / 60000) + "m" + Math.round((ms % 60000) / 1000) + "s";
}

// Cross-game skill window: a rolling list of recent per-move win% losses kept in
// localStorage, so 每手失分 reflects accumulated play strength rather than a
// single high-variance game. Capped at the most recent SKILL_WINDOW moves.
const SKILL_KEY = "skz_skill_losses";
const SKILL_WINDOW = 200;
function skillWindow() {
    try {
        const v = JSON.parse(localStorage.getItem(SKILL_KEY));
        return Array.isArray(v) ? v.filter(x => typeof x === "number" && isFinite(x)) : [];
    } catch (_) { return []; }
}
function commitSkillLosses(drops) {
    if (!drops || !drops.length) return;
    const w = skillWindow().concat(drops);
    try { localStorage.setItem(SKILL_KEY, JSON.stringify(w.slice(Math.max(0, w.length - SKILL_WINDOW)))); } catch (_) {}
}
// This game's per-human-move win% losses (play mode only): for each human move
// at step k, how much win% it conceded measured before→after the move — the
// human's win% on their own turn (k-1, their ponder) minus the eval of the
// position right after they moved (k, the AI's deep reply-search). That post-move
// eval is exactly the swing you watch live when the AI re-evaluates your move, so
// the worst-move stat tracks it. Needs the pre-move ponder (k-1); a move whose
// step-k eval never landed (e.g. a game-winning move with no AI reply) is absent
// from `pts` and thus skipped. Negative deltas (apparent gains) clamp to 0. Empty
// on the analysis board (no defined human).
function humanMoveLosses() {
    if (currentMode === "analysis") return [];
    const pts = valueHistory.filter(p => p[valueTab]).map(p => ({ step: p.step, ...p[valueTab] }));
    if (!pts.length) return [];
    const winOf = (p) => (humanSide === 1 ? p.w : p.l) * 100;
    const winAt = new Map(pts.map(p => [p.step, winOf(p)]));
    const out = [];
    for (const p of pts) {
        const k = p.step;
        if ((k % 2 === 1) !== (humanSide === 1)) continue;   // not the human's move
        if (!winAt.has(k - 1)) continue;                     // no pre-move eval to compare
        out.push({ step: k, drop: Math.max(0, winAt.get(k - 1) - winAt.get(k)) });
    }
    return out;
}
function renderGameStats() {
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    const isAnalysis = currentMode === "analysis";
    // Whose win% the chart-derived stats describe: the human in play, else Black.
    const side = isAnalysis ? 1 : humanSide;

    // Move count tracks the live board so it ticks up the instant a stone lands,
    // before that ply's eval arrives.
    let moves = 0;
    if (state && state.board) for (const row of state.board) for (const v of row) if (v) moves++;
    set("gs_moves", String(moves));

    // Perspective tag: a stone dot + "我方" (play) / "黑方" (analysis).
    const persp = document.getElementById("stats_persp");
    if (persp) {
        const cls = side === 1 ? "blk" : "wht";
        const label = isAnalysis ? t("stats_persp_black") : t("stats_persp_me");
        persp.innerHTML = `<span class="sw ${cls}"></span>${label}`;
    }

    // Per-human-move win% losses for this game (play mode only); feeds both the
    // cross-game skill and the worst-move stat — see humanMoveLosses().
    const losses = humanMoveLosses();

    // 每手失分 (棋力): the rolling mean win% lost per move over a window of recent
    // human moves persisted across games (localStorage) plus this game's moves
    // shown live, so it reads as accumulated strength rather than one noisy game.
    // Lower = stronger. newGame() banks this game's losses into the window.
    const all = skillWindow().concat(losses.map(l => l.drop));
    if (all.length) {
        const avg = all.reduce((a, b) => a + b, 0) / all.length;
        set("gs_skill", avg >= 0.05 ? "−" + avg.toFixed(1) + "%" : "0%");
    } else {
        set("gs_skill", t("wdl_dash"));
    }

    // Avg thinking time over the human's own moves (play mode only).
    let tSum = 0, tN = 0;
    if (!isAnalysis) for (const m of moveTimes) if (m.movedBy === humanSide) { tSum += m.ms; tN++; }
    set("gs_avgtime", tN ? fmtDuration(tSum / tN) : t("wdl_dash"));

    // 最大失误: this game's single human move that conceded the most win% — the max
    // of the same per-move losses. Attributed to the move at step k that fell.
    let worstDrop = 0, worstStep = 0;
    for (const l of losses) if (l.drop > worstDrop) { worstDrop = l.drop; worstStep = l.step; }
    set("gs_blunder", worstDrop > 0.5
        ? t("stat_blunder_val", worstStep, "−" + Math.round(worstDrop))
        : t("wdl_dash"));
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
    // While reviewing, the legend tracks the reviewed ply's stored value so it
    // stays consistent with the board on screen; live otherwise.
    const h = (reviewIndex !== null && reviewIndex < history.length) ? history[reviewIndex] : null;
    const root = h ? h.rootWDL : lastRootWDL;
    const nn = h ? h.nnWDL : lastNNWDL;
    renderWinLegend(valueTab === "root" ? root : nn);
    drawValueChart();
}
// Set the current ply's root/nn values (already in Black's frame) + repaint.
function setCurrentValues(rootBlack, nnBlack) {
    lastRootWDL = rootBlack || null;
    lastNNWDL = nnBlack || null;
    renderValuePanel();
}

// --- Candidate move list (right column) ----------------------------------
const candListEl = document.getElementById("cand_list");
const candLegendEl = document.getElementById("cand_legend");
let candSig = "";
const MAX_CANDS = 12;
let searchSimsTotal = 0;        // cumulative root visits reported by the worker
const ANALYSIS_CHUNK = 128;     // V7.19 play.cfg ANALYZE_CHUNK_SIMS
const ANALYSIS_CAP_MIN = 2000;  // analysis board deepens at least this many root visits
// Per-move thinking time for play mode (AI move + "my-turn" analysis), in ms;
// picked from the toolbar dropdown. Persisted in localStorage("skz_think_ms").
const THINK_MS_OPTIONS = [500, 1000, 2000, 3000, 5000, 10000];
let thinkMs = (function () {
    try {
        const v = parseInt(localStorage.getItem("skz_think_ms"), 10);
        return THINK_MS_OPTIONS.includes(v) ? v : 500;
    } catch (_) { return 500; }
})();
let searchEnabled = (function () {
    try { return localStorage.getItem("skz_search_enabled") !== "0"; }
    catch (_) { return true; }
})();
let pda = (function () {
    try {
        const raw = localStorage.getItem("skz_pda");
        if (raw === null) return 0.5;
        const v = Number(raw);
        return Number.isFinite(v) && v >= -1 && v <= 1
            && Math.abs(v * 10 - Math.round(v * 10)) < 1e-9 ? v : 0.5;
    } catch (_) { return 0.5; }
})();
function colLetter(c) { return String.fromCharCode(65 + c); }
function rowNumber(r) { return r + 1; }
// Board notation: columns A.. left→right, rows 1..N top→bottom (H8 = center of 15).
function coordLabel(r, c) { return colLetter(c) + rowNumber(r); }
// Rank searched candidates by V7.19's LCB-adjusted play-selection
// distribution. With thinking disabled, rank by the exact effective root
// prior used for the NN-only move (main/optimistic logits blended in workers/search.js).
function computeCandidates() {
    if (!state || !state.board) return [];
    const vis = state.mcts_visits;
    const rankG = state.mcts_play_selection || vis;
    const wrG = state.mcts_winrate;
    const policyG = state.policy_prior;
    const policyOnly = state.policy_only === true && !!policyG;
    if (!policyOnly && !vis && !wrG) return [];
    const out = [];
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
        if (state.board[r][c] !== 0) continue;
        if (policyOnly) {
            const prior = (policyG[r] && policyG[r][c]) || 0;
            if (prior > 0) {
                out.push({
                    r, c, vf: prior, rank: prior, wr: null,
                    source: "policy",
                });
            }
            continue;
        }
        const vf = (vis && vis[r]) ? (vis[r][c] || 0) : 0;
        const rank = (rankG && rankG[r]) ? (rankG[r][c] || 0) : vf;
        const wRaw = (wrG && wrG[r]) ? wrG[r][c] : null;
        const wr = (wRaw != null && Number.isFinite(wRaw)) ? wRaw : null;
        if (rank > 0 || vf > 0 || (wr != null && wr > 0)) {
            out.push({ r, c, vf, rank, wr, source: "search" });
        }
    }
    if (out.length === 0) return [];
    out.sort((a, b) => (b.rank - a.rank) || (b.vf - a.vf)
        || ((b.wr ?? -1) - (a.wr ?? -1)));
    const maxRank = out[0].rank || 0;
    out.forEach((o, i) => {
        o.frac = maxRank > 0 ? o.rank / maxRank : 1;
        o.best = (i === 0);
    });
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
function fmtPrior(prior) {
    const pct = Math.max(0, prior) * 100;
    return (pct >= 10 ? pct.toFixed(0) : pct.toFixed(1)) + "%";
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
function renderCandidates() {
    const cands = state ? computeCandidates() : [];
    const policyOnly = cands.length > 0 && cands[0].source === "policy";
    if (candLegendEl) {
        candLegendEl.textContent = t(
            policyOnly ? "cand_legend_policy" : "cand_legend");
    }
    // Only rebuild when the data changed, so a row's :hover stays put.
    const sig = cands.map(o =>
        o.source + ":" + o.r + "," + o.c + ":"
        + Math.round((o.wr ?? -1) * 1000) + ":"
        + Math.round(o.vf * 100000)).join("|");
    if (sig === candSig) return;
    candSig = sig;
    if (cands.length === 0) {
        candListEl.innerHTML = '<div class="cand-empty">' + t("cand_empty") + "</div>";
        return;
    }
    candListEl.innerHTML = cands.map((o, i) => {
        const rank = String.fromCharCode(65 + i);            // A, B, C, …
        const wr = o.wr != null ? Math.round(o.wr * 100) + "%" : t("wdl_dash");
        const barW = policyOnly
            ? Math.round(o.frac * 100)
            : (o.wr != null ? Math.round(o.wr * 100) : 0);
        const detail = policyOnly ? fmtPrior(o.rank) : fmtVisits(o.vf);
        return '<div class="cand-row' + (o.best ? " best" : "") +
               '" data-r="' + o.r + '" data-c="' + o.c + '">' +
               '<span class="cand-rank">' + rank + "</span>" +
               '<span class="cand-coord">' + coordLabel(o.r, o.c) + "</span>" +
               '<span class="cand-wr">' + wr + "</span>" +
               '<span class="cand-track"><span style="width:' + barW + '%"></span></span>' +
               '<span class="cand-visits">' + detail + "</span>" +
               "</div>";
    }).join("");
}
candListEl.addEventListener("mouseover", (ev) => {
    const row = ev.target.closest(".cand-row"); if (!row) return;
    hoverCand = { r: +row.dataset.r, c: +row.dataset.c };
    draw();
});
candListEl.addEventListener("mouseleave", () => {
    // mouseleave (not mouseout) fires once when the pointer truly exits the list —
    // robust to the row DOM being rebuilt mid-hover by renderCandidates(), which
    // would otherwise strand the board ring after the pointer left.
    if (hoverCand) { hoverCand = null; draw(); }
});
candListEl.addEventListener("click", (ev) => {
    const row = ev.target.closest(".cand-row"); if (!row) return;
    playCandidate(+row.dataset.r, +row.dataset.c);
});

// --- Model dropdown (loads from /gomoku/models/manifest.json) ---
let manifest = { default: null, models: [] };
let currentModelId = null;

async function loadManifest() {
    const r = await fetch("/gomoku/models/manifest.json", { cache: "no-cache" });
    if (!r.ok) throw new Error("manifest.json fetch failed: " + r.status);
    manifest = await r.json();
    const menu = document.getElementById("model_menu");
    menu.innerHTML = "";
    const items = manifest.models.slice().sort(
        (a, b) => (a.order ?? a.elo ?? 0) - (b.order ?? b.elo ?? 0));
    for (const m of items) {
        const o = document.createElement("button");
        o.type = "button";
        o.className = "cs-option";
        o.setAttribute("role", "option");
        o.dataset.id = m.id;
        const name = document.createElement("span");
        name.className = "cs-opt-name";
        name.textContent = `${m.id.toUpperCase()} ${m.label}`;
        const elo = document.createElement("span");
        elo.className = "cs-opt-elo";
        elo.textContent = formatModelElo(m);
        o.append(name, elo);
        menu.appendChild(o);
    }
    currentModelId = null;
    setModelTrigger(null);
    populateDifficultyOptions(items);
}

// Reflect the active model on the dropdown trigger — the short "LV3 高手" only
// (the ELO lives in the menu rows, so the toolbar control stays narrow) — and
// mark the matching menu row as selected.
function setModelTrigger(id) {
    const m = modelById(id);
    const labelEl = document.getElementById("model_trigger_label");
    if (labelEl) labelEl.textContent = m ? `${m.id.toUpperCase()} ${m.label}` : t("difficulty_choose");
    for (const opt of document.querySelectorAll("#model_menu .cs-option")) {
        opt.setAttribute("aria-selected", opt.dataset.id === id ? "true" : "false");
    }
    // Grey out rule buttons the newly selected model wasn't trained on
    // (manifest.json's per-model `rules`); models missing `rules` leave every
    // button enabled.
    for (const b of document.querySelectorAll(".seg-btn[data-rule]")) {
        b.disabled = !!m && !!m.rules && !m.rules.includes(b.dataset.rule);
    }
}

function modelById(id) {
    return manifest.models.find(m => m.id === id);
}

// Parse the model id without waiting for the manifest, so direct routes can
// show a model-specific loading screen from the first frame.
function modelIdFromPath() {
    const parts = window.location.pathname.split("/").filter(Boolean);
    if (parts.length !== 2 || parts[0] !== "gomoku") return null;
    return /^lv[1-6]$/.test(parts[1]) ? parts[1] : null;
}

function modelFromPath() {
    const id = modelIdFromPath();
    return id ? modelById(id) || null : null;
}

// Build the model URL with a cache-busting query string derived from the
function formatModelElo(m) {
    if (!m || !Number.isFinite(m.elo)) return t("difficulty_elo", "—");
    return t("difficulty_elo", (m.elo > 0 ? "+" : "") + m.elo);
}

function populateDifficultyOptions(items) {
    const options = document.getElementById("difficulty_options");
    if (!options) return;
    options.innerHTML = "";
    for (const m of items) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "difficulty-option";
        const main = document.createElement("span");
        main.className = "difficulty-option-main";
        const level = document.createElement("span");
        level.className = "difficulty-option-level";
        level.textContent = m.id.toUpperCase();
        const label = document.createElement("span");
        label.className = "difficulty-option-label";
        label.textContent = m.label;
        const elo = document.createElement("span");
        elo.className = "difficulty-option-elo";
        elo.textContent = formatModelElo(m);
        main.append(level, label);
        button.append(main, elo);
        button.addEventListener("click", () => chooseInitialModel(m));
        options.appendChild(button);
    }
}

function showDifficultyPicker() {
    const modal = document.getElementById("difficulty_modal");
    if (!modal) return;
    modal.classList.remove("hidden");
    document.body.classList.add("difficulty-open");
}

function chooseInitialModel(m) {
    if (!m || currentModelId !== null) return;
    currentModelId = m.id;
    setModelTrigger(m.id);
    document.getElementById("difficulty_modal")?.classList.add("hidden");
    document.body.classList.remove("difficulty-open");
    showLoadingOverlay("loading_model", m.id.toUpperCase(), m.label);
    worker.postMessage({ type: "init", modelUrl: modelUrl(m), boardSize: N, rule: currentRule });
}

// Build the model URL with a cache-busting query string derived from the
// model's `params` field (e.g., "b10c128 iter44"). _headers marks .onnx as
// immutable for a year, so without this, replacing a model file leaves
// browsers serving the stale cached blob. Same params → same URL → cache
// hit; new iter → new URL → fresh fetch.
function modelUrl(m) {
    const v = encodeURIComponent(m.params || m.file);
    return `/gomoku/models/${m.file}?v=${v}`;
}

// V7.19's fixed 15x15 network canvas supports runtime boards 11..15.
const BOARD_SIZES = [11, 12, 13, 14, 15];

// --- Loading overlay (only shown for first model load) ---
const INITIALIZATION_SLOW_MS = 10_000;
const INITIALIZATION_PROGRESS_LIMIT = 92;
let initializationFeedbackActive = false;
let initializationProgressTimer = null;
let initializationSlowTimer = null;

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
function stopInitializationFeedback() {
    initializationFeedbackActive = false;
    if (initializationProgressTimer !== null) {
        clearInterval(initializationProgressTimer);
        initializationProgressTimer = null;
    }
    if (initializationSlowTimer !== null) {
        clearTimeout(initializationSlowTimer);
        initializationSlowTimer = null;
    }
}
function showInitializationSlowHint(
    isWeChat = /MicroMessenger/i.test(navigator.userAgent)
) {
    setLoadingHeadline(isWeChat
        ? "loading_initializing_slow_wechat"
        : "loading_initializing_slow");
    setLoadingProgress(INITIALIZATION_PROGRESS_LIMIT);
    if (initializationProgressTimer !== null) {
        clearInterval(initializationProgressTimer);
        initializationProgressTimer = null;
    }
}
function startInitializationFeedback(id, label) {
    // fetchModelWithProgress can emit 100% for both the last chunk and stream
    // completion. Start the simulated session-build phase only once.
    if (initializationFeedbackActive) return;
    initializationFeedbackActive = true;
    setLoadingHeadline("loading_initializing", id, label);
    const startedAt = performance.now();
    setLoadingProgress(4);
    initializationProgressTimer = setInterval(() => {
        // Session creation exposes no real progress. Ease quickly through the
        // normal 3–4 second desktop wait, then asymptotically stop at 92%.
        const elapsed = performance.now() - startedAt;
        const pct = 4 + 88 * (1 - Math.exp(-elapsed / 2500));
        setLoadingProgress(Math.min(INITIALIZATION_PROGRESS_LIMIT, pct));
    }, 200);
    initializationSlowTimer = setTimeout(() => {
        if (initializationFeedbackActive) showInitializationSlowHint();
    }, INITIALIZATION_SLOW_MS);
}
function hideLoadingOverlay() {
    stopInitializationFeedback();
    const o = document.getElementById("loading_overlay");
    if (o) { o.style.display = "none"; o.classList.add("hidden"); }
    lastLoadingMsg = null;
}
let lastLoadingMsg = null; // { key, args }
// Render the literal "SkyZero" in a string as the brand wordmark (neutral "Sky"
// + rainbow "Zero"), mirroring the other game pages.
const BRAND_WM = '<span class="brand-wm"><span class="bw-sky">Sky</span><span class="bw-zero">Zero</span></span>';
const withBrand = (s) => s.replace("SkyZero", BRAND_WM);
function showLoadingOverlay(key, ...args) {
    stopInitializationFeedback();
    const o = document.getElementById("loading_overlay");
    if (!o) return;
    o.style.display = "";
    o.classList.remove("hidden");
    lastLoadingMsg = { key, args };
    const tEl = document.getElementById("loading_text");
    if (tEl) tEl.innerHTML = withBrand(t(key, ...args));
    setLoadingProgress(0);
}
// Swap just the headline (keeps the bar/percent as-is). Used to flip from
// "loading" to "initializing" once the download hits 100% — the session-build
// step (ort WASM runtime + graph compile) isn't tracked by the bar, so a full
// bar would otherwise read as frozen. Routes through lastLoadingMsg so a
// language switch mid-init re-renders the right string.
function setLoadingHeadline(key, ...args) {
    lastLoadingMsg = { key, args };
    const tEl = document.getElementById("loading_text");
    if (tEl) tEl.innerHTML = withBrand(t(key, ...args));
}
function rerenderLoadingOverlay() {
    if (!lastLoadingMsg) return;
    const tEl = document.getElementById("loading_text");
    if (tEl) tEl.innerHTML = withBrand(t(lastLoadingMsg.key, ...lastLoadingMsg.args));
}
