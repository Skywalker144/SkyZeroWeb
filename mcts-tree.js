// AlphaZero MCTS tree visualizer — front end.
//
// Talks to worker-tree.js (real ONNX inference + KataGo-PUCT search), then draws
// the search tree vertically on a canvas: every node is a board position with
// its N / win-rate / prior / U. Pruning to top-k + top-p happens here so the
// sliders re-render instantly without re-searching.

const ASSET_V = "" + Date.now();   // cache-bust worker + its importScripts (dev-friendly)

// Canvas palette per theme. The CSS variables drive the chrome; the canvas can't
// read those mid-draw cheaply, so we mirror the colors here and swap on theme change.
const THEMES = {
    light: {
        nodeFill: "#ffffff", rootStroke: "#8b949e", edge: "rgba(140,148,158,.5)",
        textPrimary: "#1f2328", textMuted: "#59636e",
        boardBg: "#e8c583", boardLine: "rgba(107,90,58,.5)",
        stoneBlack: "#111", stoneBlackEdge: "#000", stoneWhite: "#fff", stoneWhiteEdge: "#8b949e",
        lastRing: "#cf222e", accent: "#0969da", path: "#d29922",
        termWin: "#1a7f37", termLoss: "#cf222e", termDraw: "#9a6700", pruned: "#8250df", winLight: 42,
    },
    dark: {
        nodeFill: "#161b22", rootStroke: "#6e7681", edge: "rgba(110,118,129,.5)",
        textPrimary: "#e6edf3", textMuted: "#9198a1",
        boardBg: "#c9a460", boardLine: "rgba(74,58,31,.6)",
        stoneBlack: "#0d1117", stoneBlackEdge: "#000", stoneWhite: "#fafafa", stoneWhiteEdge: "#6e7681",
        lastRing: "#f85149", accent: "#4493f8", path: "#e3b341",
        termWin: "#3fb950", termLoss: "#f85149", termDraw: "#d29922", pruned: "#ab7df8", winLight: 52,
    },
};
let T = THEMES.light;
const NODE_W = 84, NODE_H = 112, GAP_X = 16, GAP_Y = 54;

// ---- DOM ----
const $ = id => document.getElementById(id);
const canvas = $("tree"), g = canvas.getContext("2d");
const detailCanvas = $("detailBoard"), dctx = detailCanvas.getContext("2d");
const elModel = $("model"), elRule = $("rule"), elSize = $("size"), elRunN = $("runN");
const elTopk = $("topk"), elTopp = $("topp"), elTopkv = $("topkv"), elToppv = $("toppv");
const elStatus = $("status"), detailTable = $("detailTable");

// ---- state ----
let worker = null, busy = false;
let SIZE = 9, RULE = "renju", modelFile = "level3.onnx";
let tree = null, byId = new Map(), childrenOf = new Map();
let visKids = new Map(), prunedOf = new Map(), pos = new Map();
let rootId = null, contentW = 0, contentH = 0;
let scale = 1, offX = 0, offY = 0, dpr = window.devicePixelRatio || 1;
let selectedId = null, hoverId = null, lastPath = null;

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// ============================================================ worker
function boot() {
    if (worker) worker.terminate();
    worker = new Worker("worker-tree.js?v=" + ASSET_V);
    worker.onmessage = onMsg;
    worker.onerror = e => {
        setBusy(false);
        const m = (e.message || "(no message)") + " @ " + (e.filename || "?") + ":" + (e.lineno || "?");
        elStatus.textContent = "Worker 错误: " + m;
        console.error("worker.onerror:", m);
    };
    setBusy(true);
    elStatus.textContent = `加载模型 ${modelFile} …`;
    // _headers marks *.onnx immutable; version by the model's stable `params`
    // (not Date.now) so the multi-MB blob is cached across loads, like main.js.
    const ver = (elModel.selectedOptions[0] && elModel.selectedOptions[0].dataset.ver) || modelFile;
    worker.postMessage({
        type: "init",
        modelUrl: "models/" + modelFile + "?v=" + encodeURIComponent(ver),
        boardSize: SIZE, rule: RULE,
    });
}

function onMsg(e) {
    const d = e.data;
    if (d.type === "ready") { setTree(d.tree, null); setBusy(false); fitView(); }
    else if (d.type === "step") { setTree(d.tree, d.trace.path); setBusy(false); }
    else if (d.type === "done") { setTree(d.tree, null); setBusy(false); }
    else if (d.type === "progress") { elStatus.textContent = `搜索中… ${d.done}/${d.total}`; }
    else if (d.type === "error") { setBusy(false); elStatus.textContent = "错误: " + d.message; console.error("worker reported:", d.message); }
}

function setBusy(b) {
    busy = b;
    for (const el of [$("btnStep"), $("btnRun"), $("btnReset"), elModel, elRule, elSize])
        el.disabled = b;
}

function statusText() {
    if (!tree) return "";
    return `${tree.totalSims - 1} 次模拟 · ${tree.nodes.length} 个已访问节点 · 根总访问 ${tree.rootVisits}`;
}

// ============================================================ ingest + prune + layout
function setTree(t, path) {
    tree = t; SIZE = t.boardSize;
    byId = new Map(); childrenOf = new Map();
    for (const n of t.nodes) byId.set(n.id, n);
    for (const n of t.nodes) {
        if (n.parentId == null) { rootId = n.id; continue; }
        if (!childrenOf.has(n.parentId)) childrenOf.set(n.parentId, []);
        childrenOf.get(n.parentId).push(n);   // worker pre-sorted by n desc
    }
    lastPath = path ? new Set(path) : null;
    if (selectedId == null || !byId.has(selectedId)) selectedId = rootId;
    applyPruneAndLayout();
    render();
    renderDetail();
    elStatus.textContent = statusText();
}

function prune() {
    visKids = new Map(); prunedOf = new Map();
    const K = +elTopk.value, P = +elTopp.value;
    for (const [pid, kids] of childrenOf) {
        // top-p: keep children whose network prior >= P; top-k caps the count.
        // kids are visit-sorted, so this keeps the highest-visited among them.
        const keep = [];
        for (const k of kids) {
            if (keep.length >= K) break;
            if (k.prior >= P) keep.push(k);
        }
        if (keep.length === 0 && kids.length) keep.push(kids[0]);   // never orphan a branch
        visKids.set(pid, keep);
        prunedOf.set(pid, kids.length - keep.length);
    }
}

function layout() {
    pos = new Map();
    let leafX = 0;
    (function place(id, depth) {
        const kids = visKids.get(id) || [];
        let x;
        if (kids.length === 0) { x = leafX; leafX += NODE_W + GAP_X; }
        else {
            for (const k of kids) place(k.id, depth + 1);
            x = (pos.get(kids[0].id).x + pos.get(kids[kids.length - 1].id).x) / 2;
        }
        pos.set(id, { x, y: depth * (NODE_H + GAP_Y), depth });
    })(rootId, 0);
    contentW = 0; contentH = 0;
    for (const p of pos.values()) { contentW = Math.max(contentW, p.x + NODE_W); contentH = Math.max(contentH, p.y + NODE_H); }
}

function applyPruneAndLayout() { prune(); layout(); }

// ============================================================ rendering
function resizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    }
}

function roundRect(c, x, y, w, h, r) {
    c.beginPath();
    c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r);
    c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath();
}

function drawBoard(c, x, y, px, node) {
    const n = SIZE, cell = px / n;
    c.fillStyle = T.boardBg; c.fillRect(x, y, px, px);
    c.strokeStyle = T.boardLine; c.lineWidth = Math.max(0.3, cell * 0.03);
    c.beginPath();
    for (let i = 0; i < n; i++) {
        const t = x + (i + 0.5) * cell, u = y + (i + 0.5) * cell;
        c.moveTo(x + 0.5 * cell, u); c.lineTo(x + (n - 0.5) * cell, u);
        c.moveTo(t, y + 0.5 * cell); c.lineTo(t, y + (n - 0.5) * cell);
    }
    c.stroke();
    if (!node) return;
    const b = node.board;
    for (let i = 0; i < n * n; i++) {
        const v = b[i]; if (!v) continue;
        const r = (i / n) | 0, col = i % n;
        const cx = x + (col + 0.5) * cell, cy = y + (r + 0.5) * cell;
        c.beginPath(); c.arc(cx, cy, cell * 0.42, 0, 2 * Math.PI);
        c.fillStyle = v === 1 ? T.stoneBlack : T.stoneWhite; c.fill();
        c.lineWidth = Math.max(0.4, cell * 0.05); c.strokeStyle = v === 1 ? T.stoneBlackEdge : T.stoneWhiteEdge; c.stroke();
    }
    if (node.action != null) {   // ring the last move
        const cx = x + (node.ac + 0.5) * cell, cy = y + (node.ar + 0.5) * cell;
        c.beginPath(); c.arc(cx, cy, cell * 0.47, 0, 2 * Math.PI);
        c.strokeStyle = T.lastRing; c.lineWidth = Math.max(1, cell * 0.1); c.stroke();
    }
}

function drawNode(node, p) {
    const x = p.x, y = p.y;
    g.fillStyle = T.nodeFill; roundRect(g, x, y, NODE_W, NODE_H, 8); g.fill();

    let stroke, lw = 1.3;
    if (node.depth === 0) stroke = T.rootStroke;
    else { const h = Math.round((node.winrate == null ? 0.5 : node.winrate) * 120); stroke = `hsl(${h} 60% ${T.winLight}%)`; }
    if (lastPath && lastPath.has(node.id)) { stroke = T.path; lw = 3; }
    if (node.id === hoverId && node.id !== selectedId && !(lastPath && lastPath.has(node.id))) { stroke = T.accent; lw = 2.4; }
    if (node.id === selectedId) { stroke = T.accent; lw = 3.4; }
    g.lineWidth = lw; g.strokeStyle = stroke; roundRect(g, x, y, NODE_W, NODE_H, 8); g.stroke();

    const bp = NODE_W - 10;
    drawBoard(g, x + 5, y + 5, bp, node);

    g.textAlign = "center";
    g.fillStyle = T.textPrimary; g.font = "600 11px sans-serif";
    const ty = y + 5 + bp + 13;
    g.fillText(`N ${node.n}`, x + NODE_W / 2, ty);
    const wr = node.winrate != null ? Math.round(node.winrate * 100) + "%" : "—";
    const pr = node.prior != null ? node.prior.toFixed(2) : "—";
    g.fillStyle = T.textMuted; g.font = "10px sans-serif";
    g.fillText(`胜${wr} P${pr}`, x + NODE_W / 2, ty + 13);

    if (node.isTerminal) {
        g.fillStyle = node.term === 1 ? T.termWin : node.term === -1 ? T.termLoss : T.termDraw;
        g.font = "600 9px sans-serif"; g.fillText("终局", x + NODE_W - 16, y + 11);
    }
    const pruned = prunedOf.get(node.id) || 0;
    if (pruned > 0) {
        g.fillStyle = T.pruned; g.font = "600 11px sans-serif"; g.textAlign = "center";
        g.fillText(`+${pruned}`, x + NODE_W / 2, y + NODE_H + GAP_Y * 0.55);
    }
}

function drawEdge(pp, cp, child) {
    const x1 = pp.x + NODE_W / 2, y1 = pp.y + NODE_H, x2 = cp.x + NODE_W / 2, y2 = cp.y;
    const onPath = lastPath && lastPath.has(pp.id ?? -1) && lastPath.has(cp.id ?? -1);
    const parent = byId.get(child.parentId);
    const share = parent && parent.n > 1 ? child.n / (parent.n - 1) : 0;
    g.beginPath(); g.moveTo(x1, y1);
    g.bezierCurveTo(x1, (y1 + y2) / 2, x2, (y1 + y2) / 2, x2, y2);
    g.strokeStyle = onPath ? T.path : T.edge;
    g.lineWidth = onPath ? 3 : 1 + 3.5 * share;
    g.stroke();
}

function render() {
    if (!tree) return;
    resizeCanvas();
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    g.translate(offX, offY); g.scale(scale, scale);
    // pos carries no id; attach for edge path test
    for (const [id, p] of pos) p.id = id;
    for (const [pid, kids] of visKids) {
        const pp = pos.get(pid); if (!pp) continue;
        for (const k of kids) { const cp = pos.get(k.id); if (cp) drawEdge(pp, cp, k); }
    }
    for (const [id, p] of pos) drawNode(byId.get(id), p);
}

function fitView() {
    const cw = canvas.clientWidth, ch = canvas.clientHeight;
    if (contentW <= 0 || contentH <= 0) { scale = 1; offX = cw / 2 - NODE_W / 2; offY = 24; render(); return; }
    scale = clamp(Math.min((cw - 40) / contentW, (ch - 40) / contentH), 0.15, 1.3);
    offX = (cw - contentW * scale) / 2;
    offY = 24;
    render();
}

// ============================================================ detail panel
function moveName(r, c) { return String.fromCharCode(65 + c) + (SIZE - r); }

function renderDetail() {
    const node = selectedId != null ? byId.get(selectedId) : null;
    dctx.clearRect(0, 0, 200, 200);
    drawBoard(dctx, 0, 0, 200, node);
    const rows = [];
    if (node) {
        rows.push(["访问次数 N", node.n]);
        rows.push(["网络先验 P", node.prior != null ? node.prior.toFixed(3) : "—"]);
        rows.push(["Q (落子方效用)", node.q != null ? node.q.toFixed(3) : "—"]);
        rows.push(["U (探索项)", node.U != null ? node.U.toFixed(3) : "(根)"]);
        rows.push(["Q+U (PUCT 分)", node.score != null ? node.score.toFixed(3) : "(根)"]);
        rows.push(["胜率 WR", node.winrate != null ? (node.winrate * 100).toFixed(1) + "%" : "—"]);
        if (node.wdl) rows.push(["W/D/L", node.wdl.map(v => (v * 100).toFixed(0)).join(" / ") + " %"]);
        rows.push(["着法", node.action != null ? moveName(node.ar, node.ac) : "(根·空盘)"]);
        rows.push(["轮到", node.toPlay === 1 ? "黑 ●" : "白 ○"]);
        rows.push(["深度", node.depth]);
        if (node.isTerminal) rows.push(["终局", node.term === 1 ? "走子方胜" : node.term === -1 ? "走子方负" : "和棋"]);
    }
    detailTable.innerHTML = rows.map(([k, v]) => `<tr><td class="k">${k}</td><td class="v">${v}</td></tr>`).join("");
}

// ============================================================ canvas interaction
function nodeAt(mx, my) {
    const tx = (mx - offX) / scale, ty = (my - offY) / scale;
    for (const [id, p] of pos)
        if (tx >= p.x && tx <= p.x + NODE_W && ty >= p.y && ty <= p.y + NODE_H) return id;
    return null;
}

let dragging = false, moved = false, sx = 0, sy = 0;
canvas.addEventListener("mousedown", e => { dragging = true; moved = false; sx = e.offsetX; sy = e.offsetY; canvas.classList.add("dragging"); });
canvas.addEventListener("mousemove", e => {
    if (dragging) {
        const dx = e.offsetX - sx, dy = e.offsetY - sy;
        if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
        offX += dx; offY += dy; sx = e.offsetX; sy = e.offsetY; render();
    } else {
        const id = nodeAt(e.offsetX, e.offsetY);
        if (id !== hoverId) { hoverId = id; canvas.style.cursor = id != null ? "pointer" : "grab"; render(); }
    }
});
canvas.addEventListener("mouseup", e => {
    dragging = false; canvas.classList.remove("dragging");
    if (!moved) { const id = nodeAt(e.offsetX, e.offsetY); if (id != null) { selectedId = id; renderDetail(); render(); } }
});
canvas.addEventListener("mouseleave", () => { dragging = false; hoverId = null; canvas.classList.remove("dragging"); render(); });
canvas.addEventListener("wheel", e => {
    e.preventDefault();
    const f = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    const tx = (e.offsetX - offX) / scale, ty = (e.offsetY - offY) / scale;
    scale = clamp(scale * f, 0.12, 3);
    offX = e.offsetX - tx * scale; offY = e.offsetY - ty * scale;
    render();
}, { passive: false });

// ============================================================ controls
$("btnStep").addEventListener("click", () => { if (!busy) { setBusy(true); elStatus.textContent = "模拟中…"; worker.postMessage({ type: "step" }); } });
$("btnRun").addEventListener("click", () => { if (!busy) { setBusy(true); elStatus.textContent = "连跑中…"; worker.postMessage({ type: "run", n: clamp(+elRunN.value || 50, 1, 2000) }); } });
$("btnReset").addEventListener("click", () => { if (!busy) { setBusy(true); selectedId = null; lastPath = null; worker.postMessage({ type: "reset", boardSize: SIZE, rule: RULE }); } });
$("btnFit").addEventListener("click", fitView);
elModel.addEventListener("change", () => { modelFile = elModel.value; selectedId = null; lastPath = null; boot(); });
elRule.addEventListener("change", () => { RULE = elRule.value; if (!busy) { setBusy(true); selectedId = null; lastPath = null; worker.postMessage({ type: "reset", boardSize: SIZE, rule: RULE }); } });
elSize.addEventListener("change", () => { SIZE = +elSize.value; if (!busy) { setBusy(true); selectedId = null; lastPath = null; worker.postMessage({ type: "reset", boardSize: SIZE, rule: RULE }); } });
elTopk.addEventListener("input", () => { elTopkv.textContent = elTopk.value; applyPruneAndLayout(); render(); });
elTopp.addEventListener("input", () => { elToppv.textContent = (+elTopp.value).toFixed(2); applyPruneAndLayout(); render(); });
window.addEventListener("resize", render);

// ============================================================ theme
// Tri-state auto/light/dark seg toggle, mirroring the gomoku/2048 pages. The
// pre-paint script (in <head>) already resolved the initial theme from skz_theme;
// here we sync the seg buttons, swap the canvas palette, and persist changes.
function pickPalette() { T = THEMES[document.documentElement.dataset.theme === "dark" ? "dark" : "light"]; }
function applyTheme(mode) {
    const resolved = mode === "auto"
        ? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")
        : mode;
    document.documentElement.dataset.theme = resolved;
    document.documentElement.dataset.themeMode = mode;
    document.querySelectorAll("#theme_seg .seg-btn").forEach(b =>
        b.setAttribute("aria-pressed", String(b.dataset.theme === mode)));
    pickPalette();
    if (tree) { render(); renderDetail(); }
}
(function initThemeSeg() {
    let mode = "auto";
    try { const v = localStorage.getItem("skz_theme"); if (v === "light" || v === "dark") mode = v; } catch (_) {}
    applyTheme(mode);
    document.querySelectorAll("#theme_seg .seg-btn").forEach(b => {
        b.addEventListener("click", () => {
            const m = b.dataset.theme;
            try { if (m === "auto") localStorage.removeItem("skz_theme"); else localStorage.setItem("skz_theme", m); } catch (_) {}
            applyTheme(m);
        });
    });
    try {
        matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
            if (document.documentElement.dataset.themeMode === "auto") applyTheme("auto");
        });
    } catch (_) {}
})();

// ============================================================ boot
async function loadModels() {
    try {
        const r = await fetch("models/manifest.json?v=" + ASSET_V);
        const m = await r.json();
        for (const md of m.models) {
            const o = document.createElement("option");
            o.value = md.file; o.dataset.ver = md.params || md.file; o.textContent = `${md.label} (ELO ${md.elo})`;
            if (md.id === m.default) o.selected = true;
            elModel.appendChild(o);
        }
        modelFile = elModel.value || "level3.onnx";
    } catch (e) {
        modelFile = "level3.onnx";
        const o = document.createElement("option"); o.value = modelFile; o.dataset.ver = modelFile; o.textContent = "默认"; elModel.appendChild(o);
    }
}

(async () => { await loadModels(); boot(); })();
