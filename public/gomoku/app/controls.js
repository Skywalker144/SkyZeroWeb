// Model dropdown (custom listbox). The native <select> rendered the full
// "LV3 高手 · ELO +1132" string and ran wide; this trigger shows just the short
// name while each ELO lives in the popped-open menu rows.
(function initModelDropdown() {
    const trigger = document.getElementById("model_trigger");
    const menu = document.getElementById("model_menu");
    if (!trigger || !menu) return;
    function setOpen(open) {
        if (open) {
            const thinkSel = document.getElementById("think_select");
            const thinkTrigger = document.getElementById("think_trigger");
            const thinkPop = document.getElementById("think_pop");
            if (thinkSel) thinkSel.classList.remove("open");
            if (thinkTrigger) thinkTrigger.setAttribute("aria-expanded", "false");
            if (thinkPop) {
                thinkPop.style.left = "";
                thinkPop.style.top = "";
            }
        }
        menu.classList.toggle("hidden", !open);
        trigger.setAttribute("aria-expanded", open ? "true" : "false");
    }
    trigger.addEventListener("click", (ev) => {
        ev.stopPropagation();
        setOpen(menu.classList.contains("hidden"));
    });
    menu.addEventListener("click", (ev) => {
        const opt = ev.target.closest(".cs-option");
        if (!opt) return;
        setOpen(false);
        const id = opt.dataset.id;
        const m = modelById(id);
        if (!m || id === currentModelId) return;
        currentModelId = id;
        setModelTrigger(id);
        showLoadingOverlay("loading_model", m.id.toUpperCase(), m.label);
        searchId++;
        aiThinking = false;
        worker.postMessage({ type: "swap-model", modelUrl: modelUrl(m) });
    });
    document.addEventListener("click", () => setOpen(false));
    document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") setOpen(false); });
})();

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
    updateReviewBar();
});

// --- Heatmap drawer ---
(function initHeatDrawer() {
    const btn = document.getElementById("heat_drawer_btn");
    const body = document.getElementById("heat_drawer_body");
    const drawer = document.getElementById("heat_drawer");
    if (!btn || !body || !drawer) return;
    const refit = () => {
        for (const id of Object.keys(heatCtxs)) {
            fitHeatCanvas(id);
            drawHeatById(id, state ? state[HEAT_GRID_KEYS[id]] : null);
        }
    };
    btn.addEventListener("click", () => {
        const open = body.classList.toggle("hidden") === false;
        btn.setAttribute("aria-expanded", open ? "true" : "false");
        if (open) refit();
    });
})();

// --- "Analyze in search tree" button: hand the live position to /mcts-tree ---
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
        location.href = "/mcts-tree";
    });
})();
