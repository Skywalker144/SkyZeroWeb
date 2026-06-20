// Lightweight i18n. Default language is Chinese; toggle in the topbar.
// Persisted via localStorage key 'skz_lang'. Public API: getLang, setLang, t,
// applyI18n, registerI18nCallback.

const I18N_STRINGS = {
    en: {
        page_title: "SkyZero · Gomoku",
        brand_title: "SkyZero · Gomoku",
        back_home: "Back to home",
        aria_theme: "Toggle color theme",
        aria_lang: "Switch language",
        status_idle: "idle",
        status_your_turn: "Your turn",
        status_ai_thinking: "AI thinking…",
        status_black_wins: "Black wins!",
        status_white_wins: "White wins!",
        status_draw: "Draw!",
        status_analyzing: "Analyzing…",
        status_analysis_ready: "Analysis ready",
        tb_label_mode: "Mode",
        tb_label_side: "Side",
        mode_play: "Play",
        mode_analysis: "Analysis",
        btn_settings: "Settings",
        cand_title: "Candidates",
        cand_legend: "win% · visits",
        cand_empty: "No analysis",
        tree_analyze: "Analyze this position in the search tree",
        show_on_board: "On board",
        show_on_board_aria: "Show your-turn analysis on the board (the AI's turn always shows)",
        label_cand_palette: "Marker colors",
        aria_cand_palette: "Candidate marker colors",
        palette_violet: "Violet",
        palette_amber: "Amber",
        palette_blue: "Blue",
        palette_teal: "Teal",
        palette_rose: "Rose",
        legend_black_win: "Black",
        legend_draw: "Draw",
        legend_white_win: "White",
        heat_drawer_title: "NN heatmaps",
        label_model: "Model",
        label_human_side: "Human side",
        side_black: "Black",
        side_white: "White",
        label_board_size: "Board size",
        label_rule: "Rule",
        rule_renju: "Renju",
        rule_freestyle: "Freestyle",
        label_search: "Search",
        label_search_hint: "(off = AI plays purely from intuition)",
        settings_group_game: "Game",
        settings_group_search: "Search",
        tb_label_think: "Think time",
        aria_think_time: "Thinking time per move",
        aria_search_toggle: "Toggle search (off = pure network)",
        label_value_estimates: "SkyZero win rate",
        legend_axis: "win rate · 0…100%",
        wdl_dash: "—",
        btn_new_game: "New game",
        btn_undo: "Undo",
        heat_visits_dist: "Visits Dist",
        heat_nn_policy: "NN Policy",
        heat_nn_opp_policy: "NN Opp Policy",
        heat_nn_futurepos_8: "NN Futurepos +8",
        heat_nn_futurepos_32: "NN Futurepos +32",
        heat_show_all: "Show all",
        heat_collapse: "Collapse",
        heatmap_default_title: "Heatmap",
        aria_expand: "Expand",
        aria_close: "Close",
        aria_pin_heat: "Pin to board",
        chart_no_data: "no data",
        stats_title: "Game stats",
        stats_persp_me: "You",
        stats_persp_black: "Black",
        stat_skill: "Loss/move",
        stat_moves: "Moves",
        stat_avgtime: "Avg time",
        stat_blunder: "Worst move",
        stat_blunder_val: (m, d) => `#${m}, ${d}%`,
        loading_initial: "Loading model…",
        loading_manifest: "Loading SkyZero engine…",
        loading_model: "Loading SkyZero engine…",
        err_manifest_load: (msg) => `manifest load failed: ${msg}`,
        err_manifest_empty: "manifest empty — add models",
        err_worker_failed: (msg, where) => `Worker failed: ${msg}${where}`,
        err_unknown: "unknown error",
        err_worker_msg: "Worker message error",
        err_prefix: (msg) => `Error: ${msg}`,
        theme_label_auto: "Auto",
        theme_label_light: "Light",
        theme_label_dark: "Dark",
        size_confirm_title: "Resize board?",
        size_confirm_body: (n) => `Switching to ${n}×${n} doesn't fit the current stones. The board will be reset. Continue?`,
        btn_confirm_reset: "Reset",
        btn_cancel: "Cancel",
        btn_edit_position: "Setup",
        btn_edit_done: "Done",
        btn_edit_undo: "Undo",
        edit_tool_alternate: "Alternate",
        edit_tool_black: "Black",
        edit_tool_white: "White",
        edit_tool_erase: "Erase",
        aria_edit_tool: "Edit tool",
        status_editing: "Editing position",
        status_edit_invalid: (b, w) => `Invalid stone count (B=${b}, W=${w}); need B==W or B==W+1`,
        review_label: (cur, total) => `Review · move ${cur} / ${total}`,
        review_first: "Jump to start",
        review_prev: "Previous move",
        review_next: "Next move",
        review_live: "Live",
    },
    zh: {
        page_title: "SkyZero · Gomoku",
        brand_title: "SkyZero · Gomoku",
        back_home: "返回首页",
        aria_theme: "切换配色主题",
        aria_lang: "切换语言",
        status_idle: "空闲",
        status_your_turn: "轮到您",
        status_ai_thinking: "AI 思考中…",
        status_black_wins: "黑方获胜!",
        status_white_wins: "白方获胜!",
        status_draw: "和棋!",
        status_analyzing: "分析中…",
        status_analysis_ready: "分析就绪",
        tb_label_mode: "模式",
        tb_label_side: "执子",
        mode_play: "对弈",
        mode_analysis: "分析",
        btn_settings: "设置",
        cand_title: "候选着法",
        cand_legend: "胜率 · 访问",
        cand_empty: "暂无分析",
        tree_analyze: "在搜索树中分析当前局面",
        show_on_board: "棋盘显示",
        show_on_board_aria: "在棋盘上显示我方回合的分析(AI 回合始终显示)",
        label_cand_palette: "候选配色",
        aria_cand_palette: "候选点位配色",
        palette_violet: "紫罗兰",
        palette_amber: "琥珀",
        palette_blue: "蓝青",
        palette_teal: "翠绿",
        palette_rose: "玫瑰",
        legend_black_win: "黑胜",
        legend_draw: "平局",
        legend_white_win: "白胜",
        heat_drawer_title: "NN 热力图",
        label_model: "模型",
        label_human_side: "我方执子",
        side_black: "黑棋",
        side_white: "白棋",
        label_board_size: "棋盘大小",
        label_rule: "规则",
        rule_renju: "连珠",
        rule_freestyle: "无禁手",
        label_search: "搜索",
        label_search_hint: "(关闭则 AI 直接凭直觉落子)",
        settings_group_game: "对局",
        settings_group_search: "搜索",
        tb_label_think: "思考时间",
        aria_think_time: "每步思考时间",
        aria_search_toggle: "切换搜索(关闭即纯网络)",
        label_value_estimates: "SkyZero胜率",
        legend_axis: "胜率 · 0…100%",
        wdl_dash: "—",
        btn_new_game: "新对局",
        btn_undo: "悔棋",
        heat_visits_dist: "访问分布",
        heat_nn_policy: "网络策略",
        heat_nn_opp_policy: "对手策略",
        heat_nn_futurepos_8: "未来落点 +8",
        heat_nn_futurepos_32: "未来落点 +32",
        heat_show_all: "展开全部",
        heat_collapse: "收起",
        heatmap_default_title: "热力图",
        aria_expand: "展开",
        aria_close: "关闭",
        aria_pin_heat: "固定到棋盘",
        chart_no_data: "暂无数据",
        stats_title: "本局统计",
        stats_persp_me: "我方",
        stats_persp_black: "黑方",
        stat_skill: "每手失分",
        stat_moves: "总手数",
        stat_avgtime: "平均每手用时",
        stat_blunder: "最大失误",
        stat_blunder_val: (m, d) => `第${m}手 ${d}%`,
        loading_initial: "正在加载模型…",
        loading_manifest: "加载SkyZero引擎中…",
        loading_model: "加载SkyZero引擎中…",
        err_manifest_load: (msg) => `清单加载失败:${msg}`,
        err_manifest_empty: "清单为空 — 请添加模型",
        err_worker_failed: (msg, where) => `Worker 失败:${msg}${where}`,
        err_unknown: "未知错误",
        err_worker_msg: "Worker 消息错误",
        err_prefix: (msg) => `错误:${msg}`,
        theme_label_auto: "自动",
        theme_label_light: "明亮",
        theme_label_dark: "暗黑",
        size_confirm_title: "切换棋盘大小?",
        size_confirm_body: (n) => `切换到 ${n}×${n} 无法容纳当前所有棋子,棋盘将被重置。是否继续?`,
        btn_confirm_reset: "重置",
        btn_cancel: "取消",
        btn_edit_position: "编辑棋形",
        btn_edit_done: "完成",
        btn_edit_undo: "撤回",
        edit_tool_alternate: "轮流",
        edit_tool_black: "放黑",
        edit_tool_white: "放白",
        edit_tool_erase: "擦除",
        aria_edit_tool: "编辑工具",
        status_editing: "编辑棋形中",
        status_edit_invalid: (b, w) => `棋子数不合法(黑 ${b} / 白 ${w});需 黑 = 白 或 黑 = 白 + 1`,
        review_label: (cur, total) => `复盘 · 第 ${cur} / ${total} 手`,
        review_first: "回到开局",
        review_prev: "上一手",
        review_next: "下一手",
        review_live: "返回对局",
    },
};

const I18N_DEFAULT = "zh";
const I18N_LANGS = ["zh", "en"];

function getLang() {
    try {
        const v = localStorage.getItem("skz_lang");
        if (v && I18N_STRINGS[v]) return v;
    } catch (_) {}
    return I18N_DEFAULT;
}

function setLang(lang) {
    if (!I18N_STRINGS[lang]) return;
    try { localStorage.setItem("skz_lang", lang); } catch (_) {}
    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
    applyI18n();
    for (const cb of i18nCallbacks) {
        try { cb(); } catch (_) {}
    }
}

function t(key, ...args) {
    const dict = I18N_STRINGS[getLang()] || I18N_STRINGS[I18N_DEFAULT];
    const v = dict[key];
    if (v == null) return key;
    return typeof v === "function" ? v(...args) : v;
}

const i18nCallbacks = [];
function registerI18nCallback(fn) { i18nCallbacks.push(fn); }

function applyI18n() {
    document.title = t("page_title");
    for (const el of document.querySelectorAll("[data-i18n]")) {
        el.textContent = t(el.dataset.i18n);
    }
    for (const el of document.querySelectorAll("[data-i18n-title]")) {
        el.title = t(el.dataset.i18nTitle);
    }
    for (const el of document.querySelectorAll("[data-i18n-aria]")) {
        el.setAttribute("aria-label", t(el.dataset.i18nAria));
    }
    for (const el of document.querySelectorAll("[data-i18n-aria-group]")) {
        el.setAttribute("aria-label", t(el.dataset.i18nAriaGroup));
    }
    updateLangSegPressed();
}

function updateLangSegPressed() {
    const cur = getLang();
    const seg = document.getElementById("lang_seg");
    if (!seg) return;
    for (const b of seg.querySelectorAll(".seg-btn[data-lang]")) {
        b.setAttribute("aria-pressed", b.dataset.lang === cur ? "true" : "false");
    }
}

function initLangSeg() {
    const seg = document.getElementById("lang_seg");
    if (!seg) return;
    for (const b of seg.querySelectorAll(".seg-btn[data-lang]")) {
        b.addEventListener("click", () => {
            if (b.dataset.lang !== getLang()) setLang(b.dataset.lang);
        });
    }
}

document.addEventListener("DOMContentLoaded", () => {
    document.documentElement.lang = getLang() === "zh" ? "zh-CN" : "en";
    applyI18n();
    initLangSeg();
    for (const cb of i18nCallbacks) {
        try { cb(); } catch (_) {}
    }
});
