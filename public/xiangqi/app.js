const $ = id => document.getElementById(id);
const STRINGS = {
    zh: {
        title: "中国象棋", auto: "自动主题", light: "明亮", dark: "暗黑",
        gameStats: "本局统计", totalMoves: "总手数", searchVisits: "AI 搜索", candidates: "候选着法",
        candidateLegend: "胜率 · 访问", noAnalysis: "暂无分析", side: "执子", redFirst: "红棋",
        blackSecond: "黑棋", budget: "思考", newGame: "新对局", undo: "悔棋",
        retry: "重试", source: "规则源码", loading: "加载模型…", preparing: "准备棋局…",
        thinking: "AI 思考中…", yourTurn: "轮到你", check: "将军", win: "你赢了！",
        loss: "AI 获胜", draw: "和棋", red: "红方", black: "黑方",
        evaluation: "胜率", error: "加载或运行失败，请重试。", board: "中国象棋棋盘", empty: "空位",
        target: "可走"
    },
    en: {
        title: "Xiangqi", auto: "Auto theme", light: "Light", dark: "Dark",
        gameStats: "Game statistics", totalMoves: "Moves", searchVisits: "AI visits", candidates: "Candidate moves",
        candidateLegend: "Win % · visits", noAnalysis: "No analysis yet", side: "Side", redFirst: "Red",
        blackSecond: "Black", budget: "Think", newGame: "New game", undo: "Undo",
        retry: "Retry", source: "Rules source", loading: "Loading model…", preparing: "Preparing position…",
        thinking: "AI is thinking…", yourTurn: "Your turn", check: "Check", win: "You win!",
        loss: "AI wins", draw: "Draw", red: "Red", black: "Black",
        evaluation: "Win rate", error: "Loading or execution failed. Please retry.", board: "Xiangqi board", empty: "Empty",
        target: "Legal destination"
    }
};
const NAMES = {1: ['', '車', '仕', '炮', '兵', '馬', '相', '帥'], '-1': ['', '車', '士', '砲', '卒', '馬', '象', '將']};
let lang = localStorage.getItem('skz_lang') === 'en' ? 'en' : 'zh';
let human=1, selected=null, moves=[];
let session;
const t = key => STRINGS[lang][key] || key;

function renderBoard() {
    const focus = document.activeElement?.dataset.square;
    const point = (x, y) => [42 + 57 * (human === 1 ? x : 8 - x), 43 + 57 * (human === 1 ? 9 - y : y)];
    const line = (x, y, u, v) => {
        const [a, b] = point(x, y), [c, d] = point(u, v);
        return `<path d="M${a} ${b}L${c} ${d}"/>`;
    };
    let grid = '';
    for (let y = 0; y < 10; y++) grid += line(0, y, 8, y);
    for (let x = 0; x < 9; x++) grid += x === 0 || x === 8 ? line(x, 0, x, 9) : line(x, 0, x, 4) + line(x, 5, x, 9);
    for (const y of [0, 7]) grid += line(3, y, 5, y + 2) + line(5, y, 3, y + 2);
    const targets = new Set((session.state?.legal || []).filter(action => {
        const [x, y] = moves[action]; return y * 10 + x === selected;
    }).map(action => { const [, , x, y] = moves[action]; return y * 10 + x; }));
    const last = session.actions.length ? moves[session.actions.at(-1)] : null;
    const lastSquares = last ? [last[1] * 10 + last[0], last[3] * 10 + last[2]] : [];
    const disabled = !session.ready || session.busy || session.thinking || session.error || !session.state || session.state.finished || session.state.player !== human;
    let html = `<g class="grid">${grid}</g><text x="160" y="309" class="river">楚河</text><text x="390" y="309" class="river">汉界</text>`;
    for (let y = 0; y < 10; y++) for (let x = 0; x < 9; x++) {
        const square = y * 10 + x, piece = session.state?.board[square] || 0;
        const [px, py] = point(x, y);
        const name = piece ? NAMES[Math.sign(piece)][Math.abs(piece)] : t('empty');
        const own = piece * human > 0;
        const target = targets.has(square);
        html += `<g class="square" data-square="${square}" role="button" tabindex="${!disabled && (own || target) ? 0 : -1}" aria-disabled="${!!disabled}" aria-label="${piece ? t(piece > 0 ? 'red' : 'black') : ''} ${name} ${'abcdefghi'[x]}${y}${target ? ' · ' + t('target') : ''}"><circle class="hit" cx="${px}" cy="${py}" r="28"/>`;
        if (lastSquares.includes(square)) html += `<circle class="last-marker" cx="${px}" cy="${py}" r="27"/>`;
        if (piece) html += `<g class="${piece > 0 ? 'red' : 'black'}-piece"><circle class="piece" cx="${px}" cy="${py}" r="23"/><circle class="piece-inner" cx="${px}" cy="${py}" r="19.5"/><text class="piece-name" x="${px}" y="${py}">${name}</text></g>`;
        if (selected === square) html += `<circle class="selection" cx="${px}" cy="${py}" r="27"/>`;
        if (target) html += `<circle class="${piece ? 'capture' : 'destination'}" cx="${px}" cy="${py}" r="${piece ? 26 : 6}"/>`;
        html += `<circle class="focus-ring" fill="none" cx="${px}" cy="${py}" r="28"/></g>`;
    }
    $('board').innerHTML = html;
    $('board').setAttribute('aria-label', t('board'));
    $('board').setAttribute('aria-disabled', String(!!disabled));
    if (focus !== undefined) $('board').querySelector(`[data-square="${focus}"]`)?.focus();
}

function render() {
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    document.title = `${t('title')} · SkyZero`;
    document.querySelectorAll('[data-i18n]').forEach(node => { node.textContent = t(node.dataset.i18n); });
    $('language').textContent = lang === 'zh' ? 'EN' : '中';
    let status = !session.ready ? t('loading') : session.busy ? t('preparing') : session.thinking ? t('thinking') : t('yourTurn');
    if (session.state?.finished) status = t(session.state.winner === 0 ? 'draw' : session.state.winner === human ? 'win' : 'loss');
    if (session.state?.check && !session.state.finished) status = `${t('check')} · ${status}`;
    if (session.error) status = t('error');
    $('status').title = session.error;
    $('status').textContent = status;
    $('status-pill').dataset.variant = session.error ? 'error' : !session.ready || session.busy || session.thinking ? 'thinking' : session.state?.finished ? 'done' : 'active';
    BoardGameAnalysis.render({ply: session.actions.length, result: session.lastSearch, t, coordinate: action => { const [x, y, u, v] = moves[action]; return `${'abcdefghi'[x]}${y} → ${'abcdefghi'[u]}${v}`; }});
    $('undo').disabled = !session.ready || session.actions.length <= (human === -1 ? 1 : 0);
    $('retry').hidden = !session.error;
    renderBoard();
}

function choose(square) {
    if (!session.ready || session.busy || session.thinking || session.error || !session.state || session.state.finished || session.state.player !== human) return;
    const action = session.state.legal.find(action => {
        const [x, y, u, v] = moves[action]; return y * 10 + x === selected && v * 10 + u === square;
    });
    if (action !== undefined) {
        session.actions.push(action);
        session.syncPosition();
    } else {
        selected = session.state.board[square] * human > 0 && square !== selected ? square : null;
        renderBoard();
    }
}

session=new BoardGameSession({game:'xiangqi',version:'20261105b',getHuman:()=>human,render,analysis:BoardGameAnalysis,
    onReady:data=>{moves=data.moves;},
    onPositionRequested:()=>{selected=null;},
    onManifest:manifest=>{$('rules-source').href='/xiangqi/'+manifest.rules.source;}
});
$('board').addEventListener('click', event => {
    const node = event.target.closest('[data-square]');
    if (node) choose(Number(node.dataset.square));
});
$('board').addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        const node = event.target.closest('[data-square]');
        if (node) choose(Number(node.dataset.square));
    }
});
$('new-game').onclick=()=>session.newGame();
$('undo').onclick=()=>session.undo();
$('side').onchange = () => { human = Number($('side').value); session.newGame(); };
$('budget').onchange = () => { if (session.thinking) session.syncPosition(); };
$('retry').onclick=()=>session.boot();
$('language').onclick = () => { lang = lang === 'zh' ? 'en' : 'zh'; localStorage.setItem('skz_lang', lang); render(); };
function applyTheme() {
    const mode = $('theme').value;
    localStorage.setItem('skz_theme', mode);
    document.documentElement.dataset.theme = mode === 'auto' ? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light') : mode;
}
$('theme').value = localStorage.getItem('skz_theme') || 'auto';
$('theme').onchange = applyTheme;
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if ($('theme').value === 'auto') applyTheme(); });
session.boot();
