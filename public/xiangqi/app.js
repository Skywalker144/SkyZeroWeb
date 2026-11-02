const $ = id => document.getElementById(id);
const STRINGS = {
    zh: {title: '中国象棋', subtitle: '楚河汉界，一步见高下。', auto: '自动主题', light: '明亮', dark: '暗黑',
        gameInfo: '对局信息', gameStats: '本局统计', totalMoves: '总手数', searchVisits: 'AI 搜索', candidates: '候选着法', candidateLegend: '上次搜索 · 评估 · 访问', noAnalysis: '暂无分析', side: '你的执色', redFirst: '红方 · 先手', blackSecond: '黑方 · 后手',
        budget: 'AI 思考时间', newGame: '新对局', undo: '悔棋', retry: '重试', lastMove: '上一手',
        howTo: '怎么玩', rules: '先选择棋子，再点击落点。圆点表示可走位置，圆环表示可吃的棋子。红方先行，将死或困毙对手获胜。',
        repetition: '长将、长捉与重复局面由完整棋规裁定。', local: 'AI 在你的浏览器中运行。', source: '规则源码',
        loading: '加载模型…', preparing: '准备棋局…', thinking: 'AI 思考中…', yourTurn: '轮到你', check: '将军',
        win: '你赢了！', loss: 'AI 获胜', draw: '和棋', red: '红方', black: '黑方', you: '你',
        moves: '步', model: '自训练模型 · 第 {iteration} 轮', evaluation: 'AI 局面评估',
        reason6: '重复局面', reason7: '六十回合未吃子', reason9: '子力不足', reason5: '将死 / 困毙',
        error: '加载或运行失败，请重试。', board: '中国象棋棋盘', empty: '空位', target: '可走'},
    en: {title: 'Xiangqi', subtitle: 'Across the river, every move matters.', auto: 'Auto theme', light: 'Light', dark: 'Dark',
        gameInfo: 'Game information', gameStats: 'Game statistics', totalMoves: 'Moves', searchVisits: 'AI visits', candidates: 'Candidate moves', candidateLegend: 'Last search · estimate · visits', noAnalysis: 'No analysis yet', side: 'Your side', redFirst: 'Red · first', blackSecond: 'Black · second',
        budget: 'AI thinking time', newGame: 'New game', undo: 'Undo', retry: 'Retry', lastMove: 'Last move',
        howTo: 'How to play', rules: 'Select a piece, then a destination. Dots mark legal moves; rings mark captures. Red moves first. Checkmate or stalemate wins.',
        repetition: 'Full rules adjudicate perpetual check, chase and repetition.', local: 'The AI runs in your browser.', source: 'Rules source',
        loading: 'Loading model…', preparing: 'Preparing position…', thinking: 'AI is thinking…', yourTurn: 'Your turn', check: 'Check',
        win: 'You win!', loss: 'AI wins', draw: 'Draw', red: 'Red', black: 'Black', you: 'You',
        moves: 'moves', model: 'Self-trained model · iteration {iteration}', evaluation: 'AI position estimate',
        reason6: 'Repetition', reason7: '60-move rule', reason9: 'Insufficient material', reason5: 'Checkmate / stalemate',
        error: 'Loading or execution failed. Please retry.', board: 'Xiangqi board', empty: 'Empty', target: 'Legal destination'},
};
const NAMES = {1: ['', '車', '仕', '炮', '兵', '馬', '相', '帥'], '-1': ['', '車', '士', '砲', '卒', '馬', '象', '將']};
let lang = localStorage.getItem('skz_lang') === 'en' ? 'en' : 'zh';
let human = 1, selected = null, state = null, moves = [], actions = [];
let worker, manifest, ready = false, busy = false, thinking = false, error = '', lastSearch = null;
let requestId = 0, activeId = 0;
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
    const targets = new Set((state?.legal || []).filter(action => {
        const [x, y] = moves[action]; return y * 10 + x === selected;
    }).map(action => { const [, , x, y] = moves[action]; return y * 10 + x; }));
    const last = actions.length ? moves[actions.at(-1)] : null;
    const lastSquares = last ? [last[1] * 10 + last[0], last[3] * 10 + last[2]] : [];
    const disabled = !ready || busy || thinking || error || !state || state.finished || state.player !== human;
    let html = `<g class="grid">${grid}</g><text x="160" y="309" class="river">楚河</text><text x="390" y="309" class="river">汉界</text>`;
    for (let y = 0; y < 10; y++) for (let x = 0; x < 9; x++) {
        const square = y * 10 + x, piece = state?.board[square] || 0;
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
    let status = !ready ? t('loading') : busy ? t('preparing') : thinking ? t('thinking') : t('yourTurn');
    if (state?.finished) status = t(state.winner === 0 ? 'draw' : state.winner === human ? 'win' : 'loss');
    if (state?.check && !state.finished) status = `${t('check')} · ${status}`;
    if (error) status = t('error');
    $('status').textContent = status;
    $('status-pill').dataset.variant = error ? 'error' : !ready || busy || thinking ? 'thinking' : state?.finished ? 'done' : 'active';
    BoardGameAnalysis.render({ply: actions.length, result: lastSearch, t, coordinate: action => { const [x, y, u, v] = moves[action]; return `${'abcdefghi'[x]}${y} → ${'abcdefghi'[u]}${v}`; }});
    $('undo').disabled = !ready || actions.length <= (human === -1 ? 1 : 0);
    $('retry').hidden = !error;
    $('search-info').textContent = error || (state?.finished ? t('reason' + state.reason) : lastSearch
        ? `${t('evaluation')} ${Math.round((lastSearch.value + 1) * 50)}% · ${(lastSearch.elapsedMs / 1000).toFixed(1)} s` : '');
    $('your-side').textContent = `${t('you')} · ${t(human === 1 ? 'red' : 'black')}`;
    $('opponent-side').textContent = `AI · ${t(human === 1 ? 'black' : 'red')}`;
    $('model-detail').textContent = manifest ? t('model').replace('{iteration}', manifest.model.source.checkpoint.iteration) : '';
    const last = actions.length ? moves[actions.at(-1)] : null;
    $('last-move').textContent = last ? `${'abcdefghi'[last[0]]}${last[1]} → ${'abcdefghi'[last[2]]}${last[3]}` : '—';
    renderBoard();
}

function syncPosition() {
    selected = null;
    thinking = false;
    busy = ready;
    if (ready) {
        worker.postMessage({type: 'cancel'});
        activeId = ++requestId;
        worker.postMessage({id: activeId, type: 'position', actions: [...actions]});
    }
    render();
}

function newGame() {
    actions = [];
    BoardGameAnalysis.reset();
    state = null;
    lastSearch = null;
    if (ready) error = '';
    syncPosition();
}

async function boot() {
    worker?.terminate();
    ready = false;
    busy = false;
    thinking = false;
    error = '';
    render();
    try {
        const response = await fetch('/xiangqi/models/manifest.json?v=20261023');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        manifest = await response.json();
        $('rules-source').href = '/xiangqi/' + manifest.rules.source;
        worker = new Worker('/xiangqi/worker.js?v=20261102b');
        worker.onerror = event => { error = event.message; ready = false; busy = thinking = false; render(); };
        worker.onmessage = ({data}) => {
            if (data.id !== activeId) return;
            if (data.type === 'ready') {
                moves = data.moves;
                ready = true;
                syncPosition();
            } else if (data.type === 'position') {
                state = data.state;
                busy = false;
                if (!state.finished && state.player !== human) {
                    thinking = true;
                    lastSearch = null;
                    activeId = ++requestId;
                    worker.postMessage({id: activeId, type: 'search', actions: [...actions], thinkMs: Number($('budget').value)});
                }
                render();
            } else if (data.type === 'progress') {
                lastSearch = data.result;
                render();
            } else if (data.type === 'move') {
                lastSearch = data.result;
                BoardGameAnalysis.record(actions.length + 1, lastSearch);
                actions.push(data.result.action);
                syncPosition();
            } else if (data.type === 'error') {
                error = data.message;
                busy = thinking = false;
                render();
            }
        };
        activeId = ++requestId;
        worker.postMessage({id: activeId, type: 'init', manifest});
    } catch (problem) { error = problem.message; render(); }
}

function choose(square) {
    if (!ready || busy || thinking || error || !state || state.finished || state.player !== human) return;
    const action = state.legal.find(action => {
        const [x, y, u, v] = moves[action]; return y * 10 + x === selected && v * 10 + u === square;
    });
    if (action !== undefined) {
        actions.push(action);
        syncPosition();
    } else {
        selected = state.board[square] * human > 0 && square !== selected ? square : null;
        renderBoard();
    }
}

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
$('new-game').onclick = newGame;
$('undo').onclick = () => {
    if (!ready || actions.length <= (human === -1 ? 1 : 0)) return;
    const remove = (actions.length % 2 ? -1 : 1) === human ? 2 : 1;
    actions.splice(-remove);
    BoardGameAnalysis.trim(actions.length);
    lastSearch = null;
    error = '';
    syncPosition();
};
$('side').onchange = () => { human = Number($('side').value); newGame(); };
$('budget').onchange = () => { if (thinking) syncPosition(); };
$('retry').onclick = boot;
$('language').onclick = () => { lang = lang === 'zh' ? 'en' : 'zh'; localStorage.setItem('skz_lang', lang); render(); };
function applyTheme() {
    const mode = $('theme').value;
    localStorage.setItem('skz_theme', mode);
    document.documentElement.dataset.theme = mode === 'auto' ? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light') : mode;
}
$('theme').value = localStorage.getItem('skz_theme') || 'auto';
$('theme').onchange = applyTheme;
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if ($('theme').value === 'auto') applyTheme(); });
boot();
